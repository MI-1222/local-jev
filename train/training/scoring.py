"""厳密適格スコアリング規則(Strictly Proper Scoring Rules)モジュール。

モデルが出力する予測確率分布に対して過信や過小評価を抑制し、
真の事後確率への較正性(Calibration)を保証するための数理関数および損失層を提供する。

実装規則:
1. 有界対数スコア ($S_{\\log}$ / $\\tilde{S}_{\\log}$): 下限 $-10.0$ の有界クランプ。
2. 球面スコア ($S_{\\text{sph}}$): $L_2$ 正規化による大域的有界性保証。
3. Brier スコア ($S_{\\text{brier}}$): 較正誤差評価用損失および RLCD 報酬。
4. 順位確率スコア ($S_{\\text{rps}}$): Score 順序尺度専用の離散 Wasserstein-1 距離。
5. 統括エンジン (`ProperScoringEngine`): プリミティブ別動的ルーティング (Choice/Noul vs Score)。
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.schema import QuestionType
from training.loss import DEFAULT_MASK_VALUE
from training.scoring_config import ScoringConfig


def get_normalized_probabilities(
    logits: Tensor,
    op_mask: Tensor,
    temperature: float = 1.0,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> Tensor:
    """有効候補マスクと温度を適用して正規化された Softmax 確率分布を算出する。

    内部ロジック:
    1. 温度パラメータ tau でロジットを除算する。
    2. パディング領域 (~op_mask) に負のマスキング値を代入して Softmax を適用する。
    3. 無効候補の確率質量を厳密にゼロにクリップして返却する。

    Args:
        logits (Tensor): 未マスクのロジットテンソル `[batch_size, num_options]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        temperature (float): 温度パラメータ tau。
        mask_value (float): 無効候補に代入する負の値。

    Returns:
        Tensor: 正規化された確率分布 `[batch_size, num_options]`。
    """
    temp = max(temperature, 1e-4)
    scaled_logits = logits / temp
    masked_logits = scaled_logits.masked_fill(~op_mask, mask_value)
    probs = F.softmax(masked_logits, dim=-1)
    return probs * op_mask.float()


def compute_bounded_log_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    min_log_score: float = -10.0,
    eps: float = 1e-6,
    normalize: bool = True,
) -> Tensor:
    """有界対数スコア (Bounded Logarithmic Score) を算出する。

    数理仕様:
    正解ラベルインデックス $y$ に対する確率 $p_y$ を下限値 $\\max(\\epsilon, e^{\\text{min\\_log\\_score}})$
    でクランプし、対数スコアの発散 ($-\\infty$) および勾配爆発を防止する。
    $$S_{\\log} = \\max(\\ln(p_y), \\text{min\\_log\\_score})$$
    normalize=True の場合、$[0, 1]$ 区間に線形スケーリングする:
    $$\\tilde{S}_{\\log} = 1.0 - \\frac{S_{\\log}}{\\text{min\\_log\\_score}}$$
    完全正解時 ($p_y = 1$) は $\\tilde{S}_{\\log} = 1.0$、
    完全誤信時 ($p_y \\le e^{-10.0}$) は $\\tilde{S}_{\\log} = 0.0$ となる。

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        min_log_score (float): 対数スコアの下限値 (デフォルト: -10.0)。
        eps (float): 確率クリッピング補助微小値。
        normalize (bool): [0, 1] 区間に正規化するかどうかのフラグ。

    Returns:
        Tensor: サンプルごとのスコア `[batch_size]`。
    """
    effective_floor = max(eps, math.exp(min_log_score))
    labels_clamped = labels.clamp(min=0, max=probs.size(-1) - 1)
    p_y = probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    clamped_p = torch.clamp(p_y, min=effective_floor, max=1.0)
    log_p = torch.clamp(torch.log(clamped_p), min=min_log_score, max=0.0)

    if normalize:
        log_floor = math.log(effective_floor)
        scaled_score = 1.0 - (log_p / log_floor)
        return torch.clamp(scaled_score, min=0.0, max=1.0)

    return log_p


def compute_spherical_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    eps_div: float = 1e-12,
) -> Tensor:
    """球面スコア (Spherical Score) を算出する。

    数理仕様:
    予測確率ベクトルの $L_2$ ノルムで正規化された正解確率を算出する。
    値域は常に $[0, 1]$ に収まり、急峻な勾配発散を起こさない安定したアンカーとして機能する。
    $$S_{\\text{sph}} = \\frac{p_y}{\\sqrt{\\sum_{k \\in \\text{valid}} p_k^2 + \\epsilon_{\\text{div}}}}$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        eps_div (float): ゼロ除算防止微小値。

    Returns:
        Tensor: サンプルごとのスコア `[batch_size]`。
    """
    probs_valid = probs * op_mask.float()
    norm_sq = torch.sum(probs_valid**2, dim=-1)
    norm = torch.sqrt(norm_sq + eps_div)

    labels_clamped = labels.clamp(min=0, max=probs.size(-1) - 1)
    p_y = probs_valid.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    score = p_y / norm
    return torch.clamp(score, min=0.0, max=1.0)


def compute_brier_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
) -> Tensor:
    """Brier スコア損失 (Quadratic Loss) を算出する。

    数理仕様:
    予測確率ベクトル $p$ と真の正解ワンホットベクトル $y^*$ との平均二乗誤差 (MSE) を算出する。
    値域は $[0, 2]$ であり、完全正解時に $0.0$ (最良)、完全誤答時に $2.0$ (最悪) となる。
    $$\\mathcal{L}_{\\text{brier}} = \\sum_{k \\in \\text{valid}} (p_k - \\mathbf{1}(k = y^*))^2$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。

    Returns:
        Tensor: サンプルごとの Brier 損失 `[batch_size]`。
    """
    probs_valid = probs * op_mask.float()
    num_options = probs.size(-1)
    labels_clamped = labels.clamp(min=0, max=num_options - 1)
    targets = F.one_hot(labels_clamped, num_classes=num_options).float()
    targets_valid = targets * op_mask.float()

    diff_sq = (probs_valid - targets_valid) ** 2
    return torch.sum(diff_sq, dim=-1)


def compute_brier_reward(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    normalize: bool = False,
) -> Tensor:
    """Brier 報酬 (Quadratic Reward) を算出する。

    数理仕様:
    RLCD 強化学習ポリシー更新に適した報酬形式 (大きいほど望ましい) で算出する。
    厳密適格スコアの二次規則:
    $$R_{\\text{brier}} = 2 p_{y^*} - \\sum_{k \\in \\text{valid}} p_k^2 - 1 = -\\mathcal{L}_{\\text{brier}}$$
    値域:
    - normalize=False の場合: $[-2.0, 0.0]$ (完全正解で $0.0$、完全誤答で $-2.0$)。
    - normalize=True の場合: $[0.0, 1.0]$ ($1.0 - 0.5 \\mathcal{L}_{\\text{brier}}$)。

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        normalize (bool): [0, 1] 区間に正規化するかどうかのフラグ。

    Returns:
        Tensor: サンプルごとの Brier 報酬 `[batch_size]`。
    """
    brier_loss = compute_brier_score(probs, labels, op_mask)
    raw_reward = -brier_loss
    if normalize:
        return torch.clamp(1.0 + 0.5 * raw_reward, min=0.0, max=1.0)
    return raw_reward


def compute_rps_loss(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    is_cdf: bool = False,
) -> Tensor:
    """順位確率スコア損失 (Ranked Probability Score Loss) を算出する。

    数理仕様:
    順序尺度 (Score 型) において、予測確率分布の累積分布関数 (CDF) $P_m$ と、
    真の累積ステップ関数 $Y_m = \\mathbf{1}(y \\le m)$ の二乗誤差累積を評価する。
    サンプルの有効候補数 $K_i$ を用いて厳密に正規化する。
    $$\\mathcal{L}_{\\text{rps}} = \\frac{1}{K_i - 1} \\sum_{m=0}^{K_i - 2} (P_m - Y_m)^2$$
    完全正解時 ($p_y = 1$) は $\\mathcal{L}_{\\text{rps}} = 0.0$ となり、
    正解から遠い誤答ほど二乗オーダーで重い損失が課される。

    Args:
        probs (Tensor): 正規化済み確率分布または累積確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        is_cdf (bool): probs が既に累積分布関数 (CDF) であるかどうかのフラグ。

    Returns:
        Tensor: サンプルごとの RPS 損失 `[batch_size]`。
    """
    device = probs.device
    num_options = probs.size(-1)

    if is_cdf:
        cum_probs = probs * op_mask.float()
    else:
        probs_valid = probs * op_mask.float()
        cum_probs = torch.cumsum(probs_valid, dim=-1)

    thresholds = torch.arange(num_options, device=device).unsqueeze(0)
    labels_expanded = labels.unsqueeze(-1)
    cum_targets = (labels_expanded <= thresholds).float()

    diff_sq = (cum_probs - cum_targets) ** 2

    k_counts = op_mask.sum(dim=-1)
    step_mask = (thresholds < (k_counts.unsqueeze(-1) - 1)) & (
        k_counts.unsqueeze(-1) > 1
    )

    rps_sum = torch.sum(diff_sq * step_mask.float(), dim=-1)
    denominator = torch.clamp((k_counts - 1).float(), min=1.0)
    rps_loss = rps_sum / denominator

    single_choice_mask = k_counts <= 1
    return torch.where(
        single_choice_mask,
        torch.zeros_like(rps_loss),
        torch.clamp(rps_loss, min=0.0, max=1.0),
    )


def compute_rps_reward(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    is_cdf: bool = False,
) -> Tensor:
    """順位確率スコア報酬 (RPS Reward) を算出する。

    数理仕様:
    RLCD 強化学習ポリシー更新に適した報酬形式 (大きいほど望ましい) として、
    $$R_{\\text{rps}} = 1.0 - \\mathcal{L}_{\\text{rps}}$$
    を算出する。完全正解時は $1.0$、最悪逆転時は $0.0$ となる。

    Args:
        probs (Tensor): 正規化済み確率分布または累積確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        is_cdf (bool): probs が既に累積分布関数 (CDF) であるかどうかのフラグ。

    Returns:
        Tensor: サンプルごとの RPS 報酬 `[batch_size]`。
    """
    rps_loss = compute_rps_loss(probs, labels, op_mask, is_cdf=is_cdf)
    return torch.clamp(1.0 - rps_loss, min=0.0, max=1.0)


def compute_ranked_probability_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    is_cdf: bool = False,
) -> Tensor:
    """順位確率スコア報酬 (Ranked Probability Score: RPS) を算出する。

    後方互換性維持のための委譲ラッパー関数。
    内部で `compute_rps_reward` を呼び出して $[0, 1]$ の報酬スコアを返却する。

    Args:
        probs (Tensor): 正規化済み確率分布または累積確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        is_cdf (bool): probs が既に累積分布関数 (CDF) であるかどうかのフラグ。

    Returns:
        Tensor: サンプルごとのスコア `[batch_size]`。
    """
    return compute_rps_reward(probs, labels, op_mask, is_cdf=is_cdf)


class ProperScoringEngine:
    """厳密適格スコアリング規則に基づく報酬および損失計算統括エンジン。

    決定モデルの出力ロジットまたは確率分布を受け取り、
    質問プリミティブ (Choice, Score, Noul) に適合した厳密適格スコアを動的にディスパッチする。

    Attributes:
        config (ScoringConfig): スコアリングハイパーパラメータ設定。
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """統括エンジンを初期化する。

        Args:
            config (ScoringConfig | None): スコアリング設定。
        """
        self.config = config or ScoringConfig()

    def compute_reward(
        self,
        probs: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: Sequence[str | QuestionType] | None = None,
    ) -> dict[str, Tensor]:
        """各決定プリミティブに応じた複合適格報酬を算出する。

        ルーティング仕様:
        - Choice: alpha * S_log + (1 - alpha) * S_sph (brier_weight > 0 の場合は Brier もブレンド)
        - Score: beta * S_rps + (1 - beta) * S_log
        - Noul: brier_weight > 0 の場合は Brier 報酬、それ以外は alpha * S_log + (1 - alpha) * S_sph

        Args:
            probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。

        Returns:
            dict[str, Tensor]: composite, s_log, s_sph, s_rps, s_brier を格納した辞書。
        """
        cfg = self.config
        s_log = compute_bounded_log_score(
            probs=probs,
            labels=labels,
            op_mask=op_mask,
            min_log_score=cfg.min_log_score,
            eps=cfg.eps,
            normalize=cfg.normalize_log,
        )
        s_sph = compute_spherical_score(
            probs=probs,
            labels=labels,
            op_mask=op_mask,
            eps_div=cfg.eps_div,
        )
        s_rps = compute_rps_reward(
            probs=probs,
            labels=labels,
            op_mask=op_mask,
        )
        s_brier = compute_brier_reward(
            probs=probs,
            labels=labels,
            op_mask=op_mask,
            normalize=True,
        )

        batch_size = probs.size(0)
        device = probs.device

        if question_types is None:
            if cfg.brier_weight > 0.0:
                base_choice = cfg.alpha * s_log + (1.0 - cfg.alpha) * s_sph
                composite = (
                    1.0 - cfg.brier_weight
                ) * base_choice + cfg.brier_weight * s_brier
            else:
                composite = cfg.alpha * s_log + (1.0 - cfg.alpha) * s_sph
        else:
            composite_list: list[Tensor] = []
            for i in range(batch_size):
                q_type = question_types[i]
                if isinstance(q_type, QuestionType):
                    type_val = q_type.value
                else:
                    type_val = str(q_type).lower()

                if type_val == QuestionType.SCORE.value:
                    c_i = cfg.beta * s_rps[i] + (1.0 - cfg.beta) * s_log[i]
                elif type_val == QuestionType.NOUL.value:
                    if cfg.brier_weight > 0.0:
                        c_i = (
                            cfg.brier_weight * s_brier[i]
                            + (1.0 - cfg.brier_weight) * s_sph[i]
                        )
                    else:
                        c_i = cfg.alpha * s_log[i] + (1.0 - cfg.alpha) * s_sph[i]
                else:
                    base_choice = cfg.alpha * s_log[i] + (1.0 - cfg.alpha) * s_sph[i]
                    if cfg.brier_weight > 0.0:
                        c_i = (
                            1.0 - cfg.brier_weight
                        ) * base_choice + cfg.brier_weight * s_brier[i]
                    else:
                        c_i = base_choice
                composite_list.append(c_i)

            composite = torch.stack(composite_list).to(device)

        return {
            "composite": composite,
            "s_log": s_log,
            "s_sph": s_sph,
            "s_rps": s_rps,
            "s_brier": s_brier,
        }

    def compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: Sequence[str | QuestionType] | None = None,
    ) -> Tensor:
        """候補ロジットから正規化確率を算出し、厳密適格損失 (1.0 - mean(composite)) を導出する。

        Args:
            logits (Tensor): 候補ロジット `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        probs = get_normalized_probabilities(
            logits=logits,
            op_mask=op_mask,
            temperature=self.config.temperature,
        )
        rewards = self.compute_reward(
            probs=probs,
            labels=labels,
            op_mask=op_mask,
            question_types=question_types,
        )
        return 1.0 - rewards["composite"].mean()


def compute_composite_scores(
    logits: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    question_types: Sequence[str | QuestionType] | None = None,
    config: ScoringConfig | None = None,
) -> dict[str, Tensor]:
    """質問タイプ別の動的ルーティングを適用して複合スコアを算出する。

    内部ロジック:
    1. 温度付き Softmax で正規化確率分布を算出する。
    2. ProperScoringEngine を用いて適格スコア群を一括計算する。
    3. composite, s_log, s_sph, s_rps, s_brier, probs を格納した辞書を返却する。

    Args:
        logits (Tensor): 候補ロジットテンソル `[batch_size, num_options]`。
        labels (Tensor): 正解インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。
        config (ScoringConfig | None): スコアリング設定。None の場合はデフォルト設定。

    Returns:
        dict[str, Tensor]: composite, s_log, s_sph, s_rps, s_brier, probs を格納した辞書。
    """
    cfg = config or ScoringConfig()
    engine = ProperScoringEngine(cfg)
    probs = get_normalized_probabilities(
        logits=logits,
        op_mask=op_mask,
        temperature=cfg.temperature,
    )
    reward_dict = engine.compute_reward(
        probs=probs,
        labels=labels,
        op_mask=op_mask,
        question_types=question_types,
    )
    reward_dict["probs"] = probs
    return reward_dict


class ProperScoringLoss(nn.Module):
    """厳密適格スコアリング規則に基づく損失層。

    Attributes:
        engine (ProperScoringEngine): スコアリング統括エンジン。
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """損失層を初期化する。

        Args:
            config (ScoringConfig | None): スコアリング設定。
        """
        super().__init__()
        self.config = config or ScoringConfig()
        self.engine = ProperScoringEngine(self.config)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: Sequence[str | QuestionType] | None = None,
    ) -> Tensor:
        """複合スコアから損失 (1.0 - mean(composite)) を算出する。

        Args:
            logits (Tensor): 候補ロジット `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        return self.engine.compute_loss(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            question_types=question_types,
        )


class ProperScoringEvaluator:
    """厳密適格スコアの多面的集計と較正性評価を行うトラッカークラス。

    Attributes:
        config (ScoringConfig): 評価設定。
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """評価トラッカーを初期化する。

        Args:
            config (ScoringConfig | None): 評価設定。
        """
        self.config = config or ScoringConfig()
        self.reset()

    def reset(self) -> None:
        """集計カウンタを初期状態にリセットする。"""
        self.total_samples = 0
        self.sum_composite = 0.0
        self.sum_log = 0.0
        self.sum_sph = 0.0
        self.sum_rps = 0.0
        self.sum_brier = 0.0

        self.type_stats: dict[str, dict[str, float]] = {
            QuestionType.CHOICE.value: {
                "count": 0,
                "composite": 0.0,
                "log": 0.0,
                "sph": 0.0,
            },
            QuestionType.SCORE.value: {
                "count": 0,
                "composite": 0.0,
                "log": 0.0,
                "rps": 0.0,
            },
            QuestionType.NOUL.value: {
                "count": 0,
                "composite": 0.0,
                "log": 0.0,
                "sph": 0.0,
            },
        }

        self.confidence_tiers: dict[str, dict[str, float]] = {
            "high (>=0.85)": {"count": 0, "composite": 0.0, "accuracy_sum": 0.0},
            "mid (0.50~0.85)": {"count": 0, "composite": 0.0, "accuracy_sum": 0.0},
            "low (<0.50)": {"count": 0, "composite": 0.0, "accuracy_sum": 0.0},
        }

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: Sequence[str | QuestionType] | None = None,
    ) -> None:
        """単一バッチの予測スコアを集計する。

        Args:
            logits (Tensor): 候補ロジット `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。
        """
        results = compute_composite_scores(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            question_types=question_types,
            config=self.config,
        )

        batch_size = logits.size(0)
        composite = results["composite"].detach().cpu().tolist()
        s_log = results["s_log"].detach().cpu().tolist()
        s_sph = results["s_sph"].detach().cpu().tolist()
        s_rps = results["s_rps"].detach().cpu().tolist()
        s_brier = results["s_brier"].detach().cpu().tolist()
        probs = results["probs"].detach().cpu()

        preds = logits.masked_fill(~op_mask, DEFAULT_MASK_VALUE).argmax(dim=-1).cpu()
        labels_cpu = labels.cpu()

        self.total_samples += batch_size
        self.sum_composite += sum(composite)
        self.sum_log += sum(s_log)
        self.sum_sph += sum(s_sph)
        self.sum_rps += sum(s_rps)
        self.sum_brier += sum(s_brier)

        for i in range(batch_size):
            comp_val = composite[i]
            log_val = s_log[i]
            sph_val = s_sph[i]
            rps_val = s_rps[i]
            is_correct = float(preds[i] == labels_cpu[i])

            conf_val = float(probs[i].max().item())
            if conf_val >= 0.85:
                tier = "high (>=0.85)"
            elif conf_val >= 0.50:
                tier = "mid (0.50~0.85)"
            else:
                tier = "low (<0.50)"

            self.confidence_tiers[tier]["count"] += 1
            self.confidence_tiers[tier]["composite"] += comp_val
            self.confidence_tiers[tier]["accuracy_sum"] += is_correct

            if question_types is not None:
                qt = question_types[i]
                type_str = qt.value if isinstance(qt, QuestionType) else str(qt).lower()
                if type_str in self.type_stats:
                    self.type_stats[type_str]["count"] += 1
                    self.type_stats[type_str]["composite"] += comp_val
                    self.type_stats[type_str]["log"] += log_val
                    if "sph" in self.type_stats[type_str]:
                        self.type_stats[type_str]["sph"] += sph_val
                    if "rps" in self.type_stats[type_str]:
                        self.type_stats[type_str]["rps"] += rps_val

    def compute(self) -> dict[str, Any]:
        """累積集計結果から各種平均指標を算出する。

        Returns:
            dict[str, Any]: 全体・タイプ別・確信度別のスコア平均値。
        """
        if self.total_samples == 0:
            return {
                "total_samples": 0,
                "mean_composite": 0.0,
                "mean_log": 0.0,
                "mean_sph": 0.0,
                "mean_rps": 0.0,
                "mean_brier": 0.0,
                "by_type": {},
                "by_confidence": {},
            }

        n = float(self.total_samples)
        by_type_res: dict[str, dict[str, float]] = {}
        for t_name, stat in self.type_stats.items():
            cnt = stat["count"]
            if cnt > 0:
                by_type_res[t_name] = {
                    "count": cnt,
                    "mean_composite": round(stat["composite"] / cnt, 4),
                    "mean_log": round(stat["log"] / cnt, 4),
                }
                if "sph" in stat:
                    by_type_res[t_name]["mean_sph"] = round(stat["sph"] / cnt, 4)
                if "rps" in stat:
                    by_type_res[t_name]["mean_rps"] = round(stat["rps"] / cnt, 4)

        by_conf_res: dict[str, dict[str, float]] = {}
        for c_tier, c_stat in self.confidence_tiers.items():
            c_cnt = c_stat["count"]
            if c_cnt > 0:
                by_conf_res[c_tier] = {
                    "count": c_cnt,
                    "ratio": round(c_cnt / n, 4),
                    "mean_composite": round(c_stat["composite"] / c_cnt, 4),
                    "accuracy": round(c_stat["accuracy_sum"] / c_cnt, 4),
                }

        return {
            "total_samples": self.total_samples,
            "mean_composite": round(self.sum_composite / n, 4),
            "mean_log": round(self.sum_log / n, 4),
            "mean_sph": round(self.sum_sph / n, 4),
            "mean_rps": round(self.sum_rps / n, 4),
            "mean_brier": round(self.sum_brier / n, 4),
            "by_type": by_type_res,
            "by_confidence": by_conf_res,
        }
