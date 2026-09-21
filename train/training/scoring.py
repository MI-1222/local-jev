"""厳密適格スコアリング規則(Strictly Proper Scoring Rules)モジュール。

モデルが出力する予測確率分布に対して、過信や過小評価を抑制し、
真の事後確率への較正性(Calibration)を保証するための数理関数および損失層を提供する。

実装規則:
1. 有界対数スコア ($S_{\\log}$ / $\\tilde{S}_{\\log}$)
2. 球面スコア ($S_{\\text{sph}}$)
3. 順位確率スコア ($S_{\\text{rps}}$: Score 順序尺度専用)
4. プリミティブ別動的ルーティング (Choice/Noul vs Score)
"""

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
    # パディング候補の確率を厳密にゼロにする
    return probs * op_mask.float()


def compute_bounded_log_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    eps: float = 1e-6,
    normalize: bool = True,
) -> Tensor:
    """有界対数スコア (Bounded Logarithmic Score) を算出する。

    数理仕様:
    正解ラベルインデックス $y$ に対する確率 $p_y$ を極小値 $\\epsilon$ でクランプし、
    対数スコアの発散 ($-\\infty$) を防止する。
    $$S_{\\log} = \\ln (\\max(p_y, \\epsilon))$$
    normalize=True の場合、$[0, 1]$ 区間に線形スケーリングする:
    $$\\tilde{S}_{\\log} = 1 - \\frac{\\ln(\\max(p_y, \\epsilon))}{\\ln \\epsilon}$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        eps (float): 下限クリッピング微小値。
        normalize (bool): [0, 1] 区間に正規化するかどうか。

    Returns:
        Tensor: サンプルごとのスコア `[batch_size]`。
    """
    labels_clamped = labels.clamp(min=0, max=probs.size(-1) - 1)
    p_y = probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    clamped_p = torch.clamp(p_y, min=eps, max=1.0)
    log_p = torch.log(clamped_p)

    if normalize:
        # ln(eps) は負の値のため、1.0 - log_p / ln(eps) で p=1 のとき 1.0, p=eps のとき 0.0
        log_eps = torch.log(torch.tensor(eps, device=probs.device, dtype=probs.dtype))
        scaled_score = 1.0 - (log_p / log_eps)
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


def compute_ranked_probability_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
) -> Tensor:
    """順位確率スコア (Ranked Probability Score: RPS) を算出する。

    数理仕様:
    順序尺度 (Score 型) において、予測確率分布の累積分布関数 (CDF) $P_m$ と、
    真の累積ステップ関数 $Y_m = \\mathbf{1}(y \\le m)$ の二乗誤差累積を評価する。
    サンプルの有効候補数 $K_i$ を用いて厳密に正規化する。
    $$S_{\\text{rps}} = 1 - \\frac{1}{K_i - 1} \\sum_{m=0}^{K_i - 2} (P_m - Y_m)^2$$

    完全正解時 ($p_y = 1$) は $S_{\\text{rps}} = 1.0$ となり、
    正解から近い誤答ほど高いスコアが与えられる。

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。

    Returns:
        Tensor: サンプルごとのスコア `[batch_size]`。
    """
    device = probs.device
    num_options = probs.size(-1)

    probs_valid = probs * op_mask.float()
    cum_probs = torch.cumsum(probs_valid, dim=-1)

    thresholds = torch.arange(num_options, device=device).unsqueeze(0)
    labels_expanded = labels.unsqueeze(-1)
    cum_targets = (labels_expanded <= thresholds).float()

    diff_sq = (cum_probs - cum_targets) ** 2

    # サンプルごとの有効候補数 K_i
    k_counts = op_mask.sum(dim=-1)

    # 有効評価ステップ m = 0 ... K_i - 2 (計 K_i - 1 ステップ) のマスク
    step_mask = (thresholds < (k_counts.unsqueeze(-1) - 1)) & (
        k_counts.unsqueeze(-1) > 1
    )

    rps_sum = torch.sum(diff_sq * step_mask.float(), dim=-1)
    denominator = torch.clamp((k_counts - 1).float(), min=1.0)

    rps_loss = rps_sum / denominator
    score = 1.0 - rps_loss

    # 有効候補数が 1 以下の場合はスコア 1.0
    single_choice_mask = k_counts <= 1
    score = torch.where(
        single_choice_mask,
        torch.ones_like(score),
        score,
    )

    return torch.clamp(score, min=0.0, max=1.0)


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
    2. 有界対数スコア、球面スコア、順位確率スコアを一括ベクトル計算する。
    3. 質問タイプに応じて複合スコアを集約する:
       - Choice / Noul: alpha * S_log + (1 - alpha) * S_sph
       - Score: beta * S_rps + (1 - beta) * S_log

    Args:
        logits (Tensor): 候補ロジットテンソル `[batch_size, num_options]`。
        labels (Tensor): 正解インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        question_types (list[str | QuestionType] | None): サンプルごとの質問タイプ。
        config (ScoringConfig | None): スコアリング設定。None の場合はデフォルト設定。

    Returns:
        dict[str, Tensor]: composite, s_log, s_sph, s_rps, probs を格納した辞書。
    """
    cfg = config or ScoringConfig()
    probs = get_normalized_probabilities(
        logits=logits,
        op_mask=op_mask,
        temperature=cfg.temperature,
    )

    s_log = compute_bounded_log_score(
        probs=probs,
        labels=labels,
        op_mask=op_mask,
        eps=cfg.eps,
        normalize=cfg.normalize_log,
    )
    s_sph = compute_spherical_score(
        probs=probs,
        labels=labels,
        op_mask=op_mask,
        eps_div=cfg.eps_div,
    )
    s_rps = compute_ranked_probability_score(
        probs=probs,
        labels=labels,
        op_mask=op_mask,
    )

    batch_size = logits.size(0)
    device = logits.device

    if question_types is None:
        # デフォルトは Choice 型ルーティング
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
            else:
                c_i = cfg.alpha * s_log[i] + (1.0 - cfg.alpha) * s_sph[i]
            composite_list.append(c_i)

        composite = torch.stack(composite_list).to(device)

    return {
        "composite": composite,
        "s_log": s_log,
        "s_sph": s_sph,
        "s_rps": s_rps,
        "probs": probs,
    }


class ProperScoringLoss(nn.Module):
    """厳密適格スコアリング規則に基づく損失層。

    Attributes:
        config (ScoringConfig): スコアリング設定。
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """損失層を初期化する。

        Args:
            config (ScoringConfig | None): スコアリング設定。
        """
        super().__init__()
        self.config = config or ScoringConfig()

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
        results = compute_composite_scores(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            question_types=question_types,
            config=self.config,
        )
        # スコア (高いほど良い) を損失 (低いほど良い) に変換
        return 1.0 - results["composite"].mean()


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

        # タイプ別集計
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

        # 確信度帯別集計 (High >= 0.85, Mid 0.50~0.85, Low < 0.50)
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
        probs = results["probs"].detach().cpu()

        preds = logits.masked_fill(~op_mask, DEFAULT_MASK_VALUE).argmax(dim=-1).cpu()
        labels_cpu = labels.cpu()

        self.total_samples += batch_size
        self.sum_composite += sum(composite)
        self.sum_log += sum(s_log)
        self.sum_sph += sum(s_sph)
        self.sum_rps += sum(s_rps)

        for i in range(batch_size):
            comp_val = composite[i]
            log_val = s_log[i]
            sph_val = s_sph[i]
            rps_val = s_rps[i]
            is_correct = float(preds[i] == labels_cpu[i])

            # 確信度 (最大予測確率)
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

            # 質問タイプ別
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
            "by_type": by_type_res,
            "by_confidence": by_conf_res,
        }
