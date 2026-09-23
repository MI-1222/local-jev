"""RLCD (Reinforcement Learning from Calibrated Decisions) 損失関数およびサンプリングモジュール。

単一フォワードパス決定モデルに対するロジット摂動サンプリング、
厳密適格スコア複合報酬に基づくグループ内相対アドバンテージ算出 (GRPO)、
パディング考慮型 KL ペナルティ、およびサロゲートポリシー損失を提供する。
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.schema import QuestionType
from training.loss import DEFAULT_MASK_VALUE
from training.rlcd_config import RLCDConfig
from training.scoring import compute_composite_scores, get_normalized_probabilities


def sample_perturbed_logits(
    logits: Tensor,
    op_mask: Tensor,
    num_generations: int = 4,
    perturbation_std: float = 0.10,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> Tensor:
    """決定ロジット空間にガウシアン摂動を注入し、G 個の候補ロジットを生成する。

    内部ロジック:
    1. ロジットテンソル `[B, K]` を `[B, G, K]` に拡張する。
    2. 平均 0、標準偏差 perturbation_std の正規乱数ノイズを加算する。
    3. パディング無効候補領域 (`~op_mask`) に mask_value を厳密に再適用する。

    Args:
        logits (Tensor): ベース決定ロジット `[batch_size, num_options]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        num_generations (int): 生成する摂動系統数 G。
        perturbation_std (float): 注入するノイズの標準偏差 sigma。
        mask_value (float): 無効候補に代入する負のマスキング値。

    Returns:
        Tensor: 摂動ロジットテンソル `[batch_size, num_generations, num_options]`。
    """
    batch_size, num_options = logits.shape
    device = logits.device
    dtype = logits.dtype

    # [B, 1, K] -> [B, G, K]
    expanded_logits = logits.unsqueeze(1).expand(
        batch_size, num_generations, num_options
    )

    if perturbation_std > 0.0:
        noise = (
            torch.randn(
                (batch_size, num_generations, num_options),
                device=device,
                dtype=dtype,
            )
            * perturbation_std
        )
        perturbed_logits = expanded_logits + noise
    else:
        perturbed_logits = expanded_logits.clone()

    # パディング領域にノイズが乗ったまま Softmax に渡されるのを防ぐため再マスク
    mask_expanded = op_mask.unsqueeze(1).expand(
        batch_size, num_generations, num_options
    )
    return perturbed_logits.masked_fill(~mask_expanded, mask_value)


def compute_group_advantages(
    rewards: Tensor,
    eps_div: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """同一サンプル内の G 個の摂動報酬から相対アドバンテージを算出する。

    数理仕様:
    タスク間の報酬スケール差による歪みを防ぐため、バッチ全体ではなく
    必ず同一サンプル内 (dim=1) で平均 mu と標準偏差 sigma を計算する。
    $$A_{i, g} = \\frac{R_{i, g} - \\mu_i}{\\sigma_i + \\epsilon_{\\text{div}}}$$
    全摂動が同一報酬となり標準偏差がゼロ (sigma < eps_div) の場合は、
    勾配破壊を防ぐためアドバンテージをゼロテンソルとする。

    Args:
        rewards (Tensor): 摂動ごとの報酬テンソル `[batch_size, num_generations]`。
        eps_div (float): ゼロ除算防止微小値。

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - advantages: 標準化アドバンテージ `[batch_size, num_generations]`。
            - mean_reward: サンプルごとの平均報酬 `[batch_size, 1]`。
            - std_reward: サンプルごとの報酬標準偏差 `[batch_size, 1]`。
    """
    mean_reward = torch.mean(rewards, dim=-1, keepdim=True)
    std_reward = torch.std(rewards, dim=-1, keepdim=True, unbiased=False)

    diff = rewards - mean_reward
    advantages = diff / (std_reward + eps_div)

    # 分散ゼロ (全摂動が同等) のサンプルはアドバンテージをゼロとして安全にスキップ
    zero_var_mask = std_reward < eps_div
    advantages = torch.where(zero_var_mask, torch.zeros_like(advantages), advantages)

    return advantages, mean_reward, std_reward


def compute_masked_kl_divergence(
    p_theta: Tensor,
    p_ref: Tensor,
    op_mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """有効候補空間上のみで順方向 KL ダイバージェンス D_KL(p_theta || p_ref) を算出する。

    数理仕様:
    パディング位置による NaN/Inf を防ぐため、op_mask == True の候補のみを集約する。
    $$D_{\\text{KL}}(p_\\theta \\parallel p_{\\text{ref}}) = \\sum_{k \\in \\text{valid}} p_\\theta(k) \\ln \\left( \\frac{p_\\theta(k) + \\epsilon}{p_{\\text{ref}}(k) + \\epsilon} \\right)$$

    Args:
        p_theta (Tensor): 現在のポリシーの確率分布 `[batch_size, num_options]`。
        p_ref (Tensor): 凍結参照モデル (SFT) の確率分布 `[batch_size, num_options]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        eps (float): 対数計算時の微小値。

    Returns:
        Tensor: サンプルごとの KL ダイバージェンス `[batch_size]`。
    """
    p_theta_clamped = torch.clamp(p_theta, min=eps, max=1.0)
    p_ref_clamped = torch.clamp(p_ref, min=eps, max=1.0)

    # 有効候補位置での KL 要素
    kl_elements = p_theta * (torch.log(p_theta_clamped) - torch.log(p_ref_clamped))
    kl_masked = kl_elements * op_mask.float()
    return torch.sum(kl_masked, dim=-1)


def compute_entropy(
    probs: Tensor,
    op_mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """有効候補空間におけるシャノンエントロピー H(p) を算出する。

    数理仕様:
    確信度 100% 張り付き (過信) を防ぎ、適切な探索を維持するためのエントロピー。
    $$H(p) = -\\sum_{k \\in \\text{valid}} p(k) \\ln (p(k) + \\epsilon)$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[batch_size, num_options]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        eps (float): 対数計算時の微小値。

    Returns:
        Tensor: サンプルごとのエントロピー `[batch_size]`。
    """
    probs_clamped = torch.clamp(probs, min=eps, max=1.0)
    entropy_elements = -probs * torch.log(probs_clamped)
    entropy_masked = entropy_elements * op_mask.float()
    return torch.sum(entropy_masked, dim=-1)


def compute_listwise_dpo_loss(
    policy_logits: Tensor,
    ref_logits: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    beta: float = 0.10,
    temperature: float = 1.0,
    rankings: Tensor | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Plackett-Luce ランキング選好モデルに基づく Listwise DPO 損失を算出する。

    数理仕様:
    参照モデルのロジット z_ref と学習対象モデルのロジット z_theta から暗黙の報酬 (Implicit Reward) を定義する:
    $$r(x, c_k) = \\beta \\cdot (z_\\theta(x)_k - z_{\\text{ref}}(x)_k)$$
    サンプリング温度 tau によるスケーリング:
    $$\\hat{r}(x, c_k) = \\frac{r(x, c_k)}{\\tau}$$

    1. 正解ターゲット y* が最上位 (rank 1) かつその他が下位の標準選好設定 (rankings is None):
       有効候補集合 C_valid に対し、正解クラスへの Plackett-Luce 負の対数尤度を算出する:
       $$\\mathcal{L}_{\\text{DPO}} = -\\ln \\left( \\frac{\\exp(\\hat{r}_{y^*})}{\\sum_{k \\in C_{\\text{valid}}} \\exp(\\hat{r}_k)} \\right)$$
       これは暗黙報酬ベクトル \\hat{r} に対するマスク付き Cross-Entropy 損失と数理的に等価である。

    2. 明示的な完全順序選好リスト \\pi = [c_{\\pi(1)}, \\dots, c_{\\pi(M)}] が与えられた場合 (rankings is not None):
       再帰的な Plackett-Luce 分布に基づき、順位ごとの対数尤度を累積する:
       $$\\mathcal{L}_{\\text{Listwise-DPO}} = -\\sum_{i=1}^{M-1} \\ln \\left( \\frac{\\exp(\\hat{r}_{\\pi(i)})}{\\sum_{j=i}^M \\exp(\\hat{r}_{\\pi(j)})} \\right)$$

    Args:
        policy_logits (Tensor): 学習対象ポリシーの未マスクロジット `[batch_size, num_options]`。
        ref_logits (Tensor): 凍結参照モデルの未マスクロジット `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
        beta (float): 暗黙報酬のスケーリング係数 beta。
        temperature (float): スケーリング温度 tau。
        rankings (Tensor | None): 各サンプルの候補選好順位インデックス `[batch_size, num_ranks]`。
        mask_value (float): 無効候補に代入する負のマスキング値。

    Returns:
        tuple[Tensor, dict[str, Tensor]]:
            - loss: サンプル平均の Listwise DPO 損失スカラー。
            - metrics: モニタリング用の中間メトリクス辞書。
    """
    batch_size, num_options = policy_logits.shape
    device = policy_logits.device
    temp = max(temperature, 1e-4)

    # 暗黙の報酬テンソル r = beta * (z_theta - z_ref)
    implicit_reward = beta * (policy_logits - ref_logits.detach())
    scaled_reward = implicit_reward / temp
    masked_reward = scaled_reward.masked_fill(~op_mask, mask_value)

    labels_clamped = labels.clamp(min=0, max=num_options - 1)
    # 正解ターゲットの暗黙報酬
    r_target = implicit_reward.gather(
        dim=-1, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    if rankings is None:
        # 正解 y* を Top-1 とする Plackett-Luce 損失 (マスク付き Cross-Entropy)
        loss_dpo = F.cross_entropy(masked_reward, labels_clamped, reduction="mean")
    else:
        # 明示的な順位テンソルに基づく多段階 Plackett-Luce 損失
        sample_losses = []
        for b in range(batch_size):
            rank_list = rankings[b]
            valid_ranks = [idx.item() for idx in rank_list if op_mask[b, idx]]
            if len(valid_ranks) <= 1:
                sample_losses.append(torch.tensor(0.0, device=device))
                continue

            pl_loss = torch.tensor(0.0, device=device)
            for i in range(len(valid_ranks) - 1):
                numerator = scaled_reward[b, valid_ranks[i]]
                remaining_indices = torch.tensor(valid_ranks[i:], device=device)
                denominator = torch.logsumexp(
                    scaled_reward[b, remaining_indices], dim=0
                )
                pl_loss = pl_loss - (numerator - denominator)

            sample_losses.append(pl_loss / max(1, len(valid_ranks) - 1))
        loss_dpo = torch.stack(sample_losses).mean()

    # 負例候補の平均暗黙報酬の算出
    valid_mask_float = op_mask.float()
    target_onehot = F.one_hot(labels_clamped, num_classes=num_options).float()
    non_target_mask = (valid_mask_float - target_onehot).clamp(min=0.0)
    non_target_count = non_target_mask.sum(dim=-1).clamp(min=1.0)
    r_non_target = (implicit_reward * non_target_mask).sum(dim=-1) / non_target_count
    reward_margin = r_target - r_non_target

    metrics = {
        "loss_dpo": loss_dpo.detach(),
        "implicit_reward_mean": implicit_reward.mean().detach(),
        "implicit_reward_target": r_target.mean().detach(),
        "implicit_reward_non_target": r_non_target.mean().detach(),
        "reward_margin": reward_margin.mean().detach(),
    }
    return loss_dpo, metrics


class RLCDLoss(nn.Module):
    """RLCD (Reinforcement Learning from Calibrated Decisions) 損失層。

    GRPO (Group Relative Policy Optimization) によるオンラインロジット摂動方策更新、
    または Listwise DPO (Plackett-Luce モデル) によるオフライン選好最適化を提供する。

    Attributes:
        config (RLCDConfig): RLCD ハイパーパラメータ設定。
    """

    def __init__(self, config: RLCDConfig | None = None) -> None:
        """RLCD 損失層を初期化する。

        Args:
            config (RLCDConfig | None): RLCD 設定オブジェクト。
        """
        super().__init__()
        self.config = config or RLCDConfig()

    def forward(
        self,
        policy_logits: Tensor,
        ref_logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: Sequence[str | QuestionType] | None = None,
        rankings: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """RLCD ポリシー更新損失および詳細メトリクスを算出する。

        内部ロジック:
        1. config.optimization_mode に応じて "grpo" または "listwise_dpo" のパイプラインを実行する。
        2. "grpo" モード:
           - G 系統のロジット摂動テンソル z^(g) を生成し、無効候補マスクを再適用する。
           - 厳密適格スコア複合報酬 R_g を算出し、サンプル内標準化アドバンテージ A_g (ゼロ分散ガード付き) を導出する。
           - 正解ラベル確率比に対するクリップサロゲート損失を計算する。
        3. "listwise_dpo" モード:
           - 参照モデルとの暗黙報酬差分から Plackett-Luce ランキング選好損失を計算する。
        4. 有効候補マスクを考慮した KL ペナルティ D_KL(p_theta || p_ref) およびエントロピーボーナスを加算する。
        5. config.sft_aux_coeff > 0 の場合、決定境界崩壊を防ぐ補助分類損失を合成する。

        Args:
            policy_logits (Tensor): 学習対象ポリシーの未マスクロジット `[batch_size, num_options]`。
            ref_logits (Tensor): 凍結参照モデルの未マスクロジット `[batch_size, num_options]`。
            labels (Tensor): 正解候補インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。
            rankings (Tensor | None): 明示的な選好順序インデックス `[batch_size, num_ranks]`。

        Returns:
            tuple[Tensor, dict[str, Tensor]]:
                - total_loss: バックプロパゲーション用のスカラー損失。
                - metrics: モニタリング用の各種中間値テンソル辞書。
        """
        batch_size, num_options = policy_logits.shape
        num_gen = self.config.num_generations

        # 1. ベースポリシーおよび参照ポリシーの正規化確率分布
        p_theta = get_normalized_probabilities(
            logits=policy_logits,
            op_mask=op_mask,
            temperature=self.config.sampling_temperature,
        )
        p_ref = get_normalized_probabilities(
            logits=ref_logits,
            op_mask=op_mask,
            temperature=self.config.sampling_temperature,
        )

        labels_clamped = labels.clamp(min=0, max=num_options - 1)
        p_theta_y = p_theta.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(
            -1
        )

        metrics: dict[str, Tensor] = {
            "p_target_mean": p_theta_y.mean().detach(),
        }

        # 2. 最適化モード別の損失算出
        if self.config.optimization_mode == "listwise_dpo":
            loss_main, dpo_metrics = compute_listwise_dpo_loss(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                labels=labels,
                op_mask=op_mask,
                beta=self.config.dpo_beta,
                temperature=self.config.dpo_temperature,
                rankings=rankings,
            )
            metrics.update(dpo_metrics)
            metrics["loss_main"] = loss_main.detach()
        else:
            # "grpo" モード: ロジット摂動サンプリング [B, G, K]
            perturbed_logits = sample_perturbed_logits(
                logits=policy_logits,
                op_mask=op_mask,
                num_generations=num_gen,
                perturbation_std=self.config.perturbation_std,
            )

            op_mask_gen = op_mask.unsqueeze(1).expand(batch_size, num_gen, num_options)
            p_gen = get_normalized_probabilities(
                logits=perturbed_logits.reshape(-1, num_options),
                op_mask=op_mask_gen.reshape(-1, num_options),
                temperature=self.config.sampling_temperature,
            ).reshape(batch_size, num_gen, num_options)

            # 各摂動に対する報酬計算 (B * G を一括計算)
            flat_logits = perturbed_logits.reshape(-1, num_options)
            flat_labels = labels.unsqueeze(1).expand(batch_size, num_gen).reshape(-1)
            flat_mask = op_mask_gen.reshape(-1, num_options)

            flat_q_types: list[str | QuestionType] | None = None
            if question_types is not None:
                flat_q_types = []
                for qt in question_types:
                    flat_q_types.extend([qt] * num_gen)

            reward_dict = compute_composite_scores(
                logits=flat_logits,
                labels=flat_labels,
                op_mask=flat_mask,
                question_types=flat_q_types,
                config=self.config.scoring_config,
            )
            rewards = reward_dict["composite"].reshape(batch_size, num_gen)

            # グループ内相対アドバンテージ算出
            advantages, mean_reward, std_reward = compute_group_advantages(rewards)

            # クリップ付きサロゲート損失の算出 (正解ターゲットの確率比)
            labels_gen = (
                labels_clamped.unsqueeze(1).unsqueeze(-1).expand(batch_size, num_gen, 1)
            )
            p_gen_y = p_gen.gather(dim=-1, index=labels_gen).squeeze(-1)

            # 重要度比 r_g(y) = p_theta(y) / (p_gen(y) + eps)
            ratio = p_theta_y.unsqueeze(-1) / (p_gen_y.detach() + 1e-8)

            surr1 = ratio * advantages.detach()
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )
                * advantages.detach()
            )
            loss_main = -torch.mean(torch.min(surr1, surr2))

            metrics.update(
                {
                    "loss_policy": loss_main.detach(),
                    "loss_main": loss_main.detach(),
                    "mean_reward": mean_reward.mean().detach(),
                    "std_reward": std_reward.mean().detach(),
                    "mean_advantage": advantages.mean().detach(),
                }
            )

        # 3. パディング考慮型 KL ペナルティ
        kl_div = compute_masked_kl_divergence(
            p_theta=p_theta,
            p_ref=p_ref.detach(),
            op_mask=op_mask,
        )
        loss_kl = torch.mean(kl_div)

        # 4. エントロピーボーナス (過信・モード崩壊防止)
        entropy = compute_entropy(probs=p_theta, op_mask=op_mask)
        loss_entropy = -torch.mean(entropy)

        # 5. SFT 補助分類損失 (オプション)
        loss_sft = torch.tensor(0.0, device=policy_logits.device)
        if self.config.sft_aux_coeff > 0.0:
            masked_policy_logits = policy_logits.masked_fill(
                ~op_mask, DEFAULT_MASK_VALUE
            )
            loss_sft = F.cross_entropy(masked_policy_logits, labels_clamped)

        # 6. 総合損失の集約
        total_loss = (
            loss_main
            + self.config.kl_coeff * loss_kl
            + self.config.entropy_coeff * loss_entropy
            + self.config.sft_aux_coeff * loss_sft
        )

        metrics.update(
            {
                "loss_total": total_loss.detach(),
                "loss_kl": loss_kl.detach(),
                "loss_entropy": loss_entropy.detach(),
                "loss_sft_aux": loss_sft.detach(),
            }
        )

        return total_loss, metrics
