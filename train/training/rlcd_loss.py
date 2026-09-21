"""RLCD (Reinforcement Learning from Calibrated Decisions) 損失関数およびサンプリングモジュール。

単一フォワードパス決定モデルに対するロジット摂動サンプリング、
厳密適格スコア複合報酬に基づくグループ内相対アドバンテージ算出 (GRPO)、
パディング考慮型 KL ペナルティ、およびサロゲートポリシー損失を提供する。
"""

from collections.abc import Sequence

import torch
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


class RLCDLoss(nn.Module):
    """RLCD (Reinforcement Learning from Calibrated Decisions) 損失層。

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
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """RLCD ポリシー更新損失および詳細メトリクスを算出する。

        内部ロジック:
        1. ベースモデルおよび参照モデルから温度付き正規化確率分布 p_theta, p_ref を導出する。
        2. G 系統のロジット摂動テンソル z^(g) を生成し、無効候補マスクを再適用する。
        3. 各摂動に対する確率分布 p^(g) を算出し、厳密適格スコア複合報酬 R_g を一括計算する。
        4. サンプル内の G 摂動間で平均・分散を求め、標準化アドバンテージ A_g を算出する。
        5. 正解ラベルに対する確率比 r_g(y) = p_theta(y) / p^(g)(y) を用い、PPO スタイルのクリップサロゲート損失を計算する。
        6. 有効候補マスクを考慮した KL ペナルティ D_KL(p_theta || p_ref) およびエントロピーを合算する。

        Args:
            policy_logits (Tensor): 学習対象ポリシーの未マスクロジット `[batch_size, num_options]`。
            ref_logits (Tensor): 凍結参照モデルの未マスクロジット `[batch_size, num_options]`。
            labels (Tensor): 正解候補インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (Sequence[str | QuestionType] | None): サンプルごとの質問タイプ。

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

        # 2. ロジット摂動サンプリング [B, G, K]
        perturbed_logits = sample_perturbed_logits(
            logits=policy_logits,
            op_mask=op_mask,
            num_generations=num_gen,
            perturbation_std=self.config.perturbation_std,
        )

        # 3. 摂動ロジットの Softmax 確率 [B, G, K]
        op_mask_gen = op_mask.unsqueeze(1).expand(batch_size, num_gen, num_options)
        p_gen = get_normalized_probabilities(
            logits=perturbed_logits.reshape(-1, num_options),
            op_mask=op_mask_gen.reshape(-1, num_options),
            temperature=self.config.sampling_temperature,
        ).reshape(batch_size, num_gen, num_options)

        # 4. 各摂動に対する報酬計算 (B * G を一括計算)
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

        # 5. グループ内相対アドバンテージ算出
        advantages, mean_reward, std_reward = compute_group_advantages(rewards)

        # 6. サロゲートポリシー損失の算出 (正解ターゲットの確率比)
        labels_clamped = labels.clamp(min=0, max=num_options - 1)
        # p_theta(y): [B, 1]
        p_theta_y = p_theta.gather(dim=-1, index=labels_clamped.unsqueeze(-1))
        # p_gen(y): [B, G]
        labels_gen = (
            labels_clamped.unsqueeze(1).unsqueeze(-1).expand(batch_size, num_gen, 1)
        )
        p_gen_y = p_gen.gather(dim=-1, index=labels_gen).squeeze(-1)

        # 重要度比 r_g(y) = p_theta(y) / (p_gen(y) + eps)
        ratio = p_theta_y / (p_gen_y.detach() + 1e-8)

        # クリップ付きサロゲート損失
        surr1 = ratio * advantages.detach()
        surr2 = (
            torch.clamp(
                ratio,
                1.0 - self.config.clip_range,
                1.0 + self.config.clip_range,
            )
            * advantages.detach()
        )
        loss_policy = -torch.mean(torch.min(surr1, surr2))

        # 7. パディング考慮型 KL ペナルティ
        kl_div = compute_masked_kl_divergence(
            p_theta=p_theta,
            p_ref=p_ref.detach(),
            op_mask=op_mask,
        )
        loss_kl = torch.mean(kl_div)

        # 8. エントロピーボーナス (過信・モード崩壊防止)
        entropy = compute_entropy(probs=p_theta, op_mask=op_mask)
        loss_entropy = -torch.mean(entropy)

        # 9. 総合損失の集約
        total_loss = (
            loss_policy
            + self.config.kl_coeff * loss_kl
            + self.config.entropy_coeff * loss_entropy
        )

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_policy": loss_policy.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_entropy": loss_entropy.detach(),
            "mean_reward": mean_reward.mean().detach(),
            "std_reward": std_reward.mean().detach(),
            "mean_advantage": advantages.mean().detach(),
            "p_target_mean": p_theta_y.mean().detach(),
        }

        return total_loss, metrics
