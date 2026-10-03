"""粗密マルチタスク複合損失および階層整合性正則化モジュール。

大分類判定損失 (L_coarse)、細分類判定損失 (L_fine)、および
大分類と細分類の予測確率・ロジット整合性を担保する階層整合性損失 (L_consistency) を統合し、
大分類の勾配逆流による細分類精度の共倒れ (Gradient Collapse) を防ぐ
ストップグラディエント (detach) 機構を実装する。
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from training.loss import DEFAULT_MASK_VALUE, LabelSmoothedFocalLoss


class HierarchicalConsistencyLoss(nn.Module):
    """大分類と細分類の予測整合性を強制する正則化損失層。

    数理仕様:
    1. 周辺化確率整合性 (Marginal Consistency):
       細分類 Softmax 確率 p_F を各スロットの親大分類 ID (fine_parent_indices) に基づき
       クラスタごとに集約した周辺化確率 \\tilde{p}_{C_j} と、大分類予測確率 p_{C_j} (detach 適用)
       の平均二乗誤差を算出する：
       $$\\mathcal{L}_{\\text{marginal}} = \\frac{1}{M} \\sum_{j=0}^{M-1} \\left( \\text{sg}(p_{C_j}) - \\sum_{k \\in \\mathcal{K}_j} p_{F_k} \\right)^2$$

    2. 非選択クラスタ抑制マージンペナルティ (Logit Suppression):
       正解親クラスタ以外のディストラクター候補 k に対し、大分類ロジット z_{C, parent(k)} (detach 適用)
       とのマージン超過ペナルティを課す：
       $$\\mathcal{L}_{\\text{suppression}} = \\frac{1}{|\\mathcal{K}_{\\text{neg}}|} \\sum_{k \\notin C^*} \\max(0, z_{F, k} - \\text{sg}(z_{C, \\text{parent}(k)}) + m)^2$$

    Attributes:
        margin (float): ディストラクター抑制用の安全マージン m。
        use_marginal (bool): 周辺化確率二乗誤差を損失に含めるかどうか。
        use_suppression (bool): ロジット抑制マージンペナルティを含めるかどうか。
        suppression_weight (float): ロジット抑制損失の内部結合係数。
    """

    def __init__(
        self,
        margin: float = 1.0,
        use_marginal: bool = True,
        use_suppression: bool = True,
        suppression_weight: float = 0.5,
    ) -> None:
        """整合性損失層を初期化する。

        Args:
            margin (float): ロジット抑制の安全マージン (デフォルト: 1.0)。
            use_marginal (bool): 周辺化確率整合性を適用するか (デフォルト: True)。
            use_suppression (bool): 非所属クラスタ候補抑制を適用するか (デフォルト: True)。
            suppression_weight (float): 抑制損失の重み係数 (デフォルト: 0.5)。
        """
        super().__init__()
        self.margin = margin
        self.use_marginal = use_marginal
        self.use_suppression = use_suppression
        self.suppression_weight = suppression_weight

    def forward(
        self,
        coarse_logits: Tensor,
        fine_logits: Tensor,
        coarse_op_mask: Tensor,
        fine_op_mask: Tensor,
        fine_parent_indices: Tensor,
        coarse_labels: Tensor | None = None,
        mask_value: float = DEFAULT_MASK_VALUE,
    ) -> Tensor:
        """階層整合性正則化損失を算出する。

        大分類ロジット・確率には detach() を適用し、細分類側のみにアラインメント勾配を流す
        非対称勾配フロー (Stop-Gradient) を徹底する。

        Args:
            coarse_logits (Tensor): 大分類ロジット `[batch_size, num_coarse_options]`。
            fine_logits (Tensor): 細分類ロジット `[batch_size, num_fine_options]`。
            coarse_op_mask (Tensor): 大分類の有効候補マスク `[batch_size, num_coarse_options]`。
            fine_op_mask (Tensor): 細分類の有効候補マスク `[batch_size, num_fine_options]`。
            fine_parent_indices (Tensor): 細分類候補各スロットの親大分類 ID `[batch_size, num_fine_options]`。
            coarse_labels (Tensor | None): 正解大分類インデックス `[batch_size]`。ロジット抑制時に使用。
            mask_value (float): パディング無効化用の負値。

        Returns:
            Tensor: スカラー整合性正則化損失テンソル。
        """
        device = coarse_logits.device
        batch_size = coarse_logits.size(0)

        # パディング領域をマスクした Softmax 確率の計算
        masked_coarse_logits = coarse_logits.masked_fill(~coarse_op_mask, mask_value)
        masked_fine_logits = fine_logits.masked_fill(~fine_op_mask, mask_value)

        p_coarse = F.softmax(masked_coarse_logits, dim=-1)
        p_fine = F.softmax(masked_fine_logits, dim=-1)

        # 勾配崩壊防止のストップグラディエント
        p_coarse_detached = p_coarse.detach()
        coarse_logits_detached = coarse_logits.detach()

        sample_losses: list[Tensor] = []

        for i in range(batch_size):
            num_c = int(coarse_op_mask[i].sum().item())
            num_f = int(fine_op_mask[i].sum().item())
            if num_c == 0 or num_f == 0:
                continue

            loss_components: list[Tensor] = []

            # 1. 周辺化確率整合性 (Marginal Consistency)
            if self.use_marginal:
                p_c_i = p_coarse_detached[i, :num_c]
                p_f_i = p_fine[i, :num_f]
                parent_idx_i = fine_parent_indices[i, :num_f]

                # 各大分類クラスタ配下の細分類確率の総和を算出
                summed_p_fine = torch.zeros(num_c, device=device, dtype=p_fine.dtype)
                for k in range(num_f):
                    pid = int(parent_idx_i[k].item())
                    if 0 <= pid < num_c:
                        summed_p_fine[pid] = summed_p_fine[pid] + p_f_i[k]

                # 親クラスタが存在するクラスタのみで MSE 損失
                valid_clusters = (summed_p_fine > 0.0) | (p_c_i > 0.0)
                if valid_clusters.any():
                    marginal_loss = F.mse_loss(
                        summed_p_fine[valid_clusters], p_c_i[valid_clusters]
                    )
                    loss_components.append(marginal_loss)

            # 2. 非選択クラスタ候補抑制 (Logit Suppression)
            if self.use_suppression and coarse_labels is not None:
                true_c = int(coarse_labels[i].item())
                parent_idx_i = fine_parent_indices[i, :num_f]
                z_f_i = fine_logits[i, :num_f]

                suppression_penalties: list[Tensor] = []
                for k in range(num_f):
                    pid = int(parent_idx_i[k].item())
                    # 正解クラスタ以外に属する候補のみペナルティ対象
                    if pid != true_c and 0 <= pid < num_c:
                        z_c_parent = coarse_logits_detached[i, pid]
                        diff = z_f_i[k] - z_c_parent + self.margin
                        penalty = torch.relu(diff).pow(2)
                        suppression_penalties.append(penalty)

                if suppression_penalties:
                    supp_loss = torch.stack(suppression_penalties).mean()
                    loss_components.append(self.suppression_weight * supp_loss)

            if loss_components:
                sample_losses.append(torch.stack(loss_components).sum())

        if not sample_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(sample_losses).mean()


class HierarchicalMultiTaskLoss(nn.Module):
    """大分類・細分類協調学習用マルチタスク複合損失層。

    目的関数:
    $$\\mathcal{L}_{\\text{multitask}} = \\mathcal{L}_{\\text{coarse}}(\\mathbf{z}_C, \\mathbf{y}_C) + \\beta \\cdot \\mathcal{L}_{\\text{fine}}(\\mathbf{z}_F, \\mathbf{y}_F) + \\gamma \\cdot \\mathcal{L}_{\\text{consistency}}$$

    Attributes:
        label_smoothing (float): Label Smoothing 係数 \\epsilon。
        focal_gamma (float): Focal 変調係数 \\gamma。
        beta (float): 細分類損失の重み \\beta。
        gamma (float): 階層整合性正則化の重み \\gamma。
        coarse_loss_fn (LabelSmoothedFocalLoss): 大分類用損失層。
        fine_loss_fn (LabelSmoothedFocalLoss): 細分類用損失層。
        consistency_loss_fn (HierarchicalConsistencyLoss): 階層整合性損失層。
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        focal_gamma: float = 0.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        consistency_margin: float = 1.0,
        use_marginal_consistency: bool = True,
        use_suppression_consistency: bool = True,
    ) -> None:
        """マルチタスク損失層を初期化する。

        Args:
            label_smoothing (float): ラベル平滑化係数 (デフォルト: 0.05)。
            focal_gamma (float): Focal Loss 変調係数 (デフォルト: 0.0)。
            beta (float): 細分類損失の結合重み (デフォルト: 1.0)。
            gamma (float): 整合性正則化の結合重み (デフォルト: 0.1)。
            consistency_margin (float): ロジット抑制の安全マージン (デフォルト: 1.0)。
            use_marginal_consistency (bool): 周辺化確率整合性を使用するか。
            use_suppression_consistency (bool): ロジット抑制整合性を使用するか。
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.beta = beta
        self.gamma = gamma

        self.coarse_loss_fn = LabelSmoothedFocalLoss(
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        self.fine_loss_fn = LabelSmoothedFocalLoss(
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        self.consistency_loss_fn = HierarchicalConsistencyLoss(
            margin=consistency_margin,
            use_marginal=use_marginal_consistency,
            use_suppression=use_suppression_consistency,
        )

    def set_consistency_weight(self, gamma: float) -> None:
        """整合性正則化重み gamma を動的に更新する (ウォームアップ用)。

        Args:
            gamma (float): 新しい重み係数。
        """
        self.gamma = gamma

    def forward(
        self,
        coarse_logits: Tensor,
        coarse_labels: Tensor,
        coarse_op_mask: Tensor,
        fine_logits: Tensor,
        fine_labels: Tensor,
        fine_op_mask: Tensor,
        fine_parent_indices: Tensor,
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """複合マルチタスク損失を算出する。

        Args:
            coarse_logits (Tensor): 大分類ロジット `[batch_size, num_coarse_options]`。
            coarse_labels (Tensor): 正解大分類インデックス `[batch_size]`。
            coarse_op_mask (Tensor): 大分類有効候補マスク `[batch_size, num_coarse_options]`。
            fine_logits (Tensor): 細分類ロジット `[batch_size, num_fine_options]`。
            fine_labels (Tensor): 正解細分類インデックス `[batch_size]`。
            fine_op_mask (Tensor): 細分類有効候補マスク `[batch_size, num_fine_options]`。
            fine_parent_indices (Tensor): 親大分類 ID テンソル `[batch_size, num_fine_options]`。
            return_dict (bool): メトリクス内訳辞書を併せて返却するかどうか。

        Returns:
            Tensor | tuple[Tensor, dict[str, float]]:
                - 通常時: スカラー総損失テンソル。
                - 辞書返却時: (total_loss, loss_dict) のタプル。
        """
        batch_size = coarse_logits.size(0)
        device = coarse_logits.device

        # 1. 大分類損失の計算 (各サンプルの有効候補スライス)
        coarse_losses: list[Tensor] = []
        for i in range(batch_size):
            num_c = int(coarse_op_mask[i].sum().item())
            if num_c > 0:
                l_c = coarse_logits[i, :num_c]
                t_c = coarse_labels[i]
                coarse_losses.append(self.coarse_loss_fn(l_c, t_c))

        loss_coarse = (
            torch.stack(coarse_losses).mean()
            if coarse_losses
            else torch.tensor(0.0, device=device)
        )

        # 2. 細分類損失の計算
        fine_losses: list[Tensor] = []
        for i in range(batch_size):
            num_f = int(fine_op_mask[i].sum().item())
            if num_f > 0:
                l_f = fine_logits[i, :num_f]
                t_f = fine_labels[i]
                fine_losses.append(self.fine_loss_fn(l_f, t_f))

        loss_fine = (
            torch.stack(fine_losses).mean()
            if fine_losses
            else torch.tensor(0.0, device=device)
        )

        # 3. 階層整合性正則化損失の計算
        loss_consistency = torch.tensor(0.0, device=device)
        if self.gamma > 0.0:
            loss_consistency = self.consistency_loss_fn(
                coarse_logits=coarse_logits,
                fine_logits=fine_logits,
                coarse_op_mask=coarse_op_mask,
                fine_op_mask=fine_op_mask,
                fine_parent_indices=fine_parent_indices,
                coarse_labels=coarse_labels,
            )

        total_loss = loss_coarse + self.beta * loss_fine + self.gamma * loss_consistency

        if not return_dict:
            return total_loss

        loss_dict: dict[str, float] = {
            "loss": float(total_loss.detach().item()),
            "loss_coarse": float(loss_coarse.detach().item()),
            "loss_fine": float(loss_fine.detach().item()),
            "loss_consistency": float(loss_consistency.detach().item()),
            "gamma": float(self.gamma),
        }

        return total_loss, loss_dict
