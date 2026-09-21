"""マスク考慮型クロスエントロピー損失モジュール。

可変候補数に対応するバッチにおいて、パディングされた無効候補のロジットを
マスキングし、有効な候補のみを分母とする正確な Softmax 確率と勾配を算出する。
"""

import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_MASK_VALUE = -1e4
"""無効候補に適用する負の無限大代替値。半精度アンダーフローや NaN を防止する。"""


def compute_masked_loss(
    logits: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> Tensor:
    """有効候補マスクを適用してクロスエントロピー損失を算出する。

    数理仕様:
    $$z_i^{\\text{masked}} = \\begin{cases} z_i & (\\text{op\\_mask}_i = \\text{True}) \\\\ \\text{mask\\_value} & (\\text{op\\_mask}_i = \\text{False}) \\end{cases}$$
    $$\\mathcal{L} = -\\log \\left( \\frac{\\exp(z_{\\text{label}}^{\\text{masked}})}{\\sum_k \\exp(z_k^{\\text{masked}})} \\right)$$

    Args:
        logits (Tensor): 未マスクのロジットテンソル `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]` (有効: True, 無効: False)。
        mask_value (float): 無効候補に代入する値(デフォルト: -1e4)。

    Returns:
        Tensor: スカラー損失テンソル。
    """
    masked_logits = logits.masked_fill(~op_mask, mask_value)
    return F.cross_entropy(masked_logits, labels)


class MaskedCrossEntropyLoss(nn.Module):
    """パディング候補を自動遮断するクロスエントロピー損失層。

    Attributes:
        mask_value (float): 無効候補をマスクする負の値。
    """

    def __init__(self, mask_value: float = DEFAULT_MASK_VALUE) -> None:
        """損失層を初期化する。

        Args:
            mask_value (float): 無効候補をマスクする負の値。
        """
        super().__init__()
        self.mask_value = mask_value

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
    ) -> Tensor:
        """有効候補マスクを適用して損失を算出する。

        Args:
            logits (Tensor): ロジットテンソル `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        return compute_masked_loss(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            mask_value=self.mask_value,
        )
