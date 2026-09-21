"""Jev SFT 学習パイプラインパッケージ。

設定、マスク付き損失関数、多面的評価指標、および Accelerate ベースの Trainer を提供する。
"""

from training.config import SFTConfig
from training.loss import MaskedCrossEntropyLoss, compute_masked_loss
from training.metrics import MetricsTracker
from training.trainer import SFTDataset, SFTTrainer, sft_collate_fn

__all__ = [
    "MaskedCrossEntropyLoss",
    "MetricsTracker",
    "SFTConfig",
    "SFTDataset",
    "SFTTrainer",
    "compute_masked_loss",
    "sft_collate_fn",
]
