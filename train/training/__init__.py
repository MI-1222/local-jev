"""Jev SFT 学習パイプラインパッケージ。

設定、マスク付き損失関数、多面的評価指標、および Accelerate ベースの Trainer を提供する。
"""

from training.config import SFTConfig
from training.loss import MaskedCrossEntropyLoss, compute_masked_loss
from training.metrics import MetricsTracker
from training.scoring import (
    ProperScoringEvaluator,
    ProperScoringLoss,
    compute_bounded_log_score,
    compute_composite_scores,
    compute_ranked_probability_score,
    compute_spherical_score,
    get_normalized_probabilities,
)
from training.scoring_config import ScoringConfig
from training.trainer import SFTDataset, SFTTrainer, sft_collate_fn

__all__ = [
    "MaskedCrossEntropyLoss",
    "MetricsTracker",
    "ProperScoringEvaluator",
    "ProperScoringLoss",
    "SFTConfig",
    "SFTDataset",
    "SFTTrainer",
    "ScoringConfig",
    "compute_bounded_log_score",
    "compute_composite_scores",
    "compute_masked_loss",
    "compute_ranked_probability_score",
    "compute_spherical_score",
    "get_normalized_probabilities",
    "sft_collate_fn",
]
