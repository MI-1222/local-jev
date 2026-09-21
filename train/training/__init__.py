"""Jev SFT 学習パイプラインパッケージ。

設定、マスク付き損失関数、多面的評価指標、および Accelerate ベースの Trainer を提供する。
"""

from training.config import SFTConfig
from training.loss import MaskedCrossEntropyLoss, compute_masked_loss
from training.metrics import MetricsTracker
from training.rlcd_config import RLCDConfig
from training.rlcd_loss import (
    RLCDLoss,
    compute_entropy,
    compute_group_advantages,
    compute_masked_kl_divergence,
    sample_perturbed_logits,
)
from training.rlcd_trainer import (
    RLCDTrainer,
    compute_expected_calibration_error,
)
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
    "RLCDConfig",
    "RLCDLoss",
    "RLCDTrainer",
    "SFTConfig",
    "SFTDataset",
    "SFTTrainer",
    "ScoringConfig",
    "compute_bounded_log_score",
    "compute_composite_scores",
    "compute_entropy",
    "compute_expected_calibration_error",
    "compute_group_advantages",
    "compute_masked_kl_divergence",
    "compute_masked_loss",
    "compute_ranked_probability_score",
    "compute_spherical_score",
    "get_normalized_probabilities",
    "sample_perturbed_logits",
    "sft_collate_fn",
]
