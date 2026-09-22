"""Local-Jev 事後温度較正 (Post-hoc Temperature Calibration) パッケージ。

モデル推論結果に対する質問プリミティブ別・候補数バケット別の最適温度 tau* 探索、
期待較正誤差 (ECE) や負の対数尤度 (NLL) の多面的評価、
および Rust ランタイム (`local-jev-core`) 向け契約ファイル (`calibration.json`) の出力を提供する。
"""

from calibration.config import (
    CHOICE_BUCKETS,
    NOUL_BUCKET,
    SCORE_BUCKETS,
    CalibrationRunConfig,
    matches_bucket_expr,
)
from calibration.evaluator import (
    CalibrationEvaluator,
    compute_accuracy,
    compute_brier_score,
    compute_ece,
    compute_masked_nll,
    compute_reliability_diagram_data,
    get_masked_probabilities,
)
from calibration.optimizer import (
    LogitCache,
    TemperatureOptimizer,
    collect_env_metadata,
    compute_bucket_chance_level,
)

__all__ = [
    "CHOICE_BUCKETS",
    "NOUL_BUCKET",
    "SCORE_BUCKETS",
    "CalibrationEvaluator",
    "CalibrationRunConfig",
    "LogitCache",
    "TemperatureOptimizer",
    "collect_env_metadata",
    "compute_accuracy",
    "compute_brier_score",
    "compute_bucket_chance_level",
    "compute_ece",
    "compute_masked_nll",
    "compute_reliability_diagram_data",
    "get_masked_probabilities",
    "matches_bucket_expr",
]
