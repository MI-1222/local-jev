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
from calibration.diagram import (
    export_all_diagrams,
    plot_calibration_comparison,
    plot_reliability_diagram,
)
from calibration.evaluator import (
    CalibrationEvaluator,
    compute_accuracy,
    compute_accuracy_at_confidence,
    compute_adaptive_ece,
    compute_binary_diagram_data,
    compute_binary_ece,
    compute_brier_score,
    compute_ece,
    compute_expected_score_mae,
    compute_masked_nll,
    compute_reliability_diagram_data,
    compute_rps,
    compute_wilson_score_interval,
    get_masked_probabilities,
)
from calibration.optimizer import (
    LogitCache,
    TemperatureOptimizer,
    collect_env_metadata,
    compute_bucket_chance_level,
)
from calibration.reporter import (
    CalibrationReporter,
    evaluate_exit_criteria,
)

__all__ = [
    "CHOICE_BUCKETS",
    "NOUL_BUCKET",
    "SCORE_BUCKETS",
    "CalibrationEvaluator",
    "CalibrationReporter",
    "CalibrationRunConfig",
    "LogitCache",
    "TemperatureOptimizer",
    "collect_env_metadata",
    "compute_accuracy",
    "compute_accuracy_at_confidence",
    "compute_adaptive_ece",
    "compute_binary_diagram_data",
    "compute_binary_ece",
    "compute_brier_score",
    "compute_bucket_chance_level",
    "compute_ece",
    "compute_expected_score_mae",
    "compute_masked_nll",
    "compute_reliability_diagram_data",
    "compute_rps",
    "compute_wilson_score_interval",
    "evaluate_exit_criteria",
    "export_all_diagrams",
    "get_masked_probabilities",
    "matches_bucket_expr",
    "plot_calibration_comparison",
    "plot_reliability_diagram",
]
