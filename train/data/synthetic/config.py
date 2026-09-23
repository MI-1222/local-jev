"""Evol-Instruct 合成データ生成パイプラインの設定モジュール。

タスク比率、モデル指定、バリデーション閾値、ファイルパス等のハイパーパラメータを管理する。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PrimitiveRatioConfig:
    """決定プリミティブおよび候補数の生成比率設定。

    Attributes:
        choice_ratio (float): Choice 型の全体比率 (デフォルト: 0.60)。
        score_ratio (float): Score 型の全体比率 (デフォルト: 0.20)。
        noul_ratio (float): Noul 型の全体比率 (デフォルト: 0.20)。
        choice_2_3_ratio (float): Choice 内での 2〜3 択の相対比率 (デフォルト: 0.20)。
        choice_4_6_ratio (float): Choice 内での 4〜6 択の相対比率 (デフォルト: 0.50)。
        choice_7_16_ratio (float): Choice 内での 7〜16 択の相対比率 (デフォルト: 0.30)。
        score_3_5_ratio (float): Score 内での 3〜5 段階の相対比率 (デフォルト: 0.75)。
        score_10_ratio (float): Score 内での 10 段階の相対比率 (デフォルト: 0.25)。
    """

    choice_ratio: float = 0.60
    score_ratio: float = 0.20
    noul_ratio: float = 0.20
    choice_2_3_ratio: float = 0.20
    choice_4_6_ratio: float = 0.50
    choice_7_16_ratio: float = 0.30
    score_3_5_ratio: float = 0.75
    score_10_ratio: float = 0.25

    def __post_init__(self) -> None:
        """比率の総和が 1.0 に近いことを検証する。"""
        total_primitive = self.choice_ratio + self.score_ratio + self.noul_ratio
        if not (0.99 <= total_primitive <= 1.01):
            raise ValueError(
                f"プリミティブ比率の総和は 1.0 である必要があります: {total_primitive}。"
            )


@dataclass(frozen=True)
class QualityFilterConfig:
    """品質ゲート用ハイパーパラメータ設定。

    Attributes:
        min_state_chars (int): State テキストの最小文字数 (デフォルト: 200)。
        max_state_chars (int): State テキストの最大文字数 (デフォルト: 800)。
        max_criteria_chars (int): 各 Criteria 説明文の最大文字数 (デフォルト: 80)。
        similarity_threshold (float): 意味的重複排除のコサイン類似度閾値 (デフォルト: 0.92)。
        require_hard_negative (bool): ハードネガティブの混入を必須とするか (デフォルト: True)。
    """

    min_state_chars: int = 200
    max_state_chars: int = 800
    max_criteria_chars: int = 80
    similarity_threshold: float = 0.92
    require_hard_negative: bool = True


@dataclass
class SyntheticPipelineConfig:
    """合成データ生成パイプライン全体の実行設定。

    Attributes:
        generator_model (str): 生成用フロンティアLLMモデル名。
        validator_model (str): クロスバリデーション用別系統LLMモデル名。
        primitive_ratio (PrimitiveRatioConfig): プリミティブ生成比率設定。
        quality_filter (QualityFilterConfig): 品質ゲート設定。
        output_dir (Path): 生成データ出力先ディレクトリ。
        total_target_samples (int): 目標生成有効サンプル数。
        max_concurrency (int): 非同期API同時実行数セマフォ。
        request_timeout_seconds (float): APIリクエストタイムアウト秒数。
        max_retries (int): API呼び出しの最大リトライ回数。
        seed (int): 乱数シード。
        dry_run (bool): APIを呼び出さずモック動作させるフラグ。
    """

    generator_model: str = "claude-3-7-sonnet-20250219"
    validator_model: str = "gpt-4o"
    primitive_ratio: PrimitiveRatioConfig = field(default_factory=PrimitiveRatioConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    output_dir: Path = field(
        default_factory=lambda: Path("train/data/synthetic/output")
    )
    total_target_samples: int = 1000
    max_concurrency: int = 5
    request_timeout_seconds: float = 60.0
    max_retries: int = 3
    seed: int = 42
    dry_run: bool = False
