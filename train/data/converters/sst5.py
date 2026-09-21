"""SST-5 (Stanford Sentiment Treebank 5段階感情強度) データセットコンバータモジュール。

5段階の感情極性・強度データを、Jev 統一スキーマ (Score型: 順序尺度評価)
へ正規化・変換する。
"""

import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

# 5段階評価の Criteria 定義 (0オリジン昇順キーと自然言語意味論の付与)
SST5_CRITERIA: dict[str, str] = {
    "0": "非常に否定的・最悪の評価 (Very Negative)",
    "1": "やや否定的・不満な評価 (Negative)",
    "2": "中立的・可もなく不可もない評価 (Neutral)",
    "3": "やや肯定的・概ね良好な評価 (Positive)",
    "4": "非常に肯定的・最高水準の評価 (Very Positive)",
}

# 最小・最大段階および全段階数定義
SST5_SCALE_MIN: int = 0
SST5_SCALE_MAX: int = 4
SST5_NUM_LEVELS: int = 5


class SST5Converter(BaseDatasetConverter):
    """SST-5 データセットコンバータ。

    SetFit/sst5 コーパスを走査し、順序尺度 (Score型) の UnifiedSample へ変換する。
    推論時の加重平均期待値算出と整合するよう、キーは '0' 〜 '4' の昇順を厳密に維持する。

    Attributes:
        seed (int): 指示文サンプリング等に用いる乱数シード。
        max_samples_per_class (int | None): クラスごとの最大取得サンプル数 (クラス平準化用)。
    """

    def __init__(
        self,
        seed: int = 42,
        max_samples_per_class: int | None = None,
    ) -> None:
        """コンバータを初期化する。

        Args:
            seed (int): 乱数シード。
            max_samples_per_class (int | None): クラスごとの取得上限数。
                None の場合は全件抽出する。
        """
        super().__init__(seed=seed)
        self.max_samples_per_class = max_samples_per_class

    def convert_dataset(
        self,
        dataset: Dataset,
        split: str,
    ) -> Iterator[UnifiedSample]:
        """与えられた Dataset を走査して UnifiedSample (Score型) を順次生成する。

        Args:
            dataset (Dataset): 対象の Dataset オブジェクト。
            split (str): スプリット名 ('train', 'validation', 'test' 等)。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        rng = random.Random(self.seed)
        class_counts: dict[int, int] = defaultdict(int)

        for idx, row in enumerate(dataset):
            raw_label = row["label"]
            label_id = int(raw_label)

            if not (SST5_SCALE_MIN <= label_id <= SST5_SCALE_MAX):
                continue

            if self.max_samples_per_class is not None:
                if class_counts[label_id] >= self.max_samples_per_class:
                    continue
                class_counts[label_id] += 1

            text = row["text"]
            target_key = str(label_id)
            instructions = sample_instruction("score", rng=rng)

            metadata: dict[str, Any] = {
                "split": split,
                "original_score": label_id,
                "scale_min": SST5_SCALE_MIN,
                "scale_max": SST5_SCALE_MAX,
                "scale_levels": SST5_NUM_LEVELS,
            }
            if row.get("label_text"):
                metadata["original_label_text"] = row["label_text"]

            yield UnifiedSample(
                dataset_name="sst5",
                sample_id=f"sst5_{split}_{idx}",
                question_type=QuestionType.SCORE,
                state=text,
                instructions=instructions,
                criteria=SST5_CRITERIA.copy(),
                target=target_key,
                metadata=metadata,
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から SetFit/sst5 をロードして変換する。

        Args:
            split (str): スプリット名 ('train', 'validation', 'test')。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        ds = load_dataset("SetFit/sst5", split=split)
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split)
