"""JGLUE JCommonsenseQA (日本語常識推論 5 択質問応答) コンバータモジュール。

日常常識に基づく 5 者択一質問応答データを Jev 統一スキーマ (Choice 型) へ変換する。
サンプルごとに動的に変化する選択肢テキスト群を Criteria 候補としてバインドする。
"""

import random
from collections.abc import Iterator
from typing import Any

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

JCQA_NUM_CHOICES: int = 5
JCQA_CHOICE_KEYS: list[str] = [f"choice{i}" for i in range(JCQA_NUM_CHOICES)]


class JGlueJCommonsenseQAConverter(BaseDatasetConverter):
    """JGLUE JCommonsenseQA データセットコンバータ。

    Attributes:
        seed (int): 指示文サンプリング等に用いる乱数シード。
    """

    def __init__(self, seed: int = 42) -> None:
        """JGLUE JCommonsenseQA コンバータを初期化する。

        Args:
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)

    def convert_dataset(
        self,
        dataset: Dataset,
        split: str,
        metadata_extra: dict[str, str] | None = None,
    ) -> Iterator[UnifiedSample]:
        """与えられた Dataset を走査して UnifiedSample (Choice 型) を順次生成する。

        Args:
            dataset (Dataset): 対象の Dataset オブジェクト。
            split (str): スプリット名。
            metadata_extra (dict[str, str] | None): 追加メタデータ辞書。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        rng = random.Random(self.seed)

        for idx, row in enumerate(dataset):
            raw_label = row["label"]
            if isinstance(raw_label, int):
                target_key = f"choice{raw_label}"
            elif str(raw_label).isdigit():
                target_key = f"choice{int(raw_label)}"
            else:
                target_key = str(raw_label)

            criteria: dict[str, str] = {
                key: str(row[key]) for key in JCQA_CHOICE_KEYS if key in row
            }

            # 5 択が揃っていない不完全サンプルはスキップ
            if len(criteria) != JCQA_NUM_CHOICES or target_key not in criteria:
                continue

            question = row["question"]
            q_id = row.get("q_id", f"{split}_{idx}")
            instructions = sample_instruction("commonsense_qa", rng=rng)

            metadata: dict[str, Any] = {
                "split": split,
                "q_id": q_id,
                "original_label": raw_label,
            }
            if metadata_extra:
                metadata.update(metadata_extra)

            yield UnifiedSample(
                dataset_name="jglue_jcommonsenseqa",
                sample_id=f"jglue_jcqa_{split}_{q_id}",
                question_type=QuestionType.CHOICE,
                state=question,
                instructions=instructions,
                criteria=criteria,
                target=target_key,
                metadata=metadata,
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から JGLUE JCommonsenseQA をロードして変換する。

        JGLUE 公式の test スプリットには正解ラベルが付与されていないため、
        test / val 指定時はラベルが存在する validation スプリットを代用する。

        Args:
            split (str): スプリット名 ('train', 'validation', 'test' 等)。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        is_fallback = split in ["validation", "val", "test"] and split != "validation"
        hf_split = "validation" if split in ["validation", "val", "test"] else split
        metadata_extra: dict[str, str] = {
            "requested_split": split,
            "actual_split": hf_split,
        }
        if is_fallback:
            metadata_extra["fallback_reason"] = "test_split_unlabeled"

        ds = load_dataset(
            "shunk031/JGLUE",
            data_dir="JCommonsenseQA",
            revision="refs/convert/parquet",
            split=hf_split,
        )
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split, metadata_extra=metadata_extra)
