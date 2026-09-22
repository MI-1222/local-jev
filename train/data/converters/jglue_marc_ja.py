"""JGLUE MARC-ja (多言語 Amazon レビュー日本語版) コンバータモジュール。

カスタマーレビューの感情極性判定データを Jev 統一スキーマへ変換する。
Noul 型 (肯定レビューか否かの真偽判定) と Choice 型 (肯定・否定の 2 択) をサポートする。
"""

import random
from collections.abc import Iterator
from typing import Any, Literal

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

# Choice モード向け Criteria 定義
MARC_JA_CHOICE_CRITERIA: dict[str, str] = {
    "positive": "肯定的な評価(星4〜5相当の満足・好意的なレビュー)",
    "negative": "否定的な評価(星1〜2相当の不満・低評価なレビュー)",
}

MARC_JA_LABEL_NAMES = ["positive", "negative", "neutral"]


class JGlueMarcJaConverter(BaseDatasetConverter):
    """JGLUE MARC-ja データセットコンバータ。

    Attributes:
        mode (Literal['choice', 'noul']): 出力する決定プリミティブモード。
    """

    def __init__(
        self,
        mode: Literal["choice", "noul"] = "noul",
        seed: int = 42,
    ) -> None:
        """JGLUE MARC-ja コンバータを初期化する。

        Args:
            mode (Literal['choice', 'noul']): 'noul' (真偽判定) または 'choice' (2択)。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.mode = mode

    def convert_dataset(
        self,
        dataset: Dataset,
        split: str,
        metadata_extra: dict[str, str] | None = None,
    ) -> Iterator[UnifiedSample]:
        """与えられた Dataset を走査して UnifiedSample を順次生成する。

        中立(neutral)評価は二値判定境界の曖昧化を防ぐためスキップする。

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
                if not (0 <= raw_label < len(MARC_JA_LABEL_NAMES)):
                    continue
                label_name = MARC_JA_LABEL_NAMES[raw_label]
            elif str(raw_label).isdigit():
                label_int = int(str(raw_label))
                if not (0 <= label_int < len(MARC_JA_LABEL_NAMES)):
                    continue
                label_name = MARC_JA_LABEL_NAMES[label_int]
            else:
                label_name = str(raw_label).lower()

            # 中立(neutral)および未確定・対象外サンプルは除外
            if label_name not in ["positive", "negative"]:
                continue

            sentence = row["sentence"]
            review_id = row.get("review_id", f"{split}_{idx}")

            base_metadata: dict[str, Any] = {
                "split": split,
                "review_id": review_id,
                "original_label": label_name,
            }
            if metadata_extra:
                base_metadata.update(metadata_extra)

            if self.mode == "noul":
                target_str = "true" if label_name == "positive" else "false"
                instructions = sample_instruction("sentiment", rng=rng)

                yield UnifiedSample(
                    dataset_name="jglue_marc_ja",
                    sample_id=f"jglue_marc_ja_noul_{split}_{review_id}",
                    question_type=QuestionType.NOUL,
                    state=sentence,
                    instructions=instructions,
                    criteria={},
                    target=target_str,
                    metadata=base_metadata,
                )
            elif self.mode == "choice":
                instructions = sample_instruction("sentiment", rng=rng)

                yield UnifiedSample(
                    dataset_name="jglue_marc_ja",
                    sample_id=f"jglue_marc_ja_choice_{split}_{review_id}",
                    question_type=QuestionType.CHOICE,
                    state=sentence,
                    instructions=instructions,
                    criteria=MARC_JA_CHOICE_CRITERIA.copy(),
                    target=label_name,
                    metadata=base_metadata,
                )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から JGLUE MARC-ja をロードして変換する。

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
            data_dir="MARC-ja",
            revision="refs/convert/parquet",
            split=hf_split,
        )
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split, metadata_extra=metadata_extra)
