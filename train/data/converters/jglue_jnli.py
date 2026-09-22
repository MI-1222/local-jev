"""JGLUE JNLI (日本語自然言語推論) コンバータモジュール。

前提文と仮説文の論理的含意関係データを Jev 統一スキーマへ変換する。
Choice 型 (3 値推論: 含意・矛盾・中立) と Noul 型 (言明の真偽二値判定) をサポートする。
"""

import random
from collections.abc import Iterator
from typing import Any, Literal

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

# Choice モード向け Criteria 定義
JNLI_CHOICE_CRITERIA: dict[str, str] = {
    "entailment": "含意(前提文から仮説の内容が論理的に必然として導かれる)",
    "contradiction": "矛盾(前提文と仮説の内容が両立せず、明確に背反する)",
    "neutral": "中立(前提文の情報だけでは仮説が真か偽か判断できない)",
}

JNLI_LABEL_NAMES: list[str] = ["entailment", "contradiction", "neutral"]


class JGlueJNLIConverter(BaseDatasetConverter):
    """JGLUE JNLI データセットコンバータ。

    State は '前提: {sentence1}\\n仮説: {sentence2}' の形式で結合される。
    系列長超過時は State 末尾の仮説側から優先的に切り詰められるため、
    極端な長文入力時は文脈欠損に留意する必要がある。

    Attributes:
        mode (Literal['choice', 'noul']): 出力する決定プリミティブモード。
    """

    def __init__(
        self,
        mode: Literal["choice", "noul"] = "choice",
        seed: int = 42,
    ) -> None:
        """JGLUE JNLI コンバータを初期化する。

        Args:
            mode (Literal['choice', 'noul']): 'choice' (3値関係判定) または 'noul' (真偽判定)。
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
                if not (0 <= raw_label < len(JNLI_LABEL_NAMES)):
                    continue
                label_name = JNLI_LABEL_NAMES[raw_label]
            elif str(raw_label).isdigit():
                label_int = int(str(raw_label))
                if not (0 <= label_int < len(JNLI_LABEL_NAMES)):
                    continue
                label_name = JNLI_LABEL_NAMES[label_int]
            else:
                label_name = str(raw_label)

            if label_name not in JNLI_LABEL_NAMES:
                continue

            premise = row["sentence1"]
            hypothesis = row["sentence2"]
            pair_id = row.get("sentence_pair_id", f"{split}_{idx}")

            base_metadata: dict[str, Any] = {
                "split": split,
                "sentence_pair_id": pair_id,
                "original_label": label_name,
            }
            if metadata_extra:
                base_metadata.update(metadata_extra)

            if self.mode == "choice":
                state = f"前提: {premise}\n仮説: {hypothesis}"
                instructions = sample_instruction("nli_choice", rng=rng)

                yield UnifiedSample(
                    dataset_name="jglue_jnli",
                    sample_id=f"jglue_jnli_choice_{split}_{pair_id}",
                    question_type=QuestionType.CHOICE,
                    state=state,
                    instructions=instructions,
                    criteria=JNLI_CHOICE_CRITERIA.copy(),
                    target=label_name,
                    metadata=base_metadata,
                )
            elif self.mode == "noul":
                # Noul 型では中立 (neutral) を除外し、真(true) / 偽(false) の二値決定とする
                if label_name == "neutral":
                    continue

                target_str = "true" if label_name == "entailment" else "false"
                state = premise
                instructions = sample_instruction(
                    "noul", rng=rng, hypothesis=hypothesis
                )

                noul_metadata = base_metadata.copy()
                noul_metadata["hypothesis"] = hypothesis

                yield UnifiedSample(
                    dataset_name="jglue_jnli",
                    sample_id=f"jglue_jnli_noul_{split}_{pair_id}",
                    question_type=QuestionType.NOUL,
                    state=state,
                    instructions=instructions,
                    criteria={},
                    target=target_str,
                    metadata=noul_metadata,
                )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から JGLUE JNLI をロードして変換する。

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
            data_dir="JNLI",
            revision="refs/convert/parquet",
            split=hf_split,
        )
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split, metadata_extra=metadata_extra)
