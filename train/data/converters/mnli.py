"""MNLI (Multi-Genre Natural Language Inference) コンバータモジュール。

自然言語推論データを Jev 統一スキーマへ変換する。
Choice 型 (3 値推論: 含意・中立・矛盾) と
Noul 型 (言明の真偽二値判定) の両モードをサポートする。
"""

import random
from collections.abc import Iterator
from typing import Literal

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

MNLI_CHOICE_CRITERIA: dict[str, str] = {
    "entailment": "含意(前提文から仮説の内容が論理的に必然として導かれる)",
    "neutral": "中立(前提文の情報だけでは仮説が真か偽か判断できない)",
    "contradiction": "矛盾(前提文と仮説の内容が両立せず、明確に背反する)",
}

MNLI_LABEL_NAMES = ["entailment", "neutral", "contradiction"]


class MNLIConverter(BaseDatasetConverter):
    """MNLI データセットコンバータ。

    Attributes:
        mode (Literal['choice', 'noul']): 出力する決定プリミティブモード。
    """

    def __init__(
        self,
        mode: Literal["choice", "noul"] = "choice",
        seed: int = 42,
    ) -> None:
        """MNLI コンバータを初期化する。

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
    ) -> Iterator[UnifiedSample]:
        """与えられた Dataset を走査して UnifiedSample を生成する。

        Args:
            dataset (Dataset): 対象の Dataset オブジェクト。
            split (str): スプリット名。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        rng = random.Random(self.seed)

        for idx, row in enumerate(dataset):
            label_id = row["label"]
            if label_id == -1:
                # 未ラベリング・不確定サンプルは除外
                continue

            label_name = MNLI_LABEL_NAMES[label_id]
            premise = row["premise"]
            hypothesis = row["hypothesis"]

            if self.mode == "choice":
                state = f"前提: {premise}\n仮説: {hypothesis}"
                instructions = sample_instruction("nli_choice", rng=rng)

                yield UnifiedSample(
                    dataset_name="mnli",
                    sample_id=f"mnli_choice_{split}_{idx}",
                    question_type=QuestionType.CHOICE,
                    state=state,
                    instructions=instructions,
                    criteria=MNLI_CHOICE_CRITERIA.copy(),
                    target=label_name,
                    metadata={"split": split, "original_label_id": label_id},
                )
            elif self.mode == "noul":
                # Noul 型では中立 (neutral) を除外し、真(true) / 偽(false) の決定データとする
                if label_name == "neutral":
                    continue

                target_str = "true" if label_name == "entailment" else "false"
                state = premise
                instructions = sample_instruction(
                    "noul", rng=rng, hypothesis=hypothesis
                )

                yield UnifiedSample(
                    dataset_name="mnli_noul",
                    sample_id=f"mnli_noul_{split}_{idx}",
                    question_type=QuestionType.NOUL,
                    state=state,
                    instructions=instructions,
                    criteria={},
                    target=target_str,
                    metadata={
                        "split": split,
                        "hypothesis": hypothesis,
                        "original_label": label_name,
                    },
                )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から MNLI をロードして変換する。

        MNLI は 'validation' という名前のスプリットを持たないため、
        validation 指定時は 'validation_matched' をマッピングする。

        Args:
            split (str): スプリット名 ('train', 'validation', 'validation_matched', etc.)。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        hf_split = "validation_matched" if split in ["validation", "val"] else split
        try:
            ds = load_dataset("nyu-mll/glue", "mnli", split=hf_split)
        except (RuntimeError, ValueError, OSError):
            ds = load_dataset("glue", "mnli", split=hf_split)
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split)
