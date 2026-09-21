"""CLINC150 データセットコンバータモジュール。

150 種類の意図分類および範囲外 (Out-of-Scope, oos) データを、
Jev 統一スキーマ (Choice 型) へ正規化・変換する。
oos を「該当なし」の貴重な学習シードとして保持する。
"""

import random
from collections.abc import Iterator
from typing import Any

from datasets import ClassLabel, Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample


def clean_clinc150_label(raw_label: str) -> str:
    """CLINC150 ラベル名を自然言語説明文へ変換する。

    Args:
        raw_label (str): 生のインテント名。

    Returns:
        str: 自然言語化された説明文。
    """
    if raw_label == "oos":
        return "None of the above, out-of-scope, or unsupported general inquiry"
    clean = raw_label.replace("_", " ").capitalize()
    return f"Inquiry or request related to {clean}"


class Clinc150Converter(BaseDatasetConverter):
    """CLINC150 データセットコンバータ。

    Attributes:
        dataset_config (str): Hugging Face データセットのサブセット設定 ('plus', 'small' 等)。
        max_negative_options (int | None): 訓練時サブサンプリングする負例候補数。
    """

    def __init__(
        self,
        dataset_config: str = "plus",
        max_negative_options: int | None = 7,
        seed: int = 42,
    ) -> None:
        """CLINC150 コンバータを初期化する。

        Args:
            dataset_config (str): データセットサブセット。
            max_negative_options (int | None): 訓練時負例候補数。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.dataset_config = dataset_config
        self.max_negative_options = max_negative_options

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
        label_col = "intent" if "intent" in dataset.features else "label"
        features: Any = dataset.features[label_col]
        raw_label_names: list[str] = (
            features.names if isinstance(features, ClassLabel) else []
        )

        all_descriptions = {lbl: clean_clinc150_label(lbl) for lbl in raw_label_names}

        for idx, row in enumerate(dataset):
            target_idx = row[label_col]
            target_id = (
                raw_label_names[target_idx] if raw_label_names else str(target_idx)
            )
            text = row["text"]

            # 訓練時の候補数絞り込み
            if (
                split == "train"
                and self.max_negative_options is not None
                and raw_label_names
                and self.max_negative_options < len(raw_label_names) - 1
            ):
                negatives = [lbl for lbl in raw_label_names if lbl != target_id]
                selected_negs = rng.sample(negatives, self.max_negative_options)
                candidate_ids = [target_id] + selected_negs
            else:
                candidate_ids = (
                    list(raw_label_names)
                    if raw_label_names
                    else list(all_descriptions.keys())
                )

            criteria = {
                cid: all_descriptions.get(cid, clean_clinc150_label(cid))
                for cid in candidate_ids
            }
            instructions = sample_instruction("intent", rng=rng)

            yield UnifiedSample(
                dataset_name="clinc150",
                sample_id=f"clinc150_{split}_{idx}",
                question_type=QuestionType.CHOICE,
                state=text,
                instructions=instructions,
                criteria=criteria,
                target=target_id,
                metadata={
                    "split": split,
                    "original_label_id": target_idx,
                    "is_oos": (target_id == "oos"),
                    "num_options": len(criteria),
                },
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から CLINC150 をロードして変換する。

        Args:
            split (str): スプリット名。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        ds = load_dataset("clinc/clinc_oos", self.dataset_config, split=split)
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split)
