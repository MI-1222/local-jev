"""JGLUE JSTS (日本語意味的テキスト類似度) コンバータモジュール。

文ペアの意味的類似度判定データを Jev 統一スキーマ (Score 型: 順序尺度 6 段階) へ変換する。
アノテータ平均値 (0.0〜5.0 の実数値) を四捨五入により 0〜5 の離散スコアへ正規化する。
"""

import random
from collections.abc import Iterator
from typing import Any

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

# 6 段階評価の Criteria 定義 (0 オリジン昇順キー "0"〜"5")
JSTS_CRITERIA: dict[str, str] = {
    "0": "全く異なる話題・意味的共通点なし",
    "1": "トピックは近いが意味内容は一致しない",
    "2": "一部の内容や詳細要素のみ一致している",
    "3": "重要な情報はおおむね一致している",
    "4": "細部表現を除きほぼ同一の意味を表している",
    "5": "完全に同一の意味・言い換えである",
}

JSTS_SCALE_MIN: int = 0
JSTS_SCALE_MAX: int = 5
JSTS_NUM_LEVELS: int = 6


class JGlueJSTSConverter(BaseDatasetConverter):
    """JGLUE JSTS データセットコンバータ。

    State は '文1: {sentence1}\\n文2: {sentence2}' の形式で結合される。
    系列長超過時は State 末尾の文2側から優先的に切り詰められるため、
    極端な長文入力時は意味比較に必要な文脈欠損に留意する必要がある。

    Attributes:
        seed (int): 指示文サンプリング等に用いる乱数シード。
    """

    def __init__(self, seed: int = 42) -> None:
        """JGLUE JSTS コンバータを初期化する。

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
        """与えられた Dataset を走査して UnifiedSample (Score 型) を順次生成する。

        Args:
            dataset (Dataset): 対象の Dataset オブジェクト。
            split (str): スプリット名。
            metadata_extra (dict[str, str] | None): 追加メタデータ辞書。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        rng = random.Random(self.seed)

        for idx, row in enumerate(dataset):
            raw_label = float(row["label"])
            # 四捨五入で 0〜5 の整数に離散化し、範囲をクリップ
            discrete_score = max(JSTS_SCALE_MIN, min(JSTS_SCALE_MAX, round(raw_label)))

            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]
            pair_id = row.get("sentence_pair_id", f"{split}_{idx}")

            state = f"文1: {sentence1}\n文2: {sentence2}"
            instructions = sample_instruction("similarity", rng=rng)
            target_key = str(discrete_score)

            metadata: dict[str, Any] = {
                "split": split,
                "sentence_pair_id": pair_id,
                "original_score": raw_label,
                "discrete_score": discrete_score,
                "scale_min": JSTS_SCALE_MIN,
                "scale_max": JSTS_SCALE_MAX,
                "scale_levels": JSTS_NUM_LEVELS,
            }
            if metadata_extra:
                metadata.update(metadata_extra)

            yield UnifiedSample(
                dataset_name="jglue_jsts",
                sample_id=f"jglue_jsts_{split}_{pair_id}",
                question_type=QuestionType.SCORE,
                state=state,
                instructions=instructions,
                criteria=JSTS_CRITERIA.copy(),
                target=target_key,
                metadata=metadata,
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から JGLUE JSTS をロードして変換する。

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
            data_dir="JSTS",
            revision="refs/convert/parquet",
            split=hf_split,
        )
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split, metadata_extra=metadata_extra)
