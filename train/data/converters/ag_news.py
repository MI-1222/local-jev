"""AG News データセットコンバータモジュール。

4 クラスのニュース記事分類データを、Jev 統一スキーマ (Choice 型)
へ正規化・変換する。
"""

import random
from collections.abc import Iterator

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

AG_NEWS_CRITERIA: dict[str, str] = {
    "world": "世界情勢・国際ニュース・外交・紛争・各国の政治動向",
    "sports": "スポーツ競技・試合結果・選手動向・大会情報",
    "business": "経済・金融・企業業績・市場動向・株式およびビジネスニュース",
    "sci_tech": "科学技術・IT・インターネット・宇宙・最新テクノロジー",
}

AG_NEWS_LABEL_IDS: list[str] = ["world", "sports", "business", "sci_tech"]


class AGNewsConverter(BaseDatasetConverter):
    """AG News データセットコンバータ。"""

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
            label_id = int(row["label"])
            target_key = AG_NEWS_LABEL_IDS[label_id]
            text = row["text"]
            instructions = sample_instruction("topic", rng=rng)

            yield UnifiedSample(
                dataset_name="ag_news",
                sample_id=f"ag_news_{split}_{idx}",
                question_type=QuestionType.CHOICE,
                state=text,
                instructions=instructions,
                criteria=AG_NEWS_CRITERIA.copy(),
                target=target_key,
                metadata={
                    "split": split,
                    "original_label_id": label_id,
                },
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から AG News をロードして変換する。

        AG News は検証スプリットを持たないため、validation 指定時は test を使用する。

        Args:
            split (str): スプリット名 ('train', 'test', 'validation')。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        hf_split = "test" if split in ["validation", "val"] else split
        try:
            ds = load_dataset("fancyzhx/ag_news", split=hf_split)
        except (RuntimeError, ValueError, OSError):
            ds = load_dataset("ag_news", split=hf_split)
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split)
