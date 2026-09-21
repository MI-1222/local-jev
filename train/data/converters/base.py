"""データセットコンバータ基底モジュール。

公開 NLP コーパスを走査し、Jev 統一スキーマ (`UnifiedSample`)
へ標準化変換するための抽象基底クラスを定義する。
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from datasets import Dataset

from data.schema import UnifiedSample


class BaseDatasetConverter(ABC):
    """NLP コーパスコンバータの抽象基底クラス。

    Attributes:
        seed (int): 指示文サンプリングや候補シャッフルに用いる乱数シード。
    """

    def __init__(self, seed: int = 42) -> None:
        """基底コンバータを初期化する。

        Args:
            seed (int): 乱数シード値。
        """
        self.seed = seed

    @abstractmethod
    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """指定されたスプリットを走査し、UnifiedSample を順次生成する。

        Args:
            split (str): データセットスプリット名 ('train', 'validation', 'test' 等)。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        raise NotImplementedError

    def build_dataset(self, split: str) -> Dataset:
        """指定スプリットをメモリ上に走査・変換し、Hugging Face Dataset を構築する。

        Args:
            split (str): データセットスプリット名。

        Returns:
            Dataset: 統一スキーマ辞書列から生成された Hugging Face Dataset。
        """
        samples = [sample.to_dict() for sample in self.convert_split(split)]
        return Dataset.from_list(samples)
