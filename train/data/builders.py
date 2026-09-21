"""データセット一括構築・統合ビルダーモジュール。

複数の公開コーパス (Banking77, CLINC150, MNLI, AG News) のコンバータを統合し、
マルチタスク学習用の中間データセットを一括生成・結合するパイプラインを提供する。
"""

import logging
from collections.abc import Iterator

from datasets import Dataset

from data.converters.ag_news import AGNewsConverter
from data.converters.banking77 import Banking77Converter
from data.converters.base import BaseDatasetConverter
from data.converters.clinc150 import Clinc150Converter
from data.converters.mnli import MNLIConverter
from data.schema import UnifiedSample

logger = logging.getLogger(__name__)


class UnifiedDatasetBuilder:
    """複数 NLP コーパスの統合データセットビルダー。

    Attributes:
        converters (dict[str, BaseDatasetConverter]): 登録済みコンバータマップ。
    """

    def __init__(self, seed: int = 42) -> None:
        """ビルダーを初期化し、標準コンバータを登録する。

        Args:
            seed (int): 乱数シード。
        """
        self.seed = seed
        self.converters: dict[str, BaseDatasetConverter] = {
            "banking77": Banking77Converter(seed=seed),
            "clinc150": Clinc150Converter(seed=seed),
            "mnli_choice": MNLIConverter(mode="choice", seed=seed),
            "mnli_noul": MNLIConverter(mode="noul", seed=seed),
            "ag_news": AGNewsConverter(seed=seed),
        }

    def register_converter(self, name: str, converter: BaseDatasetConverter) -> None:
        """新規コンバータを登録する。

        Args:
            name (str): コンバータ識別名。
            converter (BaseDatasetConverter): コンバーラインスタンス。
        """
        self.converters[name] = converter

    def stream_samples(
        self,
        dataset_names: list[str] | None = None,
        split: str = "train",
        max_samples_per_dataset: int | None = None,
    ) -> Iterator[UnifiedSample]:
        """指定されたデータセット群から UnifiedSample を順次ストリーミング抽出する。

        Args:
            dataset_names (list[str] | None): 対象データセット名リスト。None の場合は全登録データセット。
            split (str): 抽出対象スプリット名。
            max_samples_per_dataset (int | None): データセットごとの最大取得サンプル数。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        targets = dataset_names or list(self.converters.keys())

        for name in targets:
            if name not in self.converters:
                logger.warning(
                    "登録されていないデータセットです: %s。スキップします。",
                    name,
                )
                continue

            converter = self.converters[name]
            logger.info("データセット '%s' (split: %s) の変換を開始...", name, split)
            count = 0
            try:
                for sample in converter.convert_split(split):
                    yield sample
                    count += 1
                    if (
                        max_samples_per_dataset is not None
                        and count >= max_samples_per_dataset
                    ):
                        break
            except Exception as e:
                logger.error(
                    "データセット '%s' のロード中にエラーが発生しました: %s",
                    name,
                    e,
                )
                raise

    def build_combined_dataset(
        self,
        dataset_names: list[str] | None = None,
        split: str = "train",
        max_samples_per_dataset: int | None = None,
    ) -> Dataset:
        """指定データセット群を走査・変換し、結合された単一の Hugging Face Dataset を構築する。

        Args:
            dataset_names (list[str] | None): 対象データセット名リスト。
            split (str): スプリット名。
            max_samples_per_dataset (int | None): データセットごとの最大取得件数。

        Returns:
            Dataset: 全サンプルの辞書列から構成された統合 Dataset。
        """
        samples = [
            sample.to_dict()
            for sample in self.stream_samples(
                dataset_names=dataset_names,
                split=split,
                max_samples_per_dataset=max_samples_per_dataset,
            )
        ]
        return Dataset.from_list(samples)
