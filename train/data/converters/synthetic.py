"""合成データセットコンバータモジュール。

Evol-Instruct パイプラインによって出力された JSONL 形式の合成データを読み込み、
Jev 統一中間表現 (`UnifiedSample`) の列としてストリーミング供給する。
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from data.converters.base import BaseDatasetConverter
from data.schema import UnifiedSample

logger = logging.getLogger(__name__)


class SyntheticDatasetConverter(BaseDatasetConverter):
    """合成データセット用コンバータ。

    Attributes:
        file_path (Path): 読み込み対象の JSONL ファイルパス。
        mode (Literal['all', 'choice', 'score', 'noul']): 対象プリミティブ絞り込みモード。
        seed (int): 乱数シード。
    """

    def __init__(
        self,
        file_path: str | Path = "train/data/synthetic/output/synthetic_all.jsonl",
        mode: Literal["all", "choice", "score", "noul"] = "all",
        seed: int = 42,
    ) -> None:
        """合成データコンバータを初期化する。

        Args:
            file_path (str | Path): 読み込み対象 JSONL パス。
            mode (Literal['all', 'choice', 'score', 'noul']): 抽出モード。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.file_path = Path(file_path)
        self.mode = mode

    def convert_split(self, split: str = "train") -> Iterator[UnifiedSample]:
        """JSONL ファイルを走査し、UnifiedSample を順次生成する。

        合成データは主に 'train' スプリット用として使用されるが、
        'validation' または 'test' が指定された場合は決定論的ハッシュで分割供給可能とする。

        Args:
            split (str): スプリット名 ('train', 'validation', 'test')。

        Yields:
            Iterator[UnifiedSample]: 統一サンプルインスタンス。
        """
        if not self.file_path.exists():
            logger.warning(
                "合成データファイルが存在しません: %s。スキップします。",
                self.file_path,
            )
            return

        with open(self.file_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = UnifiedSample.from_dict(data)
                except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                    logger.warning(
                        "行 %d のサンプルパースに失敗しました: %s。スキップします。",
                        line_idx + 1,
                        e,
                    )
                    continue

                # プリミティブ絞り込み
                if self.mode != "all" and sample.question_type.value != self.mode:
                    continue

                # 簡易スプリット振り分け (train: 90%, validation: 10%)
                sample_hash = hash(sample.sample_id) % 100
                if split == "validation" or split == "test":
                    if sample_hash >= 10:
                        continue
                else:  # train
                    if sample_hash < 10:
                        continue

                yield sample
