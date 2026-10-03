"""粗密マルチタスク SFT 学習設定モジュール。

大分類・細分類協調学習ループのハイパーパラメータ、階層整合性損失係数、
および検証ループの Soft-Beam 推論シミュレーション設定を管理し、
JSON / YAML による完全な再現性を担保する。
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from contract import MAX_SEQUENCE_LENGTH
from models.backbone import DEFAULT_BACKBONE_MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalSFTConfig:
    """粗密マルチタスク SFT の設定 dataclass。

    Attributes:
        model_name_or_path (str): 事前学習済みバックボーンのモデル ID またはローカルパス。
        mlp_hidden_size (int | None): 決定ヘッドの MLP 中間次元数。None の場合はバックボーンと同一。
        max_sequence_length (int): 最大許容トークン系列長。
        batch_size (int): ミニバッチサイズ。
        gradient_accumulation_steps (int): 勾配累積ステップ数。
        learning_rate_backbone (float): 事前学習済みバックボーン Transformer 層の学習率。
        learning_rate_embed (float): 特殊トークンを含む埋め込み層の学習率。
        learning_rate_head (float): SAB デシジョンヘッドの学習率。
        weight_decay (float): 重み減衰率。
        num_epochs (int): 学習エポック数。
        warmup_ratio (float): ウォームアップステップ比率。
        max_grad_norm (float): 勾配クリッピング最大ノルム。
        beta (float): マルチタスク損失における細分類損失重み \\beta。
        gamma (float): マルチタスク損失における階層整合性正則化重み \\gamma。
        gamma_warmup_epochs (int): 整合性重み \\gamma を 0.0 から目標値へウォームアップするエポック数。
        label_smoothing (float): Label Smoothing 係数 \\epsilon。
        focal_gamma (float): Focal Loss 変調係数。
        soft_beam_ratio (float): 学習時に Soft-Beam 模倣サンプルを生成する比率。
        soft_beam_threshold (float): 検証シミュレーションで Soft-Beam を発動するマージン閾値 \\tau_{beam}。
        negative_ratio (float): 負例・受け皿サンプルの混入比率。
        seed (int): 乱数シード。
        output_dir (str): 成果物出力ディレクトリ。
    """

    model_name_or_path: str = DEFAULT_BACKBONE_MODEL_ID
    mlp_hidden_size: int | None = None
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate_backbone: float = 2e-5
    learning_rate_embed: float = 5e-5
    learning_rate_head: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # 粗密マルチタスク固有ハイパーパラメータ
    beta: float = 1.0
    gamma: float = 0.1
    gamma_warmup_epochs: int = 1
    label_smoothing: float = 0.05
    focal_gamma: float = 0.0
    soft_beam_ratio: float = 0.30
    soft_beam_threshold: float = 0.35
    negative_ratio: float = 0.15

    seed: int = 42
    output_dir: str = "runs/hierarchical_sft"

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。

        Returns:
            dict[str, Any]: 設定辞書。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalSFTConfig":
        """辞書オブジェクトから設定インスタンスを復元する。

        Args:
            data (dict[str, Any]): 設定辞書。

        Returns:
            HierarchicalSFTConfig: 復元された設定。
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, file_path: str | Path) -> None:
        """JSON ファイルへ保存する。

        Args:
            file_path (str | Path): 出力先ファイルパス。
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "HierarchicalSFTConfig":
        """JSON ファイルから設定を読み込む。

        Args:
            file_path (str | Path): 入力ファイルパス。

        Returns:
            HierarchicalSFTConfig: 復元された設定。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_yaml(self, file_path: str | Path) -> None:
        """YAML ファイルへ保存する。

        Args:
            file_path (str | Path): 出力先ファイルパス。
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "HierarchicalSFTConfig":
        """YAML ファイルから設定を読み込む。

        Args:
            file_path (str | Path): 入力ファイルパス。

        Returns:
            HierarchicalSFTConfig: 復元された設定。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
