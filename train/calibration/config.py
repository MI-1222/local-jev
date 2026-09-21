"""事後温度較正設定モジュール。

質問プリミティブ別・候補数バケット別の最適温度係数探索に必要な
ハイパーパラメータ、探索範囲、バケット定義、および YAML/JSON シリアライズ機能を提供する。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

CHOICE_BUCKETS: tuple[str, ...] = ("2", "3-5", "6-10", "11+")
"""Choice 型の標準候補数バケット一覧。"""

SCORE_BUCKETS: tuple[str, ...] = ("2-5", "6-10")
"""Score 型の標準段階数バケット一覧。"""

NOUL_BUCKET: str = "noul"
"""Noul 型の単一バケット識別名。"""


def matches_bucket_expr(expr: str, count: int) -> bool:
    """バケット定義文字列と候補数が一致するか判定する。

    Rust 側 `crates/local-jev-core/src/contract/calibration.rs` の判定規則と厳密に整合する。

    Args:
        expr (str): バケット表現 (例: "2", "3-5", "11+")。
        count (int): 有効候補数または評価段階数。

    Returns:
        bool: 一致する場合 True。
    """
    trimmed = expr.strip()
    if trimmed.isdigit():
        return int(trimmed) == count

    if trimmed.endswith("+"):
        lower_str = trimmed[:-1].strip()
        if lower_str.isdigit():
            return count >= int(lower_str)

    if "-" in trimmed:
        parts = trimmed.split("-", 1)
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        if start_str.isdigit() and end_str.isdigit():
            return int(start_str) <= count <= int(end_str)

    return False


@dataclass
class CalibrationRunConfig:
    """事後温度較正の実行設定データクラス。

    Attributes:
        checkpoint_path (str): 評価対象のモデルチェックポイントディレクトリ。
        tau_min (float): 温度パラメータ探索範囲の下限。
        tau_max (float): 温度パラメータ探索範囲の上限。
        min_samples_per_bucket (int): バケット最適化を実行するために必要な最小サンプル数。
        default_temperature (float): サンプル不足バケットや未定義時に適用するフォールバック温度。
        num_bins (int): 期待較正誤差 (ECE) 算出時の確信度分割ビン数。
        max_val_samples (int): 検証データセットの最大利用サンプル数 (0 の場合は全件利用)。
        batch_size (int): 検証推論時のバッチサイズ。
        seed (int): 乱数シード。
        output_dir (str): 較正結果成果物の保存先親ディレクトリ。
    """

    checkpoint_path: str = ""
    tau_min: float = 0.05
    tau_max: float = 5.0
    min_samples_per_bucket: int = 15
    default_temperature: float = 1.0
    num_bins: int = 10
    max_val_samples: int = 0
    batch_size: int = 16
    seed: int = 42
    output_dir: str = "runs/calibration"

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書形式へ変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な設定辞書。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationRunConfig":
        """辞書から設定インスタンスを生成する。

        Args:
            data (dict[str, Any]): 設定辞書。

        Returns:
            CalibrationRunConfig: 生成された設定インスタンス。
        """
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_yaml(self) -> str:
        """YAML 文字列へ変換する。

        Returns:
            str: 整形された YAML 文字列。
        """
        return yaml.dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "CalibrationRunConfig":
        """YAML 文字列から設定インスタンスを復元する。

        Args:
            yaml_str (str): YAML 形式の文字列。

        Returns:
            CalibrationRunConfig: 復元された設定インスタンス。
        """
        data = yaml.safe_load(yaml_str) or {}
        return cls.from_dict(data)

    def to_json(self, indent: int = 2) -> str:
        """JSON 文字列へ変換する。

        Args:
            indent (int): インデント幅。

        Returns:
            str: 整形された JSON 文字列。
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "CalibrationRunConfig":
        """JSON 文字列から設定インスタンスを復元する。

        Args:
            json_str (str): JSON 形式の文字列。

        Returns:
            CalibrationRunConfig: 復元された設定インスタンス。
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: Path | str) -> None:
        """設定をファイルへ保存する。

        拡張子 (.yaml, .yml, .json) に応じて保存フォーマットを自動判定する。

        Args:
            path (Path | str): 保存先ファイルパス。
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.suffix.lower() in [".yaml", ".yml"]:
            target_path.write_text(self.to_yaml(), encoding="utf-8")
        else:
            target_path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationRunConfig":
        """ファイルから設定を読み込む。

        Args:
            path (Path | str): 読み込み元ファイルパス。

        Returns:
            CalibrationRunConfig: 読み込まれた設定インスタンス。
        """
        target_path = Path(path)
        content = target_path.read_text(encoding="utf-8")
        if target_path.suffix.lower() in [".yaml", ".yml"]:
            return cls.from_yaml(content)
        return cls.from_json(content)
