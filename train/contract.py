"""成果物引き渡し契約(Artifact Contract)モジュール。

Python 学習・エクスポート側と Rust ランタイム推論エンジン側で合意された、
ONNX モデルのテンソル名・形状仕様およびキャリブレーション設定(`calibration.json`)を定義する。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TENSOR_INPUT_IDS: str = "input_ids"
"""入力トークンID列テンソル名。"""

TENSOR_ATTENTION_MASK: str = "attention_mask"
"""アテンションマスクテンソル名。"""

TENSOR_OP_INDICES: str = "op_indices"
"""オプションマーカー位置インデックステンソル名。"""

TENSOR_LOGITS: str = "logits"
"""決定ヘッド出力ロジットテンソル名。"""

TOKEN_OPTION_MARKER: str = "[OP]"
"""各候補の先頭に付与される決定アンカー特殊トークン。"""

MAX_SEQUENCE_LENGTH: int = 8192
"""許容される最大入力トークン系列長。"""

MAX_NUM_OPTIONS: int = 255
"""単一質問あたりの最大候補数。"""

MIN_NUM_OPTIONS: int = 1
"""単一質問あたりの最小候補数。"""


@dataclass
class TemperatureMap:
    """質問プリミティブ別・候補数バケット別の最適温度パラメータマップ。

    Attributes:
        choice (dict[str, float]): Choice 型の候補数バケット別温度テーブル。
        score (dict[str, float]): Score 型の段階数バケット別温度テーブル。
        noul (float): Noul 型の二値判定温度係数。
    """

    choice: dict[str, float] = field(
        default_factory=lambda: {
            "2": 1.05,
            "3-5": 1.12,
            "6-10": 1.20,
            "11+": 1.35,
        }
    )
    score: dict[str, float] = field(
        default_factory=lambda: {
            "2-5": 1.00,
            "6-10": 1.08,
        }
    )
    noul: float = 1.0


@dataclass
class CalibrationConfig:
    """成果物引き渡し用 `calibration.json` のデータ構造。

    Attributes:
        version (str): キャリブレーションスキーマのバージョン。
        default_temperature (float): バケットに合致しない場合に使用するフォールバック温度。
        temperature_map (TemperatureMap): プリミティブ別の温度テーブル。
    """

    version: str = "1.0"
    default_temperature: float = 1.0
    temperature_map: TemperatureMap = field(default_factory=TemperatureMap)

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な辞書表現。
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON 文字列へ変換する。

        Args:
            indent (int): インデント幅。

        Returns:
            str: 整形された JSON 文字列。
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Path | str) -> None:
        """ファイルへ保存する。

        Args:
            path (Path | str): 出力先ファイルパス。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationConfig":
        """JSON ファイルから設定を読み込む。

        Args:
            path (Path | str): 読み込み元ファイルパス。

        Returns:
            CalibrationConfig: 読み込まれたキャリブレーション設定。
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        temp_map = TemperatureMap(
            choice=data.get("temperature_map", {}).get("choice", {}),
            score=data.get("temperature_map", {}).get("score", {}),
            noul=data.get("temperature_map", {}).get("noul", 1.0),
        )
        return cls(
            version=data.get("version", "1.0"),
            default_temperature=data.get("default_temperature", 1.0),
            temperature_map=temp_map,
        )
