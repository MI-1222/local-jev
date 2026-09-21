"""成果物引き渡し契約(Artifact Contract)モジュール。

Python 学習・エクスポート側と Rust ランタイム推論エンジン側で合意された、
ONNX モデルのテンソル名・形状仕様およびキャリブレーション設定(calibration.json)を定義する。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ONNX 入出力テンソル名定数
TENSOR_INPUT_IDS: str = "input_ids"
TENSOR_ATTENTION_MASK: str = "attention_mask"
TENSOR_OP_INDICES: str = "op_indices"
TENSOR_LOGITS: str = "logits"

# 特殊マーカートークン
TOKEN_OPTION_MARKER: str = "[OP]"

# 系列長および候補数の制約定数
MAX_SEQUENCE_LENGTH: int = 8192
MAX_NUM_OPTIONS: int = 255
MIN_NUM_OPTIONS: int = 1


@dataclass
class TemperatureMap:
    """質問プリミティブ別・候補数バケット別の最適温度パラメータマップ。"""

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
    """成果物引き渡し用 calibration.json のデータ構造。"""

    version: str = "1.0"
    default_temperature: float = 1.0
    temperature_map: TemperatureMap = field(default_factory=TemperatureMap)

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON 文字列へ変換する。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Path | str) -> None:
        """ファイルへ保存する。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationConfig":
        """JSON ファイルから読み込む。"""
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
