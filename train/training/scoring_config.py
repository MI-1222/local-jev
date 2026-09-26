"""厳密適格スコアリング規則の設定モジュール。

各決定プリミティブ(Choice, Score, Noul)に対するスコアリング規則のブレンド比率、
数値ガード係数、温度パラメータを管理し、YAML および JSON による相互シリアライズを担保する。
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """厳密適格スコアリング規則の包括的設定 dataclass。

    Attributes:
        alpha (float): Choice / Noul における対数スコア比率 (1 - alpha が球面スコア比率)。
        beta (float): Score における RPS 比率 (1 - beta が対数スコア比率)。
        brier_weight (float): 複合報酬における Brier スコア比率。
        min_log_score (float): 対数スコア計算時の下限クリッピング値 (-10.0)。
        eps (float): 対数スコア計算時の確率クリッピング微小値。
        eps_div (float): 球面スコア計算時のゼロ除算防止微小値。
        temperature (float): ロジットに適用する温度パラメータ tau。
        normalize_log (bool): 対数スコアを [0, 1] 区間に線形スケーリングするかどうかのフラグ。
        output_dir (str): スコア評価結果および設定の保存先ディレクトリ。
    """

    alpha: float = 0.5
    beta: float = 0.7
    brier_weight: float = 0.0
    min_log_score: float = -10.0
    eps: float = 1e-6
    eps_div: float = 1e-12
    temperature: float = 1.0
    normalize_log: bool = True
    output_dir: str = "runs/scoring"

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書型に変換する。

        Returns:
            dict[str, Any]: 全フィールドの辞書表現。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        """辞書型データから設定インスタンスを復元する。

        Args:
            data (dict[str, Any]): 設定パラメータ辞書。

        Returns:
            ScoringConfig: 復元された設定インスタンス。
        """
        known_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered_data)

    def to_yaml(self) -> str:
        """設定を YAML 文字列として出力する。

        Returns:
            str: 整形された YAML 文字列。
        """
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save_yaml(self, path: Path | str) -> Path:
        """設定を YAML ファイルとして保存する。

        Args:
            path (Path | str): 保存先ファイルパス。

        Returns:
            Path: 保存されたファイルパス。
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(self.to_yaml(), encoding="utf-8")
        logger.info("Scoring 設定を YAML に保存しました: %s。", save_path)
        return save_path

    @classmethod
    def from_yaml(cls, path_or_content: Path | str) -> "ScoringConfig":
        """YAML ファイルパスまたは YAML 文字列から設定を読み込む。

        Args:
            path_or_content (Path | str): ファイルパスまたは YAML 文字列。

        Returns:
            ScoringConfig: 復元された設定インスタンス。
        """
        path = Path(path_or_content)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(path_or_content)
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise TypeError(f"YAML のパース結果が辞書型ではありません: {type(data)}。")
        return cls.from_dict(data)

    def to_json(self, indent: int = 2) -> str:
        """設定を JSON 文字列として出力する。

        Args:
            indent (int): インデント幅。

        Returns:
            str: 整形された JSON 文字列。
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: Path | str) -> Path:
        """設定を JSON ファイルとして保存する。

        Args:
            path (Path | str): 保存先ファイルパス。

        Returns:
            Path: 保存されたファイルパス。
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Scoring 設定を JSON に保存しました: %s。", save_path)
        return save_path

    @classmethod
    def from_json(cls, path_or_content: Path | str) -> "ScoringConfig":
        """JSON ファイルパスまたは JSON 文字列から設定を読み込む。

        Args:
            path_or_content (Path | str): ファイルパスまたは JSON 文字列。

        Returns:
            ScoringConfig: 復元された設定インスタンス。
        """
        path = Path(path_or_content)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(path_or_content)
        data = json.loads(content)
        if not isinstance(data, dict):
            raise TypeError(f"JSON のパース結果が辞書型ではありません: {type(data)}。")
        return cls.from_dict(data)

    def save(self, path: Path | str) -> Path:
        """拡張子に応じて YAML または JSON で設定を自動保存する。

        Args:
            path (Path | str): 保存先ファイルパス。

        Returns:
            Path: 保存されたファイルパス。
        """
        save_path = Path(path)
        if save_path.suffix.lower() in [".yaml", ".yml"]:
            return self.save_yaml(save_path)
        return self.save_json(save_path)
