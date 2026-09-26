"""RLCD (Reinforcement Learning from Calibrated Decisions) 設定モジュール。

ロジット摂動サンプリング、グループ相対ポリシー最適化 (GRPO)、
厳密適格スコア複合報酬、および KL 正則化のハイパーパラメータを管理し、
YAML および JSON による相互シリアライズを担保する。
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training.scoring_config import ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class RLCDConfig:
    """RLCD 強化学習および較正ポリシー更新の包括的設定 dataclass。

    Attributes:
        optimization_mode (str): 最適化方式 ("grpo" または "listwise_dpo")。
        num_generations (int): 同一サンプルから生成する摂動ロジットのグループサイズ G。
        perturbation_std (float): ロジット空間に注入するガウシアンノイズの標準偏差 sigma。
        clip_range (float): サロゲートポリシー目的関数のクリップ範囲 epsilon。
        kl_coeff (float): 参照モデル (SFT モデル) からの乖離を抑制する KL ペナルティ係数 beta_KL。
        entropy_coeff (float): 確信度 100% 張り付きを抑止するエントロピーボーナス係数 beta_ent。
        sampling_temperature (float): ロジット摂動後の Softmax に適用するサンプリング温度 tau。
        sft_aux_coeff (float): SFT 補助分類損失のブレンド係数 alpha_sft。
        dpo_beta (float): Listwise DPO における暗黙報酬スケーリング係数 beta。
        dpo_temperature (float): Listwise DPO における Plackett-Luce 温度パラメータ。
        learning_rate (float): 基本学習率。
        lr_head (float | None): デシジョンヘッドの個別学習率 (未指定時は learning_rate)。
        lr_backbone (float | None): バックボーンの個別学習率 (未指定時は learning_rate * backbone_lr_ratio)。
        lr_embed (float | None): 埋め込み層の個別学習率 (未指定時は learning_rate * 0.5)。
        freeze_backbone (bool): バックボーンの重みを完全凍結するかどうかのフラグ。
        backbone_lr_ratio (float): ヘッド学習率に対するバックボーン学習率の比率。
        weight_decay (float): AdamW オプティマイザの重み減衰率。
        warmup_ratio (float): コサイン学習率スケジューラのウォームアップ割合。
        max_grad_norm (float): 勾配クリッピングの最大 L2 ノルム閾値。
        epochs (int): RLCD 学習エポック数。
        batch_size (int): デバイスあたりのマイクロバッチサイズ。
        gradient_accumulation_steps (int): 勾配蓄積ステップ数。
        seed (int): 乱数シード。
        output_dir (str): 学習アーティファクトおよび設定の保存先ディレクトリ。
        scoring_config (ScoringConfig): 報酬計算に使用する厳密適格スコアリング規則設定。
    """

    optimization_mode: str = "grpo"
    num_generations: int = 4
    perturbation_std: float = 0.10
    clip_range: float = 0.20
    kl_coeff: float = 0.05
    entropy_coeff: float = 0.01
    sampling_temperature: float = 1.0
    sft_aux_coeff: float = 0.0
    dpo_beta: float = 0.10
    dpo_temperature: float = 1.0
    learning_rate: float = 2e-5
    lr_head: float | None = None
    lr_backbone: float | None = None
    lr_embed: float | None = None
    freeze_backbone: bool = False
    backbone_lr_ratio: float = 0.10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    seed: int = 42
    output_dir: str = "runs/rlcd"
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)

    def __post_init__(self) -> None:
        """設定パラメータの妥当性を検証する。"""
        valid_modes = {"grpo", "listwise_dpo"}
        if self.optimization_mode.lower() not in valid_modes:
            raise ValueError(
                f"無効な optimization_mode です: {self.optimization_mode}。"
                f"利用可能なモード: {sorted(valid_modes)}。"
            )
        self.optimization_mode = self.optimization_mode.lower()

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書型に変換する。

        Returns:
            dict[str, Any]: 全フィールドの辞書表現。内包された ScoringConfig も辞書化される。
        """
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RLCDConfig":
        """辞書型データから設定インスタンスを復元する。

        Args:
            data (dict[str, Any]): 設定パラメータ辞書。

        Returns:
            RLCDConfig: 復元された設定インスタンス。
        """
        data_copy = dict(data)
        if "scoring_config" in data_copy and isinstance(
            data_copy["scoring_config"], dict
        ):
            data_copy["scoring_config"] = ScoringConfig.from_dict(
                data_copy["scoring_config"]
            )

        known_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data_copy.items() if k in known_keys}
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
        logger.info("RLCD 設定を YAML に保存しました: %s。", save_path)
        return save_path

    @classmethod
    def from_yaml(cls, path_or_content: Path | str) -> "RLCDConfig":
        """YAML ファイルパスまたは YAML 文字列から設定を読み込む。

        Args:
            path_or_content (Path | str): ファイルパスまたは YAML 文字列。

        Returns:
            RLCDConfig: 復元された設定インスタンス。
        """
        is_file = False
        if isinstance(path_or_content, Path):
            is_file = path_or_content.is_file()
        elif isinstance(path_or_content, str) and "\n" not in path_or_content:
            try:
                is_file = Path(path_or_content).is_file()
            except OSError:
                is_file = False

        if is_file:
            content = Path(path_or_content).read_text(encoding="utf-8")
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
        logger.info("RLCD 設定を JSON に保存しました: %s。", save_path)
        return save_path

    @classmethod
    def from_json(cls, path_or_content: Path | str) -> "RLCDConfig":
        """JSON ファイルパスまたは JSON 文字列から設定を読み込む。

        Args:
            path_or_content (Path | str): ファイルパスまたは JSON 文字列。

        Returns:
            RLCDConfig: 復元された設定インスタンス。
        """
        is_file = False
        if isinstance(path_or_content, Path):
            is_file = path_or_content.is_file()
        elif isinstance(path_or_content, str) and "\n" not in path_or_content:
            try:
                is_file = Path(path_or_content).is_file()
            except OSError:
                is_file = False

        if is_file:
            content = Path(path_or_content).read_text(encoding="utf-8")
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
