"""成果物バンドル生成モジュール。

Rust ランタイム (`local-jev-runtime`) が外部通信なしで即座に起動できるよう、
ONNX モデル、トークナイザー、モデル設定、温度較正設定、およびメタデータを
単一ディレクトリ配下へ統合パッケージングする。
"""

import datetime
import json
import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import onnx
import onnxruntime
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from contract import CalibrationConfig
from export.validator import ValidationResult

logger = logging.getLogger(__name__)


def get_git_info() -> tuple[str, bool]:
    """Git コミットハッシュおよび作業ツリーの変更有無を取得する。

    Returns:
        tuple[str, bool]: (コミットハッシュ, 変更有無フラグ)。
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit, len(status) > 0
    except (subprocess.SubprocessError, OSError):
        return "unknown", False


def create_artifact_bundle(
    output_dir: Path | str,
    onnx_path: Path | str,
    tokenizer: PreTrainedTokenizerFast,
    backbone: PreTrainedModel,
    calibration_path: Path | str | None = None,
    validation_result: ValidationResult | None = None,
) -> Path:
    """Rust ランタイム用成果物バンドルを生成する。

    内部処理手順:
    1. 出力先ディレクトリを作成する。
    2. `model.onnx` を指定ディレクトリに配置する。
    3. トークナイザー設定 (`tokenizer.json`, `tokenizer_config.json` 等) を出力する。
    4. バックボーンの `config.json` を出力する。
    5. `calibration.json` を配置する (未指定時はデフォルト設定を生成)。
    6. 環境情報と検証結果を記録した `export_metadata.json` を保存する。

    Args:
        output_dir (Path | str): バンドル配置先ディレクトリ。
        onnx_path (Path | str): エクスポート済み ONNX ファイルパス。
        tokenizer (PreTrainedTokenizerFast): 特殊トークン登録済みトークナイザー。
        backbone (PreTrainedModel): バックボーンモデル。
        calibration_path (Path | str | None): `calibration.json` のファイルパス。
        validation_result (ValidationResult | None): パリティ検証結果オブジェクト。

    Returns:
        Path: 生成された成果物バンドルのディレクトリパス。
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("成果物バンドルを生成中: %s...", dest_dir)

    # 1. model.onnx の配置
    source_onnx = Path(onnx_path)
    target_onnx = dest_dir / "model.onnx"
    if source_onnx.resolve() != target_onnx.resolve():
        shutil.copy2(source_onnx, target_onnx)
    logger.info("ONNX モデルを配置しました: %s", target_onnx)

    # 2. トークナイザー設定の出力
    tokenizer.save_pretrained(dest_dir)
    logger.info("トークナイザー設定を出力しました: %s", dest_dir)

    # 3. バックボーン config.json の出力
    backbone.config.save_pretrained(dest_dir)
    logger.info("モデル設定 (config.json) を出力しました: %s", dest_dir)

    # 4. calibration.json の配置
    target_calib = dest_dir / "calibration.json"
    if calibration_path is not None and Path(calibration_path).exists():
        shutil.copy2(calibration_path, target_calib)
        logger.info("既存の calibration.json をコピーしました: %s", calibration_path)
    else:
        default_config = CalibrationConfig()
        default_config.save(target_calib)
        logger.info(
            "デフォルトの calibration.json を生成・配置しました: %s", target_calib
        )

    # 5. export_metadata.json の作成と保存
    commit_hash, is_dirty = get_git_info()
    metadata: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": commit_hash,
        "git_dirty": is_dirty,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": onnxruntime.__version__,
        },
        "model_info": {
            "backbone_type": backbone.config.model_type
            if hasattr(backbone.config, "model_type")
            else "unknown",
            "hidden_size": getattr(backbone.config, "hidden_size", None),
            "vocab_size": getattr(backbone.config, "vocab_size", None),
        },
        "validation_summary": validation_result.summary()
        if validation_result is not None
        else None,
    }

    metadata_path = dest_dir / "export_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("エクスポートメタデータを保存しました: %s", metadata_path)

    logger.info("成果物バンドルの構築が完了しました: %s", dest_dir)
    return dest_dir
