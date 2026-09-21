"""ONNX エクスポートおよび成果物バンドル生成 CLI スクリプト。

学習済みチェックポイントおよび事後較正パラメータから、
Rust 本番推論ランタイム向けの配布バンドル (models/default/) を一括生成する。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from contract import TOKEN_OPTION_MARKER
from export.bundle import create_artifact_bundle
from export.exporter import DEFAULT_OPSET_VERSION, export_onnx_model
from export.validator import DEFAULT_ATOL, DEFAULT_RTOL, validate_onnx_parity
from models.backbone import (
    DEFAULT_MODERNBERT_MODEL_ID,
)
from models.decision_head import JevDecisionModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_export")


def find_latest_checkpoint(base_dir: Path | str) -> Path | None:
    """指定ディレクトリ配下から最新のチェックポイントを探索する。

    Args:
        base_dir (Path | str): 探索ベースディレクトリ (`runs/sft` 等)。

    Returns:
        Path | None: 発見されたチェックポイントディレクトリパス。
    """
    search_path = Path(base_dir)
    if not search_path.exists():
        return None

    candidates = sorted(
        search_path.glob("**/best_checkpoint"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_latest_calibration(base_dir: Path | str) -> Path | None:
    """指定ディレクトリ配下から最新の calibration.json を探索する。

    Args:
        base_dir (Path | str): 探索ベースディレクトリ (`runs/calibration` 等)。

    Returns:
        Path | None: 発見された calibration.json ファイルパス。
    """
    search_path = Path(base_dir)
    if not search_path.exists():
        return None

    candidates = sorted(
        search_path.glob("**/calibration.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    backbone_name_or_path: str = DEFAULT_MODERNBERT_MODEL_ID,
) -> tuple[JevDecisionModel, PreTrainedTokenizerFast]:
    """チェックポイントディレクトリからモデルとトークナイザーを復元する。

    Args:
        checkpoint_dir (Path): チェックポイントディレクトリ。
        backbone_name_or_path (str): バックボーン識別子。

    Returns:
        tuple[JevDecisionModel, PreTrainedTokenizerFast]:
            - 復元された JevDecisionModel インスタンス。
            - 復元されたトークナイザー。

    Raises:
        FileNotFoundError: model.pt が見つからない場合。
    """
    model_pt_path = checkpoint_dir / "model.pt"
    if not model_pt_path.exists():
        raise FileNotFoundError(
            f"チェックポイントファイルが見つかりません: {model_pt_path}。"
        )

    # トークナイザーの読み込み元 (チェックポイント同梱優先、なければベース)
    tok_dir = checkpoint_dir / "tokenizer"
    tok_path = str(tok_dir) if tok_dir.exists() else backbone_name_or_path

    logger.info(
        "バックボーン (%s) およびトークナイザー (%s) を初期化中...",
        backbone_name_or_path,
        tok_path,
    )
    # トークナイザーの準備
    loaded_tokenizer = AutoTokenizer.from_pretrained(tok_path)
    tokenizer = cast(PreTrainedTokenizerFast, loaded_tokenizer)
    if TOKEN_OPTION_MARKER not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [TOKEN_OPTION_MARKER]}
        )

    # バックボーンの準備
    backbone = AutoModel.from_pretrained(
        backbone_name_or_path,
        dtype=torch.float32,
    )
    backbone.resize_token_embeddings(len(tokenizer))

    model = JevDecisionModel(backbone=backbone)
    logger.info("チェックポイント重みをロード中: %s...", model_pt_path)
    checkpoint_data: Any = torch.load(
        model_pt_path, map_location="cpu", weights_only=False
    )

    state_dict: dict[str, Any]
    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data
    else:
        raise ValueError(
            f"サポートされていないチェックポイント形式です: {type(checkpoint_data)}。"
        )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning("未ロードキー (欠落): %s", missing_keys)
    if unexpected_keys:
        logger.warning("未知のキー (余剰): %s", unexpected_keys)

    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def run_pipeline(args: argparse.Namespace) -> int:
    """エクスポートパイプラインを実行する。

    Args:
        args (argparse.Namespace): パース済みコマンドライン引数。

    Returns:
        int: 終了コード (0: 成功, 1: 失敗)。
    """
    train_root = Path(__file__).resolve().parent.parent
    project_root = train_root.parent

    # 1. チェックポイントディレクトリの解決
    checkpoint_dir: Path | None
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
    else:
        checkpoint_dir = find_latest_checkpoint(train_root / "runs" / "sft")
        if checkpoint_dir is None:
            checkpoint_dir = find_latest_checkpoint(train_root / "runs" / "rlcd")

    if checkpoint_dir is None or not checkpoint_dir.exists():
        logger.error(
            "有効なチェックポイントが見つかりません。--checkpoint でパスを指定してください。"
        )
        return 1

    logger.info("使用チェックポイント: %s", checkpoint_dir)

    # 2. calibration.json の解決
    calibration_path: Path | None
    if args.calibration:
        calibration_path = Path(args.calibration)
    else:
        calibration_path = find_latest_calibration(train_root / "runs" / "calibration")

    if calibration_path:
        logger.info("使用較正設定: %s", calibration_path)
    else:
        logger.warning(
            "較正設定が見つかりません。デフォルト設定を自動生成してバンドルします。"
        )

    # 3. モデルとトークナイザーのロード
    try:
        model, tokenizer = load_model_from_checkpoint(checkpoint_dir)
    except Exception:
        logger.exception("モデルの復元に失敗しました。")
        return 1

    # 4. 出力先ディレクトリの設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "models" / "default"

    temp_onnx_path = output_dir / "model.onnx"

    # 5. ONNX グラフのエクスポート
    logger.info("=== STEP 1: ONNX グラフエクスポート ===")
    try:
        export_onnx_model(
            model=model,
            output_path=temp_onnx_path,
            opset_version=args.opset,
        )
    except Exception:
        logger.exception("ONNX エクスポートに失敗しました。")
        return 1

    # 6. 数値整合性検証 (Parity Test)
    validation_result = None
    if not args.skip_validation:
        logger.info("=== STEP 2: 数値整合性検証 (Parity Test) ===")
        try:
            validation_result = validate_onnx_parity(
                model=model,
                onnx_path=temp_onnx_path,
                tolerance_atol=args.tolerance,
                tolerance_rtol=args.rtol,
            )
            if not validation_result.passed:
                logger.error(
                    "数値整合性検証で許容誤差 (atol=%s, rtol=%s) を超える差分が検出されました。",
                    args.tolerance,
                    args.rtol,
                )
                return 1
        except Exception:
            logger.exception("数値整合性検証中に例外が発生しました。")
            return 1
    else:
        logger.info("数値整合性検証をスキップしました。")

    # 7. 成果物バンドル化
    logger.info("=== STEP 3: 成果物バンドル生成 ===")
    try:
        backbone = cast(PreTrainedModel, model.backbone)
        bundle_path = create_artifact_bundle(
            output_dir=output_dir,
            onnx_path=temp_onnx_path,
            tokenizer=tokenizer,
            backbone=backbone,
            calibration_path=calibration_path,
            validation_result=validation_result,
        )
    except Exception:
        logger.exception("成果物バンドルの構築に失敗しました。")
        return 1

    logger.info("==========================================")
    logger.info("🎉 ONNX エクスポート & バンドル化が完了しました！")
    logger.info("出力成果物ディレクトリ: %s", bundle_path.resolve())
    logger.info("==========================================")
    return 0


def main() -> None:
    """CLI エントリーポイント関数。"""
    parser = argparse.ArgumentParser(
        description="Jev 決定モデルの ONNX エクスポートおよび配布バンドル生成"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="ロードする学習済みチェックポイントディレクトリパス",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="同梱する calibration.json のファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="成果物バンドルの出力先ディレクトリパス (デフォルト: models/default)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help=f"ONNX Opset バージョン (デフォルト: {DEFAULT_OPSET_VERSION})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_ATOL,
        help=f"パリティ検証の最大絶対誤差許容値 (デフォルト: {DEFAULT_ATOL})",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"パリティ検証の相対誤差許容値 (デフォルト: {DEFAULT_RTOL})",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="ONNX Runtime による数値整合性検証をスキップする",
    )

    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
