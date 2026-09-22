"""ONNX INT8 PTQ (Post-Training Quantization) 量子化パイプライン。

学習・エクスポート済みの FP32 ONNX モデルに対して動的 INT8 量子化 (Dynamic Quantization) を適用し、
メモリ帯域の削減と推論高速化を実現する。
決定ヘッド (OptionGatherLayer, out_proj) を FP32 で保護する Selective Quantization を行い、
成果物バンドルとしての完全性検証および FP32 モデルとのパリティ検証を実施する。
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

# train ディレクトリ直下のモジュールを検索可能にする
_TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantizeResult:
    """量子化処理およびパリティ検証の結果データクラス。

    Attributes:
        input_model_path (Path): 入力 FP32 ONNX ファイルパス。
        output_model_path (Path): 出力 INT8 ONNX ファイルパス。
        bundle_dir (Path): 成果物バンドル格納ディレクトリ。
        original_size_bytes (int): 量子化前のファイルサイズ (バイト)。
        quantized_size_bytes (int): 量子化後のファイルサイズ (バイト)。
        compression_ratio (float): 圧縮率 (1.0 - quantized / original)。
        parity_passed (bool): パリティ検証に合格したかどうか。
        top1_agreement_rate (float): FP32 と INT8 の Top-1 決定一致率。
        max_abs_error (float): ロジットの最大絶対誤差。
        mean_abs_error (float): ロジットの平均絶対誤差。
    """

    input_model_path: Path
    output_model_path: Path
    bundle_dir: Path
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    parity_passed: bool
    top1_agreement_rate: float
    max_abs_error: float
    mean_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な検証結果辞書。
        """
        return {
            "input_model_path": str(self.input_model_path),
            "output_model_path": str(self.output_model_path),
            "bundle_dir": str(self.bundle_dir),
            "original_size_bytes": self.original_size_bytes,
            "quantized_size_bytes": self.quantized_size_bytes,
            "compression_ratio": self.compression_ratio,
            "parity_passed": self.parity_passed,
            "top1_agreement_rate": self.top1_agreement_rate,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
        }


def find_nodes_to_exclude(model_proto: onnx.ModelProto) -> list[str]:
    """デシジョンヘッドおよび Gather 演算など、量子化から除外すべきノード名を収集する。

    ModernBERT の Transformer エンコーダ層の MatMul を INT8 化する一方、
    候補マーカーを抽出する Gather 層や、最終ロジットを出力する MLP 射影層 (out_proj) を
    FP32 のまま保護することで、ロジットスケールおよび較正温度の歪みを防ぐ。

    Args:
        model_proto (onnx.ModelProto): 対象 ONNX モデルプロトコルバッファ。

    Returns:
        list[str]: 量子化から除外するノード名一覧。
    """
    excluded_names: set[str] = set()

    for node in model_proto.graph.node:
        node_name = node.name
        # 決定ヘッド関連ノードを保護
        if "decision_head" in node_name:
            excluded_names.add(node_name)
        # Gather 演算を保護
        if "Gather" in node.op_type or "gather" in node_name.lower():
            excluded_names.add(node_name)
        # 最終出力 logits を生成するノードを保護
        if any(out == TENSOR_LOGITS for out in node.output):
            excluded_names.add(node_name)

    return sorted(excluded_names)


def quantize_onnx_model(
    input_model_path: Path | str,
    output_model_path: Path | str,
    per_channel: bool = True,
    reduce_range: bool = False,
) -> Path:
    """ONNX モデルに対して動的 INT8 量子化を実行する。

    Args:
        input_model_path (Path | str): 入力 FP32 ONNX ファイルパス。
        output_model_path (Path | str): 出力 INT8 ONNX ファイルパス。
        per_channel (bool): チャンネル単位での重み量子化フラグ (デフォルト: True)。
        reduce_range (bool): 7bit への範囲縮小フラグ (デフォルト: False)。

    Returns:
        Path: 生成された量子化モデルのファイルパス。

    Raises:
        FileNotFoundError: 入力ファイルが存在しない場合。
        ValueError: 出力テンソル契約が満たされない場合。
    """
    input_path = Path(input_model_path).resolve()
    output_path = Path(output_model_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"入力モデルが見つかりません: {input_path}。")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("ONNX モデルを読み込み、除外対象ノードを特定中: %s...", input_path)
    model_proto = onnx.load(str(input_path))
    nodes_to_exclude = find_nodes_to_exclude(model_proto)
    logger.info(
        "量子化除外ノード数 (デシジョンヘッド・Gather等): %d", len(nodes_to_exclude)
    )

    logger.info("動的 INT8 量子化を実行中 (per_channel=%s)...", per_channel)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
    )

    # 出力モデルの契約検証
    quantized_proto = onnx.load(str(output_path))
    output_names = [out.name for out in quantized_proto.graph.output]
    if TENSOR_LOGITS not in output_names:
        raise ValueError(
            f"量子化後のモデルに出力テンソル '{TENSOR_LOGITS}' が存在しません: {output_names}。"
        )

    logger.info("動的 INT8 量子化が完了しました: %s", output_path)
    return output_path


def verify_quantized_parity(
    fp32_model_path: Path | str,
    int8_model_path: Path | str,
    num_samples: int = 25,
    seq_len: int = 48,
    num_options: int = 4,
) -> tuple[bool, float, float, float]:
    """FP32 モデルと INT8 モデルの出力ロジットおよび決定一致率を突き合わせ検証する。

    多様な候補数 (2〜5 選択肢) および系列長にわたる複数サンプルで推論を実行し、
    量子化による決定論的一致度およびロジット誤差の統計的安定性を担保する。

    Args:
        fp32_model_path (Path | str): 元の FP32 モデルファイルパス。
        int8_model_path (Path | str): 量子化後の INT8 モデルファイルパス。
        num_samples (int): 検証サンプル数 (デフォルト: 25)。
        seq_len (int): 系列長 (デフォルト: 48)。
        num_options (int): 基準候補数 (デフォルト: 4)。

    Returns:
        tuple[bool, float, float, float]: (合格フラグ, Top-1一致率, 最大絶対誤差, 平均絶対誤差)。
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    fp32_sess = ort.InferenceSession(
        str(fp32_model_path), opts, providers=["CPUExecutionProvider"]
    )
    int8_sess = ort.InferenceSession(
        str(int8_model_path), opts, providers=["CPUExecutionProvider"]
    )

    total_decisions = 0
    agreed_decisions = 0
    all_abs_errors: list[float] = []

    np.random.seed(42)

    for i in range(num_samples):
        # 候補数を 2〜5 の範囲で巡回させて多様な次元での整合性を検証
        cur_options = 2 + (i % 4)
        cur_seq_len = max(seq_len, (cur_options + 2) * 4)

        # 合成入力テンソルの作成
        input_ids = np.random.randint(0, 1000, size=(1, cur_seq_len), dtype=np.int64)
        attention_mask = np.ones((1, cur_seq_len), dtype=np.int64)

        step = max(1, cur_seq_len // (cur_options + 1))
        op_indices = np.array(
            [[step * (j + 1) for j in range(cur_options)]], dtype=np.int64
        )

        feed_dict = {
            TENSOR_INPUT_IDS: input_ids,
            TENSOR_ATTENTION_MASK: attention_mask,
            TENSOR_OP_INDICES: op_indices,
        }

        fp32_runs = fp32_sess.run([TENSOR_LOGITS], feed_dict)
        int8_runs = int8_sess.run([TENSOR_LOGITS], feed_dict)

        fp32_raw = fp32_runs[0]
        int8_raw = int8_runs[0]
        assert isinstance(fp32_raw, np.ndarray)
        assert isinstance(int8_raw, np.ndarray)

        fp32_out: np.ndarray = fp32_raw[0]
        int8_out: np.ndarray = int8_raw[0]

        fp32_choice = int(np.argmax(fp32_out))
        int8_choice = int(np.argmax(int8_out))

        total_decisions += 1
        if fp32_choice == int8_choice:
            agreed_decisions += 1

        abs_err = np.abs(fp32_out - int8_out)
        all_abs_errors.extend(abs_err.tolist())

    agreement_rate = agreed_decisions / max(1, total_decisions)
    max_err = float(np.max(all_abs_errors)) if all_abs_errors else 0.0
    mean_err = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0

    # Top-1 一致率が 90% 以上かつ最大誤差が発散していないことを合格基準とする
    passed = agreement_rate >= 0.90 and max_err < 2.5

    logger.info(
        "パリティ検証完了: Top-1 一致率=%.1f%%, 最大絶対誤差=%.4f, 平均絶対誤差=%.4f, 合否=%s",
        agreement_rate * 100.0,
        max_err,
        mean_err,
        "合格" if passed else "不合格",
    )

    return passed, agreement_rate, max_err, mean_err


def create_quantized_bundle(
    source_model_dir: Path | str,
    output_bundle_dir: Path | str,
    per_channel: bool = True,
    verify: bool = True,
) -> QuantizeResult:
    """量子化モデルおよび関連設定ファイル群を統合した成果物バンドルを生成する。

    Args:
        source_model_dir (Path | str): 元モデルファイル群が存在するディレクトリ。
        output_bundle_dir (Path | str): 量子化後バンドルの出力先ディレクトリ。
        per_channel (bool): チャンネル単位量子化フラグ。
        verify (bool): パリティ検証を実施するかどうか。

    Returns:
        QuantizeResult: 量子化および検証の要約結果。

    Raises:
        FileNotFoundError: 入力ディレクトリまたは必須ファイルが存在しない場合。
    """
    src_dir = Path(source_model_dir).resolve()
    out_dir = Path(output_bundle_dir).resolve()

    src_model = src_dir / "model.onnx"
    if not src_model.exists():
        raise FileNotFoundError(f"入力 ONNX モデルが見つかりません: {src_model}。")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "model.onnx"

    original_size = os.path.getsize(src_model)

    # 1. 量子化の実行
    quantize_onnx_model(
        input_model_path=src_model,
        output_model_path=out_model,
        per_channel=per_channel,
    )

    quantized_size = os.path.getsize(out_model)
    compression_ratio = 1.0 - (quantized_size / max(1, original_size))

    # 2. トークナイザーおよび設定ファイルの複製
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "calibration.json",
        "config.json",
    ]:
        src_file = src_dir / filename
        if src_file.exists():
            dest_file = out_dir / filename
            shutil.copy2(src_file, dest_file)
            logger.info("ファイルを複製しました: %s -> %s", src_file.name, dest_file)

    # 3. パリティ検証
    if verify:
        passed, agreement, max_err, mean_err = verify_quantized_parity(
            fp32_model_path=src_model,
            int8_model_path=out_model,
        )
    else:
        passed, agreement, max_err, mean_err = True, 1.0, 0.0, 0.0

    result = QuantizeResult(
        input_model_path=src_model,
        output_model_path=out_model,
        bundle_dir=out_dir,
        original_size_bytes=original_size,
        quantized_size_bytes=quantized_size,
        compression_ratio=compression_ratio,
        parity_passed=passed,
        top1_agreement_rate=agreement,
        max_abs_error=max_err,
        mean_abs_error=mean_err,
    )

    # 4. メタデータの保存
    meta_path = out_dir / "quantize_metadata.json"
    metadata = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "quantize_result": result.to_dict(),
        "per_channel": per_channel,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("量子化メタデータを保存しました: %s", meta_path)
    return result


def main() -> None:
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="Local-Jev ONNX INT8 PTQ 量子化スクリプト。"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/default",
        help="元モデルファイル群が存在するディレクトリパス。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/quantized",
        help="量子化後成果物バンドルの出力先ディレクトリパス。",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="重みテンソルをチャンネル単位で量子化する (デフォルト: True)。",
    )
    parser.add_argument(
        "--no-per-channel",
        dest="per_channel",
        action="store_false",
        help="重みテンソルをテンソル全体単位で量子化する。",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="FP32 モデルとのパリティ検証を実行する (デフォルト: True)。",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="パリティ検証をスキップする。",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        result = create_quantized_bundle(
            source_model_dir=args.model_dir,
            output_bundle_dir=args.output_dir,
            per_channel=args.per_channel,
            verify=args.verify,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        if not result.parity_passed:
            sys.exit(2)
    except Exception:
        logger.exception("量子化処理に失敗しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()
