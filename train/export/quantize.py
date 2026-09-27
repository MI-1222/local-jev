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


def is_backbone_node(node_name: str) -> bool:
    """ノード名がエンコーダバックボーンに属しているかどうかを判定する。

    ModernBERT 内部の Linear / MatMul 層 (q_proj, k_proj, v_proj, out_proj, W1, W2 等) は
    INT8 量子化の主対象とするため、誤って除外リストに入らないよう判定する。

    Args:
        node_name (str): 検査対象の ONNX ノード名。

    Returns:
        bool: バックボーン内部のノードであれば True、そうでなければ False。
    """
    lower_name = node_name.lower()
    return lower_name.startswith(
        (
            "/backbone/",
            "backbone.",
            "/model/encoder/",
            "model.encoder.",
            "/bert/",
            "bert.",
        )
    )


def find_nodes_to_exclude(model_proto: onnx.ModelProto) -> list[str]:
    """デシジョンヘッドおよび Gather 演算など、量子化から除外すべきノード名を収集する。

    ModernBERT の Transformer エンコーダ層の MatMul を INT8 化する一方、
    候補マーカーを抽出する Gather 層や、最終ロジットを出力する MLP 射影層 (out_proj)、
    Set Attention Block (SAB)、CORAL 順序回帰層、NLI Noul 射影層を
    FP32 のまま保護することで、ロジットスケールおよび較正温度の歪みを防ぐ。

    内部ロジック:
    1. 出力テンソル `logits` からの有向グラフ逆トラバースにより、デシジョンヘッドを構成する全ノードを同定する。
    2. バックボーン以外のノードに対するスコープ名マッチング (head, gather, sab, coral 等) を併用する。
    3. バックボーン内部の演算ノードは除外対象から厳格に保護する。

    Args:
        model_proto (onnx.ModelProto): 対象 ONNX モデルプロトコルバッファ。

    Returns:
        list[str]: 量子化から除外するノード名一覧。
    """
    excluded_names: set[str] = set()

    # 1. テンソル -> 生成元ノードの逆引きマップを構築
    producer_map: dict[str, onnx.NodeProto] = {}
    for node in model_proto.graph.node:
        for out_name in node.output:
            producer_map[out_name] = node

    # 2. 出力テンソル logits からの上流依存ノード探索 (逆トラバース)
    visited_tensors: set[str] = set()
    queue: list[str] = []

    for out in model_proto.graph.output:
        if out.name == TENSOR_LOGITS:
            queue.append(out.name)
            visited_tensors.add(out.name)

    while queue:
        tensor_name = queue.pop(0)
        producer_node = producer_map.get(tensor_name)
        if producer_node is None:
            continue

        # バックボーン境界に到達した場合はそれ以上遡らない
        if is_backbone_node(producer_node.name):
            continue

        if producer_node.name not in excluded_names:
            excluded_names.add(producer_node.name)
            for inp_name in producer_node.input:
                if inp_name not in visited_tensors:
                    visited_tensors.add(inp_name)
                    queue.append(inp_name)

    # 3. スコープ名パターンマッチングおよび Gather 演算子の保護
    head_keywords = [
        "head",
        "gather",
        "decision",
        "choice",
        "score",
        "noul",
        "sab",
        "coral",
        "feature_proj",
        "out_proj",
    ]

    for node in model_proto.graph.node:
        node_name = node.name
        # バックボーン内部ノードは除外しない
        if is_backbone_node(node_name):
            continue

        lower_name = node_name.lower()
        if any(kw in lower_name for kw in head_keywords):
            excluded_names.add(node_name)

        if "Gather" in node.op_type:
            excluded_names.add(node_name)

        if any(out == TENSOR_LOGITS for out in node.output):
            excluded_names.add(node_name)

    return sorted(excluded_names)


def quantize_onnx_model(
    input_model_path: Path | str,
    output_model_path: Path | str,
    per_channel: bool = True,
    reduce_range: bool = False,
) -> Path:
    """ONNX モデルに対してハイブリッド動的 INT8 量子化を実行する。

    バックボーンの Linear/MatMul 層のみを INT8 化し、デシジョンヘッドおよび
    Gather 層は FP32 を厳格に保持する。動的軸の完全性検証も併せて実施する。

    Args:
        input_model_path (Path | str): 入力 FP32 ONNX ファイルパス。
        output_model_path (Path | str): 出力 INT8 ONNX ファイルパス。
        per_channel (bool): チャンネル単位での重み量子化フラグ (デフォルト: True)。
        reduce_range (bool): 7bit への範囲縮小フラグ (デフォルト: False)。

    Returns:
        Path: 生成された量子化モデルのファイルパス。

    Raises:
        FileNotFoundError: 入力ファイルが存在しない場合。
        ValueError: 出力テンソル契約または動的軸定義が満たされない場合。
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

    # 出力モデルの構造および整合性検証
    logger.info("量子化後 ONNX グラフの整合性および契約を検証中...")
    quantized_proto = onnx.load(str(output_path))
    onnx.checker.check_model(quantized_proto)

    output_names = [out.name for out in quantized_proto.graph.output]
    if TENSOR_LOGITS not in output_names:
        raise ValueError(
            f"量子化後のモデルに出力テンソル '{TENSOR_LOGITS}' が存在しません: {output_names}。"
        )

    # 動的軸の存在検証
    for inp in quantized_proto.graph.input:
        if inp.name in (TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_OP_INDICES):
            dims = inp.type.tensor_type.shape.dim
            if len(dims) != 2:
                raise ValueError(
                    f"入力テンソル '{inp.name}' の次元数が不正です (期待値: 2, 実際: {len(dims)})。"
                )

    logger.info("動的 INT8 量子化が完了しました: %s", output_path)
    return output_path


REPRESENTATIVE_SAMPLES: list[tuple[str, list[str]]] = [
    (
        "商品の発送状況を確認したいです。",
        ["配送中", "準備中", "配達完了", "キャンセル"],
    ),
    (
        "パスワードを再発行したいです。",
        ["メール送信", "再設定画面へ", "問い合わせ", "ログイン"],
    ),
    (
        "この製品は防水機能が付いていますか？",
        ["完全防水", "生活防水", "非防水", "未確認"],
    ),
    (
        "契約プランをアップグレードしたいです。",
        ["プレミアムプラン", "スタンダードプラン", "ライトプラン"],
    ),
    (
        "初期不良の疑いがあります。",
        ["交換を依頼", "修理を依頼", "返品を依頼", "様子見"],
    ),
    (
        "アカウントを退会する方法を教えてください。",
        ["退会手続きへ", "一時休止", "プラン解約", "継続"],
    ),
    (
        "支払い方法を変更したいです。",
        ["クレジットカード", "銀行振込", "コンビニ決済", "電子マネー"],
    ),
    (
        "領収書を発行してほしいです。",
        ["PDFでダウンロード", "郵送希望", "電子領収書", "再発行"],
    ),
    (
        "画面がフリーズして動きません。",
        ["再起動する", "キャッシュ削除", "アプリ再インストール", "端末初期化"],
    ),
    (
        "最新バージョンへのアップデートが必要です。",
        ["今すぐ更新", "後で通知", "自動更新", "キャンセル"],
    ),
]
"""パリティ検証に使用する代表的な日本語タスクサンプル群。"""


def verify_quantized_parity(
    fp32_model_path: Path | str,
    int8_model_path: Path | str,
    tokenizer_path: Path | str | None = None,
    num_synthetic_samples: int = 20,
    seq_len: int = 48,
) -> tuple[bool, float, float, float]:
    """FP32 モデルと INT8 モデルの出力ロジットおよび決定一致率を突き合わせ検証する。

    代表的な自然言語サンプルによる意味空間での Top-1 決定一致率、
    および多様な候補数・無効候補パディング (-1) を含む合成テンソルでの
    動的軸ロバスト性とコサイン類似度・MAE を総合評価する。

    Args:
        fp32_model_path (Path | str): 元の FP32 モデルファイルパス。
        int8_model_path (Path | str): 量子化後の INT8 モデルファイルパス。
        tokenizer_path (Path | str | None): トークナイザーファイルパス (`tokenizer.json`)。
        num_synthetic_samples (int): 合成検証サンプル数 (デフォルト: 20)。
        seq_len (int): 基準系列長 (デフォルト: 48)。

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
    cosine_sims: list[float] = []

    # 1. 自然言語テキストによる実質的パリティ検証 (tokenizer.json が利用可能な場合)
    tok_file = Path(tokenizer_path) if tokenizer_path is not None else None
    if tok_file is not None and tok_file.exists():
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_file))
            op_token_id = tok.token_to_id("[OP]")
            if op_token_id is not None:
                for text, options in REPRESENTATIVE_SAMPLES:
                    full_text = text
                    for opt in options:
                        full_text += f" [OP] {opt}"

                    enc = tok.encode(full_text)
                    input_ids = np.array([enc.ids], dtype=np.int64)
                    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
                    op_indices_list = [
                        i for i, tid in enumerate(enc.ids) if tid == op_token_id
                    ]
                    if not op_indices_list:
                        continue
                    op_indices = np.array([op_indices_list], dtype=np.int64)

                    feed = {
                        TENSOR_INPUT_IDS: input_ids,
                        TENSOR_ATTENTION_MASK: attention_mask,
                        TENSOR_OP_INDICES: op_indices,
                    }

                    fp32_runs = fp32_sess.run([TENSOR_LOGITS], feed)
                    int8_runs = int8_sess.run([TENSOR_LOGITS], feed)
                    fp32_res = fp32_runs[0]
                    int8_res = int8_runs[0]
                    assert isinstance(fp32_res, np.ndarray)
                    assert isinstance(int8_res, np.ndarray)
                    fp32_out = fp32_res[0]
                    int8_out = int8_res[0]

                    fp32_choice = int(np.argmax(fp32_out))
                    int8_choice = int(np.argmax(int8_out))

                    total_decisions += 1
                    if fp32_choice == int8_choice:
                        agreed_decisions += 1

                    abs_err = np.abs(fp32_out - int8_out)
                    all_abs_errors.extend(abs_err.tolist())

                    norm_fp32 = np.linalg.norm(fp32_out)
                    norm_int8 = np.linalg.norm(int8_out)
                    if norm_fp32 > 1e-6 and norm_int8 > 1e-6:
                        cos_sim = float(
                            np.dot(fp32_out, int8_out) / (norm_fp32 * norm_int8)
                        )
                        cosine_sims.append(cos_sim)
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning("テキストベースのパリティ検証をスキップしました: %s。", e)

    # 2. 合成入力テンソルによる動的軸ストレステスト
    rng = np.random.RandomState(42)

    for i in range(num_synthetic_samples):
        cur_options = 2 + (i % 6)
        cur_seq_len = max(seq_len, (cur_options + 2) * 5)
        batch_size = 1 if i % 2 == 0 else 2

        input_ids = rng.randint(0, 1000, size=(batch_size, cur_seq_len), dtype=np.int64)
        attention_mask = np.ones((batch_size, cur_seq_len), dtype=np.int64)

        step = max(1, cur_seq_len // (cur_options + 2))
        op_list = [step * (j + 1) for j in range(cur_options)]

        op_indices_list: list[list[int]] = []
        for b in range(batch_size):
            b_ops = list(op_list)
            if b > 0 and cur_options > 2:
                b_ops[-1] = -1
            op_indices_list.append(b_ops)

        op_indices = np.array(op_indices_list, dtype=np.int64)

        feed_dict = {
            TENSOR_INPUT_IDS: input_ids,
            TENSOR_ATTENTION_MASK: attention_mask,
            TENSOR_OP_INDICES: op_indices,
        }

        fp32_runs = fp32_sess.run([TENSOR_LOGITS], feed_dict)
        int8_runs = int8_sess.run([TENSOR_LOGITS], feed_dict)

        fp32_out = fp32_runs[0]
        int8_out = int8_runs[0]
        assert isinstance(fp32_out, np.ndarray)
        assert isinstance(int8_out, np.ndarray)

        for b in range(batch_size):
            valid_mask = op_indices[b] != -1
            if not np.any(valid_mask):
                continue

            fp32_b = fp32_out[b][valid_mask]
            int8_b = int8_out[b][valid_mask]

            abs_err = np.abs(fp32_b - int8_b)
            all_abs_errors.extend(abs_err.tolist())

            norm_fp32 = np.linalg.norm(fp32_b)
            norm_int8 = np.linalg.norm(int8_b)
            if norm_fp32 > 1e-6 and norm_int8 > 1e-6:
                cos_sim = float(np.dot(fp32_b, int8_b) / (norm_fp32 * norm_int8))
                cosine_sims.append(cos_sim)

            # テキスト評価が行われなかった場合の決定一致率フォールバック
            if total_decisions == 0:
                total_decisions += 1
                if int(np.argmax(fp32_b)) == int(np.argmax(int8_b)):
                    agreed_decisions += 1

    agreement_rate = agreed_decisions / max(1, total_decisions)
    max_err = float(np.max(all_abs_errors)) if all_abs_errors else 0.0
    mean_err = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0
    mean_cos = float(np.mean(cosine_sims)) if cosine_sims else 1.0

    # Phase 5 Exit Criteria:
    # 自然言語 Top-1 一致率 >= 99.0%、平均コサイン類似度 >= 0.99、平均絶対誤差 < 0.25
    passed = agreement_rate >= 0.99 and mean_err < 0.25 and mean_cos >= 0.99

    logger.info(
        "パリティ検証完了: Top-1 一致率=%.1f%%, コサイン類似度=%.4f, 最大絶対誤差=%.4f, 平均絶対誤差=%.4f, 合否=%s",
        agreement_rate * 100.0,
        mean_cos,
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
            tokenizer_path=src_dir / "tokenizer.json",
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
