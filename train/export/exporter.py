"""ONNX グラフエクスポートモジュール。

学習・較正された JevDecisionModel を、Rust ランタイムが直接実行可能な
静的計算グラフ (model.onnx) として出力する。
動的軸 (バッチサイズ、系列長、候補数) の完全分離および静的トポロジー検証を行う。
"""

import logging
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
)
from models.decision_head import JevDecisionModel

logger = logging.getLogger(__name__)

DEFAULT_OPSET_VERSION: int = 17
"""推奨される標準 ONNX Opset バージョン。"""


def export_onnx_model(
    model: JevDecisionModel,
    output_path: Path | str,
    opset_version: int = DEFAULT_OPSET_VERSION,
    dummy_batch_size: int = 1,
    dummy_seq_len: int = 16,
    dummy_num_options: int = 3,
    verbose: bool = False,
) -> Path:
    """JevDecisionModel を ONNX 形式でエクスポートし、グラフの整合性を検証する。

    内部処理手順:
    1. モデルを推論評価モード (`model.eval()`) に設定し、勾配計算を無効化する。
    2. バックボーンの未サポートカーネルを回避するため、アテンション実装を `eager` に設定する。
    3. 契約仕様に合致する `int64` 型のダミー入力テンソルを生成する。
    4. `torch.onnx.export` を呼び出し、動的軸をバッチ・系列長・候補数に分離してエクスポートする。
    5. `onnx.checker` および `onnx.shape_inference` で静的トポロジーおよび次元伝播を検証する。

    Args:
        model (JevDecisionModel): エクスポート対象の Jev 決定モデル。
        output_path (Path | str): 出力先 ONNX ファイルパス (`model.onnx`)。
        opset_version (int): ONNX Opset バージョン (デフォルト: 17)。
        dummy_batch_size (int): トレーシングに使用するダミーバッチサイズ。
        dummy_seq_len (int): トレーシングに使用するダミー系列長。
        dummy_num_options (int): トレーシングに使用するダミー候補数。
        verbose (bool): エクスポートログの詳細出力フラグ。

    Returns:
        Path: 保存された ONNX ファイルパス。

    Raises:
        ValueError: 出力パスが無効な場合、またはトポロジー検査に失敗した場合。
    """
    export_file = Path(output_path)
    export_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. モデルの評価モード化および勾配追跡の停止
    model.eval()
    model.requires_grad_(False)

    # 2. ModernBERT のアテンション展開設定 (FlashAttention / SDPA 回避)
    if hasattr(model.backbone, "config"):
        model.backbone.config._attn_implementation = "eager"
        model.backbone.config.attn_implementation = "eager"

    # 3. ダミー入力の作成 (すべて int64 型)
    vocab_size = getattr(model.backbone.config, "vocab_size", 50368)
    dummy_input_ids: Tensor = torch.randint(
        0,
        min(vocab_size, 1000),
        (dummy_batch_size, dummy_seq_len),
        dtype=torch.long,
    )
    dummy_attention_mask: Tensor = torch.ones(
        (dummy_batch_size, dummy_seq_len),
        dtype=torch.long,
    )

    # 候補マーカーインデックスのダミー生成
    step = max(1, dummy_seq_len // (dummy_num_options + 1))
    op_pos_list = [step * (i + 1) for i in range(dummy_num_options)]
    dummy_op_indices: Tensor = torch.tensor(
        [op_pos_list] * dummy_batch_size,
        dtype=torch.long,
    )

    dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_op_indices)

    # テンソル契約の取得
    input_names = model.get_input_names()
    output_names = model.get_output_names()
    dynamic_axes = model.get_dynamic_axes()

    logger.info(
        "ONNX エクスポートを開始します (出力先: %s, opset: %d)...",
        export_file,
        opset_version,
    )
    logger.info("入力テンソル名: %s, 出力テンソル名: %s", input_names, output_names)
    logger.info("動的軸定義: %s", dynamic_axes)

    # 4. torch.onnx.export の実行 (dynamo=False による決定論的トレーシング)
    torch.onnx.export(
        model,
        dummy_inputs,
        str(export_file),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        verbose=verbose,
    )

    # 5. ONNX グラフの構文チェックと形状推論
    logger.info("エクスポートされた ONNX グラフのトポロジー検査を実行中...")
    model_proto = onnx.load_model(str(export_file))
    onnx.checker.check_model(model_proto)

    logger.info("ONNX 形状推論 (Shape Inference) を実行中...")
    inferred_proto = onnx.shape_inference.infer_shapes(model_proto)
    onnx.save_model(inferred_proto, str(export_file))

    # 契約仕様 (テンソル名と次元数) の簡易検証
    _validate_contract_names_and_ranks(inferred_proto)

    logger.info(
        "ONNX グラフのエクスポートおよび静的検証が完了しました: %s", export_file
    )
    return export_file


def _validate_contract_names_and_ranks(model_proto: Any) -> None:
    """ONNX グラフの入出力テンソル名およびランクが契約と完全一致するか検証する。

    Args:
        model_proto (Any): ONNX モデルプロトコルバッファ。

    Raises:
        ValueError: テンソル名またはランクが契約仕様を満たさない場合。
    """
    graph = model_proto.graph
    expected_inputs = [TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_OP_INDICES]
    actual_inputs = [inp.name for inp in graph.input]

    for exp in expected_inputs:
        if exp not in actual_inputs:
            raise ValueError(
                f"ONNX 入力テンソル契約違反: 期待される入力 '{exp}' がグラフ内に見つかりません: {actual_inputs}。"
            )

    expected_outputs = [TENSOR_LOGITS]
    actual_outputs = [out.name for out in graph.output]

    for exp in expected_outputs:
        if exp not in actual_outputs:
            raise ValueError(
                f"ONNX 出力テンソル契約違反: 期待される出力 '{exp}' がグラフ内に見つかりません: {actual_outputs}。"
            )

    # 出力 logits の次元数が 2 であることを確認
    logits_output = next(out for out in graph.output if out.name == TENSOR_LOGITS)
    rank = len(logits_output.type.tensor_type.shape.dim)
    if rank != 2:
        raise ValueError(
            f"ONNX 出力テンソル契約違反: '{TENSOR_LOGITS}' の次元数は 2 [batch_size, num_options] である必要がありますが、{rank} 次元です。"
        )
