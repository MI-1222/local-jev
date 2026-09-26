"""ONNX ハイブリッド動的 INT8 量子化パイプラインのテストスイート。

デシジョンヘッド (OptionGatherLayer, out_proj) の除外完全性、
動的軸の保持、および成果物バンドル・メタデータの整合性を網羅的に検証する。
"""

import json
from pathlib import Path
from typing import cast

import onnx
import pytest
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
    TOKEN_OPTION_MARKER,
)
from export.exporter import export_onnx_model
from export.quantize import (
    create_quantized_bundle,
    find_nodes_to_exclude,
    is_backbone_node,
    quantize_onnx_model,
)
from models.decision_head import JevDecisionModel


@pytest.fixture
def lightweight_model_and_tokenizer(
    tmp_path: Path,
) -> tuple[JevDecisionModel, PreTrainedTokenizerFast]:
    """テスト用の軽量バックボーンを持つ JevDecisionModel とトークナイザーを準備するフィクスチャ。

    Args:
        tmp_path (Path): pytest 提供の一時ディレクトリ。

    Returns:
        tuple[JevDecisionModel, PreTrainedTokenizerFast]: 軽量モデルとトークナイザーのタプル。
    """
    config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    config.num_hidden_layers = 2
    if hasattr(config, "layer_types") and isinstance(config.layer_types, list):
        config.layer_types = config.layer_types[:2]
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config._attn_implementation = "eager"

    raw_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    assert raw_tokenizer is not None
    tokenizer = cast(PreTrainedTokenizerFast, raw_tokenizer)
    if TOKEN_OPTION_MARKER not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [TOKEN_OPTION_MARKER]}
        )

    backbone = AutoModel.from_config(config)
    backbone.resize_token_embeddings(len(tokenizer))

    model = JevDecisionModel(backbone=backbone, mlp_hidden_size=32)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def test_is_backbone_node_classification() -> None:
    """バックボーン内外のノード名分類が正しく機能するか検証する。"""
    assert is_backbone_node("/backbone/layers.0/attn/q_proj/MatMul")
    assert is_backbone_node("backbone.embeddings.tok_embeddings.Gather")
    assert is_backbone_node("/model/encoder/layer.1/attention/self/MatMul")
    assert not is_backbone_node("/decision_head/dense/MatMul")
    assert not is_backbone_node("/decision_head/out_proj/MatMul")
    assert not is_backbone_node("/gather_layer/GatherElements")
    assert not is_backbone_node("/sab/q_proj/MatMul")


def test_find_nodes_to_exclude_logic(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """ONNX グラフからデシジョンヘッド関連ノードのみが厳格に除外されるか検証する。

    検証項目:
    - デシジョンヘッドの MatMul / Gemm が除外リストに含まれること。
    - バックボーンの MatMul が除外リストに含まれないこと。
    """
    model, _ = lightweight_model_and_tokenizer
    onnx_file = tmp_path / "model.onnx"

    export_onnx_model(
        model=model,
        output_path=onnx_file,
        opset_version=17,
    )

    model_proto = onnx.load_model(str(onnx_file))
    excluded = find_nodes_to_exclude(model_proto)

    assert len(excluded) > 0

    # バックボーンのノードが誤って除外されていないことを確認
    for name in excluded:
        assert not is_backbone_node(name), (
            f"バックボーンノードが誤って除外リストに入っています: {name}。"
        )

    # 決定ヘッドノードが除外されていることを確認
    head_nodes = [
        n.name
        for n in model_proto.graph.node
        if "decision_head" in n.name and ("MatMul" in n.name or "Gemm" in n.name)
    ]
    for hn in head_nodes:
        assert hn in excluded, (
            f"決定ヘッドのノードが除外リストに含まれていません: {hn}。"
        )


def test_quantize_onnx_model_execution_and_dynamic_axes(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """動的 INT8 量子化の実行および動的軸の保持を検証する。

    検証項目:
    - 量子化モデルが正常に生成されること。
    - 出力テンソル `logits` が存在すること。
    - 入出力テンソルの次元数が 2 であること (動的軸の維持)。
    """
    model, _ = lightweight_model_and_tokenizer
    fp32_onnx = tmp_path / "model_fp32.onnx"
    int8_onnx = tmp_path / "model_int8.onnx"

    export_onnx_model(
        model=model,
        output_path=fp32_onnx,
        opset_version=17,
    )

    quantized_path = quantize_onnx_model(
        input_model_path=fp32_onnx,
        output_model_path=int8_onnx,
        per_channel=True,
    )

    assert quantized_path.exists()
    quant_proto = onnx.load_model(str(quantized_path))
    onnx.checker.check_model(quant_proto)

    # 入出力テンソル名の確認
    inp_names = [inp.name for inp in quant_proto.graph.input]
    assert TENSOR_INPUT_IDS in inp_names
    assert TENSOR_ATTENTION_MASK in inp_names
    assert TENSOR_OP_INDICES in inp_names

    out_names = [out.name for out in quant_proto.graph.output]
    assert TENSOR_LOGITS in out_names

    # 次元数の確認
    for inp in quant_proto.graph.input:
        if inp.name in (TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_OP_INDICES):
            assert len(inp.type.tensor_type.shape.dim) == 2


def test_create_quantized_bundle_generation(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """成果物バンドル作成関数がアセット複製とメタデータ出力を行うか検証する。"""
    model, tokenizer = lightweight_model_and_tokenizer
    src_dir = tmp_path / "src_model"
    src_dir.mkdir(parents=True, exist_ok=True)

    export_onnx_model(
        model=model,
        output_path=src_dir / "model.onnx",
        opset_version=17,
    )
    tokenizer.save_pretrained(src_dir)

    calib_data = {"temperature_scale": 1.0, "threshold": 0.5}
    with open(src_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_data, f)

    out_bundle_dir = tmp_path / "out_quantized_bundle"
    result = create_quantized_bundle(
        source_model_dir=src_dir,
        output_bundle_dir=out_bundle_dir,
        per_channel=True,
        verify=False,
    )
    assert result.bundle_dir == out_bundle_dir.resolve()
    assert result.parity_passed is True

    assert (out_bundle_dir / "model.onnx").exists()
    assert (out_bundle_dir / "tokenizer.json").exists()
    assert (out_bundle_dir / "calibration.json").exists()
    assert (out_bundle_dir / "quantize_metadata.json").exists()

    with open(out_bundle_dir / "quantize_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["per_channel"] is True
    assert "timestamp" in metadata
    res = metadata["quantize_result"]
    assert res["original_size_bytes"] > 0
    assert res["quantized_size_bytes"] > 0
    assert "compression_ratio" in res
