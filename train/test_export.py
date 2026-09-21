"""ONNX グラフエクスポートおよび整合性検証テストスイート。

入出力テンソル契約、動的軸の柔軟性、および PyTorch vs ONNX Runtime の
数値パリティ (誤差 < 1e-5) を網羅的に検証する。
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
from export.bundle import create_artifact_bundle
from export.exporter import export_onnx_model
from export.validator import validate_onnx_parity
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


def test_onnx_export_contract_and_topology(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """ONNX エクスポートが契約入出力仕様および静的トポロジーを満たすか検証する。

    検証項目:
    - 指定パスに model.onnx が出力されること。
    - 入力テンソル名が `input_ids`, `attention_mask`, `op_indices` であり型が INT64 であること。
    - 出力テンソル名が `logits` であり型が FLOAT32、2次元 [batch_size, num_options] であること。
    """
    model, _ = lightweight_model_and_tokenizer
    onnx_file = tmp_path / "model.onnx"

    exported_path = export_onnx_model(
        model=model,
        output_path=onnx_file,
        opset_version=17,
    )

    assert exported_path.exists()
    model_proto = onnx.load_model(str(exported_path))

    # 入力テンソル契約の検証
    input_names = [inp.name for inp in model_proto.graph.input]
    assert TENSOR_INPUT_IDS in input_names
    assert TENSOR_ATTENTION_MASK in input_names
    assert TENSOR_OP_INDICES in input_names

    for inp in model_proto.graph.input:
        # ONNX TensorProto.INT64 = 7
        assert inp.type.tensor_type.elem_type == 7
        assert len(inp.type.tensor_type.shape.dim) == 2

    # 出力テンソル契約の検証
    output_names = [out.name for out in model_proto.graph.output]
    assert output_names == [TENSOR_LOGITS]

    logits_output = model_proto.graph.output[0]
    # ONNX TensorProto.FLOAT = 1
    assert logits_output.type.tensor_type.elem_type == 1
    assert len(logits_output.type.tensor_type.shape.dim) == 2


def test_dynamic_axes_adaptation_and_parity(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """多次元バッチ・系列長・候補数での動的伸縮耐性と PyTorch/ORT 数値パリティを検証する。

    検証項目:
    - バッチサイズ 1, 2, 4 での実行。
    - 系列長 32, 64, 128 での実行。
    - 候補数 2, 5, 8 での実行。
    - 出力テンソル形状が [B, K] と完全一致すること。
    - PyTorch と ORT の最大絶対誤差が 1e-5 未満であること。
    """
    model, _ = lightweight_model_and_tokenizer
    onnx_file = tmp_path / "dynamic_model.onnx"

    export_onnx_model(model=model, output_path=onnx_file)

    test_matrix = [
        ("単一質問最小構成", 1, 32, 2),
        ("標準複数候補", 2, 64, 5),
        ("非対称並列バッチ", 4, 128, 8),
    ]

    val_result = validate_onnx_parity(
        model=model,
        onnx_path=onnx_file,
        test_cases=test_matrix,
        tolerance_atol=1.0e-5,
    )

    assert val_result.passed is True
    assert val_result.max_abs_error < 1.0e-5
    for case in val_result.test_case_results:
        assert case.passed is True
        assert not case.has_nan_or_inf


def test_extreme_boundary_dimensions(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """極端な境界値において squeeze などによる次元消失が発生しないか検証する。

    検証項目:
    - バッチサイズ 1 かつ 候補数 1 (Noul 等の最小境界) において形状 [1, 1] が維持されること。
    - 候補数 64 の多候補において正常に動作すること。
    """
    model, _ = lightweight_model_and_tokenizer
    onnx_file = tmp_path / "boundary_model.onnx"

    export_onnx_model(model=model, output_path=onnx_file)

    boundary_cases = [
        ("単一質問単一候補", 1, 32, 1),
        ("多候補構成", 1, 128, 64),
    ]

    val_result = validate_onnx_parity(
        model=model,
        onnx_path=onnx_file,
        test_cases=boundary_cases,
        tolerance_atol=1.0e-5,
    )

    assert val_result.passed is True
    assert val_result.test_case_results[0].num_options == 1
    assert val_result.test_case_results[1].num_options == 64


def test_artifact_bundle_creation(
    lightweight_model_and_tokenizer: tuple[JevDecisionModel, PreTrainedTokenizerFast],
    tmp_path: Path,
) -> None:
    """成果物バンドルが完全なファイル一式を生成するか検証する。

    検証項目:
    - model.onnx, tokenizer.json, tokenizer_config.json, config.json, calibration.json, export_metadata.json が存在すること。
    - export_metadata.json に環境情報および検証サマリーが正しく記録されていること。
    """
    model, tokenizer = lightweight_model_and_tokenizer
    onnx_file = tmp_path / "src_model.onnx"
    export_onnx_model(model=model, output_path=onnx_file)

    val_result = validate_onnx_parity(
        model=model,
        onnx_path=onnx_file,
        test_cases=[("簡易テスト", 1, 32, 2)],
    )

    bundle_dir = tmp_path / "bundle_output"
    create_artifact_bundle(
        output_dir=bundle_dir,
        onnx_path=onnx_file,
        tokenizer=tokenizer,
        backbone=model.backbone,
        calibration_path=None,
        validation_result=val_result,
    )

    expected_files = [
        "model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "calibration.json",
        "export_metadata.json",
    ]

    for fname in expected_files:
        target_file = bundle_dir / fname
        assert target_file.exists(), f"ファイルが存在しません: {fname}。"

    metadata_path = bundle_dir / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "git_commit" in metadata
    assert "environment" in metadata
    assert metadata["validation_summary"]["passed"] is True
