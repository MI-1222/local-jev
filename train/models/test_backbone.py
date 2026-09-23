"""バックボーンローダーおよび特殊トークン初期化の単体テストモジュール。"""

from typing import cast

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from contract import TOKEN_OPTION_MARKER
from models.backbone import (
    DEFAULT_BACKBONE_MODEL_ID,
    DEFAULT_MODERNBERT_JA_MODEL_ID,
    initialize_token_embedding_with_normalized_centroid,
    prepare_backbone_and_tokenizer,
    verify_option_marker_tokenization,
)


def test_centroid_normalized_initialization_math() -> None:
    """セントロイド正規化初期化の数理的計算精度とノルム一致性を検証する。

    検証項目:
    - 初期化後のベクトルが既存トークン群の平均 L2 ノルムと等しいこと。
    - 初期化後のベクトルが既存トークン群のセントロイドと同一方向を向いていること(コサイン類似度が 1.0)。
    """
    vocab_size = 10
    hidden_dim = 8
    target_id = 10
    # 11 行の重みテンソルを作成(既存 10 トークン + 新規 1 トークン)
    torch.manual_seed(42)
    weights = torch.randn(vocab_size + 1, hidden_dim)

    # 事前計算
    existing_weights = weights[:vocab_size].clone()
    expected_centroid = existing_weights.mean(dim=0)
    expected_avg_norm = torch.norm(existing_weights, p=2, dim=1).mean().item()

    returned_norm = initialize_token_embedding_with_normalized_centroid(
        embedding_weight=weights,
        target_token_id=target_id,
        existing_vocab_size=vocab_size,
    )

    # 戻り値の平均ノルムの検証
    assert pytest.approx(returned_norm, rel=1e-5) == expected_avg_norm

    # ターゲット埋め込みの L2 ノルムの検証
    actual_norm = torch.norm(weights[target_id], p=2).item()
    assert pytest.approx(actual_norm, rel=1e-5) == expected_avg_norm

    # セントロイドとのコサイン類似度の検証(同一方向)
    cosine_sim = torch.nn.functional.cosine_similarity(
        weights[target_id].unsqueeze(0),
        expected_centroid.unsqueeze(0),
    ).item()
    assert pytest.approx(cosine_sim, rel=1e-5) == 1.0


def test_verify_option_marker_tokenization_behavior() -> None:
    """verify_option_marker_tokenization の判定ロジックを検証する。

    検証項目:
    - [OP] が正しく埋め込まれている場合に True を返すこと。
    - 期待出現数と異なる場合に False を返すこと。
    """
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODERNBERT_JA_MODEL_ID)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    tokenizer.add_special_tokens({"additional_special_tokens": [TOKEN_OPTION_MARKER]})
    op_id = tokenizer.convert_tokens_to_ids(TOKEN_OPTION_MARKER)
    assert isinstance(op_id, int)

    # 正常系: [OP] が 2 回含まれるテキストで True
    assert verify_option_marker_tokenization(tokenizer, op_id) is True

    # 異常系: 誤った op_id(存在しないIDなど)を渡した場合に False
    dummy_op_id = 999999
    assert verify_option_marker_tokenization(tokenizer, dummy_op_id) is False


@pytest.mark.slow
def test_prepare_backbone_and_tokenizer_e2e_modernbert_ja() -> None:
    """第一推奨モデル(modernbert-ja-130m)を用いたバックボーン構築のE2E検証を行う。

    検証項目:
    - デフォルトモデルとして `sbintuitions/modernbert-ja-130m` がロードされること。
    - 特殊トークン `[OP]` が語彙に追加され、埋め込み層がリサイズされること。
    - 初期化された `[OP]` 埋め込みベクトルの L2 ノルムが、既存埋め込みの平均 L2 ノルムと整合していること。
    - トークナイズ検証に合格すること。
    """
    assert DEFAULT_BACKBONE_MODEL_ID == DEFAULT_MODERNBERT_JA_MODEL_ID
    assert DEFAULT_BACKBONE_MODEL_ID == "sbintuitions/modernbert-ja-130m"

    model, tokenizer, op_token_id = prepare_backbone_and_tokenizer()

    # 語彙サイズと埋め込み層の整合性
    embedding_module = cast(nn.Embedding, model.get_input_embeddings())
    assert len(tokenizer) == embedding_module.weight.shape[0]

    # [OP] トークンIDの検証
    assert tokenizer.convert_tokens_to_ids(TOKEN_OPTION_MARKER) == op_token_id

    # 埋め込みノルムの整合性検証
    weights = embedding_module.weight.detach()
    op_vector = weights[op_token_id]
    op_norm = torch.norm(op_vector, p=2).item()

    # 既存トークンの平均ノルム
    avg_token_norm = torch.norm(weights[:op_token_id], p=2, dim=1).mean().item()
    assert pytest.approx(op_norm, rel=1e-4) == avg_token_norm

    # トークナイズ検証の実行
    assert verify_option_marker_tokenization(tokenizer, op_token_id) is True
