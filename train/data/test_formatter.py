"""プロンプトフォーマッタおよびトークナイズの単体テストモジュール。

特に State 優先トランケーションにより、Criteria ([OP] マーカー群) が
絶対に欠落しないことを厳密に検証する。
"""

import pytest

from contract import TOKEN_OPTION_MARKER
from data.formatter import format_prompt, tokenize_sample
from data.schema import QuestionType, UnifiedSample
from models.backbone import DEFAULT_MODERNBERT_MODEL_ID, prepare_backbone_and_tokenizer


@pytest.fixture(scope="module")
def modernbert_tokenizer_and_op():
    """ModernBERT トークナイザーと [OP] トークン ID を提供するフィクスチャ。"""
    _, tokenizer, op_token_id = prepare_backbone_and_tokenizer(
        DEFAULT_MODERNBERT_MODEL_ID
    )
    return tokenizer, op_token_id


def test_format_prompt_choice() -> None:
    """Choice 型のプロンプトフォーマットテスト。"""
    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b1",
        question_type=QuestionType.CHOICE,
        state="I want to activate my new card.",
        instructions="カテゴリを選択せよ。",
        criteria={
            "activate_my_card": "Activating a newly received card",
            "card_arrival": "Delivery status of card",
        },
        target="activate_my_card",
    )
    prompt, option_keys = format_prompt(sample, shuffle_options=False)

    assert "State: I want to activate my new card." in prompt
    assert "Instructions: カテゴリを選択せよ。" in prompt
    assert (
        f"Criteria: {TOKEN_OPTION_MARKER} Activating a newly received card {TOKEN_OPTION_MARKER} Delivery status of card"
        in prompt
    )
    assert option_keys == ["activate_my_card", "card_arrival"]


def test_format_prompt_noul() -> None:
    """Noul 型のプロンプトフォーマットテスト。"""
    sample = UnifiedSample(
        dataset_name="mnli",
        sample_id="m1",
        question_type=QuestionType.NOUL,
        state="The sky is blue.",
        instructions="言明「空は青い」の真偽を判定せよ。",
        criteria={},
        target="true",
    )
    prompt, option_keys = format_prompt(sample)

    assert "State: The sky is blue." in prompt
    assert f"{TOKEN_OPTION_MARKER} 真 (True)" in prompt
    assert f"{TOKEN_OPTION_MARKER} 偽 (False)" in prompt
    assert option_keys == ["true", "false"]


def test_tokenize_sample_normal(modernbert_tokenizer_and_op) -> None:
    """通常のトークナイズと [OP] マーカー抽出テスト。"""
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b2",
        question_type=QuestionType.CHOICE,
        state="I lost my card yesterday.",
        instructions="意図を分類せよ。",
        criteria={
            "card_arrival": "Card delivery status",
            "lost_card": "Reporting lost card",
            "pin_reset": "Resetting PIN code",
        },
        target="lost_card",
    )

    tensors = tokenize_sample(
        sample=sample,
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=512,
    )

    assert "input_ids" in tensors
    assert "attention_mask" in tensors
    assert "op_indices" in tensors
    assert "label" in tensors

    # [OP] の出現数と候補数が一致 (3候補)
    assert len(tensors["op_indices"]) == 3

    # 正解 target の index ('lost_card' は index 1)
    assert tensors["label"].item() == 1

    # 各 [OP] 位置のトークンIDが op_token_id と一致
    for idx in tensors["op_indices"]:
        assert tensors["input_ids"][idx].item() == op_token_id


def test_tokenize_sample_state_truncation(modernbert_tokenizer_and_op) -> None:
    """長大 State 入力時における Criteria 保持トランケーションの厳密検証。

    State を極端に長くし、max_length を小さく設定した場合でも、
    文末の Criteria ([OP] マーカー) が一切切り落とされずに保持されることを確認する。
    """
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    # 非常に長い State テキスト (数千文字)
    long_state = (
        "This is an extremely long transaction log with hundreds of details. " * 50
    )

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_long",
        question_type=QuestionType.CHOICE,
        state=long_state,
        instructions="分類せよ。",
        criteria={
            "opt_a": "First option description",
            "opt_b": "Second option description",
            "opt_c": "Third option description",
            "opt_d": "Fourth option description",
        },
        target="opt_c",
    )

    # 狭い max_length (128) を指定
    max_len = 128
    tensors = tokenize_sample(
        sample=sample,
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=max_len,
    )

    # 系列長が上限に収まっていること
    assert len(tensors["input_ids"]) <= max_len

    # 【重要】State が切り詰められても、4候補すべての [OP] マーカーが完全に保持されていること
    assert len(tensors["op_indices"]) == 4

    # 正解ラベル ('opt_c' は index 2) が正しく取得できること
    assert tensors["label"].item() == 2

    # 全 [OP] 位置のトークンが op_token_id と一致すること
    for idx in tensors["op_indices"]:
        assert tensors["input_ids"][idx].item() == op_token_id
