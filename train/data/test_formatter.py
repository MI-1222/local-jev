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


def test_format_prompt_choice_shuffling() -> None:
    """Choice 型において候補が正しくシャッフルされ、決定論的に再現されることを検証する。"""
    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_shuffle",
        question_type=QuestionType.CHOICE,
        state="Card inquiry.",
        instructions="分類せよ。",
        criteria={
            "c0": "Option Zero",
            "c1": "Option One",
            "c2": "Option Two",
            "c3": "Option Three",
            "c4": "Option Four",
        },
        target="c2",
    )

    # シャッフルなし
    _, keys_unshuffled = format_prompt(sample, shuffle_options=False)
    assert keys_unshuffled == ["c0", "c1", "c2", "c3", "c4"]

    # シャッフルあり (シード 42)
    prompt_seed42, keys_seed42 = format_prompt(sample, shuffle_options=True, seed=42)
    # 同一シードでの再現性
    prompt_seed42_repeat, keys_seed42_repeat = format_prompt(
        sample, shuffle_options=True, seed=42
    )
    assert prompt_seed42 == prompt_seed42_repeat
    assert keys_seed42 == keys_seed42_repeat

    # 異なるシード (シード 999)
    _, keys_seed999 = format_prompt(sample, shuffle_options=True, seed=999)
    # シャッフルにより順序が変化していること
    assert keys_seed42 != keys_unshuffled or keys_seed999 != keys_unshuffled
    assert set(keys_seed42) == set(keys_unshuffled)


def test_format_prompt_noul_never_shuffles() -> None:
    """Noul 型では shuffle_options が True であっても絶対にシャッフルされないことを検証する。"""
    sample = UnifiedSample(
        dataset_name="mnli",
        sample_id="m_noul_shuffle_test",
        question_type=QuestionType.NOUL,
        state="The sky is blue.",
        instructions="真偽判定。",
        criteria={},
        target="true",
    )

    for test_seed in [0, 42, 12345, 999999]:
        prompt, option_keys = format_prompt(
            sample, shuffle_options=True, seed=test_seed
        )
        assert option_keys == ["true", "false"]
        # プロンプト内にも true -> false の順で展開されていること
        assert prompt.index("真 (True)") < prompt.index("偽 (False)")


def test_format_prompt_score_never_shuffles() -> None:
    """Score 型では shuffle_options が True であっても順序尺度が維持されることを検証する。"""
    sample = UnifiedSample(
        dataset_name="score_task",
        sample_id="s_score_shuffle_test",
        question_type=QuestionType.SCORE,
        state="Good experience.",
        instructions="評価せよ。",
        criteria={
            "0": "0点 (最悪)",
            "1": "1点 (悪い)",
            "2": "2点 (普通)",
            "3": "3点 (良い)",
            "4": "4点 (最高)",
        },
        target="3",
    )

    for test_seed in [0, 42, 9999]:
        prompt, option_keys = format_prompt(
            sample, shuffle_options=True, seed=test_seed
        )
        assert option_keys == ["0", "1", "2", "3", "4"]
        assert "0点 (最悪)" in prompt


def test_tokenize_sample_choice_label_tracking(modernbert_tokenizer_and_op) -> None:
    """Choice 型のシャッフル時に正解ラベルインデックスが新位置を正しく追従することを検証する。"""
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_track",
        question_type=QuestionType.CHOICE,
        state="I want to update my PIN code.",
        instructions="分類せよ。",
        criteria={
            "c0": "Card delivery",
            "c1": "Exchange rate",
            "c2": "PIN code reset",
            "c3": "Fraud report",
            "c4": "Account balance",
        },
        target="c2",
    )

    target_indices_found: set[int] = set()

    for seed_val in range(20):
        tensors = tokenize_sample(
            sample=sample,
            tokenizer=tokenizer,
            op_token_id=op_token_id,
            shuffle_options=True,
            seed=seed_val,
        )

        label_idx = int(tensors["label"].item())
        target_indices_found.add(label_idx)

        # op_indices の個数および op_token_id の整合性確認
        assert len(tensors["op_indices"]) == 5
        for op_idx in tensors["op_indices"]:
            assert tensors["input_ids"][op_idx].item() == op_token_id

    # 複数シードを試すことで、target="c2" の位置が複数のインデックスへ動的に遷移していること
    assert len(target_indices_found) > 1


def test_tokenize_sample_target_not_in_criteria_raises(
    modernbert_tokenizer_and_op,
) -> None:
    """正解ラベル target が Criteria に含まれていない場合に ValueError が送出されることを検証する。"""
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_invalid_target",
        question_type=QuestionType.CHOICE,
        state="Help me.",
        instructions="分類せよ。",
        criteria={
            "opt_a": "Option A",
            "opt_b": "Option B",
        },
        target="opt_a",
    )
    # frozen dataclass に対して object.__setattr__ で不正なターゲットを設定し、
    # tokenize_sample 側の防壁を検証する
    object.__setattr__(sample, "target", "opt_nonexistent")

    with pytest.raises(
        ValueError, match="正解ラベル .* が候補リストに含まれていません"
    ):
        tokenize_sample(
            sample=sample,
            tokenizer=tokenizer,
            op_token_id=op_token_id,
        )


def test_tokenize_sample_fixed_len_exceeds_max_length_raises(
    modernbert_tokenizer_and_op,
) -> None:
    """Instructions と Criteria だけで max_length を超過した場合に ValueError が送出されることを検証する。"""
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_too_long",
        question_type=QuestionType.CHOICE,
        state="Short state.",
        instructions="非常に長い指示文 " * 30,
        criteria={
            "opt_a": "長い説明文 " * 20,
            "opt_b": "さらに長い説明文 " * 20,
        },
        target="opt_a",
    )

    # 極小の max_length (16) を指定して確実に超過させる
    with pytest.raises(ValueError, match="max_length .* を超過しているため"):
        tokenize_sample(
            sample=sample,
            tokenizer=tokenizer,
            op_token_id=op_token_id,
            max_length=16,
        )
