"""階層データセット生成・トークナイズ・Collate の単体テスト。"""

import random

import pytest
from transformers import AutoTokenizer

from data.hierarchical import HierarchicalMapping
from data.hierarchical_dataset import (
    HierarchicalDatasetGenerator,
    HierarchicalJevDataset,
    hierarchical_collate_fn,
)
from data.schema import QuestionType, UnifiedSample


@pytest.fixture
def dummy_mapping() -> HierarchicalMapping:
    """テスト用の小規模階層オントロジーを提供する。"""
    return HierarchicalMapping(
        name="TestHierarchy",
        coarse_categories={
            "c_card": "カード関連",
            "c_transfer": "送金関連",
        },
        fine_criteria={
            "card_lost": "カードを紛失した。",
            "card_pin": "暗証番号を忘れた。",
            "card_other": "カードその他の問題。",
            "transfer_delay": "送金が遅延している。",
            "transfer_fee": "送金手数料が高い。",
        },
        coarse_to_fine={
            "c_card": ["card_lost", "card_pin", "card_other"],
            "c_transfer": ["transfer_delay", "transfer_fee"],
        },
        fine_to_coarse={
            "card_lost": "c_card",
            "card_pin": "c_card",
            "card_other": "c_card",
            "transfer_delay": "c_transfer",
            "transfer_fee": "c_transfer",
        },
        all_coarse_keys=["c_card", "c_transfer"],
        all_fine_keys=[
            "card_lost",
            "card_pin",
            "card_other",
            "transfer_delay",
            "transfer_fee",
        ],
        negative_keys=["card_other"],
    )


from typing import cast

from transformers import PreTrainedTokenizerFast


@pytest.fixture
def mock_tokenizer() -> PreTrainedTokenizerFast:
    """特殊トークン [OP] を含むテスト用トークナイザーを提供する。"""
    raw_tok = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-130m")
    tok = cast(PreTrainedTokenizerFast, raw_tok)
    if "[OP]" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["[OP]"]})
    return tok


def test_generator_oracle_pair(dummy_mapping: HierarchicalMapping) -> None:
    """Oracle ペア生成時に大分類と細分類の正解が整合することを検証する。"""
    gen = HierarchicalDatasetGenerator(
        mapping=dummy_mapping,
        soft_beam_ratio=0.0,  # Oracle のみ
    )
    sample = UnifiedSample(
        dataset_name="Test",
        sample_id="s1",
        question_type=QuestionType.CHOICE,
        state="カードを落としてしまいました。",
        instructions="詳細意図を選択してください。",
        criteria=dict(dummy_mapping.fine_criteria),
        target="card_lost",
    )

    pair = gen.generate_pair(sample)
    assert pair.is_soft_beam is False

    # 大分類の正解は c_card
    assert pair.coarse_sample.target == "c_card"
    assert len(pair.coarse_sample.criteria) == 2
    assert "c_card" in pair.coarse_sample.criteria

    # 細分類の正解は card_lost、候補は c_card 配下の 3 件
    assert pair.fine_sample.target == "card_lost"
    assert len(pair.fine_sample.criteria) == 3
    assert "card_lost" in pair.fine_sample.criteria
    assert "card_pin" in pair.fine_sample.criteria
    assert "transfer_delay" not in pair.fine_sample.criteria


def test_generator_soft_beam_pair(dummy_mapping: HierarchicalMapping) -> None:
    """Soft-Beam 模倣ペア生成時に近傍クラスタの候補がマージされることを検証する。"""
    gen = HierarchicalDatasetGenerator(
        mapping=dummy_mapping,
        soft_beam_ratio=1.0,  # Soft-Beam のみ
    )
    sample = UnifiedSample(
        dataset_name="Test",
        sample_id="s2",
        question_type=QuestionType.CHOICE,
        state="カードの暗証番号がわかりません。",
        instructions="詳細意図を選択してください。",
        criteria=dict(dummy_mapping.fine_criteria),
        target="card_pin",
    )

    pair = gen.generate_pair(sample, rng=random.Random(42))
    assert pair.is_soft_beam is True

    # 正解クラスタ (c_card) + 近傍クラスタ (c_transfer) の候補が含まれる
    assert "card_pin" in pair.fine_sample.criteria
    assert "transfer_delay" in pair.fine_sample.criteria
    assert len(pair.fine_sample.criteria) == 5


def test_hierarchical_dataset_and_collate(
    dummy_mapping: HierarchicalMapping,
    mock_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """HierarchicalJevDataset と hierarchical_collate_fn がバッチを正しく構築することを検証する。"""
    gen = HierarchicalDatasetGenerator(
        mapping=dummy_mapping,
        soft_beam_ratio=0.5,
    )
    samples = [
        UnifiedSample(
            dataset_name="Test",
            sample_id=f"s_{i}",
            question_type=QuestionType.CHOICE,
            state=f"テスト文脈 {i}",
            instructions="指示",
            criteria=dict(dummy_mapping.fine_criteria),
            target="card_lost" if i % 2 == 0 else "transfer_delay",
        )
        for i in range(4)
    ]

    pairs = gen.generate_pairs(samples, base_seed=42)
    op_token_id = mock_tokenizer.convert_tokens_to_ids("[OP]")
    assert isinstance(op_token_id, int)

    dataset = HierarchicalJevDataset(
        pairs=pairs,
        mapping=dummy_mapping,
        tokenizer=mock_tokenizer,
        op_token_id=op_token_id,
        is_train=True,
    )
    assert len(dataset) == 4

    item0 = dataset[0]
    assert "coarse" in item0
    assert "fine" in item0
    assert "fine_parent_indices" in item0

    # Collate の検証
    pad_id = mock_tokenizer.pad_token_id or 0
    batch = hierarchical_collate_fn([dataset[0], dataset[1]], pad_token_id=pad_id)

    assert batch["coarse_input_ids"].shape[0] == 2
    assert batch["fine_input_ids"].shape[0] == 2
    assert batch["coarse_op_mask"].shape[0] == 2
    assert batch["fine_op_mask"].shape[0] == 2
    assert batch["fine_parent_indices"].shape[0] == 2
    assert batch["coarse_labels"].shape[0] == 2
    assert batch["fine_labels"].shape[0] == 2
