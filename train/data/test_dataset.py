"""位置バイアス排除 Dataset および Collate 関数の単体テストモジュール。

Choice 型のエポック連動動的シャッフル、Score 型および Noul 型の順序不変性、
評価モードでの決定論的挙動、ならびに可変長バッチのパディング動作を網羅的に検証する。
"""

import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import JevDataset, jev_collate_fn
from data.schema import QuestionType, UnifiedSample
from models.backbone import DEFAULT_MODERNBERT_MODEL_ID, prepare_backbone_and_tokenizer


@pytest.fixture(scope="module")
def modernbert_tokenizer_and_op():
    """ModernBERT トークナイザーと [OP] トークン ID を提供するフィクスチャ。

    Returns:
        tuple[PreTrainedTokenizerFast, int]: トークナイザーおよび [OP] ID。
    """
    _, tokenizer, op_token_id = prepare_backbone_and_tokenizer(
        DEFAULT_MODERNBERT_MODEL_ID
    )
    return tokenizer, op_token_id


def test_jev_dataset_epoch_shuffling_choice(modernbert_tokenizer_and_op) -> None:
    """Choice 型におけるエポック連動シャッフルと [OP] 整合性の検証。

    エポックを進めることで候補の並び順および正解ラベルインデックスが変化し、
    かつ各 [OP] マーカー位置が常に op_token_id と一致することを検証する。
    """
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_shuffle_test",
        question_type=QuestionType.CHOICE,
        state="I want to cancel my card immediately.",
        instructions="カテゴリを選択せよ。",
        criteria={
            "opt_0": "Delivery delay of new card",
            "opt_1": "Cancelling my active credit card",
            "opt_2": "Resetting PIN password",
            "opt_3": "Checking exchange rate",
            "opt_4": "Reporting suspicious charge",
        },
        target="opt_1",
    )

    dataset = JevDataset(
        samples=[sample],
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=512,
        is_train=True,
    )

    labels_across_epochs: list[int] = []

    for ep in range(10):
        dataset.set_epoch(ep)
        item = dataset[0]

        # op_indices が指す全トークンが [OP] であることを確認
        assert len(item["op_indices"]) == 5
        for op_idx in item["op_indices"]:
            assert item["input_ids"][op_idx].item() == op_token_id

        labels_across_epochs.append(int(item["label"].item()))

    # 10エポックの間で少なくとも複数の異なるラベルインデックス(位置)が現れること
    unique_labels = set(labels_across_epochs)
    assert len(unique_labels) > 1, (
        f"ラベル位置がシャッフルされていません: {labels_across_epochs}。"
    )


def test_jev_dataset_score_no_shuffling(modernbert_tokenizer_and_op) -> None:
    """Score 型において候補順序が絶対にシャッフルされないことの検証。

    順序尺度(Ordinality)を保護するため、エポックを変更しても候補順序や
    ラベルインデックス、op_indices が完全に一致し続けることを確認する。
    """
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="feedback_score",
        sample_id="s_score_test",
        question_type=QuestionType.SCORE,
        state="The customer service was terribly slow and unhelpful.",
        instructions="深刻度を1から5で判定せよ。",
        criteria={
            "1": "非常に不満 (極めて深刻)",
            "2": "不満 (問題あり)",
            "3": "普通 (平均的)",
            "4": "満足 (良好)",
            "5": "大変満足 (素晴らしい)",
        },
        target="1",
    )

    dataset = JevDataset(
        samples=[sample],
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=512,
        is_train=True,
    )

    dataset.set_epoch(0)
    item_epoch_0 = dataset[0]

    for ep in [1, 2, 5, 10]:
        dataset.set_epoch(ep)
        item = dataset[0]
        # input_ids, op_indices, label が完全一致すること
        assert torch.equal(item["input_ids"], item_epoch_0["input_ids"])
        assert torch.equal(item["op_indices"], item_epoch_0["op_indices"])
        assert item["label"].item() == item_epoch_0["label"].item()


def test_jev_dataset_noul_no_shuffling(modernbert_tokenizer_and_op) -> None:
    """Noul 型において真偽順序が固定されていることの検証。

    二値契約(true: 0, false: 1)を保持するため、エポックの進行によらず
    常に同一の順序で出力されることを確認する。
    """
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="mnli",
        sample_id="n_noul_test",
        question_type=QuestionType.NOUL,
        state="The company reported record quarterly revenue.",
        instructions="業績は好調であるか判定せよ。",
        criteria={},
        target="true",
    )

    dataset = JevDataset(
        samples=[sample],
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=512,
        is_train=True,
    )

    dataset.set_epoch(0)
    item_epoch_0 = dataset[0]
    assert item_epoch_0["label"].item() == 0  # true は常に index 0

    for ep in [1, 2, 3]:
        dataset.set_epoch(ep)
        item = dataset[0]
        assert torch.equal(item["input_ids"], item_epoch_0["input_ids"])
        assert item["label"].item() == 0


def test_jev_dataset_eval_mode_no_shuffling(modernbert_tokenizer_and_op) -> None:
    """評価モード (is_train=False) においてシャッフルが無効化されることの検証。

    検証・テスト時における再現性を担保するため、Choice 型であっても
    順序が固定されることを確認する。
    """
    tokenizer, op_token_id = modernbert_tokenizer_and_op

    sample = UnifiedSample(
        dataset_name="banking77",
        sample_id="b_eval_test",
        question_type=QuestionType.CHOICE,
        state="Where is my card?",
        instructions="分類せよ。",
        criteria={
            "c_0": "Option Alpha",
            "c_1": "Option Beta",
            "c_2": "Option Gamma",
        },
        target="c_0",
    )

    dataset = JevDataset(
        samples=[sample],
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=512,
        is_train=False,
    )

    dataset.set_epoch(0)
    item_epoch_0 = dataset[0]

    for ep in [1, 2, 5]:
        dataset.set_epoch(ep)
        item = dataset[0]
        assert torch.equal(item["input_ids"], item_epoch_0["input_ids"])
        assert item["label"].item() == item_epoch_0["label"].item()


def test_jev_collate_fn_padding(modernbert_tokenizer_and_op) -> None:
    """可変長系列および可変候補数サンプルのパディング検証。

    系列長や候補数が異なる複数のサンプルを正しく単一テンソルに結合し、
    op_mask や labels が正しく構成されることを確認する。
    """
    tokenizer, op_token_id = modernbert_tokenizer_and_op
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    sample_short = UnifiedSample(
        dataset_name="test_short",
        sample_id="s1",
        question_type=QuestionType.CHOICE,
        state="Short state.",
        instructions="指示1。",
        criteria={
            "o1": "First choice",
            "o2": "Second choice",
        },
        target="o1",
    )
    sample_long = UnifiedSample(
        dataset_name="test_long",
        sample_id="s2",
        question_type=QuestionType.CHOICE,
        state="A much longer state text that contains multiple sentences.",
        instructions="指示2。",
        criteria={
            "o1": "First choice",
            "o2": "Second choice",
            "o3": "Third choice",
            "o4": "Fourth choice",
        },
        target="o3",
    )

    dataset = JevDataset(
        samples=[sample_short, sample_long],
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=512,
        is_train=False,
    )

    batch = [dataset[0], dataset[1]]
    collated = jev_collate_fn(batch, pad_token_id=pad_id)

    # 形状の整合性確認
    assert collated["input_ids"].ndim == 2
    assert collated["attention_mask"].ndim == 2
    assert collated["op_indices"].ndim == 2
    assert collated["op_mask"].ndim == 2
    assert collated["labels"].ndim == 1

    batch_size = 2
    assert collated["input_ids"].shape[0] == batch_size
    assert collated["op_indices"].shape[0] == batch_size
    assert collated["labels"].shape[0] == batch_size

    # sample_long の方が候補数が多いため、max_options は 4
    assert collated["op_indices"].shape[1] == 4
    assert collated["op_mask"].shape[1] == 4

    # サンプル0 (候補数2) の op_mask: [True, True, False, False]
    assert collated["op_mask"][0].tolist() == [True, True, False, False]
    # サンプル1 (候補数4) の op_mask: [True, True, True, True]
    assert collated["op_mask"][1].tolist() == [True, True, True, True]

    # DataLoader との統合動作確認
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: jev_collate_fn(b, pad_token_id=pad_id),
    )
    batch_loaded = next(iter(loader))
    assert batch_loaded["input_ids"].shape == collated["input_ids"].shape
    assert batch_loaded["labels"].tolist() == collated["labels"].tolist()
