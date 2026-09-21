"""データセットコンバータおよび統一スキーマの単体テストモジュール。"""

import pytest
from datasets import ClassLabel, Dataset, Features, Value

from data.converters.ag_news import AGNewsConverter
from data.converters.banking77 import Banking77Converter, clean_banking77_label
from data.converters.clinc150 import Clinc150Converter
from data.converters.mnli import MNLIConverter
from data.converters.sst5 import SST5Converter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample


def test_unified_sample_validation_choice() -> None:
    """Choice 型の target 検証テスト。"""
    # 正常系
    sample = UnifiedSample(
        dataset_name="test",
        sample_id="test_1",
        question_type=QuestionType.CHOICE,
        state="ユーザーの質問文。",
        instructions="カテゴリを選択せよ。",
        criteria={"cat_a": "カテゴリA", "cat_b": "カテゴリB"},
        target="cat_a",
    )
    assert sample.target == "cat_a"
    assert sample.to_dict()["question_type"] == "choice"

    # target が criteria に存在しない場合は ValueError
    with pytest.raises(ValueError, match="target 'invalid' が criteria"):
        UnifiedSample(
            dataset_name="test",
            sample_id="test_2",
            question_type=QuestionType.CHOICE,
            state="ユーザーの質問文。",
            instructions="カテゴリを選択せよ。",
            criteria={"cat_a": "カテゴリA"},
            target="invalid",
        )


def test_unified_sample_validation_noul() -> None:
    """Noul 型の target 検証テスト。"""
    sample = UnifiedSample(
        dataset_name="test",
        sample_id="test_noul_1",
        question_type=QuestionType.NOUL,
        state="前提事実。",
        instructions="真偽を判定せよ。",
        criteria={},
        target="true",
    )
    assert sample.target == "true"

    with pytest.raises(ValueError, match="Noul 型の target は 'true' または 'false'"):
        UnifiedSample(
            dataset_name="test",
            sample_id="test_noul_2",
            question_type=QuestionType.NOUL,
            state="前提事実。",
            instructions="真偽を判定せよ。",
            criteria={},
            target="maybe",
        )


def test_banking77_converter_mock() -> None:
    """Banking77 コンバータのモック Dataset を用いた変換テスト。"""
    label_names = [
        "card_arrival",
        "pin_blocked",
        "lost_or_stolen_card",
        "top_up_failed",
    ]
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=label_names),
        }
    )
    mock_data = {
        "text": ["Where is my card?", "I forgot my pin"],
        "label": [0, 1],
    }
    ds = Dataset.from_dict(mock_data, features=features)

    # 全候補展開モード
    converter = Banking77Converter(max_negative_options=None, seed=42)
    samples = list(converter.convert_dataset(ds, split="test"))

    assert len(samples) == 2
    first = samples[0]
    assert first.question_type == QuestionType.CHOICE
    assert first.target == "card_arrival"
    assert "card_arrival" in first.criteria
    assert len(first.criteria) == 4
    assert first.criteria["card_arrival"] == clean_banking77_label("card_arrival")

    # サブサンプリングモード (正解1 + 負例1)
    converter_sub = Banking77Converter(max_negative_options=1, seed=42)
    samples_sub = list(converter_sub.convert_dataset(ds, split="train"))
    assert len(samples_sub[0].criteria) == 2
    assert samples_sub[0].target in samples_sub[0].criteria


def test_clinc150_converter_mock() -> None:
    """CLINC150 コンバータの oos 保持テスト。"""
    label_names = ["balance", "transfer", "oos"]
    features = Features(
        {
            "text": Value("string"),
            "intent": ClassLabel(names=label_names),
        }
    )
    mock_data = {
        "text": ["What is my balance?", "Tell me a joke"],
        "intent": [0, 2],
    }
    ds = Dataset.from_dict(mock_data, features=features)

    converter = Clinc150Converter(max_negative_options=None, seed=42)
    samples = list(converter.convert_dataset(ds, split="train"))

    assert len(samples) == 2
    oos_sample = samples[1]
    assert oos_sample.target == "oos"
    assert "oos" in oos_sample.criteria
    assert oos_sample.metadata["is_oos"] is True
    assert "None of the above" in oos_sample.criteria["oos"]


def test_mnli_converter_choice_and_noul_mock() -> None:
    """MNLI コンバータの Choice 型および Noul 型変換テスト。"""
    mock_data = {
        "premise": [
            "A man is playing soccer.",
            "The dog is sleeping.",
            "Cats are playing outside.",
        ],
        "hypothesis": [
            "A person is outdoors.",
            "The dog is running in the park.",
            "Animals are outside.",
        ],
        "label": [0, 2, 1],  # 0: entailment, 2: contradiction, 1: neutral
    }
    ds = Dataset.from_dict(mock_data)

    # Choice モード
    converter_choice = MNLIConverter(mode="choice", seed=42)
    choice_samples = list(converter_choice.convert_dataset(ds, split="test"))
    assert len(choice_samples) == 3
    assert choice_samples[0].target == "entailment"
    assert choice_samples[1].target == "contradiction"
    assert choice_samples[2].target == "neutral"
    assert len(choice_samples[0].criteria) == 3

    # Noul モード (neutral は除外される)
    converter_noul = MNLIConverter(mode="noul", seed=42)
    noul_samples = list(converter_noul.convert_dataset(ds, split="test"))
    assert len(noul_samples) == 2  # neutral が除外されて 2 件
    assert noul_samples[0].target == "true"
    assert noul_samples[1].target == "false"
    assert noul_samples[0].question_type == QuestionType.NOUL


def test_ag_news_converter_mock() -> None:
    """AG News コンバータの変換テスト。"""
    mock_data = {
        "text": ["Stock market hits all time high", "Team wins championship"],
        "label": [2, 1],  # 2: Business, 1: Sports
    }
    ds = Dataset.from_dict(mock_data)

    converter = AGNewsConverter(seed=42)
    samples = list(converter.convert_dataset(ds, split="train"))

    assert len(samples) == 2
    assert samples[0].target == "business"
    assert samples[1].target == "sports"
    assert len(samples[0].criteria) == 4


def test_prompt_pool_sampling() -> None:
    """指示文テンプレートサンプリングのテスト。"""
    inst = sample_instruction("intent")
    assert isinstance(inst, str)
    assert len(inst) > 0

    noul_inst = sample_instruction("noul", hypothesis="テスト言明")
    assert "テスト言明" in noul_inst

    score_inst = sample_instruction("score")
    assert isinstance(score_inst, str)
    assert len(score_inst) > 0

    rating_inst = sample_instruction("rating")
    assert isinstance(rating_inst, str)
    assert len(rating_inst) > 0


def test_unified_sample_validation_score() -> None:
    """Score 型のバリデーションテスト。"""
    # 正常系 (5段階)
    criteria_5 = {str(i): f"レベル{i}" for i in range(5)}
    sample = UnifiedSample(
        dataset_name="test_score",
        sample_id="score_1",
        question_type=QuestionType.SCORE,
        state="評価対象テキスト。",
        instructions="段階を評価せよ。",
        criteria=criteria_5,
        target="3",
    )
    assert sample.target == "3"
    assert sample.to_dict()["question_type"] == "score"

    # 段階数不足 (< 2)
    with pytest.raises(ValueError, match="2 段階以上 10 段階以下"):
        UnifiedSample(
            dataset_name="test_score",
            sample_id="score_invalid_len",
            question_type=QuestionType.SCORE,
            state="テキスト。",
            instructions="評価せよ。",
            criteria={"0": "単一レベル"},
            target="0",
        )

    # 段階数超過 (> 10)
    with pytest.raises(ValueError, match="2 段階以上 10 段階以下"):
        UnifiedSample(
            dataset_name="test_score",
            sample_id="score_invalid_len2",
            question_type=QuestionType.SCORE,
            state="テキスト。",
            instructions="評価せよ。",
            criteria={str(i): f"レベル{i}" for i in range(11)},
            target="0",
        )

    # キーが 0 オリジン連番でない (1〜5 の場合)
    with pytest.raises(ValueError, match="0 から始まる昇順連番"):
        UnifiedSample(
            dataset_name="test_score",
            sample_id="score_invalid_keys",
            question_type=QuestionType.SCORE,
            state="テキスト。",
            instructions="評価せよ。",
            criteria={str(i): f"レベル{i}" for i in range(1, 6)},
            target="3",
        )

    # target が criteria に含まれない
    with pytest.raises(ValueError, match="target '9' が criteria"):
        UnifiedSample(
            dataset_name="test_score",
            sample_id="score_invalid_target",
            question_type=QuestionType.SCORE,
            state="テキスト。",
            instructions="評価せよ。",
            criteria=criteria_5,
            target="9",
        )


def test_sst5_converter_mock() -> None:
    """SST-5 コンバータのモック Dataset を用いた変換テスト。"""
    mock_data = {
        "text": [
            "Terrible and utterly boring.",
            "Not good, had many flaws.",
            "Average movie, neither good nor bad.",
            "Pretty enjoyable and well acted.",
            "Masterpiece of modern cinema!",
        ],
        "label": [0, 1, 2, 3, 4],
        "label_text": [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
    }
    ds = Dataset.from_dict(mock_data)

    converter = SST5Converter(seed=42)
    samples = list(converter.convert_dataset(ds, split="test"))

    assert len(samples) == 5
    for idx, s in enumerate(samples):
        assert s.question_type == QuestionType.SCORE
        assert s.target == str(idx)
        assert len(s.criteria) == 5
        # criteria のキーが '0' 〜 '4' の昇順であること
        assert list(s.criteria.keys()) == ["0", "1", "2", "3", "4"]
        assert s.metadata["original_score"] == idx
        assert s.metadata["scale_min"] == 0
        assert s.metadata["scale_max"] == 4
        assert s.metadata["scale_levels"] == 5
        assert s.metadata["original_label_text"] == mock_data["label_text"][idx]

    # クラス平準化 (max_samples_per_class) の検証
    mock_imbalanced = {
        "text": [
            "Bad 1",
            "Bad 2",
            "Bad 3",
            "Good 1",
            "Good 2",
        ],
        "label": [0, 0, 0, 4, 4],
    }
    ds_imbalanced = Dataset.from_dict(mock_imbalanced)
    converter_balanced = SST5Converter(max_samples_per_class=1, seed=42)
    balanced_samples = list(
        converter_balanced.convert_dataset(ds_imbalanced, split="train")
    )
    # クラス0から1件、クラス4から1件の計2件のみ抽出されること
    assert len(balanced_samples) == 2
    assert [s.target for s in balanced_samples] == ["0", "4"]
