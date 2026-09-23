"""データセットコンバータおよび統一スキーマの単体テストモジュール。"""

import pytest
from datasets import ClassLabel, Dataset, Features, Value

from data.builders import UnifiedDatasetBuilder
from data.converters.ag_news import AGNewsConverter
from data.converters.banking77 import Banking77Converter, clean_banking77_label
from data.converters.clinc150 import Clinc150Converter
from data.converters.jglue_jcommonsenseqa import JGlueJCommonsenseQAConverter
from data.converters.jglue_jnli import JGlueJNLIConverter
from data.converters.jglue_jsts import JGlueJSTSConverter
from data.converters.jglue_marc_ja import JGlueMarcJaConverter
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


def test_jglue_marc_ja_converter_mock() -> None:
    """JGLUE MARC-ja コンバータのモック Dataset を用いた変換テスト。"""
    features = Features(
        {
            "sentence": Value("string"),
            "label": ClassLabel(names=["positive", "negative", "neutral"]),
            "review_id": Value("string"),
        }
    )
    mock_data = {
        "sentence": [
            "素晴らしい商品でした。大満足です。",
            "すぐに壊れてしまい使い物になりません。",
            "普通です。可もなく不可もありません。",
        ],
        "label": [0, 1, 2],
        "review_id": ["rev_pos", "rev_neg", "rev_neu"],
    }
    ds = Dataset.from_dict(mock_data, features=features)

    # Noul モードの検証 (neutral は除外されるため 2 件になる)
    conv_noul = JGlueMarcJaConverter(mode="noul", seed=42)
    samples_noul = list(conv_noul.convert_dataset(ds, split="test"))
    assert len(samples_noul) == 2

    assert samples_noul[0].question_type == QuestionType.NOUL
    assert samples_noul[0].target == "true"
    assert samples_noul[0].criteria == {}
    assert samples_noul[0].state == mock_data["sentence"][0]

    assert samples_noul[1].question_type == QuestionType.NOUL
    assert samples_noul[1].target == "false"
    assert samples_noul[1].criteria == {}
    assert samples_noul[1].state == mock_data["sentence"][1]

    # Choice モードの検証
    conv_choice = JGlueMarcJaConverter(mode="choice", seed=42)
    samples_choice = list(conv_choice.convert_dataset(ds, split="test"))
    assert len(samples_choice) == 2

    assert samples_choice[0].question_type == QuestionType.CHOICE
    assert samples_choice[0].target == "positive"
    assert "positive" in samples_choice[0].criteria
    assert "negative" in samples_choice[0].criteria

    assert samples_choice[1].question_type == QuestionType.CHOICE
    assert samples_choice[1].target == "negative"


def test_jglue_jnli_converter_mock() -> None:
    """JGLUE JNLI コンバータのモック Dataset を用いた変換テスト。"""
    features = Features(
        {
            "sentence_pair_id": Value("string"),
            "sentence1": Value("string"),
            "sentence2": Value("string"),
            "label": ClassLabel(names=["entailment", "contradiction", "neutral"]),
        }
    )
    mock_data = {
        "sentence_pair_id": ["pair_0", "pair_1", "pair_2"],
        "sentence1": [
            "犬が公園で走っています。",
            "子供が屋外で遊んでいます。",
            "男性がコーヒーを飲んでいます。",
        ],
        "sentence2": [
            "動物が運動しています。",
            "子供が部屋で寝ています。",
            "男性は20歳です。",
        ],
        "label": [0, 1, 2],  # entailment, contradiction, neutral
    }
    ds = Dataset.from_dict(mock_data, features=features)

    # Choice モードの検証 (3 件すべて保持)
    conv_choice = JGlueJNLIConverter(mode="choice", seed=42)
    samples_choice = list(conv_choice.convert_dataset(ds, split="test"))
    assert len(samples_choice) == 3

    assert samples_choice[0].question_type == QuestionType.CHOICE
    assert samples_choice[0].target == "entailment"
    assert "前提: 犬が公園で走っています。" in samples_choice[0].state
    assert "仮説: 動物が運動しています。" in samples_choice[0].state
    assert len(samples_choice[0].criteria) == 3

    assert samples_choice[1].target == "contradiction"
    assert samples_choice[2].target == "neutral"

    # Noul モードの検証 (neutral はスキップされ 2 件)
    conv_noul = JGlueJNLIConverter(mode="noul", seed=42)
    samples_noul = list(conv_noul.convert_dataset(ds, split="test"))
    assert len(samples_noul) == 2

    assert samples_noul[0].question_type == QuestionType.NOUL
    assert samples_noul[0].target == "true"
    assert samples_noul[0].state == mock_data["sentence1"][0]
    assert samples_noul[0].criteria == {}

    assert samples_noul[1].question_type == QuestionType.NOUL
    assert samples_noul[1].target == "false"
    assert samples_noul[1].state == mock_data["sentence1"][1]


def test_jglue_jsts_converter_mock() -> None:
    """JGLUE JSTS コンバータのモック Dataset を用いた変換テスト。"""
    features = Features(
        {
            "sentence_pair_id": Value("string"),
            "sentence1": Value("string"),
            "sentence2": Value("string"),
            "label": Value("float32"),
        }
    )
    mock_data = {
        "sentence_pair_id": ["sts_0", "sts_1", "sts_2", "sts_3"],
        "sentence1": [
            "猫が昼寝をしています。",
            "車が道路を走っています。",
            "料理を作っています。",
            "雨が激しく降っています。",
        ],
        "sentence2": [
            "猫が横になって眠っています。",
            "電車が駅に到着しました。",
            "晩ご飯を調理しています。",
            "快晴で太陽が輝いています。",
        ],
        "label": [4.8, 1.2, 3.5, 0.1],  # 四捨五入後: 5, 1, 4, 0
    }
    ds = Dataset.from_dict(mock_data, features=features)

    converter = JGlueJSTSConverter(seed=42)
    samples = list(converter.convert_dataset(ds, split="test"))
    assert len(samples) == 4

    expected_targets = ["5", "1", "4", "0"]
    expected_original = [4.8, 1.2, 3.5, 0.1]

    for idx, sample in enumerate(samples):
        assert sample.question_type == QuestionType.SCORE
        assert sample.target == expected_targets[idx]
        assert len(sample.criteria) == 6
        assert list(sample.criteria.keys()) == ["0", "1", "2", "3", "4", "5"]
        assert "文1: " in sample.state
        assert "文2: " in sample.state
        assert sample.metadata["discrete_score"] == int(expected_targets[idx])
        assert (
            pytest.approx(sample.metadata["original_score"], 0.01)
            == expected_original[idx]
        )


def test_jglue_jcommonsenseqa_converter_mock() -> None:
    """JGLUE JCommonsenseQA コンバータのモック Dataset を用いた変換テスト。"""
    features = Features(
        {
            "q_id": Value("int64"),
            "question": Value("string"),
            "choice0": Value("string"),
            "choice1": Value("string"),
            "choice2": Value("string"),
            "choice3": Value("string"),
            "choice4": Value("string"),
            "label": ClassLabel(
                names=["choice0", "choice1", "choice2", "choice3", "choice4"]
            ),
        }
    )
    mock_data = {
        "q_id": [101, 102],
        "question": [
            "主に子ども向けのもので、イラストのついた物語が書かれているものはどれ？",
            "水分を補給するために飲むもので最も一般的なものはどれ？",
        ],
        "choice0": ["世界", "砂漠"],
        "choice1": ["写真集", "塩"],
        "choice2": ["絵本", "水"],
        "choice3": ["論文", "油"],
        "choice4": ["図鑑", "氷"],
        "label": [2, 2],  # choice2
    }
    ds = Dataset.from_dict(mock_data, features=features)

    converter = JGlueJCommonsenseQAConverter(seed=42)
    samples = list(converter.convert_dataset(ds, split="test"))
    assert len(samples) == 2

    first = samples[0]
    assert first.question_type == QuestionType.CHOICE
    assert first.target == "choice2"
    assert first.state == mock_data["question"][0]
    assert len(first.criteria) == 5
    assert first.criteria["choice0"] == "世界"
    assert first.criteria["choice2"] == "絵本"
    assert first.metadata["q_id"] == 101

    second = samples[1]
    assert second.target == "choice2"
    assert second.criteria["choice2"] == "水"


def test_unified_dataset_builder_jglue_registration() -> None:
    """UnifiedDatasetBuilder における JGLUE コンバータ登録テスト。"""
    builder = UnifiedDatasetBuilder(seed=42)
    expected_jglue_keys = [
        "jglue_marc_ja",
        "jglue_marc_ja_choice",
        "jglue_jnli",
        "jglue_jnli_noul",
        "jglue_jsts",
        "jglue_jcommonsenseqa",
    ]
    for key in expected_jglue_keys:
        assert key in builder.converters


def test_jglue_marc_ja_label_type_resilience() -> None:
    """MARC-ja においてラベル型が int、数字文字列、英語文字列のいずれでも安全に処理されるかを検証するテスト。"""
    converter = JGlueMarcJaConverter(mode="noul", seed=42)

    # 1. int 型ラベル (0: positive, 1: negative, 2: neutral)
    ds_int = Dataset.from_dict(
        {
            "sentence": ["良い商品。", "悪い商品。", "普通。"],
            "label": [0, 1, 2],
            "review_id": ["r0", "r1", "r2"],
        }
    )
    samples_int = list(converter.convert_dataset(ds_int, split="test"))
    assert [s.target for s in samples_int] == ["true", "false"]

    # 2. 数字文字列型ラベル ("0", "1", "2")
    ds_str_num = Dataset.from_dict(
        {
            "sentence": ["良い商品。", "悪い商品。", "普通。"],
            "label": ["0", "1", "2"],
            "review_id": ["r0", "r1", "r2"],
        }
    )
    samples_str_num = list(converter.convert_dataset(ds_str_num, split="test"))
    assert [s.target for s in samples_str_num] == ["true", "false"]

    # 3. 英語文字列型ラベル ("positive", "negative", "neutral")
    ds_str_word = Dataset.from_dict(
        {
            "sentence": ["良い商品。", "悪い商品。", "普通。"],
            "label": ["positive", "negative", "neutral"],
            "review_id": ["r0", "r1", "r2"],
        }
    )
    samples_str_word = list(converter.convert_dataset(ds_str_word, split="test"))
    assert [s.target for s in samples_str_word] == ["true", "false"]


@pytest.mark.slow
def test_jglue_converters_real_smoke() -> None:
    """Hugging Face から実際の JGLUE Parquet を 1 件取得し、スキーマ整合性を検証するスモークテスト。"""
    builder = UnifiedDatasetBuilder(seed=42)
    test_datasets = [
        ("jglue_marc_ja", QuestionType.NOUL),
        ("jglue_jnli", QuestionType.CHOICE),
        ("jglue_jsts", QuestionType.SCORE),
        ("jglue_jcommonsenseqa", QuestionType.CHOICE),
    ]

    for name, expected_type in test_datasets:
        samples = list(
            builder.stream_samples(
                dataset_names=[name],
                split="validation",
                max_samples_per_dataset=1,
            )
        )
        assert len(samples) == 1
        sample = samples[0]
        assert sample.question_type == expected_type
        assert len(sample.state) > 0
        assert len(sample.instructions) > 0
        assert sample.metadata["actual_split"] == "validation"
