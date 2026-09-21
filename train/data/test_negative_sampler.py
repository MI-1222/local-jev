"""合成ネガティブデータ混入パイプラインの単体テストモジュール。

SyntheticNegativeInjector の混入比率、In-Domain / Cross-Domain 生成ロジック、
非対象型 (Score, Noul) の保護、スキーマ整合性、および再現性を検証する。
"""

import pytest

from data.builders import UnifiedDatasetBuilder
from data.converters.base import BaseDatasetConverter
from data.negative_sampler import NEGATIVE_OPTION_POOL, SyntheticNegativeInjector
from data.schema import QuestionType, UnifiedSample


def _create_dummy_choice_sample(sample_id: str, num_options: int = 4) -> UnifiedSample:
    """テスト用のダミー Choice サンプルを生成する。

    Args:
        sample_id (str): サンプル ID。
        num_options (int): 候補数。

    Returns:
        UnifiedSample: 生成されたサンプル。
    """
    criteria = {f"opt_{i}": f"説明文_{i}" for i in range(num_options)}
    return UnifiedSample(
        dataset_name="dummy_task",
        sample_id=sample_id,
        question_type=QuestionType.CHOICE,
        state=f"入力コンテキスト_{sample_id}。",
        instructions="適切な選択肢を1つ選定せよ。",
        criteria=criteria,
        target="opt_0",
        metadata={"original_id": sample_id},
    )


def _create_dummy_noul_sample(sample_id: str) -> UnifiedSample:
    """テスト用のダミー Noul サンプルを生成する。

    Args:
        sample_id (str): サンプル ID。

    Returns:
        UnifiedSample: 生成されたサンプル。
    """
    return UnifiedSample(
        dataset_name="dummy_noul",
        sample_id=sample_id,
        question_type=QuestionType.NOUL,
        state=f"前提事実_{sample_id}。",
        instructions="真偽を判定せよ。",
        criteria={},
        target="true",
        metadata={"original_id": sample_id},
    )


def _create_dummy_score_sample(sample_id: str) -> UnifiedSample:
    """テスト用のダミー Score サンプルを生成する。

    Args:
        sample_id (str): サンプル ID。

    Returns:
        UnifiedSample: 生成されたサンプル。
    """
    criteria = {str(i): f"{i}点" for i in range(1, 6)}
    return UnifiedSample(
        dataset_name="dummy_score",
        sample_id=sample_id,
        question_type=QuestionType.SCORE,
        state=f"評価対象テキスト_{sample_id}。",
        instructions="1から5段階で評価せよ。",
        criteria=criteria,
        target="3",
        metadata={"original_id": sample_id},
    )


class TestSyntheticNegativeInjector:
    """SyntheticNegativeInjector の動作検証テスト群。"""

    def test_injection_ratio_and_metadata(self) -> None:
        """指定したネガティブ混入比率 (15%) が正しく達成され、メタデータが付与されることを検証する。"""
        samples = [_create_dummy_choice_sample(f"sample_{i}") for i in range(100)]
        injector = SyntheticNegativeInjector(
            negative_ratio=0.15, in_domain_ratio=0.70, seed=42
        )

        result = injector.inject(samples, split="train")

        # 100件に対して追加合成される件数: int(100 * (0.15 / 0.85)) = 17件
        # 総件数は 117件、合成ネガティブ比率は 17 / 117 ≒ 14.5% (約15%)
        negatives = [
            s for s in result if s.metadata.get("is_synthetic_negative") is True
        ]
        assert len(negatives) == 17
        assert len(result) == 117

        # In-Domain と Cross-Domain の内訳確認 (17 * 0.70 = 11件, Cross-Domain = 6件)
        in_domains = [
            s
            for s in negatives
            if s.metadata.get("negative_type") == "in_domain_dropped"
        ]
        cross_domains = [
            s
            for s in negatives
            if s.metadata.get("negative_type") == "cross_domain_oos"
        ]
        assert len(in_domains) == 11
        assert len(cross_domains) == 6

    def test_protection_of_non_choice_types(self) -> None:
        """Score 型および Noul 型がネガティブ化の影響を受けず、完全に保護されることを検証する。"""
        choice_samples = [_create_dummy_choice_sample(f"choice_{i}") for i in range(50)]
        noul_samples = [_create_dummy_noul_sample(f"noul_{i}") for i in range(20)]
        score_samples = [_create_dummy_score_sample(f"score_{i}") for i in range(20)]

        all_samples = choice_samples + noul_samples + score_samples
        injector = SyntheticNegativeInjector(negative_ratio=0.15, seed=42)

        result = injector.inject(all_samples, split="train")

        # 合成ネガティブサンプルはすべて Choice 型であることを確認
        negatives = [
            s for s in result if s.metadata.get("is_synthetic_negative") is True
        ]
        for neg in negatives:
            assert neg.question_type == QuestionType.CHOICE

        # Noul と Score の件数が変化していないことを確認
        result_nouls = [s for s in result if s.question_type == QuestionType.NOUL]
        result_scores = [s for s in result if s.question_type == QuestionType.SCORE]
        assert len(result_nouls) == 20
        assert len(result_scores) == 20

        # Noul と Score の内容が変更されていないことを確認
        assert result_nouls == noul_samples
        assert result_scores == score_samples

    def test_in_domain_negative_excludes_original_target(self) -> None:
        """In-Domain 合成ネガティブにおいて、元の正解が Criteria から除外され、新正解が注入されていることを検証する。"""
        samples = [_create_dummy_choice_sample(f"sample_{i}") for i in range(30)]
        injector = SyntheticNegativeInjector(
            negative_ratio=0.20, in_domain_ratio=1.0, seed=42
        )

        result = injector.inject(samples, split="train")
        negatives = [
            s for s in result if s.metadata.get("is_synthetic_negative") is True
        ]

        assert len(negatives) > 0
        for neg in negatives:
            orig_target = neg.metadata["original_target"]
            # 元の正解が Criteria に残っていないこと
            assert orig_target not in neg.criteria
            # 新しい target が Criteria に存在すること (スキーマ整合性)
            assert neg.target in neg.criteria
            # サンプル ID に neg 接頭辞が含まれていること
            assert "synth_neg_indom" in neg.sample_id

    def test_cross_domain_negative_structure(self) -> None:
        """Cross-Domain 合成ネガティブにおいて、State と Criteria が結合され正しく生成されていることを検証する。"""
        samples = [_create_dummy_choice_sample(f"sample_{i}") for i in range(30)]
        injector = SyntheticNegativeInjector(
            negative_ratio=0.20, in_domain_ratio=0.0, seed=42
        )

        result = injector.inject(samples, split="train")
        negatives = [
            s for s in result if s.metadata.get("is_synthetic_negative") is True
        ]

        assert len(negatives) > 0
        for neg in negatives:
            assert neg.metadata["negative_type"] == "cross_domain_oos"
            assert "state_source" in neg.metadata
            assert "criteria_source" in neg.metadata
            assert neg.target in neg.criteria

    def test_diversity_of_negative_options(self) -> None:
        """単一の固定キー ('none' のみ) ではなく、複数の多様なネガティブキーが使用されることを検証する。"""
        samples = [_create_dummy_choice_sample(f"sample_{i}") for i in range(200)]
        injector = SyntheticNegativeInjector(
            negative_ratio=0.20, in_domain_ratio=0.50, seed=42
        )

        result = injector.inject(samples, split="train")
        negatives = [
            s for s in result if s.metadata.get("is_synthetic_negative") is True
        ]

        selected_keys = {neg.target for neg in negatives}
        pool_keys = {pair[0] for pair in NEGATIVE_OPTION_POOL}

        # プール内の複数種類のキーが実際に選択されていること (少なくとも3種類以上)
        intersected = selected_keys.intersection(pool_keys)
        assert len(intersected) >= 3

    def test_deterministic_reproducibility(self) -> None:
        """同一シードで実行した場合に完全に同一の結果が得られることを検証する。"""
        samples = [_create_dummy_choice_sample(f"sample_{i}") for i in range(50)]

        injector1 = SyntheticNegativeInjector(negative_ratio=0.15, seed=123)
        result1 = injector1.inject(samples, split="train")

        injector2 = SyntheticNegativeInjector(negative_ratio=0.15, seed=123)
        result2 = injector2.inject(samples, split="train")

        assert [s.to_dict() for s in result1] == [s.to_dict() for s in result2]

    def test_split_protection(self) -> None:
        """validation または test スプリットではネガティブ混入が行われないことを検証する。"""
        samples = [_create_dummy_choice_sample(f"sample_{i}") for i in range(50)]
        injector = SyntheticNegativeInjector(negative_ratio=0.15, seed=42)

        val_result = injector.inject(samples, split="validation")
        test_result = injector.inject(samples, split="test")

        assert val_result == samples
        assert test_result == samples

    def test_invalid_parameters(self) -> None:
        """パラメータが範囲外の場合に ValueError が発生することを検証する。"""
        with pytest.raises(ValueError):
            SyntheticNegativeInjector(negative_ratio=-0.1)

        with pytest.raises(ValueError):
            SyntheticNegativeInjector(negative_ratio=1.0)

        with pytest.raises(ValueError):
            SyntheticNegativeInjector(in_domain_ratio=1.5)

    def test_empty_samples(self) -> None:
        """空リストに対して安全に空リストを返却することを検証する。"""
        injector = SyntheticNegativeInjector(negative_ratio=0.15)
        assert injector.inject([], split="train") == []


class DummyConverter(BaseDatasetConverter):
    """テスト用モックコンバータ。"""

    def __init__(self, samples: list[UnifiedSample], seed: int = 42) -> None:
        """コンバータを初期化する。"""
        super().__init__(seed=seed)
        self.mock_samples = samples

    def convert_split(self, split: str = "train"):
        """モックサンプルを順次送出する。"""
        yield from self.mock_samples


class TestUnifiedDatasetBuilderIntegration:
    """UnifiedDatasetBuilder との連携テスト群。"""

    def test_builder_negative_injection(self) -> None:
        """UnifiedDatasetBuilder 経由で合成ネガティブが混入されることを検証する。"""
        dummy_samples = [_create_dummy_choice_sample(f"choice_{i}") for i in range(60)]
        builder = UnifiedDatasetBuilder(seed=42, negative_ratio=0.15)
        builder.converters = {"dummy": DummyConverter(dummy_samples, seed=42)}

        # 1. train スプリット (混入有効)
        dataset_train = builder.build_combined_dataset(
            dataset_names=["dummy"], split="train", inject_negatives=True
        )
        assert len(dataset_train) > 60
        negatives_in_dataset = [
            row
            for row in dataset_train
            if row["metadata"].get("is_synthetic_negative") is True
        ]
        assert len(negatives_in_dataset) > 0

        # 2. train スプリット (混入無効化フラグ)
        dataset_no_inject = builder.build_combined_dataset(
            dataset_names=["dummy"], split="train", inject_negatives=False
        )
        assert len(dataset_no_inject) == 60

        # 3. validation スプリット (自動的に混入対象外)
        dataset_val = builder.build_combined_dataset(
            dataset_names=["dummy"], split="validation", inject_negatives=True
        )
        assert len(dataset_val) == 60
