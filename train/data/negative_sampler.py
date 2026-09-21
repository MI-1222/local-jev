"""合成ネガティブデータ生成および混入モジュール。

閉域分類バイアスを排除し、未知入力や候補欠落時に「該当なし」を選択・棄却できるよう、
In-Domain 正解欠落および Cross-Domain OOS サンプルを制御された比率で合成・注入する。
"""

import random
from collections.abc import Sequence

from data.schema import QuestionType, UnifiedSample

# 「該当なし」候補の識別キーと自然言語説明文の多様化テンプレートプール
NEGATIVE_OPTION_POOL: list[tuple[str, str]] = [
    ("none", "上記のいずれにも該当しない (None of the above)"),
    ("other", "その他の項目・分類対象外の要求 (Other / Unrelated)"),
    ("out_of_scope", "サポート範囲外の問い合わせ・未対応のカテゴリ"),
    ("not_applicable", "該当する選択肢なし (Not applicable)"),
    ("unknown", "提示された候補のいずれにも当てはまらない"),
    ("unsupported", "定義済みのカテゴリに該当しない要求・質問"),
]


class SyntheticNegativeInjector:
    """学習データセットに合成ネガティブサンプルを一定比率で混入するインジェクタ。

    Attributes:
        negative_ratio (float): 全体に対する合成ネガティブサンプルの混入目標比率 (例: 0.15 = 15%)。
        in_domain_ratio (float): 合成ネガティブのうち In-Domain 欠落手法の割合 (残りが Cross-Domain)。
        seed (int): 決定論的合成のための乱数シード。
    """

    def __init__(
        self,
        negative_ratio: float = 0.15,
        in_domain_ratio: float = 0.70,
        seed: int = 42,
    ) -> None:
        """インジェクタを初期化する。

        Args:
            negative_ratio (float): 全体に対するネガティブサンプルの混入比率 (0.0 以上 1.0 未満)。
            in_domain_ratio (float): ネガティブ内での In-Domain 手法の割合 (0.0 〜 1.0)。
            seed (int): 乱数シード。

        Raises:
            ValueError: negative_ratio または in_domain_ratio が範囲外の場合。
        """
        if not (0.0 <= negative_ratio < 1.0):
            raise ValueError(
                f"negative_ratio は 0.0 以上 1.0 未満である必要があります: {negative_ratio}。"
            )
        if not (0.0 <= in_domain_ratio <= 1.0):
            raise ValueError(
                f"in_domain_ratio は 0.0 以上 1.0 以下である必要があります: {in_domain_ratio}。"
            )
        self.negative_ratio = negative_ratio
        self.in_domain_ratio = in_domain_ratio
        self.seed = seed

    def _sample_negative_option(
        self, rng: random.Random, existing_keys: set[str]
    ) -> tuple[str, str]:
        """既存キーと衝突しない「該当なし」候補キーと説明文を抽出する。

        Args:
            rng (random.Random): 乱数生成器。
            existing_keys (set[str]): 既存の候補キー集合。

        Returns:
            tuple[str, str]: (候補キー, 自然言語説明文)。
        """
        candidates = [
            pair for pair in NEGATIVE_OPTION_POOL if pair[0] not in existing_keys
        ]
        if not candidates:
            # 衝突回避のフォールバック (末尾に乱数サフィックスを付与)
            base_key, base_desc = rng.choice(NEGATIVE_OPTION_POOL)
            unique_key = f"{base_key}_{rng.randint(1, 9999)}"
            while unique_key in existing_keys:
                unique_key = f"{base_key}_{rng.randint(1, 9999)}"
            return unique_key, base_desc
        return rng.choice(candidates)

    def _create_in_domain_negative(
        self,
        sample: UnifiedSample,
        rng: random.Random,
        new_sample_id: str,
    ) -> UnifiedSample | None:
        """既存サンプルから正解を除去し、「該当なし」を注入した負例を生成する。

        Args:
            sample (UnifiedSample): 元の Choice 型サンプル。
            rng (random.Random): 乱数生成器。
            new_sample_id (str): 新規付与するサンプル ID。

        Returns:
            UnifiedSample | None: 生成された合成ネガティブサンプル。候補数が不足する場合は None。
        """
        if len(sample.criteria) < 2:
            return None

        # 正解候補を取り除いた新しい Criteria を構築
        new_criteria = {k: v for k, v in sample.criteria.items() if k != sample.target}
        neg_key, neg_desc = self._sample_negative_option(rng, set(new_criteria.keys()))
        new_criteria[neg_key] = neg_desc

        new_metadata = dict(sample.metadata)
        new_metadata.update(
            {
                "is_synthetic_negative": True,
                "negative_type": "in_domain_dropped",
                "original_target": sample.target,
            }
        )

        return UnifiedSample(
            dataset_name=sample.dataset_name,
            sample_id=new_sample_id,
            question_type=QuestionType.CHOICE,
            state=sample.state,
            instructions=sample.instructions,
            criteria=new_criteria,
            target=neg_key,
            metadata=new_metadata,
        )

    def _create_cross_domain_negative(
        self,
        state_sample: UnifiedSample,
        criteria_sample: UnifiedSample,
        rng: random.Random,
        new_sample_id: str,
    ) -> UnifiedSample:
        """無関係な State と Criteria を組み合わせ、「該当なし」を注入した OOS 負例を生成する。

        Args:
            state_sample (UnifiedSample): State 文脈を提供するサンプル。
            criteria_sample (UnifiedSample): Criteria および Instructions を提供するサンプル。
            rng (random.Random): 乱数生成器。
            new_sample_id (str): 新規付与するサンプル ID。

        Returns:
            UnifiedSample: 生成された Cross-Domain 合成ネガティブサンプル。
        """
        new_criteria = dict(criteria_sample.criteria)
        neg_key, neg_desc = self._sample_negative_option(rng, set(new_criteria.keys()))
        new_criteria[neg_key] = neg_desc

        return UnifiedSample(
            dataset_name=f"{state_sample.dataset_name}_x_{criteria_sample.dataset_name}",
            sample_id=new_sample_id,
            question_type=QuestionType.CHOICE,
            state=state_sample.state,
            instructions=criteria_sample.instructions,
            criteria=new_criteria,
            target=neg_key,
            metadata={
                "is_synthetic_negative": True,
                "negative_type": "cross_domain_oos",
                "state_source": state_sample.sample_id,
                "criteria_source": criteria_sample.sample_id,
            },
        )

    def inject(
        self,
        samples: Sequence[UnifiedSample],
        split: str = "train",
    ) -> list[UnifiedSample]:
        """データセット列に指定比率の合成ネガティブを注入する。

        Choice 型のみを対象とし、Score 型および Noul 型は無変更のまま保持する。
        検証・テストスプリット (split != 'train') に対しては変更を加えない。

        Args:
            samples (Sequence[UnifiedSample]): 元の統一サンプル列。
            split (str): 対象スプリット名 ('train', 'validation', 'test')。

        Returns:
            list[UnifiedSample]: ネガティブが混入されたサンプルリスト。
        """
        if split != "train" or self.negative_ratio <= 0.0 or not samples:
            return list(samples)

        rng = random.Random(self.seed)

        # Choice 型のサンプルと、それ以外 (Score, Noul) を分離
        choice_samples: list[UnifiedSample] = [
            s for s in samples if s.question_type == QuestionType.CHOICE
        ]
        other_samples: list[UnifiedSample] = [
            s for s in samples if s.question_type != QuestionType.CHOICE
        ]

        if not choice_samples:
            return list(samples)

        total_choice = len(choice_samples)
        # 合成後データ全体に対する negative_ratio を達成する追加件数を算出
        num_negatives = int(
            total_choice * (self.negative_ratio / (1.0 - self.negative_ratio))
        )
        if num_negatives <= 0:
            return list(samples)

        num_in_domain = int(num_negatives * self.in_domain_ratio)
        num_cross_domain = num_negatives - num_in_domain

        synthetic_samples: list[UnifiedSample] = []

        # 1. In-Domain 正解欠落サンプルの生成
        valid_in_domain_candidates = [s for s in choice_samples if len(s.criteria) >= 2]
        if valid_in_domain_candidates and num_in_domain > 0:
            # 復元抽出または非復元抽出で母集団からサンプリング
            if len(valid_in_domain_candidates) >= num_in_domain:
                sampled_for_in_domain = rng.sample(
                    valid_in_domain_candidates, num_in_domain
                )
            else:
                sampled_for_in_domain = [
                    rng.choice(valid_in_domain_candidates) for _ in range(num_in_domain)
                ]

            for idx, base_sample in enumerate(sampled_for_in_domain):
                neg_sample = self._create_in_domain_negative(
                    base_sample, rng, f"synth_neg_indom_{idx}_{base_sample.sample_id}"
                )
                if neg_sample is not None:
                    synthetic_samples.append(neg_sample)

        # 2. Cross-Domain OOS サンプルの生成
        for idx in range(num_cross_domain):
            s_state = rng.choice(choice_samples)
            s_criteria = rng.choice(choice_samples)
            if (
                s_state.dataset_name == s_criteria.dataset_name
                and len(choice_samples) > 1
            ):
                attempts = 0
                while s_criteria.sample_id == s_state.sample_id and attempts < len(
                    choice_samples
                ):
                    s_criteria = rng.choice(choice_samples)
                    attempts += 1

            cross_sample = self._create_cross_domain_negative(
                s_state, s_criteria, rng, f"synth_neg_cross_{idx}_{s_state.sample_id}"
            )
            synthetic_samples.append(cross_sample)

        combined = choice_samples + synthetic_samples
        rng.shuffle(combined)

        return combined + other_samples
