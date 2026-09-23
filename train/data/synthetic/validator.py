"""3段階品質ゲート (構文検証、重複排除、デュアルLLMクロス検証) モジュール。

生成データのスキーマ適合性、意味的多様性、およびゼロショット解答一致性を
厳格にスクリーニングし、高品質なサンプルのみを抽出する。
"""

import logging
from dataclasses import dataclass
from typing import Any

from data.schema import QuestionType, UnifiedSample
from data.synthetic.client import BaseLLMClient
from data.synthetic.config import QualityFilterConfig
from data.synthetic.deduplicator import TextEmbeddingDeduplicator
from data.synthetic.prompt_templates import (
    SYSTEM_PROMPT_VALIDATOR,
    build_validation_prompt,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """品質ゲートの評価結果。

    Attributes:
        is_valid (bool): 全ゲートを通過したか。
        gate_failed (str | None): 失敗したゲート名 ('gate1_schema', 'gate2_dedup', 'gate3_cross_val' 等)。
        failure_reason (str | None): 棄却理由の詳細説明。
        metrics (dict[str, Any]): 類似度や確信度などの詳細メトリクス。
    """

    is_valid: bool
    gate_failed: str | None = None
    failure_reason: str | None = None
    metrics: dict[str, Any] | None = None


class SyntheticQualityGate:
    """3段階品質ゲートオーケストレータ。

    Attributes:
        config (QualityFilterConfig): 品質設定。
        deduplicator (TextEmbeddingDeduplicator): 意味的重複排除エンジン。
        validator_client (BaseLLMClient): クロス検証用LLMクライアント。
    """

    def __init__(
        self,
        config: QualityFilterConfig,
        deduplicator: TextEmbeddingDeduplicator,
        validator_client: BaseLLMClient,
    ) -> None:
        """品質ゲートを初期化する。

        Args:
            config (QualityFilterConfig): 品質設定。
            deduplicator (TextEmbeddingDeduplicator): 重複排除器。
            validator_client (BaseLLMClient): 検証用LLMクライアント。
        """
        self.config = config
        self.deduplicator = deduplicator
        self.validator_client = validator_client

    def validate_gate1_schema(
        self,
        raw_dict: dict[str, Any],
        sample_id: str,
        dataset_name: str = "synthetic",
    ) -> tuple[UnifiedSample | None, str | None]:
        """第1ゲート: スキーマおよび構造制約を検証する。

        Args:
            raw_dict (dict[str, Any]): 生の生成辞書。
            sample_id (str): サンプル識別子。
            dataset_name (str): データセット名。

        Returns:
            tuple[UnifiedSample | None, str | None]:
                - 構築された UnifiedSample (失敗時は None)。
                - 失敗理由文字列 (成功時は None)。
        """
        state = str(raw_dict.get("state", "")).strip()
        state_len = len(state)
        if not (
            self.config.min_state_chars <= state_len <= self.config.max_state_chars
        ):
            return None, (
                f"State の文字数 ({state_len}文字) が許容範囲 "
                f"[{self.config.min_state_chars}, {self.config.max_state_chars}] を逸脱しています。"
            )

        instructions = str(raw_dict.get("instructions", "")).strip()
        if not instructions:
            return None, "Instructions が空です。"

        q_type_str = str(raw_dict.get("question_type", "")).lower()
        try:
            q_type = QuestionType(q_type_str)
        except ValueError:
            return None, f"未定義の question_type です: '{q_type_str}'。"

        criteria = dict(raw_dict.get("criteria", {}))
        for key, val in criteria.items():
            if len(str(val)) > self.config.max_criteria_chars:
                return None, (
                    f"Criteria '{key}' の説明文が長すぎます ({len(str(val))}文字 > "
                    f"{self.config.max_criteria_chars}文字)。"
                )

        target = str(raw_dict.get("target", "")).strip()

        # ハードネガティブの検証
        if (
            q_type == QuestionType.CHOICE
            and self.config.require_hard_negative
            and "hard_negative_key" not in raw_dict
        ):
            return (
                None,
                "Choice 型でハードネガティブキー (hard_negative_key) が指定されていません。",
            )

        sample_dict = {
            "dataset_name": dataset_name,
            "sample_id": sample_id,
            "question_type": q_type.value,
            "state": state,
            "instructions": instructions,
            "criteria": criteria,
            "target": target,
            "metadata": {
                "rationale": raw_dict.get("rationale", ""),
                "hard_negative_key": raw_dict.get("hard_negative_key"),
            },
        }

        try:
            sample = UnifiedSample.from_dict(sample_dict)
            return sample, None
        except (ValueError, KeyError, TypeError) as e:
            return None, f"UnifiedSample の契約検証に違反しました: {e}。"

    def validate_gate2_deduplication(
        self,
        sample: UnifiedSample,
    ) -> tuple[bool, float, str | None]:
        """第2ゲート: 意味的重複排除を検証する。

        Args:
            sample (UnifiedSample): 検証対象サンプル。

        Returns:
            tuple[bool, float, str | None]:
                - 重複判定 (True なら重複)。
                - 最大コサイン類似度。
                - 重複相手のサンプルID。
        """
        is_dup, max_sim, dup_id = self.deduplicator.is_duplicate(
            sample_id=sample.sample_id,
            text=sample.state,
            add_if_unique=True,
        )
        return is_dup, max_sim, dup_id

    async def validate_gate3_cross_validation(
        self,
        sample: UnifiedSample,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """第3ゲート: デュアルLLMクロスバリデーションを実行する。

        Args:
            sample (UnifiedSample): 検証対象サンプル。

        Returns:
            tuple[bool, str | None, dict[str, Any]]:
                - 解答一致フラグ。
                - 不一致または曖昧性棄却理由。
                - 検証器の出力メトリクス。
        """
        sample_dict = sample.to_dict()
        val_prompt = build_validation_prompt(sample_dict)

        try:
            val_response = await self.validator_client.validate_sample(
                system_prompt=SYSTEM_PROMPT_VALIDATOR,
                user_prompt=val_prompt,
            )
        except (RuntimeError, ValueError, TimeoutError, OSError) as e:
            return False, f"検証用 LLM 呼び出しに失敗しました: {e}。", {}

        is_ambiguous = bool(val_response.get("is_ambiguous", False))
        if is_ambiguous:
            return (
                False,
                (
                    f"検証モデルにより問題文または選択肢に曖昧性があると指摘されました: "
                    f"{val_response.get('reasoning')}。"
                ),
                val_response,
            )

        predicted = str(val_response.get("predicted_target", "")).strip()
        expected = sample.target.strip()

        # 大文字小文字や空白の揺らぎを吸収
        if predicted.lower() != expected.lower():
            return (
                False,
                (
                    f"ゼロショット検証解答の不一致: 期待値 '{expected}' に対し、"
                    f"検証モデルの判定は '{predicted}' でした (理由: {val_response.get('reasoning')})。"
                ),
                val_response,
            )

        return True, None, val_response

    async def evaluate_sample(
        self,
        raw_dict: dict[str, Any],
        sample_id: str,
    ) -> tuple[UnifiedSample | None, ValidationResult]:
        """全3段階品質ゲートを順次適用してサンプルを総合評価する。

        Args:
            raw_dict (dict[str, Any]): 生のLLM生成辞書。
            sample_id (str): サンプル識別子。

        Returns:
            tuple[UnifiedSample | None, ValidationResult]:
                - 全ゲート通過時は UnifiedSample、棄却時は None。
                - バリデーション結果オブジェクト。
        """
        # ゲート1: 構文・スキーマ
        sample, err1 = self.validate_gate1_schema(raw_dict, sample_id)
        if sample is None:
            return None, ValidationResult(
                is_valid=False,
                gate_failed="gate1_schema",
                failure_reason=err1,
            )

        # ゲート2: 意味的重複排除
        is_dup, max_sim, dup_id = self.validate_gate2_deduplication(sample)
        if is_dup:
            return None, ValidationResult(
                is_valid=False,
                gate_failed="gate2_dedup",
                failure_reason=(
                    f"既存サンプル '{dup_id}' との意味的コサイン類似度 ({max_sim:.4f}) が "
                    f"閾値 ({self.config.similarity_threshold}) を超過しました。"
                ),
                metrics={"similarity": max_sim, "duplicate_with": dup_id},
            )

        # ゲート3: デュアルLLMクロスバリデーション
        matched, err3, metrics = await self.validate_gate3_cross_validation(sample)
        if not matched:
            return None, ValidationResult(
                is_valid=False,
                gate_failed="gate3_cross_val",
                failure_reason=err3,
                metrics=metrics,
            )

        return sample, ValidationResult(
            is_valid=True,
            metrics={"similarity": max_sim, **metrics},
        )
