"""合成データ生成・検証・永続化オーケストレーションパイプラインモジュール。

バッチ生成、3段階品質ゲート、JSONL逐次追記、チェックポイント・レジューム、
およびデッドレター記録を統括する。
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data.schema import QuestionType, UnifiedSample
from data.synthetic.client import BaseLLMClient
from data.synthetic.config import SyntheticPipelineConfig
from data.synthetic.deduplicator import TextEmbeddingDeduplicator
from data.synthetic.evolver import SyntheticEvolver
from data.synthetic.taxonomy import DomainTaxonomySampler
from data.synthetic.validator import SyntheticQualityGate

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatistics:
    """パイプライン実行統計情報。

    Attributes:
        total_attempted (int): 試行シード総数。
        total_passed (int): 全ゲート通過サンプル総数。
        gate1_rejected (int): ゲート1 (スキーマ) 棄却数。
        gate2_rejected (int): ゲート2 (重複排除) 棄却数。
        gate3_rejected (int): ゲート3 (クロス検証) 棄却数。
        choice_count (int): 通過した Choice 型数。
        score_count (int): 通過した Score 型数。
        noul_count (int): 通過した Noul 型数。
    """

    total_attempted: int = 0
    total_passed: int = 0
    gate1_rejected: int = 0
    gate2_rejected: int = 0
    gate3_rejected: int = 0
    choice_count: int = 0
    score_count: int = 0
    noul_count: int = 0

    @property
    def pass_rate(self) -> float:
        """通過率を計算する。"""
        if self.total_attempted == 0:
            return 0.0
        return self.total_passed / self.total_attempted


class SyntheticPipeline:
    """合成データ生成パイプライン実行オーケストレータ。

    Attributes:
        config (SyntheticPipelineConfig): パイプライン設定。
        sampler (DomainTaxonomySampler): ドメインシードサンプラー。
        evolver (SyntheticEvolver): Evol-Instruct 生成器。
        quality_gate (SyntheticQualityGate): 3段階品質ゲート。
        stats (PipelineStatistics): 実行統計。
    """

    def __init__(
        self,
        config: SyntheticPipelineConfig,
        generator_client: BaseLLMClient,
        validator_client: BaseLLMClient,
    ) -> None:
        """パイプラインを初期化する。

        Args:
            config (SyntheticPipelineConfig): 設定オブジェクト。
            generator_client (BaseLLMClient): 生成用LLMクライアント。
            validator_client (BaseLLMClient): 検証用LLMクライアント。
        """
        self.config = config
        self.sampler = DomainTaxonomySampler(
            config=config.primitive_ratio,
            seed=config.seed,
        )
        self.evolver = SyntheticEvolver(client=generator_client, seed=config.seed)
        self.deduplicator = TextEmbeddingDeduplicator(
            threshold=config.quality_filter.similarity_threshold
        )
        self.quality_gate = SyntheticQualityGate(
            config=config.quality_filter,
            deduplicator=self.deduplicator,
            validator_client=validator_client,
        )
        self.stats = PipelineStatistics()
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

        # 出力パスの準備
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path_all = self.output_dir / "synthetic_all.jsonl"
        self.path_choice = self.output_dir / "synthetic_choice.jsonl"
        self.path_score = self.output_dir / "synthetic_score.jsonl"
        self.path_noul = self.output_dir / "synthetic_noul.jsonl"
        self.path_dead_letter = self.output_dir / "dead_letter.jsonl"

    def load_existing_sample_ids(self) -> set[str]:
        """既存の出力 JSONL から処理済み sample_id 群を読み出す (レジューム用)。

        Returns:
            set[str]: 既に永続化済みの sample_id の集合。
        """
        existing_ids: set[str] = set()
        if not self.path_all.exists():
            return existing_ids

        with open(self.path_all, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample_id = data.get("sample_id")
                    if sample_id:
                        existing_ids.add(sample_id)
                        # 重複排除インデックスにも事前ロード
                        state = data.get("state")
                        if state:
                            self.deduplicator.is_duplicate(
                                sample_id=sample_id,
                                text=state,
                                add_if_unique=True,
                            )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.debug("既存行のパースに失敗しました: %s。", e)
                    continue

        logger.info(
            "既存出力から %d 件の sample_id をロードしました (レジューム準備完了)。",
            len(existing_ids),
        )
        return existing_ids

    def _append_to_jsonl(self, file_path: Path, data: dict[str, Any]) -> None:
        """JSONL ファイルへ 1 行追記する。

        Args:
            file_path (Path): 出力先ファイルパス。
            data (dict[str, Any]): 記録データ辞書。
        """
        line = json.dumps(data, ensure_ascii=False)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def persist_sample(
        self,
        sample: UnifiedSample,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """検証を通過したサンプルを該当 JSONL 群へ永続化する。

        Args:
            sample (UnifiedSample): 永続化対象サンプル。
            extra_metadata (dict[str, Any] | None): 追加記録情報。
        """
        sample_dict = sample.to_dict()
        if extra_metadata:
            sample_dict["pipeline_metadata"] = extra_metadata

        # 全体ファイルへの追記
        self._append_to_jsonl(self.path_all, sample_dict)

        # プリミティブ別ファイルへの追記
        if sample.question_type == QuestionType.CHOICE:
            self._append_to_jsonl(self.path_choice, sample_dict)
            self.stats.choice_count += 1
        elif sample.question_type == QuestionType.SCORE:
            self._append_to_jsonl(self.path_score, sample_dict)
            self.stats.score_count += 1
        elif sample.question_type == QuestionType.NOUL:
            self._append_to_jsonl(self.path_noul, sample_dict)
            self.stats.noul_count += 1

        self.stats.total_passed += 1

    def persist_dead_letter(
        self,
        sample_id: str,
        gate_failed: str,
        reason: str,
        raw_dict: dict[str, Any] | None = None,
    ) -> None:
        """棄却されたサンプルをデッドレターログへ永続化する。

        Args:
            sample_id (str): サンプル識別子。
            gate_failed (str): 失敗したゲート名。
            reason (str): 棄却理由。
            raw_dict (dict[str, Any] | None): 生の生成辞書。
        """
        record = {
            "sample_id": sample_id,
            "gate_failed": gate_failed,
            "reason": reason,
            "raw_data": raw_dict or {},
        }
        self._append_to_jsonl(self.path_dead_letter, record)

        if gate_failed == "gate1_schema":
            self.stats.gate1_rejected += 1
        elif gate_failed == "gate2_dedup":
            self.stats.gate2_rejected += 1
        elif gate_failed == "gate3_cross_val":
            self.stats.gate3_rejected += 1

    async def process_single_seed(
        self,
        sequence_index: int,
        existing_ids: set[str],
    ) -> UnifiedSample | None:
        """1件のシード仕様を生成・評価・永続化する。

        Args:
            sequence_index (int): シーケンス番号。
            existing_ids (set[str]): 処理済み ID 集合。

        Returns:
            UnifiedSample | None: 通過したサンプル、棄却時は None。
        """
        spec = self.sampler.generate_seed_spec(sequence_index)
        if spec.seed_id in existing_ids:
            return None

        async with self._semaphore:
            self.stats.total_attempted += 1
            try:
                raw_dict, operator_name = await self.evolver.evolve_seed(spec)
            except (RuntimeError, ValueError, KeyError, OSError, TimeoutError) as e:
                self.persist_dead_letter(
                    sample_id=spec.seed_id,
                    gate_failed="generation_error",
                    reason=f"LLM 生成処理で例外が発生しました: {e}。",
                )
                return None

            sample, val_result = await self.quality_gate.evaluate_sample(
                raw_dict=raw_dict,
                sample_id=spec.seed_id,
            )

            if sample is not None and val_result.is_valid:
                extra = {
                    "operator_name": operator_name,
                    "domain": spec.domain_node.domain,
                    "category": spec.domain_node.category,
                    "constraint_node": spec.domain_node.constraint_node,
                    "metrics": val_result.metrics,
                }
                self.persist_sample(sample, extra_metadata=extra)
                existing_ids.add(spec.seed_id)
                return sample
            else:
                self.persist_dead_letter(
                    sample_id=spec.seed_id,
                    gate_failed=val_result.gate_failed or "unknown",
                    reason=val_result.failure_reason
                    or "不明な理由により棄却されました。",
                    raw_dict=raw_dict,
                )
                return None

    async def run(
        self,
        target_count: int | None = None,
        max_attempts: int | None = None,
    ) -> PipelineStatistics:
        """パイプラインを実行し、目標サンプル数に達するまでバッチ処理を反復する。

        Args:
            target_count (int | None): 目標通過サンプル数。None 時は設定値。
            max_attempts (int | None): 最大試行回数。None 時は target_count * 3。

        Returns:
            PipelineStatistics: 実行結果統計オブジェクト。
        """
        targets = target_count or self.config.total_target_samples
        attempts_limit = max_attempts or (targets * 3)

        existing_ids = self.load_existing_sample_ids()
        current_valid_count = len(existing_ids)
        logger.info(
            "パイプライン開始: 現在 %d 件 / 目標 %d 件 (最大試行: %d 件)...",
            current_valid_count,
            targets,
            attempts_limit,
        )

        seq_idx = current_valid_count
        while (
            self.stats.total_passed + current_valid_count < targets
            and self.stats.total_attempted < attempts_limit
        ):
            batch_size = min(
                self.config.max_concurrency * 2,
                targets - (self.stats.total_passed + current_valid_count),
            )
            tasks = [
                self.process_single_seed(seq_idx + i, existing_ids)
                for i in range(batch_size)
            ]
            seq_idx += batch_size

            await asyncio.gather(*tasks)
            logger.info(
                "進捗: 通過 %d 件 (新規: %d 件, Choice: %d, Score: %d, Noul: %d) / 試行 %d 件 (通過率: %.1f%%)",
                current_valid_count + self.stats.total_passed,
                self.stats.total_passed,
                self.stats.choice_count,
                self.stats.score_count,
                self.stats.noul_count,
                self.stats.total_attempted,
                self.stats.pass_rate * 100,
            )

        logger.info("パイプライン実行完了。最終統計: %s", asdict(self.stats))
        return self.stats
