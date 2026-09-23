"""Jev Evol-Instruct 合成データ生成パイプラインパッケージ。

実務トリアージ・障害対応・規約判定タスクの高品質な日本語合成データを生成・検証・管理する。
"""

from data.synthetic.client import (
    BaseLLMClient,
    HttpLLMClient,
    MockLLMClient,
)
from data.synthetic.config import (
    PrimitiveRatioConfig,
    QualityFilterConfig,
    SyntheticPipelineConfig,
)
from data.synthetic.deduplicator import TextEmbeddingDeduplicator
from data.synthetic.evolver import SyntheticEvolver
from data.synthetic.pipeline import PipelineStatistics, SyntheticPipeline
from data.synthetic.taxonomy import (
    DOMAIN_TAXONOMY,
    DomainNode,
    DomainTaxonomySampler,
    SeedSpecification,
)
from data.synthetic.validator import SyntheticQualityGate, ValidationResult

__all__ = [
    "DOMAIN_TAXONOMY",
    "BaseLLMClient",
    "DomainNode",
    "DomainTaxonomySampler",
    "HttpLLMClient",
    "MockLLMClient",
    "PipelineStatistics",
    "PrimitiveRatioConfig",
    "QualityFilterConfig",
    "SeedSpecification",
    "SyntheticEvolver",
    "SyntheticPipeline",
    "SyntheticPipelineConfig",
    "SyntheticQualityGate",
    "TextEmbeddingDeduplicator",
    "ValidationResult",
]
