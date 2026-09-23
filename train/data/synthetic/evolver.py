"""Evol-Instruct 進化生成エンジンモジュール。

シード仕様に進化オペレータを適用し、難関境界事例・ハードネガティブを含む
高品質なタスク生成リクエストを構築・実行する。
"""

import logging
import random
from typing import Any

from data.synthetic.client import BaseLLMClient
from data.synthetic.prompt_templates import (
    EVOL_OPERATORS,
    SYSTEM_PROMPT_GENERATOR,
    build_generation_prompt,
)
from data.synthetic.taxonomy import SeedSpecification

logger = logging.getLogger(__name__)


class SyntheticEvolver:
    """Evol-Instruct 手法に基づくデータ進化生成エンジン。

    Attributes:
        client (BaseLLMClient): 生成用LLMクライアント。
        rng (random.Random): オペレータ選択用乱数生成器。
    """

    def __init__(self, client: BaseLLMClient, seed: int = 42) -> None:
        """進化エンジンを初期化する。

        Args:
            client (BaseLLMClient): 生成用LLMクライアント。
            seed (int): 乱数シード。
        """
        self.client = client
        self.rng = random.Random(seed)
        self.operator_names = list(EVOL_OPERATORS.keys())

    def choose_operator(self) -> str:
        """適用する進化オペレータをランダムに選択する。

        Returns:
            str: オペレータ識別名。
        """
        return self.rng.choice(self.operator_names)

    async def evolve_seed(
        self,
        spec: SeedSpecification,
        operator_name: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """シード仕様から進化サンプルを生成する。

        Args:
            spec (SeedSpecification): シード仕様。
            operator_name (str | None): 明示的に指定するオペレータ名。None 時は自動サンプリング。

        Returns:
            tuple[dict[str, Any], str]:
                - 生成された生データ辞書。
                - 適用された進化オペレータ名。
        """
        op = operator_name or self.choose_operator()
        user_prompt = build_generation_prompt(spec, operator_name=op)

        raw_dict = await self.client.generate_sample(
            spec=spec,
            system_prompt=SYSTEM_PROMPT_GENERATOR,
            user_prompt=user_prompt,
        )
        return raw_dict, op
