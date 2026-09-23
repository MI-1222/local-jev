"""Evol-Instruct 合成データ生成パイプライン CLI エントリーポイント。

実行例:
    uv run python -m data.synthetic.run_synthetic --mock --num-samples 30 --output-dir runs/test_synthetic
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from data.synthetic.client import BaseLLMClient, HttpLLMClient, MockLLMClient
from data.synthetic.config import SyntheticPipelineConfig
from data.synthetic.pipeline import SyntheticPipeline


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: パースされた引数。
    """
    parser = argparse.ArgumentParser(
        description="Jev Evol-Instruct 合成データ生成パイプライン実行 CLI"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="生成する有効サンプルの目標総数 (デフォルト: 50)。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train/data/synthetic/output",
        help="JSONL 出力先ディレクトリ (デフォルト: train/data/synthetic/output)。",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="APIを呼び出さず、決定論的な MockLLMClient で生成・検証を実行するフラグ。",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="非同期 API 呼び出しの最大同時実行数 (デフォルト: 5)。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)。",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="生成用モデル名 (デフォルト: claude-3-7-sonnet-20250219)。",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="gpt-4o",
        help="クロスバリデーション用モデル名 (デフォルト: gpt-4o)。",
    )
    return parser.parse_args()


logger = logging.getLogger(__name__)


async def main_async() -> int:
    """非同期メイン実行ルーチン。

    Returns:
        int: 終了コード (0: 成功, 1: 異常終了)。
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = SyntheticPipelineConfig(
        generator_model=args.generator_model,
        validator_model=args.validator_model,
        output_dir=Path(args.output_dir),
        total_target_samples=args.num_samples,
        max_concurrency=args.concurrency,
        seed=args.seed,
        dry_run=args.mock,
    )

    gen_client: BaseLLMClient
    val_client: BaseLLMClient

    if args.mock:
        logger.info("MockLLMClient を使用してオフライン実行します。")
        gen_client = MockLLMClient()
        val_client = MockLLMClient()
    else:
        logger.info(
            "HTTP API クライアントを初期化します (Generator: %s, Validator: %s)。",
            config.generator_model,
            config.validator_model,
        )
        gen_client = HttpLLMClient(
            model_name=config.generator_model,
            max_concurrency=config.max_concurrency,
        )
        val_client = HttpLLMClient(
            model_name=config.validator_model,
            max_concurrency=config.max_concurrency,
        )

    pipeline = SyntheticPipeline(
        config=config,
        generator_client=gen_client,
        validator_client=val_client,
    )

    stats = await pipeline.run()
    logger.info(
        "パイプライン完了: 試行 %d 件中 %d 件通過 (Choice: %d, Score: %d, Noul: %d)",
        stats.total_attempted,
        stats.total_passed,
        stats.choice_count,
        stats.score_count,
        stats.noul_count,
    )
    return 0


def main() -> None:
    """CLI エントリーポイント関数。"""
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
