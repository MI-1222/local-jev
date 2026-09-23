"""SFT (教師あり指示学習) 実行エントリーポイントスクリプト。

コマンドライン引数および YAML/JSON 設定ファイルを受け付け、
Jev アーキテクチャの SFT 学習ループを実行して再現性のある成果物を出力する。
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# マルチワーカー実行時の Hugging Face Tokenizers デッドロックを防止
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from training.config import SFTConfig
from training.trainer import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_sft")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Args:
        args (list[str] | None): 引数リスト。None の場合は sys.argv[1:] を解析。

    Returns:
        argparse.Namespace: 解析済み引数オブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="Local-Jev SFT (教師あり指示学習) 実行スクリプト。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="ベースとなる設定ファイルパス (YAML または JSON)。",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="バックボーンモデル識別子またはローカルパス。",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="学習に使用するデータセット識別子リスト。",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="データセットあたりの最大取得サンプル数 (データバランシング用)。",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=None,
        help="合成ネガティブサンプルの混入比率。",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="事前学習済みバックボーンの学習率。",
    )
    parser.add_argument(
        "--lr-embed",
        type=float,
        default=None,
        help="[OP] を含む入力埋め込み層の学習率。",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=None,
        help="デシジョンヘッドの学習率。",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="オプティマイザの重み減衰率。",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="学習エポック数。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="ミニバッチサイズ。",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="勾配累積ステップ数。",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default=None,
        help="混合精度モード。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="成果物出力ディレクトリ。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード。",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="早期終了の許容エポック数。",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default=None,
        help="早期終了および最良判定に使用する評価メトリクス名。",
    )
    parser.add_argument(
        "--evaluate-position-bias",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Choice 型選択肢順序シャッフル不変性 (位置バイアス) を評価するかどうか。",
    )

    return parser.parse_args(args)


def build_config_from_args(parsed_args: argparse.Namespace) -> SFTConfig:
    """解析済み引数から SFTConfig を構築する。

    設定ファイルが指定されている場合はそれをロードし、
    明示的に指定された CLI 引数で上書きする。

    Args:
        parsed_args (argparse.Namespace): 解析済み引数。

    Returns:
        SFTConfig: 構築された学習設定。
    """
    if parsed_args.config is not None:
        config_path = Path(parsed_args.config)
        logger.info("設定ファイルを読み込み中: %s...", config_path)
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            config = SFTConfig.from_yaml(config_path)
        else:
            config = SFTConfig.from_json(config_path)
    else:
        config = SFTConfig()

    # CLI 引数による明示的な上書き
    if parsed_args.model_name_or_path is not None:
        config.model_name_or_path = parsed_args.model_name_or_path
    if parsed_args.dataset_names is not None:
        config.dataset_names = parsed_args.dataset_names
    if parsed_args.max_samples_per_dataset is not None:
        config.max_samples_per_dataset = parsed_args.max_samples_per_dataset
    if parsed_args.negative_ratio is not None:
        config.negative_ratio = parsed_args.negative_ratio
    if parsed_args.lr_backbone is not None:
        config.learning_rate_backbone = parsed_args.lr_backbone
    if parsed_args.lr_embed is not None:
        config.learning_rate_embed = parsed_args.lr_embed
    if parsed_args.lr_head is not None:
        config.learning_rate_head = parsed_args.lr_head
    if parsed_args.weight_decay is not None:
        config.weight_decay = parsed_args.weight_decay
    if parsed_args.num_epochs is not None:
        config.num_epochs = parsed_args.num_epochs
    if parsed_args.batch_size is not None:
        config.batch_size = parsed_args.batch_size
    if parsed_args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = parsed_args.gradient_accumulation_steps
    if parsed_args.mixed_precision is not None:
        config.mixed_precision = parsed_args.mixed_precision
    if parsed_args.output_dir is not None:
        config.output_dir = parsed_args.output_dir
    if parsed_args.seed is not None:
        config.seed = parsed_args.seed
    if parsed_args.early_stopping_patience is not None:
        config.early_stopping_patience = parsed_args.early_stopping_patience
    if parsed_args.eval_metric is not None:
        config.eval_metric = parsed_args.eval_metric
    if parsed_args.evaluate_position_bias is not None:
        config.evaluate_position_bias = parsed_args.evaluate_position_bias

    return config


def run_sft_pipeline(config: SFTConfig) -> dict[str, Any]:
    """SFT 学習パイプラインを実行する。

    Args:
        config (SFTConfig): 学習ハイパーパラメータ設定。

    Returns:
        dict[str, Any]: 実行結果サマリー辞書。
    """
    logger.info("=== SFT パイプライン開始 ===")
    logger.info("バックボーンモデル: %s", config.model_name_or_path)
    logger.info("対象データセット: %s", config.dataset_names)
    logger.info(
        "学習率: Backbone=%.2e, Embed=%.2e, Head=%.2e",
        config.learning_rate_backbone,
        config.learning_rate_embed,
        config.learning_rate_head,
    )
    logger.info(
        "エポック数: %d, バッチサイズ: %d (累積: %d)",
        config.num_epochs,
        config.batch_size,
        config.gradient_accumulation_steps,
    )

    trainer = SFTTrainer(config=config)
    result = trainer.train()
    return result


def main() -> None:
    """CLI メインエントリーポイント。"""
    args = parse_args(sys.argv[1:])
    config = build_config_from_args(args)
    run_sft_pipeline(config)


if __name__ == "__main__":
    main()
