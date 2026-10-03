"""粗密マルチタスク SFT 実行エントリーポイントスクリプト。

コマンドライン引数および設定ファイルを受け付け、
同一バックボーン・同一決定ヘッド上で大分類と細分類を協調学習させ、
二段階推論 (Soft-Beam 救済) のエンドツーエンド精度を評価・保存する。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# マルチワーカー実行時の Hugging Face Tokenizers デッドロックを防止
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.hierarchical import HierarchicalMapping
from data.hierarchical_dataset import HierarchicalDatasetGenerator
from models.backbone import prepare_backbone_and_tokenizer
from models.decision_head import JevDecisionModel
from training.hierarchical_config import HierarchicalSFTConfig
from training.hierarchical_trainer import HierarchicalSFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_hierarchical_sft")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Args:
        args (list[str] | None): 引数リスト。None の場合は sys.argv[1:] を解析。

    Returns:
        argparse.Namespace: 解析済み引数オブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="Sokuto 粗密マルチタスク SFT (教師あり指示学習) 実行スクリプト。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="ベースとなる設定ファイルパス (YAML または JSON)。",
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default=None,
        help="階層オントロジー定義 JSON パス。未指定時は Banking77 プリセットを使用。",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="バックボーンモデル識別子またはローカルパス。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="成果物出力ディレクトリ。",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="細分類損失重み beta。",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="階層整合性正則化重み gamma。",
    )
    parser.add_argument(
        "--soft-beam-ratio",
        type=float,
        default=None,
        help="Soft-Beam 模倣サンプル生成比率。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="ミニバッチサイズ。",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="学習エポック数。",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="実データセット名 (例: 'banking77')。指定時は UnifiedDatasetBuilder より実コーパスをロード。未指定時はオントロジー定義からセルフ学習サンプルを生成。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="実データセットから取得する最大サンプル数。",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="訓練用サンプルの分割比率 (デフォルト: 0.8)。",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="バックボーン Transformer 層の学習率。",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=None,
        help="決定ヘッド (SAB) の学習率。",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """粗密マルチタスク SFT の実行メイン関数。

    Args:
        args (list[str] | None): コマンドライン引数。

    Returns:
        int: 終了コード (0: 成功, 1: 異常終了)。
    """
    parsed = parse_args(args)

    if parsed.config is not None:
        config_path = Path(parsed.config)
        if config_path.suffix.lower() in (".yaml", ".yml"):
            config = HierarchicalSFTConfig.from_yaml(config_path)
        else:
            config = HierarchicalSFTConfig.from_json(config_path)
    else:
        config = HierarchicalSFTConfig()

    # コマンドライン引数による上書き
    if parsed.model_name_or_path is not None:
        config.model_name_or_path = parsed.model_name_or_path
    if parsed.output_dir is not None:
        config.output_dir = parsed.output_dir
    if parsed.beta is not None:
        config.beta = parsed.beta
    if parsed.gamma is not None:
        config.gamma = parsed.gamma
    if parsed.soft_beam_ratio is not None:
        config.soft_beam_ratio = parsed.soft_beam_ratio
    if parsed.batch_size is not None:
        config.batch_size = parsed.batch_size
    if parsed.num_epochs is not None:
        config.num_epochs = parsed.num_epochs
    if parsed.lr_backbone is not None:
        config.learning_rate_backbone = parsed.lr_backbone
    if parsed.lr_head is not None:
        config.learning_rate_head = parsed.lr_head

    # オントロジーのロード
    if parsed.mapping_path is not None:
        mapping = HierarchicalMapping.from_json(parsed.mapping_path)
    else:
        mapping = HierarchicalMapping.banking77()

    logger.info(
        "オントロジー `%s` を準備しました (大分類 %d 件, 細分類 %d 件)。",
        mapping.name,
        len(mapping.coarse_categories),
        len(mapping.fine_criteria),
    )

    # バックボーンおよびトークナイザーの準備
    backbone, tokenizer, _ = prepare_backbone_and_tokenizer(
        model_name_or_path=config.model_name_or_path,
    )

    model = JevDecisionModel(
        backbone=backbone,
        mlp_hidden_size=config.mlp_hidden_size,
    )

    # 擬似または実データセットからペアを生成
    from data.schema import QuestionType, UnifiedSample

    train_samples: list[UnifiedSample] = []
    eval_samples: list[UnifiedSample] = []

    if parsed.dataset_name is not None:
        from data.builders import UnifiedDatasetBuilder

        logger.info(
            "データセット `%s` から実コーパスをロード中...", parsed.dataset_name
        )
        builder = UnifiedDatasetBuilder(seed=config.seed, negative_ratio=0.0)

        loaded_samples: list[UnifiedSample] = []
        for sample in builder.stream_samples(
            dataset_names=[parsed.dataset_name],
            max_samples_per_dataset=parsed.max_samples,
        ):
            # ターゲットキーの正規化 (オントロジーの fine_criteria とマッチング)
            norm_target = sample.target.lower().strip()
            matched_key: str | None = None
            for fk in mapping.fine_criteria:
                if fk.lower() == norm_target or fk == sample.target:
                    matched_key = fk
                    break

            if matched_key is not None:
                adapted_sample = UnifiedSample(
                    dataset_name=sample.dataset_name,
                    sample_id=sample.sample_id,
                    question_type=QuestionType.CHOICE,
                    state=sample.state,
                    instructions=sample.instructions
                    or "問い合わせ内容の詳細意図を選択してください。",
                    criteria=dict(mapping.fine_criteria),
                    target=matched_key,
                    metadata=dict(sample.metadata),
                )
                loaded_samples.append(adapted_sample)

        logger.info("ロード完了: オントロジー適合サンプル %d 件", len(loaded_samples))
        if loaded_samples:
            split_idx = int(len(loaded_samples) * parsed.split_ratio)
            train_samples = loaded_samples[:split_idx]
            eval_samples = loaded_samples[split_idx:]
        else:
            logger.warning(
                "オントロジーに合致するサンプルが 0 件でした。オントロジー定義文フォールバックを使用します。"
            )

    # 実データ未指定または適合サンプルなしの場合はオントロジー定義文からセルフ学習
    if not train_samples:
        for fine_key, fine_desc in mapping.fine_criteria.items():
            state_text = f"ユーザーの問い合わせ: {fine_desc}"
            sample = UnifiedSample(
                dataset_name=mapping.name,
                sample_id=f"{fine_key}_sample",
                question_type=QuestionType.CHOICE,
                state=state_text,
                instructions="問い合わせ内容の詳細意図を選択してください。",
                criteria=dict(mapping.fine_criteria),
                target=fine_key,
            )
            train_samples.append(sample)
            eval_samples.append(sample)

    generator = HierarchicalDatasetGenerator(
        mapping=mapping,
        soft_beam_ratio=config.soft_beam_ratio,
        negative_ratio=config.negative_ratio,
    )

    train_pairs = generator.generate_pairs(train_samples, base_seed=config.seed)
    eval_pairs = generator.generate_pairs(eval_samples, base_seed=config.seed + 100)

    trainer = HierarchicalSFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        mapping=mapping,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
    )

    summary = trainer.train()
    logger.info("学習が完了しました: 最良 E2E Acc=%.4f", summary["best_e2e_acc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
