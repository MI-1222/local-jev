"""RLCD (Reinforcement Learning from Calibrated Decisions) 実行エントリーポイント。

事前学習・SFT 済みの Jev 決定モデルを読み込み、厳密適格スコア複合報酬に基づく
ポリシー最適化 (GRPO / Listwise DPO) を実行して確率較正度を最大化する。
"""

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from data.builders import UnifiedDatasetBuilder
from models.backbone import prepare_backbone_and_tokenizer
from models.decision_head import JevDecisionModel
from training.config import SFTConfig
from training.rlcd_config import RLCDConfig
from training.rlcd_trainer import RLCDTrainer
from training.trainer import SFTDataset, sft_collate_fn

# マルチワーカー実行時の Hugging Face Tokenizers デッドロックを防止
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_rlcd")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数オブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="Local-Jev RLCD (較正決定強化学習) 実行スクリプト。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="train/runs/sft/sft_20260923_144747_modernbert-ja-130m/best_checkpoint",
        help="参照および初期化に使用する SFT 最良チェックポイントディレクトリ。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train/runs/rlcd/run_latest",
        help="RLCD チェックポイント保存先ディレクトリ。",
    )
    parser.add_argument(
        "--optimization-mode",
        type=str,
        default="grpo",
        choices=["grpo", "listwise_dpo"],
        help="RLCD 最適化方式。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="学習エポック数。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="ミニバッチサイズ。",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="勾配蓄積ステップ数。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="デシジョンヘッドの学習率。",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="GRPO における摂動サンプル数 G。",
    )
    parser.add_argument(
        "--perturbation-std",
        type=float,
        default=0.10,
        help="ロジット摂動ノイズの標準偏差 sigma。",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.05,
        help="参照モデルからの乖離を抑止する KL ペナルティ係数。",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="バックボーンを完全凍結しヘッド層のみ更新するかどうか。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード。",
    )
    return parser.parse_args()


def main() -> None:
    """RLCD 実行メインルーチン。"""
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    logger.info("=== RLCD 学習パイプライン開始: %s ===", checkpoint_dir)

    sft_config_path = checkpoint_dir / "config.json"
    if not sft_config_path.exists():
        raise FileNotFoundError(f"SFT 設定が見つかりません: {sft_config_path}。")

    sft_config = SFTConfig.from_json(sft_config_path)
    sft_config.batch_size = args.batch_size

    # 1. バックボーンとトークナイザーの準備
    backbone, tokenizer, op_token_id = prepare_backbone_and_tokenizer(
        model_name_or_path=sft_config.model_name_or_path
    )
    if (checkpoint_dir / "tokenizer").exists():
        tokenizer = cast(
            PreTrainedTokenizerFast,
            AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer"),
        )

    # 2. Policy モデルの構築と重みロード
    policy_model = JevDecisionModel(
        backbone=backbone,
        mlp_hidden_size=sft_config.mlp_hidden_size,
    )
    model_weights = torch.load(
        checkpoint_dir / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy_model.load_state_dict(model_weights)

    # 3. Reference モデルの構築 (ディープコピーとパラメータ固定)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 4. データローダーの準備
    builder = UnifiedDatasetBuilder(
        seed=sft_config.seed,
        negative_ratio=sft_config.negative_ratio,
    )
    train_samples = list(
        builder.stream_samples(
            dataset_names=sft_config.dataset_names,
            split="train",
            max_samples_per_dataset=sft_config.max_samples_per_dataset,
        )
    )
    train_samples = builder.negative_injector.inject(train_samples, split="train")

    val_samples = list(
        builder.stream_samples(
            dataset_names=sft_config.dataset_names,
            split="validation",
            max_samples_per_dataset=sft_config.max_samples_per_dataset,
        )
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_dataset = SFTDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=sft_config.max_sequence_length,
        is_train=True,
        base_seed=sft_config.seed,
    )
    val_dataset = SFTDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        op_token_id=op_token_id,
        max_length=sft_config.max_sequence_length,
        is_train=False,
        base_seed=sft_config.seed,
    )

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
    )
    logger.info(
        "データセット準備完了: 訓練サンプル=%d, 検証サンプル=%d。",
        len(train_samples),
        len(val_samples),
    )

    # 5. RLCD 設定の構築
    rlcd_config = RLCDConfig(
        optimization_mode=args.optimization_mode,
        num_generations=args.num_generations,
        perturbation_std=args.perturbation_std,
        kl_coeff=args.kl_coeff,
        learning_rate=args.learning_rate,
        freeze_backbone=args.freeze_backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # 6. RLCDTrainer の初期化と学習実行
    trainer = RLCDTrainer(
        config=rlcd_config,
        policy_model=policy_model,
        ref_model=ref_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
    )

    result = trainer.train()
    logger.info(
        "=== RLCD 学習完了: 最良 Composite Score = %.4f ===",
        result.get("best_composite_score", 0.0),
    )


if __name__ == "__main__":
    main()
