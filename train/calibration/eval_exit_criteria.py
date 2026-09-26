"""Phase 4 Exit Criteria 一括自動評価スクリプト。

SFT / RLCD で訓練された最新の JevDecisionModel (SAB + CORAL + ASL) を読み込み、
検証データセット (10,010 件) 全体に対して Exit Criteria の 7 項目を一括評価・判定する。
"""

import argparse
import subprocess
from pathlib import Path
from typing import cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from calibration.config import CalibrationRunConfig
from calibration.optimizer import TemperatureOptimizer
from models.backbone import prepare_backbone_and_tokenizer
from models.decision_head import JevDecisionModel
from training.config import SFTConfig
from training.trainer import SFTTrainer


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数。
    """
    parser = argparse.ArgumentParser(description="Phase 4 Exit Criteria 一括評価。")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="train/runs/sft/sft_20260923_144747_modernbert-ja-130m/best_checkpoint",
        help="評価対象チェックポイントディレクトリパス。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="評価時バッチサイズ。",
    )
    return parser.parse_args()


def main() -> None:
    """一括評価メイン関数。"""
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    print(f"=== Phase 4 Exit Criteria 一括評価開始: {checkpoint_dir} ===")

    config = SFTConfig.from_json(checkpoint_dir / "config.json")
    config.evaluate_position_bias = True
    config.batch_size = args.batch_size

    backbone, tokenizer, op_token_id = prepare_backbone_and_tokenizer(
        model_name_or_path=config.model_name_or_path
    )
    if (checkpoint_dir / "tokenizer").exists():
        tokenizer = cast(
            PreTrainedTokenizerFast,
            AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer"),
        )

    model = JevDecisionModel(
        backbone=backbone,
        mlp_hidden_size=config.mlp_hidden_size,
    )
    raw_checkpoint = torch.load(
        checkpoint_dir / "model.pt", map_location="cpu", weights_only=False
    )
    state_dict = (
        raw_checkpoint["state_dict"]
        if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint
        else raw_checkpoint
    )
    model.load_state_dict(state_dict)

    trainer = SFTTrainer(
        config=config, model=model, tokenizer=tokenizer, op_token_id=op_token_id
    )
    device = trainer.accelerator.device
    model.to(device)

    # 1. 検証データセット評価 (Choice 精度, Score MAE, 位置バイアス)
    print("1/3 検証データセット評価中...")
    _, val_loader, _ = trainer.prepare_data()
    metrics = trainer.evaluate(model=model, val_loader=val_loader)
    pos_metrics = trainer.evaluate_position_bias(model=model, max_samples=200)

    # 2. キャリブレーション評価 (ECE, 確信度 0.8 精度, Brier Score)
    print("2/3 キャリブレーション信頼性評価中...")
    calib_config = CalibrationRunConfig(
        output_dir="train/runs/calibration/phase4_eval", num_bins=10
    )
    optimizer = TemperatureOptimizer(config=calib_config)
    cache = optimizer.collect_logits(model=model, dataloader=val_loader, device=device)
    _calib_contract, eval_summary = optimizer.calibrate(cache)

    post_metrics = eval_summary.get("post_calibration", {})
    pre_metrics = eval_summary.get("pre_calibration", {})
    post_ece = post_metrics.get("ece", 1.0)
    acc_08_info = post_metrics.get("accuracy_at_08", {})
    acc_08 = acc_08_info.get("accuracy", 0.0)
    cnt_08 = acc_08_info.get("count", 0)
    pre_brier = pre_metrics.get("brier_score", 0.0)
    post_brier = post_metrics.get("brier_score", 0.0)
    brier_imp = (pre_brier - post_brier) / pre_brier if pre_brier > 0 else 0.0

    choice_acc = metrics.get("choice_accuracy", 0.0)
    score_mae = metrics.get("score_mae", 999.0)
    pos_delta = pos_metrics.get("position_bias_max_delta", 1.0)
    pos_mean = pos_metrics.get("position_bias_mean_delta", 1.0)

    # 3. Rust 結合テスト実行 (ワークスペースルートで実行)
    print("3/3 ガードレール結合テスト実行中...")
    workspace_root = Path.cwd()
    rust_res = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "local-jev-server",
            "--test",
            "guardrails",
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    rust_passed = rust_res.returncode == 0

    # 4. 判定結果サマリー表示
    print("\n============================================================")
    print("📋 Phase 4 Exit Criteria 判定結果")
    print("============================================================")
    checks = [
        (
            "1. 事後較正 ECE (<= 0.08)",
            f"{post_ece:.4f} ({(post_ece * 100):.2f}%)",
            post_ece <= 0.08,
        ),
        (
            "2. 確信度0.8の実正答率 (78%〜82%)",
            f"{(acc_08 * 100):.2f}% (N={cnt_08})",
            0.78 <= acc_08 <= 0.82,
        ),
        (
            "3. Brier Score 改善率 (>= 10%)",
            f"{(brier_imp * 100):+.2f}% (事前: {pre_brier:.4f} → 事後: {post_brier:.4f})",
            brier_imp >= 0.10,
        ),
        (
            "4.1 Choice 決定精度 (>= 88%)",
            f"{(choice_acc * 100):.2f}%",
            choice_acc >= 0.88,
        ),
        (
            "4.2 Score MAE (<= 0.40)",
            f"{score_mae:.4f}",
            score_mae <= 0.40,
        ),
        (
            "5. 位置バイアス変動幅 (<= 1.0%)",
            f"最大 {(pos_delta * 100):.2f}%, 平均 {(pos_mean * 100):.2f}%",
            pos_delta <= 0.010,
        ),
        (
            "6. 記号プレフィックスロバスト化",
            "Rust ガードレールテスト通過",
            rust_passed,
        ),
        (
            "7. Noul 簡潔言明自動補正",
            "Rust ガードレールテスト通過",
            rust_passed,
        ),
    ]

    for name, val_str, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"- {name}: {val_str} [{status}]")
    print("============================================================\n")


if __name__ == "__main__":
    main()
