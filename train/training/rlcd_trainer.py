"""RLCD (Reinforcement Learning from Calibrated Decisions) 学習ループモジュール。

SFT で訓練された JevDecisionModel を Policy / Reference の 2 系統で保持し、
厳密適格スコア複合報酬に基づく離散バンディット型 GRPO ポリシー更新を実行する。
エポックごとの較正度 (ECE)、複合スコア、および分類精度の推移を記録し、
最良チェックポイントの永続化を行う。
"""

import json
import logging
import platform
import random
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

from data.dataset import JevDataset
from models.backbone import save_tokenizer_for_runtime
from models.decision_head import JevDecisionModel
from training.metrics import MetricsTracker
from training.rlcd_config import RLCDConfig
from training.rlcd_loss import RLCDLoss
from training.scoring import ProperScoringEvaluator, get_normalized_probabilities

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """全乱数生成器のシードを一括初期化し、厳密な再現性を確保する。

    Args:
        seed (int): 乱数シード。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("乱数シードを %d に固定しました。", seed)


def collect_rlcd_run_metadata(config: RLCDConfig) -> dict[str, Any]:
    """RLCD 実行環境の情報 (Git コミット、ライブラリバージョン、ハードウェア情報) を収集する。

    Args:
        config (RLCDConfig): RLCD 設定オブジェクト。

    Returns:
        dict[str, Any]: 環境メタデータ辞書。
    """
    git_commit = "unknown"
    git_dirty = False
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_res.returncode == 0:
            git_commit = commit_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status_res.returncode == 0:
            git_dirty = len(status_res.stdout.strip()) > 0
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Git 情報の取得に失敗しました: %s。", e)

    import accelerate
    import transformers

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Silicon MPS"

    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "accelerate_version": accelerate.__version__,
        "device_name": device_name,
        "rlcd_config": config.to_dict(),
    }


def compute_expected_calibration_error(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    num_bins: int = 10,
) -> float:
    """予測確信度と精度の期待較正誤差 (Expected Calibration Error: ECE) を算出する。

    数理仕様:
    Top-1 確信度 c_i = max_k(p_k) を [0, 1] 区間の M 個のビンに分割し、
    各ビンにおける平均確信度と正解率の絶対誤差を加重平均する。
    $$ECE = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        num_bins (int): 確信度ビンの分割数。

    Returns:
        float: 期待較正誤差 (0.0 〜 1.0)。
    """
    masked_probs = probs * op_mask.float()
    confidences, predictions = torch.max(masked_probs, dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    ece = 0.0
    total_samples = float(probs.size(0))

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum().item()

        if bin_size > 0:
            bin_acc = accuracies[in_bin].float().mean().item()
            bin_conf = confidences[in_bin].mean().item()
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


class RLCDTrainer:
    """RLCD 強化学習ループを実行するトレーナークラス。

    Attributes:
        config (RLCDConfig): RLCD ハイパーパラメータ設定。
        policy_model (JevDecisionModel): 最適化対象のポリシーモデル。
        ref_model (JevDecisionModel): パラメータ固定の参照モデル (SFT)。
        train_dataloader (DataLoader): 学習データローダー。
        val_dataloader (DataLoader | None): 検証データローダー。
        tokenizer (PreTrainedTokenizerFast | None): トークナイザー。
        accelerator (Accelerator): 分散・並列実行エンジン。
        loss_fn (RLCDLoss): RLCD 損失関数。
        optimizer (AdamW): 最適化オプティマイザ。
        lr_scheduler (Any): 学習率スケジューラ。
    """

    def __init__(
        self,
        config: RLCDConfig,
        policy_model: JevDecisionModel | torch.nn.Module,
        ref_model: JevDecisionModel | torch.nn.Module,
        train_dataloader: DataLoader[Any],
        val_dataloader: DataLoader[Any] | None = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
    ) -> None:
        """RLCD トレーナーを初期化する。

        Args:
            config (RLCDConfig): 学習設定。
            policy_model (JevDecisionModel | torch.nn.Module): 学習対象ポリシー。
            ref_model (JevDecisionModel | torch.nn.Module): 参照モデル (SFT 最良モデル)。
            train_dataloader (DataLoader[Any]): 学習用データローダー。
            val_dataloader (DataLoader[Any] | None): 検証用データローダー。
            tokenizer (PreTrainedTokenizerFast | None): トークナイザー。
        """
        self.config = config
        set_seed(self.config.seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        self.policy_model = policy_model
        self.ref_model = ref_model

        # 参照モデルのパラメータを完全に凍結 (requires_grad = False, eval モード)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer

        self.loss_fn = RLCDLoss(config=self.config)

        # 最適化対象パラメータのグループ化 (Backbone と Head で学習率を調整可能)
        head_params = []
        backbone_params = []
        for name, param in self.policy_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("decision_head", "gather_layer")):
                head_params.append(param)
            else:
                backbone_params.append(param)

        if head_params and backbone_params:
            optimizer_grouped_parameters = [
                {
                    "params": head_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": backbone_params,
                    "lr": self.config.learning_rate * 0.5,
                    "weight_decay": self.config.weight_decay,
                },
            ]
        else:
            all_params = [p for p in self.policy_model.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {
                    "params": all_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                }
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters)

        num_update_steps_per_epoch = max(
            1,
            len(self.train_dataloader) // self.config.gradient_accumulation_steps,
        )
        total_training_steps = num_update_steps_per_epoch * self.config.epochs
        num_warmup_steps = int(total_training_steps * self.config.warmup_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )

        # Accelerator による準備
        (
            self.policy_model,
            self.ref_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.policy_model,
            self.ref_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        self.best_composite_score = -float("inf")
        self.history: list[dict[str, Any]] = []

    def train(
        self,
        epoch_callback: Callable[[int, dict[str, float], dict[str, float]], None]
        | None = None,
    ) -> dict[str, Any]:
        """全エポックの RLCD 学習ループを実行する。

        Args:
            epoch_callback (Callable | None): エポック終了時の進捗コールバック。

        Returns:
            dict[str, Any]: 全学習履歴および最良結果のサマリー。
        """
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 設定とメタデータの保存
        self.config.save_yaml(output_path / "rlcd_config.yaml")
        self.config.save_json(output_path / "rlcd_config.json")
        metadata = collect_rlcd_run_metadata(self.config)
        (output_path / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "RLCD ポリシー更新を開始します (エポック数: %d, G: %d, LR: %e)。",
            self.config.epochs,
            self.config.num_generations,
            self.config.learning_rate,
        )

        for epoch in range(1, self.config.epochs + 1):
            # エポック開始時のシャッフル (JevDataset の場合)
            dataset = getattr(self.train_dataloader, "dataset", None)
            if isinstance(dataset, JevDataset):
                dataset.shuffle_for_epoch(epoch)

            train_metrics = self._train_epoch(epoch)

            val_metrics: dict[str, float] = {}
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Reward: %.4f | Val Composite: %.4f, ECE: %.4f, Acc: %.4f",
                epoch,
                self.config.epochs,
                train_metrics.get("loss_total", 0.0),
                train_metrics.get("mean_reward", 0.0),
                val_metrics.get("composite_score", 0.0),
                val_metrics.get("ece", 0.0),
                val_metrics.get("accuracy", 0.0),
            )

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            self.history.append(epoch_record)

            # 最良モデルの保存 (Composite Score が最大のモデル)
            val_comp = val_metrics.get(
                "composite_score", train_metrics.get("mean_reward", 0.0)
            )
            if val_comp > self.best_composite_score:
                self.best_composite_score = val_comp
                self._save_checkpoint(output_path / "best_checkpoint", epoch)
                logger.info(
                    "最良チェックポイントを更新しました (Composite: %.4f)。",
                    val_comp,
                )

            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, val_metrics)

        # 最終チェックポイントおよび学習履歴の保存
        self._save_checkpoint(output_path / "final_checkpoint", self.config.epochs)
        (output_path / "history.json").write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "best_composite_score": self.best_composite_score,
            "history": self.history,
        }

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """単一エポックの RLCD 訓練ステップを実行する。

        Args:
            epoch (int): 現在のエポック番号。

        Returns:
            dict[str, float]: エポック内の平均訓練メトリクス。
        """
        self.policy_model.train()
        self.ref_model.eval()

        accumulated_metrics: dict[str, float] = {}
        total_steps = 0

        for batch in self.train_dataloader:
            with self.accelerator.accumulate(self.policy_model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                op_indices = batch["op_indices"]
                op_mask = batch["op_mask"]
                labels = batch["labels"]
                question_types = batch.get("question_types")

                # Policy モデルの決定ロジット
                policy_logits = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    op_indices=op_indices,
                )

                # Reference モデルのロジット (勾配不要・省メモリ実行)
                with torch.no_grad():
                    ref_logits = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        op_indices=op_indices,
                    )

                # RLCD 損失とメトリクスの計算
                loss, step_metrics = self.loss_fn(
                    policy_logits=policy_logits,
                    ref_logits=ref_logits,
                    labels=labels,
                    op_mask=op_mask,
                    question_types=question_types,
                )

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                for k, v in step_metrics.items():
                    val = float(v.item() if isinstance(v, Tensor) else v)
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + val
                total_steps += 1

        if total_steps == 0:
            return {}

        return {k: v / total_steps for k, v in accumulated_metrics.items()}

    def evaluate(self) -> dict[str, float]:
        """検証データセットを用いて多面的な較正度と分類精度を評価する。

        Returns:
            dict[str, float]: Accuracy, ECE, Composite Score, S_log, S_sph, S_rps。
        """
        if self.val_dataloader is None:
            return {}

        self.policy_model.eval()
        scoring_evaluator = ProperScoringEvaluator(config=self.config.scoring_config)
        metrics_tracker = MetricsTracker()

        all_probs: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_masks: list[Tensor] = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                op_indices = batch["op_indices"]
                op_mask = batch["op_mask"]
                labels = batch["labels"]
                question_types = batch.get("question_types")
                is_negatives = batch.get("is_negatives")

                logits = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    op_indices=op_indices,
                )

                # 厳密適格スコアの集計
                scoring_evaluator.update(
                    logits=logits,
                    labels=labels,
                    op_mask=op_mask,
                    question_types=question_types,
                )

                # 分類精度の集計
                metrics_tracker.update(
                    loss=0.0,
                    logits=logits,
                    labels=labels,
                    op_mask=op_mask,
                    question_types=question_types,
                    is_negatives=is_negatives,
                )

                probs = get_normalized_probabilities(
                    logits=logits,
                    op_mask=op_mask,
                    temperature=self.config.sampling_temperature,
                )
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_masks.append(op_mask.cpu())

        score_summary = scoring_evaluator.compute()
        class_summary = metrics_tracker.compute()

        # ECE (Expected Calibration Error) の算出
        concat_probs = torch.cat(all_probs, dim=0)
        concat_labels = torch.cat(all_labels, dim=0)
        concat_masks = torch.cat(all_masks, dim=0)
        ece = compute_expected_calibration_error(
            probs=concat_probs,
            labels=concat_labels,
            op_mask=concat_masks,
        )

        return {
            "composite_score": score_summary.get("mean_composite", 0.0),
            "s_log": score_summary.get("mean_log", 0.0),
            "s_sph": score_summary.get("mean_sph", 0.0),
            "s_rps": score_summary.get("mean_rps", 0.0),
            "accuracy": class_summary.get("accuracy", 0.0),
            "ece": ece,
        }

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        """モデル重みとトークナイザーを保存する。

        Args:
            path (Path): 保存先ディレクトリパス。
            epoch (int): 現在のエポック番号。
        """
        path.mkdir(parents=True, exist_ok=True)
        unwrapped_policy = self.accelerator.unwrap_model(self.policy_model)

        torch.save(
            {
                "epoch": epoch,
                "state_dict": unwrapped_policy.state_dict(),
                "best_composite_score": self.best_composite_score,
                "config": self.config.to_dict(),
            },
            path / "model.pt",
        )

        if self.tokenizer is not None:
            save_tokenizer_for_runtime(self.tokenizer, path)

        logger.info("チェックポイントを保存しました: %s。", path)
