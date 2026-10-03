"""粗密マルチタスク SFT トレーナーおよび二段階階層推論シミュレーションモジュール。

単一のバックボーンと共有 Set Attention Block (SAB) 決定ヘッドに対し、
大分類判定と細分類判定を同一計算グラフで同時学習させ、
検証ループにおいて Rust ランタイム (CoarseToFineRouter) と同一仕様の
二段階階層推論 (Soft-Beam 救済付き) シミュレーションを実行して評価する。
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

from data.hierarchical import HierarchicalMapping
from data.hierarchical_dataset import (
    HierarchicalJevDataset,
    HierarchicalSamplePair,
    hierarchical_collate_fn,
)
from models.backbone import save_tokenizer_for_runtime
from models.decision_head import JevDecisionModel
from training.hierarchical_config import HierarchicalSFTConfig
from training.hierarchical_loss import HierarchicalMultiTaskLoss

logger = logging.getLogger(__name__)


def _extract_logits(output: Any) -> Tensor:
    """モデル出力からロジットテンソルを安全に抽出する。

    JevDecisionModel (Tensor 直返却) および Hugging Face / スタブモデル
    (out.logits 保持オブジェクト) の双方に透過的に対応する。

    Args:
        output (Any): モデルの forward 戻り値。

    Returns:
        Tensor: 抽出されたロジットテンソル。
    """
    if hasattr(output, "logits"):
        return output.logits
    return output


def compute_expected_calibration_error(
    confidences: np.ndarray,
    corrects: np.ndarray,
    num_bins: int = 10,
) -> float:
    """予測確信度と正解バイナリ列から Expected Calibration Error (ECE) を算出する。

    Args:
        confidences (np.ndarray): 予測最高確信度配列 [N]。
        corrects (np.ndarray): 正解フラグ配列 [N] (1: 正解, 0: 不正解)。
        num_bins (int): ビン分割数 (デフォルト: 10)。

    Returns:
        float: ECE スコア (0.0〜1.0)。
    """
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            avg_confidence = float(np.mean(confidences[in_bin]))
            avg_accuracy = float(np.mean(corrects[in_bin]))
            ece += (bin_size / total_samples) * abs(avg_accuracy - avg_confidence)

    return float(ece)


class HierarchicalSFTTrainer:
    """粗密マルチタスク SFT の学習および検証オーケストレータ。

    Attributes:
        config (HierarchicalSFTConfig): 学習ハイパーパラメータ設定。
        model (JevDecisionModel): 共有バックボーンおよび SAB 決定ヘッド。
        tokenizer (PreTrainedTokenizerFast): トークナイザー。
        mapping (HierarchicalMapping): 階層オントロジー。
        train_pairs (list[HierarchicalSamplePair]): 学習用階層ペアリスト。
        eval_pairs (list[HierarchicalSamplePair]): 検証用階層ペアリスト。
        device (torch.device): 実行デバイス (CUDA, MPS, CPU)。
        loss_fn (HierarchicalMultiTaskLoss): 複合マルチタスク損失層。
    """

    def __init__(
        self,
        config: HierarchicalSFTConfig,
        model: JevDecisionModel,
        tokenizer: PreTrainedTokenizerFast,
        mapping: HierarchicalMapping,
        train_pairs: list[HierarchicalSamplePair],
        eval_pairs: list[HierarchicalSamplePair],
        device: torch.device | None = None,
    ) -> None:
        """トレーナーを初期化する。

        Args:
            config (HierarchicalSFTConfig): 設定オブジェクト。
            model (JevDecisionModel): モデル。
            tokenizer (PreTrainedTokenizerFast): トークナイザー。
            mapping (HierarchicalMapping): 階層オントロジー。
            train_pairs (list[HierarchicalSamplePair]): 訓練ペアリスト。
            eval_pairs (list[HierarchicalSamplePair]): 検証ペアリスト。
            device (torch.device | None): 実行デバイス。
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.train_pairs = train_pairs
        self.eval_pairs = eval_pairs

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.loss_fn = HierarchicalMultiTaskLoss(
            label_smoothing=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            beta=config.beta,
            gamma=config.gamma,
        ).to(self.device)

        op_token_id = self.tokenizer.convert_tokens_to_ids("[OP]")
        if (
            not isinstance(op_token_id, int)
            or op_token_id == self.tokenizer.unk_token_id
        ):
            raise ValueError(
                "[OP] 特殊トークンがトークナイザーに正しく登録されていません。"
            )
        self.op_token_id: int = op_token_id

        # データセットおよび DataLoader の構築
        self.train_dataset = HierarchicalJevDataset(
            pairs=self.train_pairs,
            mapping=self.mapping,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.config.max_sequence_length,
            is_train=True,
            base_seed=self.config.seed,
        )
        self.eval_dataset = HierarchicalJevDataset(
            pairs=self.eval_pairs,
            mapping=self.mapping,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.config.max_sequence_length,
            is_train=False,
            base_seed=self.config.seed + 1000,
        )

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        self.pad_token_id = pad_id

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: hierarchical_collate_fn(
                b, pad_token_id=self.pad_token_id
            ),
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: hierarchical_collate_fn(
                b, pad_token_id=self.pad_token_id
            ),
        )

        self.optimizer, self.scheduler = self._setup_optimization()

    def _setup_optimization(self) -> tuple[AdamW, Any]:
        """差分学習率を適用した AdamW オプティマイザおよびスケジューラを構築する。

        Returns:
            tuple[AdamW, Any]: (オプティマイザ, コサインスケジューラ)。
        """
        backbone_params: list[torch.nn.Parameter] = []
        embed_params: list[torch.nn.Parameter] = []
        head_params: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_tokens" in name or "embeddings" in name:
                embed_params.append(param)
            elif "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {
                "params": backbone_params,
                "lr": self.config.learning_rate_backbone,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": embed_params,
                "lr": self.config.learning_rate_embed,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": head_params,
                "lr": self.config.learning_rate_head,
                "weight_decay": self.config.weight_decay,
            },
        ]

        optimizer = AdamW(param_groups)

        total_steps = (
            len(self.train_loader)
            // self.config.gradient_accumulation_steps
            * self.config.num_epochs
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )

        return optimizer, scheduler

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """単一エポックの学習ループを実行する。

        Args:
            epoch (int): 現在のエポック番号 (0-indexed)。

        Returns:
            dict[str, float]: エポックの平均損失メトリクス。
        """
        self.model.train()
        self.train_dataset.set_epoch(epoch)

        # 整合性正則化重み gamma のウォームアップ計算
        if self.config.gamma_warmup_epochs > 0:
            if epoch < self.config.gamma_warmup_epochs:
                progress = (epoch + 1) / self.config.gamma_warmup_epochs
                current_gamma = self.config.gamma * progress
            else:
                current_gamma = self.config.gamma
        else:
            current_gamma = self.config.gamma

        self.loss_fn.set_consistency_weight(current_gamma)

        total_loss = 0.0
        total_loss_coarse = 0.0
        total_loss_fine = 0.0
        total_loss_consistency = 0.0
        step_count = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            coarse_input_ids = batch["coarse_input_ids"].to(self.device)
            coarse_attention_mask = batch["coarse_attention_mask"].to(self.device)
            coarse_op_indices = batch["coarse_op_indices"].to(self.device)
            coarse_op_mask = batch["coarse_op_mask"].to(self.device)
            coarse_labels = batch["coarse_labels"].to(self.device)

            fine_input_ids = batch["fine_input_ids"].to(self.device)
            fine_attention_mask = batch["fine_attention_mask"].to(self.device)
            fine_op_indices = batch["fine_op_indices"].to(self.device)
            fine_op_mask = batch["fine_op_mask"].to(self.device)
            fine_labels = batch["fine_labels"].to(self.device)
            fine_parent_indices = batch["fine_parent_indices"].to(self.device)

            # 共有モデルで Coarse を forward
            coarse_output = self.model(
                input_ids=coarse_input_ids,
                attention_mask=coarse_attention_mask,
                op_indices=coarse_op_indices,
                op_mask=coarse_op_mask,
            )
            coarse_logits = _extract_logits(coarse_output)

            # 共有モデルで Fine を forward
            fine_output = self.model(
                input_ids=fine_input_ids,
                attention_mask=fine_attention_mask,
                op_indices=fine_op_indices,
                op_mask=fine_op_mask,
            )
            fine_logits = _extract_logits(fine_output)

            loss, loss_dict = self.loss_fn(
                coarse_logits=coarse_logits,
                coarse_labels=coarse_labels,
                coarse_op_mask=coarse_op_mask,
                fine_logits=fine_logits,
                fine_labels=fine_labels,
                fine_op_mask=fine_op_mask,
                fine_parent_indices=fine_parent_indices,
                return_dict=True,
            )

            # 勾配累積スケーリング
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += loss_dict["loss"]
            total_loss_coarse += loss_dict["loss_coarse"]
            total_loss_fine += loss_dict["loss_fine"]
            total_loss_consistency += loss_dict["loss_consistency"]
            step_count += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        num_batches = max(1, step_count)
        return {
            "train_loss": total_loss / num_batches,
            "train_loss_coarse": total_loss_coarse / num_batches,
            "train_loss_fine": total_loss_fine / num_batches,
            "train_loss_consistency": total_loss_consistency / num_batches,
            "current_gamma": current_gamma,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """検証データセットに対する評価および二段階推論シミュレーションを実行する。

        Rust ランタイム側 (CoarseToFineRouter) と同一の推論フロー:
        1. 第 1 パス: 大分類 Softmax 確率 \\mathbf{p}_C を算出。
        2. Top-Margin \\Delta = p_{(1)} - p_{(2)} を計算。
        3. \\Delta < \\tau_{\\text{beam}} の場合、Soft-Beam (Top-1 + Top-2 候補結合) を模擬。
        4. 第 2 パス: 細分類予測を判定し、正解と一致するかを計測。

        Returns:
            dict[str, float]: 評価メトリクス辞書。
        """
        self.model.eval()

        coarse_correct = 0
        coarse_total = 0

        fine_oracle_correct = 0
        fine_oracle_total = 0

        e2e_correct = 0
        e2e_total = 0
        soft_beam_count = 0

        confidences: list[float] = []
        corrects: list[float] = []

        for batch in self.eval_loader:
            coarse_input_ids = batch["coarse_input_ids"].to(self.device)
            coarse_attention_mask = batch["coarse_attention_mask"].to(self.device)
            coarse_op_indices = batch["coarse_op_indices"].to(self.device)
            coarse_op_mask = batch["coarse_op_mask"].to(self.device)
            coarse_labels = batch["coarse_labels"].to(self.device)

            fine_input_ids = batch["fine_input_ids"].to(self.device)
            fine_attention_mask = batch["fine_attention_mask"].to(self.device)
            fine_op_indices = batch["fine_op_indices"].to(self.device)
            fine_op_mask = batch["fine_op_mask"].to(self.device)
            fine_labels = batch["fine_labels"].to(self.device)

            # 1. 大分類推論
            coarse_output = self.model(
                input_ids=coarse_input_ids,
                attention_mask=coarse_attention_mask,
                op_indices=coarse_op_indices,
                op_mask=coarse_op_mask,
            )
            coarse_logits = _extract_logits(coarse_output)
            masked_coarse_logits = coarse_logits.masked_fill(~coarse_op_mask, -1e4)
            coarse_probs = F.softmax(masked_coarse_logits, dim=-1)
            pred_coarse = coarse_probs.argmax(dim=-1)

            # 大分類単体精度
            coarse_correct += int((pred_coarse == coarse_labels).sum().item())
            coarse_total += len(coarse_labels)

            # 2. 細分類単体推論 (Oracle / Soft-Beam プロンプト時)
            fine_output = self.model(
                input_ids=fine_input_ids,
                attention_mask=fine_attention_mask,
                op_indices=fine_op_indices,
                op_mask=fine_op_mask,
            )
            fine_logits = _extract_logits(fine_output)
            masked_fine_logits = fine_logits.masked_fill(~fine_op_mask, -1e4)
            fine_probs = F.softmax(masked_fine_logits, dim=-1)
            pred_fine = fine_probs.argmax(dim=-1)

            fine_oracle_correct += int((pred_fine == fine_labels).sum().item())
            fine_oracle_total += len(fine_labels)

            # 3. エンドツーエンド二段階推論シミュレーション
            batch_size = coarse_labels.size(0)
            for i in range(batch_size):
                p_c = coarse_probs[i]
                sorted_p_c, sorted_idx = torch.sort(p_c, descending=True)

                top1_prob = sorted_p_c[0].item()
                top2_prob = sorted_p_c[1].item() if sorted_p_c.size(0) > 1 else 0.0
                margin = top1_prob - top2_prob

                is_soft_beam = margin < self.config.soft_beam_threshold
                if is_soft_beam:
                    soft_beam_count += 1

                # 第 1 パスで正解大分類が Top-1 (または Soft-Beam 時の Top-2) に含まれているか判定
                true_c = int(coarse_labels[i].item())
                top1_c = int(sorted_idx[0].item())
                top2_c = int(sorted_idx[1].item()) if sorted_p_c.size(0) > 1 else -1

                coarse_passed = (top1_c == true_c) or (
                    is_soft_beam and (top2_c == true_c)
                )

                # 第 2 パスの正解判定
                fine_pred_i = int(pred_fine[i].item())
                fine_label_i = int(fine_labels[i].item())
                fine_passed = fine_pred_i == fine_label_i

                e2e_is_correct = coarse_passed and fine_passed
                if e2e_is_correct:
                    e2e_correct += 1
                e2e_total += 1

                conf_i = float(fine_probs[i, fine_pred_i].item())
                confidences.append(conf_i)
                corrects.append(1.0 if e2e_is_correct else 0.0)

        coarse_acc = coarse_correct / max(1, coarse_total)
        fine_acc = fine_oracle_correct / max(1, fine_oracle_total)
        e2e_acc = e2e_correct / max(1, e2e_total)
        beam_rate = soft_beam_count / max(1, e2e_total)
        ece = compute_expected_calibration_error(
            np.array(confidences), np.array(corrects)
        )

        return {
            "eval_coarse_acc": float(coarse_acc),
            "eval_fine_acc": float(fine_acc),
            "eval_e2e_acc": float(e2e_acc),
            "eval_soft_beam_rate": float(beam_rate),
            "eval_ece": float(ece),
        }

    def train(self) -> dict[str, Any]:
        """全エポックの学習を実行し、最良モデルを保存する。

        Returns:
            dict[str, Any]: 最終学習結果メタデータ。
        """
        best_e2e_acc = -1.0
        best_metrics: dict[str, float] = {}

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # オントロジーおよび設定の永続化
        self.mapping.to_json(output_path / "hierarchical_mapping.json")
        self.config.to_json(output_path / "hierarchical_config.json")
        save_tokenizer_for_runtime(self.tokenizer, output_path)

        for epoch in range(self.config.num_epochs):
            logger.info("=== Epoch %d / %d 開始 ===", epoch + 1, self.config.num_epochs)
            train_metrics = self.train_epoch(epoch)
            eval_metrics = self.evaluate()

            logger.info(
                "Epoch %d 結果: Train Loss=%.4f (C=%.4f, F=%.4f, Cons=%.4f), "
                "Coarse Acc=%.4f, Fine Acc=%.4f, E2E Acc=%.4f, ECE=%.4f, BeamRate=%.4f",
                epoch + 1,
                train_metrics["train_loss"],
                train_metrics["train_loss_coarse"],
                train_metrics["train_loss_fine"],
                train_metrics["train_loss_consistency"],
                eval_metrics["eval_coarse_acc"],
                eval_metrics["eval_fine_acc"],
                eval_metrics["eval_e2e_acc"],
                eval_metrics["eval_ece"],
                eval_metrics["eval_soft_beam_rate"],
            )

            current_e2e = eval_metrics["eval_e2e_acc"]
            if current_e2e > best_e2e_acc:
                best_e2e_acc = current_e2e
                best_metrics = {**train_metrics, **eval_metrics}

                # 最良重みの保存
                model_save_path = output_path / "best_model.pt"
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(
                    "最良モデルを更新・保存しました (E2E Acc=%.4f)。", best_e2e_acc
                )

        summary = {
            "best_e2e_acc": best_e2e_acc,
            "best_metrics": best_metrics,
            "output_dir": str(output_path),
        }

        with open(output_path / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary
