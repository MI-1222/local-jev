"""多面的評価メトリクス集計モジュール。

質問タイプ別(Choice, Score, Noul)、候補数バケット別、
および合成ネガティブ(該当なし)の適合率・再現率・MAE を統合追跡する。
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from data.schema import QuestionType
from training.loss import DEFAULT_MASK_VALUE


class MetricsTracker:
    """SFT 学習・検証における多面的評価メトリクスを追跡・集計するクラス。

    Attributes:
        total_loss (float): 累積損失値。
        total_samples (int): 累積処理サンプル数。
        total_correct (int): 累積正解サンプル数。
    """

    def __init__(self, mask_value: float = DEFAULT_MASK_VALUE) -> None:
        """メトリクストラッカーを初期化する。

        Args:
            mask_value (float): 無効候補ロジットのマスキング値。
        """
        self.mask_value = mask_value
        self.reset()

    def reset(self) -> None:
        """集計用カウンタをすべて初期状態にリセットする。"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0

        # タイプ別
        self.type_stats: dict[str, dict[str, float]] = {
            QuestionType.CHOICE.value: {"correct": 0, "total": 0},
            QuestionType.SCORE.value: {"correct": 0, "total": 0, "abs_error_sum": 0.0},
            QuestionType.NOUL.value: {"correct": 0, "total": 0},
        }

        # 候補数バケット別
        self.bucket_stats: dict[str, dict[str, int]] = {
            "bucket_2": {"correct": 0, "total": 0},
            "bucket_3_5": {"correct": 0, "total": 0},
            "bucket_6_10": {"correct": 0, "total": 0},
            "bucket_11_plus": {"correct": 0, "total": 0},
        }

        # 合成ネガティブ (該当なし判定)
        self.neg_tp = 0
        self.neg_fp = 0
        self.neg_fn = 0
        self.neg_tn = 0

    def update(
        self,
        loss: float,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: list[str | QuestionType] | None = None,
        is_negatives: list[bool] | Tensor | None = None,
    ) -> None:
        """単一バッチの予測結果を受け取り、各メトリクスを集計する。

        内部ロジック:
        1. op_mask に基づき無効候補のロジットをマスクし、Softmax 確率と Top-1 予測を取得する。
        2. 全体の正解数を更新する。
        3. 質問タイプ(Choice, Score, Noul)別に分類精度および Score の MAE を算出する。
        4. 各サンプルの有効候補数に応じて候補数バケット(2択, 3~5択, 6~10択, 11択以上)を更新する。
        5. 合成ネガティブサンプルに対する混同行列(TP, FP, FN, TN)を更新する。

        Args:
            loss (float): バッチの平均損失値。
            logits (Tensor): 候補ロジットテンソル `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。
            question_types (list[str | QuestionType] | None): サンプルごとの質問タイプ。
            is_negatives (list[bool] | Tensor | None): サンプルごとの合成ネガティブフラグ。
        """
        batch_size = logits.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        device = logits.device
        op_mask_dev = op_mask.to(device)
        labels_dev = labels.to(device)

        masked_logits = logits.masked_fill(~op_mask_dev, self.mask_value)
        preds = masked_logits.argmax(dim=-1)
        corrects = preds.eq(labels_dev)
        self.total_correct += int(corrects.sum().item())

        probs = F.softmax(masked_logits, dim=-1)

        # 有効候補数の計算
        valid_option_counts = op_mask.sum(dim=-1).tolist()

        # is_negatives のリスト化
        neg_flags: list[bool] = []
        if is_negatives is not None:
            if isinstance(is_negatives, Tensor):
                neg_flags = is_negatives.bool().tolist()
            else:
                neg_flags = [bool(x) for x in is_negatives]

        for i in range(batch_size):
            is_correct = bool(corrects[i].item())
            pred_idx = int(preds[i].item())
            label_idx = int(labels[i].item())
            num_options = int(valid_option_counts[i])

            # 質問タイプ別集計
            if question_types is not None and i < len(question_types):
                q_type_raw = question_types[i]
                q_type = (
                    q_type_raw.value
                    if isinstance(q_type_raw, QuestionType)
                    else str(q_type_raw)
                )
                if q_type in self.type_stats:
                    self.type_stats[q_type]["total"] += 1
                    if is_correct:
                        self.type_stats[q_type]["correct"] += 1

                    if q_type == QuestionType.SCORE.value:
                        # 期待値スコア = sum(k * prob[k])
                        prob_i = probs[i, :num_options]
                        indices = torch.arange(
                            num_options, device=prob_i.device, dtype=prob_i.dtype
                        )
                        expected_score = (prob_i * indices).sum().item()
                        abs_error = abs(expected_score - label_idx)
                        self.type_stats[q_type]["abs_error_sum"] += abs_error

            # 候補数バケット別集計
            if num_options <= 2:
                bucket_key = "bucket_2"
            elif num_options <= 5:
                bucket_key = "bucket_3_5"
            elif num_options <= 10:
                bucket_key = "bucket_6_10"
            else:
                bucket_key = "bucket_11_plus"

            self.bucket_stats[bucket_key]["total"] += 1
            if is_correct:
                self.bucket_stats[bucket_key]["correct"] += 1

            # 合成ネガティブ混同行列集計 (最後の候補が "該当なし" 枠)
            if neg_flags and i < len(neg_flags):
                actual_neg = neg_flags[i]
                pred_neg = pred_idx == (num_options - 1)
                if actual_neg and pred_neg:
                    self.neg_tp += 1
                elif not actual_neg and pred_neg:
                    self.neg_fp += 1
                elif actual_neg and not pred_neg:
                    self.neg_fn += 1
                else:
                    self.neg_tn += 1

    def compute(self) -> dict[str, Any]:
        """集計結果を辞書形式で算出する。

        Returns:
            dict[str, Any]: 損失、総合精度、タイプ別精度、バケット別精度を含む指標辞書。
        """
        samples = max(self.total_samples, 1)
        mean_loss = self.total_loss / samples
        accuracy = self.total_correct / samples

        result: dict[str, Any] = {
            "loss": round(mean_loss, 4),
            "accuracy": round(accuracy, 4),
            "total_samples": self.total_samples,
        }

        # タイプ別
        for q_type, stats in self.type_stats.items():
            tot = int(stats["total"])
            acc = stats["correct"] / tot if tot > 0 else 0.0
            result[f"{q_type}_accuracy"] = round(acc, 4)
            result[f"{q_type}_total"] = tot
            if q_type == QuestionType.SCORE.value and tot > 0:
                mae = stats["abs_error_sum"] / tot
                result["score_mae"] = round(mae, 4)

        # 候補数バケット別
        for b_name, stats in self.bucket_stats.items():
            tot = stats["total"]
            acc = stats["correct"] / tot if tot > 0 else 0.0
            result[f"{b_name}_accuracy"] = round(acc, 4)
            result[f"{b_name}_total"] = tot

        # 合成ネガティブメトリクス
        neg_total = self.neg_tp + self.neg_fn
        precision = (
            self.neg_tp / (self.neg_tp + self.neg_fp)
            if (self.neg_tp + self.neg_fp) > 0
            else 0.0
        )
        recall = self.neg_tp / neg_total if neg_total > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result["negative_precision"] = round(precision, 4)
        result["negative_recall"] = round(recall, 4)
        result["negative_f1"] = round(f1, 4)
        result["negative_total"] = neg_total

        return result
