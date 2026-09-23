"""較正性能評価 (Calibration Evaluator) モジュール。

期待較正誤差 (ECE: Expected Calibration Error)、負の対数尤度 (NLL)、
マルチクラス Brier スコア、分類精度、および信頼性曲線 (Reliability Diagram) データの算出を提供する。
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from training.loss import DEFAULT_MASK_VALUE


def get_masked_probabilities(
    logits: Tensor,
    op_mask: Tensor,
    temperature: float = 1.0,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> Tensor:
    """温度適用後に無効候補をマスクし、正規化された Softmax 確率分布を算出する。

    数理仕様:
    温度スケーリング後にマスクを適用することで、無効候補への確率漏れを完全に遮断する。
    $$z_i^{\\text{scaled}} = z_i / \\tau$$
    $$\\tilde{z}_i = \\begin{cases} z_i^{\\text{scaled}} & (\\text{op\\_mask}_i = \\text{True}) \\\\ \\text{mask\\_value} & (\\text{op\\_mask}_i = \\text{False}) \\end{cases}$$
    $$p_i = \\text{Softmax}(\\tilde{z}_i) \\cdot \\text{op\\_mask}_i$$

    Args:
        logits (Tensor): 未マスクの生ロジットテンソル `[N, K]`。
        op_mask (Tensor): 有効候補マスク `[N, K]` (有効: True, 無効: False)。
        temperature (float): 温度パラメータ tau。
        mask_value (float): 無効候補に代入する負の値。

    Returns:
        Tensor: 正規化確率テンソル `[N, K]`。
    """
    temp = max(temperature, 1e-4)
    scaled_logits = logits / temp
    masked_logits = scaled_logits.masked_fill(~op_mask, mask_value)
    probs = F.softmax(masked_logits, dim=-1)
    return probs * op_mask.float()


def compute_accuracy(
    logits: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> float:
    """有効候補マスクを適用した Top-1 分類精度を算出する。

    Args:
        logits (Tensor): ロジットテンソル `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        mask_value (float): 無効候補マスク値。

    Returns:
        float: 正解率 (0.0 〜 1.0)。
    """
    if logits.size(0) == 0:
        return 0.0
    masked_logits = logits.masked_fill(~op_mask, mask_value)
    preds = masked_logits.argmax(dim=-1)
    corrects = preds.eq(labels)
    return float(corrects.float().mean().item())


def compute_masked_nll(
    logits: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    temperature: float = 1.0,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> float:
    """温度適用後の有効候補クロスエントロピー損失 (負の対数尤度: NLL) を算出する。

    Args:
        logits (Tensor): 未マスクの生ロジットテンソル `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        temperature (float): 温度パラメータ tau。
        mask_value (float): 無効候補マスク値。

    Returns:
        float: 平均負の対数尤度。
    """
    if logits.size(0) == 0:
        return 0.0
    temp = max(temperature, 1e-4)
    scaled_logits = logits / temp
    masked_logits = scaled_logits.masked_fill(~op_mask, mask_value)
    loss = F.cross_entropy(masked_logits, labels)
    return float(loss.item())


def compute_ece(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    num_bins: int = 10,
) -> float:
    """期待較正誤差 (Expected Calibration Error: ECE) を算出する。

    数理仕様:
    Top-1 予測確信度 $c_i = \\max_{k} p_{ik}$ を $M$ 個の等幅ビンに分割し、
    各ビンにおける平均精度と平均確信度の差の絶対値をサンプル数で加重平均する。
    $$ECE = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        num_bins (int): 確信度分割ビン数。

    Returns:
        float: 期待較正誤差 (0.0 〜 1.0)。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return 0.0

    masked_probs = probs * op_mask.float()
    confidences, predictions = torch.max(masked_probs, dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_size = int(in_bin.sum().item())
        if bin_size > 0:
            bin_acc = float(accuracies[in_bin].float().mean().item())
            bin_conf = float(confidences[in_bin].mean().item())
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
) -> float:
    """マルチクラス Brier スコアを算出する。

    数理仕様:
    有効候補における予測確率ベクトルと真のワンホットベクトルとの二乗差の和を算出する。
    $$BS = \\frac{1}{N} \\sum_{i=1}^N \\sum_{k \\in \\text{valid}} (p_{ik} - y_{ik})^2$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。

    Returns:
        float: Brier スコア (0.0 〜 2.0)。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return 0.0

    num_classes = probs.size(-1)
    labels_clamped = labels.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(labels_clamped, num_classes=num_classes).float()

    masked_probs = probs * op_mask.float()
    masked_target = one_hot * op_mask.float()

    diff_sq = (masked_probs - masked_target) ** 2
    sum_diff_sq = diff_sq.sum(dim=-1)
    return float(sum_diff_sq.mean().item())


def compute_reliability_diagram_data(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    num_bins: int = 10,
) -> list[dict[str, Any]]:
    """信頼性曲線 (Reliability Diagram) 描画用のビン統計データを算出する。

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        num_bins (int): 分割ビン数。

    Returns:
        list[dict[str, Any]]: 各ビンの統計情報 (下限、上限、サンプル数、確信度平均、正解率、ギャップ)。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return []

    masked_probs = probs * op_mask.float()
    confidences, predictions = torch.max(masked_probs, dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    diagram_data: list[dict[str, Any]] = []

    for i in range(num_bins):
        bin_lower = float(bin_boundaries[i].item())
        bin_upper = float(bin_boundaries[i + 1].item())

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_size = int(in_bin.sum().item())
        if bin_size > 0:
            bin_acc = float(accuracies[in_bin].float().mean().item())
            bin_conf = float(confidences[in_bin].mean().item())
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0

        diagram_data.append(
            {
                "bin_index": i,
                "bin_lower": round(bin_lower, 3),
                "bin_upper": round(bin_upper, 3),
                "count": bin_size,
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "gap": round(abs(bin_acc - bin_conf), 4),
            }
        )

    return diagram_data


class CalibrationEvaluator:
    """モデルの較正性能と予測品質を一括評価するクラス。"""

    def __init__(self, num_bins: int = 10) -> None:
        """評価器を初期化する。

        Args:
            num_bins (int): ECE 算出用ビン分割数。
        """
        self.num_bins = num_bins

    def evaluate(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """指定温度における ECE, NLL, Brier スコア, 精度を一括集計する。

        Args:
            logits (Tensor): 未マスクの生ロジットテンソル `[N, K]`。
            labels (Tensor): 正解インデックス `[N]`。
            op_mask (Tensor): 有効候補マスク `[N, K]`。
            temperature (float): 適用する温度パラメータ。

        Returns:
            dict[str, Any]: 集計結果メトリクス辞書。
        """
        probs = get_masked_probabilities(logits, op_mask, temperature=temperature)
        ece = compute_ece(probs, labels, op_mask, num_bins=self.num_bins)
        nll = compute_masked_nll(logits, labels, op_mask, temperature=temperature)
        brier = compute_brier_score(probs, labels, op_mask)
        accuracy = compute_accuracy(logits, labels, op_mask)
        diagram = compute_reliability_diagram_data(
            probs, labels, op_mask, num_bins=self.num_bins
        )

        return {
            "samples": int(logits.size(0)),
            "temperature": round(temperature, 4),
            "ece": round(ece, 4),
            "nll": round(nll, 4),
            "brier_score": round(brier, 4),
            "accuracy": round(accuracy, 4),
            "reliability_diagram": diagram,
        }


def compute_batch_normalized_entropy(probs: Tensor, op_mask: Tensor) -> Tensor:
    """バッチテンソルに対する正規化シャノンエントロピー H_norm [N] を算出する。

    数理仕様:
    $$H_{\\text{norm}}(p) = \\begin{cases} 0.0 & (K_i = 1) \\\\ \\frac{-\\sum_{k \\in \\text{valid}} p_{ik} \\ln p_{ik}}{\\ln K_i} & (K_i \\ge 2) \\end{cases}$$

    Args:
        probs (Tensor): 正規化済み確率分布テンソル `[N, K]`。
        op_mask (Tensor): 有効候補マスクテンソル `[N, K]`。

    Returns:
        Tensor: [0.0, 1.0] にクランプされた正規化エントロピーテンソル `[N]`。
    """
    valid_counts = op_mask.sum(dim=-1).float()
    masked_probs = probs * op_mask.float()

    # p * ln(p) の計算 (p <= 0 は 0 に置換)
    safe_p = torch.clamp(masked_probs, min=1e-12)
    p_log_p = torch.where(masked_probs > 0.0, masked_probs * torch.log(safe_p), 0.0)
    entropy = -p_log_p.sum(dim=-1)

    max_entropy = torch.log(torch.clamp(valid_counts, min=1.0))
    # valid_counts <= 1 の場合は 0.0
    h_norm = torch.where(
        valid_counts > 1.0,
        entropy / torch.clamp(max_entropy, min=1e-12),
        torch.zeros_like(entropy),
    )
    return h_norm.clamp(0.0, 1.0)


def compute_batch_top_margin(probs: Tensor, op_mask: Tensor) -> Tensor:
    """バッチテンソルに対する Top-Margin M(p) [N] を算出する。

    数理仕様:
    $$M(p) = \\begin{cases} 1.0 & (K_i = 1) \\\\ p_{i,(1)} - p_{i,(2)} & (K_i \\ge 2) \\end{cases}$$

    Args:
        probs (Tensor): 正規化済み確率分布テンソル `[N, K]`。
        op_mask (Tensor): 有効候補マスクテンソル `[N, K]`。

    Returns:
        Tensor: [0.0, 1.0] にクランプされた Top-Margin テンソル `[N]`。
    """
    valid_counts = op_mask.sum(dim=-1)
    k_dim = probs.size(-1)

    if k_dim == 1:
        return torch.ones(probs.size(0), device=probs.device, dtype=probs.dtype)

    masked_probs = probs.masked_fill(~op_mask, -1e9)
    top2_values, _ = torch.topk(masked_probs, k=min(2, k_dim), dim=-1)

    diff = top2_values[:, 0] - top2_values[:, 1]
    # valid_counts == 1 の場合は 1.0
    margin = torch.where(
        valid_counts <= 1,
        torch.ones_like(diff),
        diff,
    )
    return margin.clamp(0.0, 1.0)


def compute_batch_composite_confidence(probs: Tensor, op_mask: Tensor) -> Tensor:
    """バッチテンソルに対する複合確信度スコア S_confidence [N] を算出する。

    数理仕様:
    $$S_{\\text{confidence}}(p) = (1.0 - H_{\\text{norm}}(p)) \\times M(p)$$

    Args:
        probs (Tensor): 正規化済み確率分布テンソル `[N, K]`。
        op_mask (Tensor): 有効候補マスクテンソル `[N, K]`。

    Returns:
        Tensor: [0.0, 1.0] にクランプされた複合確信度スコアテンソル `[N]`。
    """
    h_norm = compute_batch_normalized_entropy(probs, op_mask)
    margin = compute_batch_top_margin(probs, op_mask)
    score = (1.0 - h_norm) * margin
    return score.clamp(0.0, 1.0)
