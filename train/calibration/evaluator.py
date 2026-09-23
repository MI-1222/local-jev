"""較正性能評価 (Calibration Evaluator) モジュール。

期待較正誤差 (ECE: Expected Calibration Error)、負の対数尤度 (NLL)、
マルチクラス Brier スコア、分類精度、および信頼性曲線 (Reliability Diagram) データの算出を提供する。
"""

from typing import Any

import torch
import torch.nn.functional as F
from scipy import stats
from torch import Tensor

from training.loss import DEFAULT_MASK_VALUE
from training.scoring import compute_rps_loss


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


def compute_adaptive_ece(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    num_bins: int = 10,
) -> float:
    """等頻度ビン分割 (Adaptive / Quantile ECE) による期待較正誤差を算出する。

    数理仕様:
    サンプルが集中する高確信度帯と過疎になりがちな低確信度帯のサンプル疎密歪みを解消するため、
    確信度昇順にソートしたサンプル列を均等な要素数を持つ $M$ 個のビンに分割して較正誤差を算出する。
    $$ECE_{\\text{adaptive}} = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        num_bins (int): 分割ビン数。

    Returns:
        float: 等頻度期待較正誤差 (0.0 〜 1.0)。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return 0.0

    num_bins = max(1, min(num_bins, total_samples))
    masked_probs = probs * op_mask.float()
    confidences, predictions = torch.max(masked_probs, dim=-1)
    accuracies = predictions.eq(labels)

    # 確信度で昇順ソート
    sorted_conf, sort_indices = torch.sort(confidences)
    sorted_acc = accuracies[sort_indices]

    ece = 0.0
    # 各ビンのスライス範囲を均等分割
    indices = torch.linspace(0, total_samples, num_bins + 1).long()

    for i in range(num_bins):
        start_idx = int(indices[i].item())
        end_idx = int(indices[i + 1].item())
        bin_size = end_idx - start_idx

        if bin_size > 0:
            bin_conf = float(sorted_conf[start_idx:end_idx].mean().item())
            bin_acc = float(sorted_acc[start_idx:end_idx].float().mean().item())
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_rps(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    is_cdf: bool = False,
) -> float:
    """順位確率スコア (Ranked Probability Score: RPS) の平均値を算出する。

    数理仕様:
    Score 型（順序尺度）における累積分布関数 (CDF) 間の二乗誤差平均（離散 Wasserstein-1 距離）を算出する。
    $$RPS = \\frac{1}{N} \\sum_{i=1}^N \\mathcal{L}_{\\text{rps}}(p_i, y_i)$$

    Args:
        probs (Tensor): 正規化済み確率分布または累積確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        is_cdf (bool): probs が既に累積分布関数 (CDF) であるかどうかのフラグ。

    Returns:
        float: 平均 RPS 損失値 (0.0 〜 1.0)。
    """
    if probs.size(0) == 0:
        return 0.0

    rps_losses = compute_rps_loss(probs, labels, op_mask, is_cdf=is_cdf)
    return float(rps_losses.mean().item())


def compute_expected_score_mae(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
) -> float:
    """Score 型の連続値期待スコアと真のラベル間の平均絶対誤差 (MAE) を算出する。

    数理仕様:
    予測確率ベクトル $p_i$ による重み付き平均期待値 $\\hat{s}_i = \\sum_{k=0}^{K-1} k \\cdot p_{ik}$ を算出し、
    正解インデックス $y_i$ との絶対誤差の平均を計算する。
    $$MAE = \\frac{1}{N} \\sum_{i=1}^N |\\hat{s}_i - y_i|$$

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。

    Returns:
        float: 期待スコアの平均絶対誤差。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return 0.0

    device = probs.device
    num_classes = probs.size(-1)
    class_indices = torch.arange(num_classes, device=device, dtype=probs.dtype)

    masked_probs = probs * op_mask.float()
    expected_scores = torch.sum(masked_probs * class_indices.unsqueeze(0), dim=-1)
    abs_errors = torch.abs(expected_scores - labels.float())
    return float(abs_errors.mean().item())


def compute_wilson_score_interval(
    positive_count: int,
    total_count: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """二項分布に対する Wilson スコア信頼区間を算出する。

    Args:
        positive_count (int): 陽性（または正解）サンプル数。
        total_count (int): 総サンプル数。
        confidence_level (float): 信頼水準 (デフォルト: 0.95)。

    Returns:
        tuple[float, float]: 信頼区間の下限と上限 (0.0 〜 1.0)。
    """
    if total_count <= 0:
        return 0.0, 1.0

    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p_hat = positive_count / total_count
    n = total_count

    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denominator
    margin = (z / denominator) * (
        (p_hat * (1.0 - p_hat) / n + (z**2) / (4 * (n**2))) ** 0.5
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)


def compute_binary_diagram_data(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    num_bins: int = 10,
    positive_class_idx: int = 1,
) -> list[dict[str, Any]]:
    """Noul 型（二値判定）専用の絶対確率に対する信頼性ダイアグラムデータを算出する。

    数理仕様:
    正例クラス（インデックス 1）の言明確率 $P(\\text{true}) \\in [0.0, 1.0]$ そのものを $M$ 分割し、
    各ビン内の実際の Positive ラベル比率と平均予測確率を比較する。
    Wilson スコア信頼区間も併せて算出する。

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        num_bins (int): 分割ビン数。
        positive_class_idx (int): 陽性クラスのインデックス (デフォルト: 1)。

    Returns:
        list[dict[str, Any]]: 各ビンの二値較正統計辞書リスト。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return []

    masked_probs = probs * op_mask.float()
    if masked_probs.size(-1) > positive_class_idx:
        p_true = masked_probs[:, positive_class_idx]
    else:
        p_true = torch.zeros(total_samples, device=probs.device)

    is_positive = labels.eq(positive_class_idx)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    diagram_data: list[dict[str, Any]] = []

    for i in range(num_bins):
        bin_lower = float(bin_boundaries[i].item())
        bin_upper = float(bin_boundaries[i + 1].item())

        if i == 0:
            in_bin = (p_true >= bin_lower) & (p_true <= bin_upper)
        else:
            in_bin = (p_true > bin_lower) & (p_true <= bin_upper)

        bin_size = int(in_bin.sum().item())
        if bin_size > 0:
            pos_count = int(is_positive[in_bin].sum().item())
            bin_acc = float(pos_count / bin_size)
            bin_conf = float(p_true[in_bin].mean().item())
            ci_lower, ci_upper = compute_wilson_score_interval(pos_count, bin_size)
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0
            ci_lower, ci_upper = (0.0, 1.0)

        diagram_data.append(
            {
                "bin_index": i,
                "bin_lower": round(bin_lower, 3),
                "bin_upper": round(bin_upper, 3),
                "count": bin_size,
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "gap": round(abs(bin_acc - bin_conf), 4),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    return diagram_data


def compute_binary_ece(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    num_bins: int = 10,
    positive_class_idx: int = 1,
) -> float:
    """Noul 型（二値判定）の言明確率 P(true) に対する二値期待較正誤差を算出する。

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        num_bins (int): 分割ビン数。
        positive_class_idx (int): 陽性クラスのインデックス。

    Returns:
        float: 二値期待較正誤差 (0.0 〜 1.0)。
    """
    diagram = compute_binary_diagram_data(
        probs, labels, op_mask, num_bins=num_bins, positive_class_idx=positive_class_idx
    )
    total_samples = int(probs.size(0))
    if total_samples == 0:
        return 0.0

    ece = sum(
        (b["count"] / total_samples) * b["gap"] for b in diagram if b["count"] > 0
    )
    return float(round(ece, 4))


def compute_accuracy_at_confidence(
    probs: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    target_conf: float = 0.8,
    window: float = 0.05,
) -> dict[str, Any]:
    """目標確信度周辺のローカルウィンドウにおける実正答率と統計量を算出する。

    数理仕様:
    Roadmap 4.4 Exit Criteria の「確信度 0.8 近傍で実正答率が 78% 〜 82%」を検証するため、
    $[\\text{target\\_conf} - \\text{window}, \\text{target\\_conf} + \\text{window}]$ の
    確信度区間に入るサンプルを抽出し、精度および Wilson 信頼区間を計算する。

    Args:
        probs (Tensor): 正規化済み確率分布 `[N, K]`。
        labels (Tensor): 正解インデックス `[N]`。
        op_mask (Tensor): 有効候補マスク `[N, K]`。
        target_conf (float): 目標確信度 (デフォルト: 0.8)。
        window (float): 許容ウィンドウ半幅 (デフォルト: 0.05, 区間 [0.75, 0.85])。

    Returns:
        dict[str, Any]: ウィンドウ内のサンプル数、平均確信度、精度、信頼区間、判定結果。
    """
    total_samples = probs.size(0)
    if total_samples == 0:
        return {
            "target_conf": target_conf,
            "window": window,
            "count": 0,
            "confidence": 0.0,
            "accuracy": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "in_target_range": False,
        }

    masked_probs = probs * op_mask.float()
    confidences, predictions = torch.max(masked_probs, dim=-1)
    accuracies = predictions.eq(labels)

    lower_bound = max(0.0, target_conf - window)
    upper_bound = min(1.0, target_conf + window)

    in_window = (confidences >= lower_bound) & (confidences <= upper_bound)
    count = int(in_window.sum().item())

    if count > 0:
        mean_conf = float(confidences[in_window].mean().item())
        mean_acc = float(accuracies[in_window].float().mean().item())
        pos_count = int(accuracies[in_window].sum().item())
        ci_lower, ci_upper = compute_wilson_score_interval(pos_count, count)
        in_target_range = bool(0.78 <= mean_acc <= 0.82)
    else:
        mean_conf = target_conf
        mean_acc = 0.0
        ci_lower, ci_upper = (0.0, 1.0)
        in_target_range = False

    return {
        "target_conf": round(target_conf, 4),
        "window": round(window, 4),
        "window_range": [round(lower_bound, 3), round(upper_bound, 3)],
        "count": count,
        "confidence": round(mean_conf, 4),
        "accuracy": round(mean_acc, 4),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "in_target_range": in_target_range,
    }


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
        adaptive_ece = compute_adaptive_ece(
            probs, labels, op_mask, num_bins=self.num_bins
        )
        nll = compute_masked_nll(logits, labels, op_mask, temperature=temperature)
        brier = compute_brier_score(probs, labels, op_mask)
        accuracy = compute_accuracy(logits, labels, op_mask)
        diagram = compute_reliability_diagram_data(
            probs, labels, op_mask, num_bins=self.num_bins
        )
        conf_08 = compute_accuracy_at_confidence(
            probs, labels, op_mask, target_conf=0.8, window=0.05
        )

        return {
            "samples": int(logits.size(0)),
            "temperature": round(temperature, 4),
            "ece": round(ece, 4),
            "adaptive_ece": round(adaptive_ece, 4),
            "nll": round(nll, 4),
            "brier_score": round(brier, 4),
            "accuracy": round(accuracy, 4),
            "reliability_diagram": diagram,
            "accuracy_at_08": conf_08,
        }

    def evaluate_primitive(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_type: str,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """質問プリミティブ固有の幾何・数理特性に応じた較正メトリクスを一括算出する。

        Choice: Top-1 ECE, Adaptive ECE, Brier Score
        Score: RPS (Ranked Probability Score), ECE, 連続値期待スコア MAE
        Noul: 二値絶対確率 ECE, 二値 Brier Score, 二値ダイアグラムデータ

        Args:
            logits (Tensor): 未マスクの生ロジットテンソル `[N, K]`。
            labels (Tensor): 正解インデックス `[N]`。
            op_mask (Tensor): 有効候補マスク `[N, K]`。
            question_type (str): 質問タイプ ("choice", "score", "noul")。
            temperature (float): 適用する温度パラメータ。

        Returns:
            dict[str, Any]: プリミティブ特化の較正評価結果辞書。
        """
        base_eval = self.evaluate(logits, labels, op_mask, temperature=temperature)
        probs = get_masked_probabilities(logits, op_mask, temperature=temperature)
        q_type = question_type.lower()

        if q_type == "score":
            rps = compute_rps(probs, labels, op_mask)
            score_mae = compute_expected_score_mae(probs, labels, op_mask)
            base_eval["rps"] = round(rps, 4)
            base_eval["expected_score_mae"] = round(score_mae, 4)
        elif q_type == "noul":
            binary_ece = compute_binary_ece(
                probs, labels, op_mask, num_bins=self.num_bins
            )
            binary_diagram = compute_binary_diagram_data(
                probs, labels, op_mask, num_bins=self.num_bins
            )
            base_eval["binary_ece"] = binary_ece
            base_eval["binary_reliability_diagram"] = binary_diagram

        return base_eval


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
