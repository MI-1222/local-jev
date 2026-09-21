"""ONNX 数値整合性検証 (Parity Test) モジュール。

PyTorch モデルと ONNX Runtime (ORT) の推論結果を多次元バリエーションで突き合わせ、
契約形状、動的軸の自由度、および数値的一致 (許容誤差 < 1e-5) を機械的に検証する。
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
)
from models.decision_head import JevDecisionModel

logger = logging.getLogger(__name__)

DEFAULT_ATOL: float = 1.0e-5
"""最大絶対誤差の許容閾値。"""

DEFAULT_RTOL: float = 1.0e-4
"""相対誤差 (rtol) の許容閾値。"""

DEFAULT_MSE_TOL: float = 1.0e-8
"""平均二乗誤差 (MSE) の許容閾値。"""


@dataclass
class TestCaseResult:
    """単一テストケースの検証結果。

    Attributes:
        name (str): テストケース識別名。
        batch_size (int): 入力バッチサイズ。
        seq_len (int): 入力トークン系列長。
        num_options (int): 評価候補数。
        max_abs_error (float): PyTorch 出力と ORT 出力の最大絶対誤差。
        max_rel_error (float): PyTorch 出力と ORT 出力の最大相対誤差。
        mse (float): PyTorch 出力と ORT 出力の平均二乗誤差。
        has_nan_or_inf (bool): 出力テンソルに非有限値が存在するか。
        duration_ms (float): 推論実行所要時間 (ミリ秒)。
        passed (bool): 合格フラグ。
        error_message (str | None): 不合格時の詳細エラー理由。
    """

    name: str
    batch_size: int
    seq_len: int
    num_options: int
    max_abs_error: float
    max_rel_error: float
    mse: float
    has_nan_or_inf: bool
    duration_ms: float
    passed: bool
    error_message: str | None = None


@dataclass
class ValidationResult:
    """ONNX パリティ検証全体の集約結果。

    Attributes:
        passed (bool): 全テストケースに合格したか。
        max_abs_error (float): 全テストケースを通じて検出された最大絶対誤差。
        test_case_results (list[TestCaseResult]): 各テストケースの個別結果リスト。
        onnx_path (str): 検証対象となった ONNX ファイルパス。
    """

    passed: bool
    max_abs_error: float
    test_case_results: list[TestCaseResult] = field(default_factory=list)
    onnx_path: str = ""

    def summary(self) -> dict[str, Any]:
        """サマリー辞書を生成する。

        Returns:
            dict[str, Any]: 集約メトリクス情報。
        """
        return {
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "num_test_cases": len(self.test_case_results),
            "passed_test_cases": sum(1 for r in self.test_case_results if r.passed),
            "test_cases": [
                {
                    "name": r.name,
                    "shape": [r.batch_size, r.seq_len, r.num_options],
                    "max_abs_error": r.max_abs_error,
                    "max_rel_error": r.max_rel_error,
                    "mse": r.mse,
                    "has_nan_or_inf": r.has_nan_or_inf,
                    "duration_ms": round(r.duration_ms, 2),
                    "passed": r.passed,
                }
                for r in self.test_case_results
            ],
        }


def get_default_test_matrix() -> list[tuple[str, int, int, int]]:
    """標準の多次元検証マトリクス定義を取得する。

    Returns:
        list[tuple[str, int, int, int]]:
            (テストケース名, バッチサイズ, 系列長, 候補数) のタプルリスト。
    """
    return [
        ("単一質問(最小構成)", 1, 64, 2),
        ("標準推論", 4, 256, 5),
        ("非対称バッチ", 3, 128, 16),
        ("中規模候補構成", 2, 512, 32),
        ("長系列コンテキスト", 1, 1024, 4),
    ]


def validate_onnx_parity(
    model: JevDecisionModel,
    onnx_path: Path | str,
    test_cases: list[tuple[str, int, int, int]] | None = None,
    tolerance_atol: float = DEFAULT_ATOL,
    tolerance_rtol: float = DEFAULT_RTOL,
    tolerance_mse: float = DEFAULT_MSE_TOL,
    seed: int = 42,
) -> ValidationResult:
    """PyTorch モデルと ONNX Runtime の推論結果を多次元形状で比較検証する。

    Args:
        model (JevDecisionModel): 基準となる PyTorch 決定モデル。
        onnx_path (Path | str): 検証対象の ONNX ファイルパス。
        test_cases (list[tuple[str, int, int, int]] | None):
            検証ケース (名前, バッチ, 系列長, 候補数) のリスト。None の場合は標準マトリクスを使用。
        tolerance_atol (float): 許容される最大絶対誤差。
        tolerance_rtol (float): 許容される相対誤差。
        tolerance_mse (float): 許容される平均二乗誤差。
        seed (int): 乱数シード。

    Returns:
        ValidationResult: 検証結果オブジェクト。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX ファイルが見つかりません: {onnx_file}。")

    logger.info("ONNX Runtime セッションを初期化中: %s...", onnx_file)
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    cases = test_cases if test_cases is not None else get_default_test_matrix()
    results: list[TestCaseResult] = []
    overall_max_diff: float = 0.0
    all_passed: bool = True

    model.eval()
    model.requires_grad_(False)
    vocab_size = getattr(model.backbone.config, "vocab_size", 50368)

    for name, batch_size, seq_len, num_options in cases:
        logger.info(
            "テストケース実行中: '%s' [B=%d, L=%d, K=%d]...",
            name,
            batch_size,
            seq_len,
            num_options,
        )

        # 1. テストテンソルの準備
        input_ids = torch.randint(
            0,
            min(vocab_size, 1000),
            (batch_size, seq_len),
            dtype=torch.long,
        )
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        # 各候補のマーカー位置インデックスを系列長内に均等配置
        step = max(1, (seq_len - 2) // num_options)
        op_positions = [min(seq_len - 1, 1 + i * step) for i in range(num_options)]
        op_indices = torch.tensor([op_positions] * batch_size, dtype=torch.long)

        # 2. PyTorch モデルでの推論
        with torch.no_grad():
            t0 = time.perf_counter()
            pt_logits_tensor: Tensor = model(input_ids, attention_mask, op_indices)
            duration_ms = (time.perf_counter() - t0) * 1000.0

        pt_logits = pt_logits_tensor.cpu().numpy()

        # 3. ONNX Runtime での推論
        ort_inputs = {
            TENSOR_INPUT_IDS: input_ids.cpu().numpy(),
            TENSOR_ATTENTION_MASK: attention_mask.cpu().numpy(),
            TENSOR_OP_INDICES: op_indices.cpu().numpy(),
        }
        ort_outputs = session.run([TENSOR_LOGITS], ort_inputs)
        ort_logits: np.ndarray = np.asarray(ort_outputs[0])

        # 4. 形状および数値整合性判定
        expected_shape = (batch_size, num_options)
        if ort_logits.shape != expected_shape:
            err = (
                f"形状不一致: 期待される形状 {expected_shape} に対し、"
                f"ORT 出力形状は {ort_logits.shape} です。"
            )
            logger.error("'%s' 失敗: %s", name, err)
            results.append(
                TestCaseResult(
                    name=name,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_options=num_options,
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    mse=float("inf"),
                    has_nan_or_inf=True,
                    duration_ms=duration_ms,
                    passed=False,
                    error_message=err,
                )
            )
            all_passed = False
            continue

        has_nan_or_inf = bool(
            np.isnan(ort_logits).any()
            or np.isinf(ort_logits).any()
            or np.isnan(pt_logits).any()
            or np.isinf(pt_logits).any()
        )

        abs_diff = np.abs(pt_logits - ort_logits)
        max_abs_error = float(np.max(abs_diff))
        rel_diff = abs_diff / (np.abs(pt_logits) + 1.0e-8)
        max_rel_error = float(np.max(rel_diff))
        mse = float(np.mean(abs_diff**2))

        overall_max_diff = max(overall_max_diff, max_abs_error)

        # 要素ごとの許容誤差判定: |pt - ort| <= atol + rtol * |ort|
        is_close_array = np.isclose(
            pt_logits, ort_logits, atol=tolerance_atol, rtol=tolerance_rtol
        )
        is_close_all = bool(is_close_array.all())

        passed = is_close_all and (mse <= tolerance_mse) and not has_nan_or_inf

        error_message = None
        if not passed:
            all_passed = False
            reasons = []
            if max_abs_error > tolerance_atol:
                reasons.append(
                    f"max_abs_error ({max_abs_error:.4e} > {tolerance_atol})"
                )
            if mse > tolerance_mse:
                reasons.append(f"mse ({mse:.4e} > {tolerance_mse})")
            if has_nan_or_inf:
                reasons.append("NaN/Inf 検出")
            error_message = f"閾値超過: {', '.join(reasons)}。"
            logger.error("'%s' 失敗: %s", name, error_message)
        else:
            logger.info(
                "'%s' 合格: max_abs = %.4e, max_rel = %.4e, MSE = %.4e, 所要時間 = %.2f ms",
                name,
                max_abs_error,
                max_rel_error,
                mse,
                duration_ms,
            )

        results.append(
            TestCaseResult(
                name=name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_options=num_options,
                max_abs_error=max_abs_error,
                max_rel_error=max_rel_error,
                mse=mse,
                has_nan_or_inf=has_nan_or_inf,
                duration_ms=duration_ms,
                passed=passed,
                error_message=error_message,
            )
        )

    logger.info(
        "パリティ検証完了: 合格=%s, 最大絶対誤差=%.4e (全 %d ケース)",
        all_passed,
        overall_max_diff,
        len(cases),
    )
    return ValidationResult(
        passed=all_passed,
        max_abs_error=overall_max_diff,
        test_case_results=results,
        onnx_path=str(onnx_file),
    )
