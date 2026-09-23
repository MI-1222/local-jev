"""キャリブレーション信頼性評価 (4.4) 単体・結合テストスイート。

Adaptive ECE、RPS (Ranked Probability Score)、Noul 二値絶対確率較正、
確信度 0.8 ウィンドウ精度評価、信頼性ダイアグラム描画・画像生成、
Phase 4 Exit Criteria 自動合否判定、および総合レポート生成を包括的に検証する。
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from matplotlib.figure import Figure

from calibration.diagram import (
    export_all_diagrams,
    plot_calibration_comparison,
    plot_reliability_diagram,
)
from calibration.evaluator import (
    compute_accuracy_at_confidence,
    compute_adaptive_ece,
    compute_binary_diagram_data,
    compute_binary_ece,
    compute_expected_score_mae,
    compute_rps,
    compute_wilson_score_interval,
)
from calibration.optimizer import (
    LogitCache,
    TemperatureOptimizer,
)
from calibration.reporter import (
    CalibrationReporter,
    evaluate_exit_criteria,
)


def test_compute_adaptive_ece_perfect_and_overconfident() -> None:
    """Adaptive ECE が完全較正データでゼロ近傍となり、過信データで正しく較正誤差を検出することを検証する。"""
    # 1. 完全較正データのシミュレーション
    # 確信度 0.6 のサンプル 50 件 (正答率 60%)、確信度 0.9 のサンプル 50 件 (正答率 90%)
    n = 100
    probs_perfect = torch.zeros(n, 2)
    labels_perfect = torch.zeros(n, dtype=torch.long)
    op_mask_perfect = torch.ones(n, 2, dtype=torch.bool)

    # 前半 50 件: 確信度 0.6 (クラス 0: 0.6, クラス 1: 0.4)、うち 30 件正解
    probs_perfect[:50, 0] = 0.6
    probs_perfect[:50, 1] = 0.4
    labels_perfect[:30] = 0
    labels_perfect[30:50] = 1

    # 後半 50 件: 確信度 0.9 (クラス 0: 0.9, クラス 1: 0.1)、うち 45 件正解
    probs_perfect[50:, 0] = 0.9
    probs_perfect[50:, 1] = 0.1
    labels_perfect[50:95] = 0
    labels_perfect[95:] = 1

    adapt_ece_perfect = compute_adaptive_ece(
        probs_perfect, labels_perfect, op_mask_perfect, num_bins=2
    )
    assert adapt_ece_perfect == pytest.approx(0.0, abs=1e-4)

    # 2. 意図的過信データ (確信度 0.99 だが正答率 50%)
    probs_overconf = torch.zeros(n, 2)
    labels_overconf = torch.zeros(n, dtype=torch.long)
    probs_overconf[:, 0] = 0.99
    probs_overconf[:, 1] = 0.01
    labels_overconf[:50] = 0
    labels_overconf[50:] = 1

    adapt_ece_overconf = compute_adaptive_ece(
        probs_overconf, labels_overconf, op_mask_perfect, num_bins=5
    )
    # 0.99 - 0.50 = 0.49
    assert adapt_ece_overconf == pytest.approx(0.49, abs=0.02)


def test_compute_rps_and_score_mae() -> None:
    """Score 型の RPS (Ranked Probability Score) および期待スコア MAE の算出精度を検証する。"""
    # 3 段階評価 (0, 1, 2)
    # 正解が 0 のとき、予測が [1.0, 0.0, 0.0] なら RPS = 0.0, MAE = 0.0
    probs_perfect = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    labels_perfect = torch.tensor([0, 1], dtype=torch.long)
    op_mask = torch.ones_like(probs_perfect, dtype=torch.bool)

    rps_perfect = compute_rps(probs_perfect, labels_perfect, op_mask)
    mae_perfect = compute_expected_score_mae(probs_perfect, labels_perfect, op_mask)
    assert rps_perfect == pytest.approx(0.0, abs=1e-5)
    assert mae_perfect == pytest.approx(0.0, abs=1e-5)

    # 正解が 0 のとき、予測が [0.0, 0.0, 1.0] (2 段階の完全誤信)
    probs_wrong = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    labels_wrong = torch.tensor([0], dtype=torch.long)
    op_mask_wrong = torch.ones(1, 3, dtype=torch.bool)

    # CDF: P = [0, 0, 1], Y = [1, 1, 1], diff = [1, 1], diff_sq = [1, 1], sum = 2.0 / (3 - 1) = 1.0
    rps_wrong = compute_rps(probs_wrong, labels_wrong, op_mask_wrong)
    mae_wrong = compute_expected_score_mae(probs_wrong, labels_wrong, op_mask_wrong)
    assert rps_wrong == pytest.approx(1.0, abs=1e-4)
    assert mae_wrong == pytest.approx(2.0, abs=1e-4)


def test_wilson_score_interval_and_binary_diagram() -> None:
    """Wilson スコア信頼区間および Noul 二値信頼性ダイアグラムの統計量を検証する。"""
    # Wilson 区間の検証: n=100, k=80 (p=0.8)
    lower, upper = compute_wilson_score_interval(80, 100, confidence_level=0.95)
    assert 0.70 <= lower < 0.80
    assert 0.80 < upper <= 0.88

    # Noul 二値ダイアグラムデータの検証
    n = 20
    probs = torch.zeros(n, 2)
    labels = torch.zeros(n, dtype=torch.long)
    op_mask = torch.ones(n, 2, dtype=torch.bool)

    # P(True) = 0.85 に 10 件配置 (うち 8 件 True=1)
    probs[:10, 1] = 0.85
    probs[:10, 0] = 0.15
    labels[:8] = 1

    # P(True) = 0.15 に 10 件配置 (うち 1 件 True=1)
    probs[10:, 1] = 0.15
    probs[10:, 0] = 0.85
    labels[10] = 1

    diagram = compute_binary_diagram_data(probs, labels, op_mask, num_bins=10)
    assert len(diagram) == 10

    # 有効ビンの確認
    valid_bins = [b for b in diagram if b["count"] > 0]
    assert len(valid_bins) == 2

    bin_high = next(b for b in valid_bins if b["bin_lower"] >= 0.8)
    assert bin_high["count"] == 10
    assert bin_high["confidence"] == pytest.approx(0.85, abs=1e-3)
    assert bin_high["accuracy"] == pytest.approx(0.80, abs=1e-3)

    bin_ece = compute_binary_ece(probs, labels, op_mask, num_bins=10)
    assert bin_ece >= 0.0


def test_compute_accuracy_at_confidence_window() -> None:
    """確信度 0.8 近傍 ([0.75, 0.85]) における精度判定ロジックを検証する。"""
    n = 50
    probs = torch.zeros(n, 2)
    labels = torch.zeros(n, dtype=torch.long)
    op_mask = torch.ones(n, 2, dtype=torch.bool)

    # 40 件を確信度 0.80 に配置、そのうち 32 件正解 (32/40 = 80%)
    probs[:40, 0] = 0.80
    probs[:40, 1] = 0.20
    labels[:32] = 0
    labels[32:40] = 1

    # 10 件を確信度 0.50 (ウィンドウ外) に配置
    probs[40:, 0] = 0.50
    probs[40:, 1] = 0.50

    res = compute_accuracy_at_confidence(
        probs, labels, op_mask, target_conf=0.8, window=0.05
    )
    assert res["count"] == 40
    assert res["confidence"] == pytest.approx(0.80, abs=1e-3)
    assert res["accuracy"] == pytest.approx(0.80, abs=1e-3)
    assert res["in_target_range"] is True


def test_diagram_plotting_headless() -> None:
    """ヘッドレス環境で信頼性ダイアグラムおよび比較プロットが正常に描画・保存されることを検証する。"""
    # ダミーダイアグラムデータ
    diagram_data = [
        {
            "bin_index": 0,
            "bin_lower": 0.0,
            "bin_upper": 0.5,
            "count": 20,
            "confidence": 0.45,
            "accuracy": 0.40,
            "gap": 0.05,
            "ci_lower": 0.30,
            "ci_upper": 0.52,
        },
        {
            "bin_index": 1,
            "bin_lower": 0.5,
            "bin_upper": 1.0,
            "count": 80,
            "confidence": 0.85,
            "accuracy": 0.82,
            "gap": 0.03,
            "ci_lower": 0.74,
            "ci_upper": 0.89,
        },
    ]

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        single_plot_path = tmp_path / "diagram.png"

        # 1. 単一ダイアグラム描画
        fig1 = plot_reliability_diagram(
            diagram_data=diagram_data,
            title="Unit Test Diagram",
            ece=0.035,
            brier_score=0.12,
            accuracy=0.74,
            save_path=single_plot_path,
        )
        assert isinstance(fig1, Figure)
        assert single_plot_path.exists()
        assert single_plot_path.stat().st_size > 1000  # 空ファイルでないこと

        # 2. 比較プロット描画
        pre_summary = {
            "temperature": 1.0,
            "ece": 0.12,
            "brier_score": 0.22,
            "accuracy": 0.74,
            "reliability_diagram": diagram_data,
        }
        post_summary = {
            "temperature": 1.45,
            "ece": 0.035,
            "brier_score": 0.18,
            "accuracy": 0.74,
            "reliability_diagram": diagram_data,
        }
        cmp_plot_path = tmp_path / "comparison.png"
        fig2 = plot_calibration_comparison(
            pre_summary,
            post_summary,
            title="Comparison Test",
            save_path=cmp_plot_path,
        )
        assert isinstance(fig2, Figure)
        assert cmp_plot_path.exists()
        assert cmp_plot_path.stat().st_size > 1000

        # 3. export_all_diagrams の統合テスト
        full_eval = {
            "pre_calibration": pre_summary,
            "post_calibration": post_summary,
            "buckets": {
                "choice_3-5": {
                    "question_type": "choice",
                    "bucket": "3-5",
                    "post_calibration": post_summary,
                },
                "noul": {
                    "question_type": "noul",
                    "bucket": "noul",
                    "post_calibration": {
                        **post_summary,
                        "binary_reliability_diagram": diagram_data,
                        "binary_ece": 0.03,
                    },
                },
            },
        }
        exported = export_all_diagrams(full_eval, tmp_path / "plots")
        assert "overall_comparison" in exported
        assert "choice_3-5" in exported
        assert "noul" in exported
        assert "noul_binary" in exported
        for p in exported.values():
            assert Path(p).exists()
            assert Path(p).stat().st_size > 500


def test_evaluate_exit_criteria_and_report_generation() -> None:
    """Exit Criteria 判定ロジックおよびレポート生成 (Markdown / JSON) の完全性を検証する。"""
    # 合格ケースのモック
    passing_summary: dict[str, Any] = {
        "total_samples": 500,
        "pre_calibration": {
            "ece": 0.14,
            "brier_score": 0.30,
            "accuracy": 0.88,
        },
        "post_calibration": {
            "ece": 0.05,  # <= 0.08 を達成
            "brier_score": 0.24,  # (0.30 - 0.24)/0.30 = 20% 改善 (>= 10%)
            "accuracy": 0.88,  # 精度不変
            "accuracy_at_08": {
                "count": 50,
                "accuracy": 0.80,  # 78% 〜 82% 内
            },
        },
        "buckets": {
            "choice_2": {
                "question_type": "choice",
                "bucket": "2",
                "samples": 200,
                "temperature": 1.2,
                "post_calibration": {
                    "ece": 0.04,
                    "brier_score": 0.22,
                    "accuracy": 0.89,
                },
            },
        },
    }

    criteria_pass = evaluate_exit_criteria(passing_summary)
    assert criteria_pass["overall_status"] == "PASS"
    assert criteria_pass["all_criteria_met"] is True

    # 不合格ケース (ECE 未達)
    failing_summary = dict(passing_summary)
    failing_summary["post_calibration"] = {
        **passing_summary["post_calibration"],
        "ece": 0.11,  # > 0.08
    }
    criteria_fail = evaluate_exit_criteria(failing_summary)
    assert criteria_fail["overall_status"] == "FAIL"

    # レポート生成のテスト
    with TemporaryDirectory() as tmp_dir:
        reporter = CalibrationReporter(tmp_dir)
        json_p, md_p = reporter.generate_report(passing_summary)
        assert json_p.exists()
        assert md_p.exists()

        md_content = md_p.read_text(encoding="utf-8")
        assert "Phase 4 Exit Criteria" in md_content
        assert "✅ PASS" in md_content
        assert "事後期待較正誤差 (ECE)" in md_content


def test_optimizer_save_run_artifacts_integration() -> None:
    """TemperatureOptimizer.save_run_artifacts がダイアグラムと判定レポートを完全生成することを検証する。"""
    n = 60
    logits = torch.randn(n, 3) * 2.0
    labels = torch.randint(0, 3, (n,))
    op_mask = torch.ones(n, 3, dtype=torch.bool)
    qtypes = ["choice"] * n
    counts = [3] * n

    cache = LogitCache(
        logits=logits,
        op_mask=op_mask,
        labels=labels,
        question_types=qtypes,
        candidate_counts=counts,
    )

    optimizer = TemperatureOptimizer()
    calib_config, overall_summary = optimizer.calibrate(cache)

    with TemporaryDirectory() as tmp_dir:
        run_path = optimizer.save_run_artifacts(tmp_dir, calib_config, overall_summary)
        assert (run_path / "calibration.json").exists()
        assert (run_path / "calibration_run_config.yaml").exists()
        assert (run_path / "calibration_metrics.json").exists()
        assert (run_path / "calibration_report.json").exists()
        assert (run_path / "calibration_report.md").exists()
        assert (run_path / "plots").is_dir()
