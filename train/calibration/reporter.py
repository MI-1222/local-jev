"""較正評価 Exit Criteria 判定およびレポート生成モジュール。

Phase 4 ロードマップの達成条件（ECE <= 0.08, 確信度0.8での精度 78-82%, Brier Score 10% 改善など）
を機械的に合否判定し、機械可読な JSON および人間可読な Markdown レポートを出力する。
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def evaluate_exit_criteria(
    eval_summary: dict[str, Any],
    min_confidence_samples: int = 15,
) -> dict[str, Any]:
    """Phase 4 Exit Criteria の合否判定を機械的に実行する。

    判定項目:
    1. 全プリミティブ事後 ECE が 0.08 (8%) 以下であること。
    2. 確信度 0.8 付近 ([0.75, 0.85]) の実正答率が 78% 〜 82% に収まること。
    3. Brier Score が SFT (事前) 比で 10% 以上改善 (減少) していること。
    4. 温度スケーリング前後で決定精度が不変 (Accuracy Invariance) であること。

    Args:
        eval_summary (dict[str, Any]): 最適化・評価結果サマリー辞書。
        min_confidence_samples (int): 確信度 0.8 検証に最低限必要なサンプル数。

    Returns:
        dict[str, Any]: 各判定項目の判定結果および総合判定辞書。
    """
    pre_overall = eval_summary.get("pre_calibration", {})
    post_overall = eval_summary.get("post_calibration", {})

    post_ece = post_overall.get("ece", 1.0)
    pre_brier = pre_overall.get("brier_score", 0.0)
    post_brier = post_overall.get("brier_score", 0.0)
    pre_acc = pre_overall.get("accuracy", 0.0)
    post_acc = post_overall.get("accuracy", 0.0)

    # 1. ECE <= 0.08 判定
    ece_target = 0.08
    ece_pass = bool(post_ece <= ece_target)

    # 2. 確信度 0.8 での正答率 78% 〜 82% 判定
    acc_08_info = post_overall.get("accuracy_at_08", {})
    cnt_08 = acc_08_info.get("count", 0)
    acc_08 = acc_08_info.get("accuracy", 0.0)
    if cnt_08 >= min_confidence_samples:
        acc_08_pass = bool(0.78 <= acc_08 <= 0.82)
        acc_08_status = "PASS" if acc_08_pass else "FAIL"
    elif cnt_08 > 0:
        acc_08_pass = bool(0.78 <= acc_08 <= 0.82)
        acc_08_status = "WARNING_FEW_SAMPLES" if acc_08_pass else "FAIL"
    else:
        acc_08_pass = False
        acc_08_status = "NO_SAMPLES"

    # 3. Brier Score 改善率 >= 10%
    if pre_brier > 1e-6:
        brier_improvement_rate = (pre_brier - post_brier) / pre_brier
        brier_pass = bool(brier_improvement_rate >= 0.10)
    else:
        brier_improvement_rate = 0.0
        brier_pass = False

    # 4. 精度不変性 (Accuracy Invariance)
    acc_diff = abs(post_acc - pre_acc)
    acc_invariant = bool(acc_diff <= 1e-4)

    # プリミティブ別の詳細チェック
    buckets = eval_summary.get("buckets", {})
    choice_pass = True
    score_pass = True
    noul_pass = True

    for b_info in buckets.values():
        q_type = b_info.get("question_type", "")
        p_eval = b_info.get("post_calibration", {})
        b_ece = p_eval.get("ece", 1.0)
        if q_type == "choice" and b_ece > ece_target:
            choice_pass = False
        elif q_type == "score" and b_ece > ece_target:
            score_pass = False
        elif q_type == "noul":
            bin_ece = p_eval.get("binary_ece", b_ece)
            if bin_ece > ece_target:
                noul_pass = False

    all_criteria_met = bool(
        ece_pass
        and (acc_08_pass or acc_08_status == "WARNING_FEW_SAMPLES")
        and brier_pass
        and acc_invariant
    )

    criteria_results = {
        "overall_status": "PASS" if all_criteria_met else "FAIL",
        "all_criteria_met": all_criteria_met,
        "items": [
            {
                "id": "ece_threshold",
                "name": "事後期待較正誤差 (ECE)",
                "target": f"<= {ece_target:.2f} (8.0%)",
                "measured": f"{post_ece:.4f} ({(post_ece * 100):.2f}%)",
                "status": "PASS" if ece_pass else "FAIL",
                "passed": ece_pass,
            },
            {
                "id": "confidence_08_accuracy",
                "name": "確信度 0.8 での実正答率整合性",
                "target": "78.0% 〜 82.0% ([0.75, 0.85] ウィンドウ)",
                "measured": f"{(acc_08 * 100):.2f}% (N={cnt_08} 件)",
                "status": acc_08_status,
                "passed": acc_08_pass,
            },
            {
                "id": "brier_improvement",
                "name": "Brier Score 改善率",
                "target": ">= 10.0% 減少 (事前比)",
                "measured": f"{(brier_improvement_rate * 100):+.2f}% (事前: {pre_brier:.4f} → 事後: {post_brier:.4f})",
                "status": "PASS" if brier_pass else "FAIL",
                "passed": brier_pass,
            },
            {
                "id": "accuracy_invariance",
                "name": "決定精度の不変性 (単調変換性)",
                "target": "|Acc_post - Acc_pre| <= 0.0001",
                "measured": f"ΔAcc: {post_acc - pre_acc:+.4f} (事前: {pre_acc:.4f} → 事後: {post_acc:.4f})",
                "status": "PASS" if acc_invariant else "FAIL",
                "passed": acc_invariant,
            },
        ],
        "primitive_breakdown": {
            "choice_criteria_met": choice_pass,
            "score_criteria_met": score_pass,
            "noul_criteria_met": noul_pass,
        },
    }

    return criteria_results


class CalibrationReporter:
    """較正結果の評価レポートを生成・永続化するレポータークラス。"""

    def __init__(self, output_dir: str | Path) -> None:
        """レポーターを初期化する。

        Args:
            output_dir (str | Path): 出力ディレクトリパス。
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        eval_summary: dict[str, Any],
        diagram_paths: dict[str, str] | None = None,
    ) -> tuple[Path, Path]:
        """JSON および Markdown レポートを生成して保存する。

        Args:
            eval_summary (dict[str, Any]): 評価メトリクス集計データ。
            diagram_paths (dict[str, str] | None): 生成された画像ファイルへの相対/絶対パス辞書。

        Returns:
            tuple[Path, Path]: (JSON レポートパス, Markdown レポートパス)。
        """
        criteria_results = evaluate_exit_criteria(eval_summary)
        full_report = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "exit_criteria": criteria_results,
            "metrics_summary": eval_summary,
            "diagrams": diagram_paths or {},
        }

        # 1. JSON レポート保存
        json_path = self.output_dir / "calibration_report.json"
        json_path.write_text(
            json.dumps(full_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # 2. Markdown レポート生成・保存
        md_text = self._build_markdown(
            criteria_results, eval_summary, diagram_paths or {}
        )
        md_path = self.output_dir / "calibration_report.md"
        md_path.write_text(md_text, encoding="utf-8")

        return json_path, md_path

    def _build_markdown(
        self,
        criteria: dict[str, Any],
        eval_summary: dict[str, Any],
        diagram_paths: dict[str, str],
    ) -> str:
        """Markdown 本文を組み立てる。

        Args:
            criteria (dict[str, Any]): 判定基準辞書。
            eval_summary (dict[str, Any]): メトリクス集計辞書。
            diagram_paths (dict[str, str]): 画像パス。

        Returns:
            str: Markdown 文字列。
        """
        now_str = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        overall_status = criteria["overall_status"]
        status_badge = "✅ PASS" if overall_status == "PASS" else "❌ FAIL"

        lines = [
            "# 📈 Local-Jev: キャリブレーション信頼性評価レポート",
            "",
            f"- **評価実行日時**: {now_str}",
            f"- **Phase 4 Exit Criteria 判定**: **{status_badge}**",
            f"- **総評価サンプル数**: {eval_summary.get('total_samples', 0)} 件",
            "",
            "## 1. Phase 4 Exit Criteria (達成条件) 自動合否判定",
            "",
            "| 判定項目 | 目標基準 | 実測値 | 判定ステータス |",
            "| :--- | :--- | :--- | :---: |",
        ]

        for item in criteria["items"]:
            s = item["status"]
            if s == "PASS":
                s_icon = "✅ PASS"
            elif s == "WARNING_FEW_SAMPLES":
                s_icon = "⚠️ WARNING (少数サンプル)"
            else:
                s_icon = "❌ FAIL"

            lines.append(
                f"| **{item['name']}** | {item['target']} | {item['measured']} | {s_icon} |"
            )

        lines.extend(
            [
                "",
                "## 2. プリミティブ別・バケット別較正性能",
                "",
                "| 質問タイプ | バケット | サンプル数 | 温度 $\\tau$ | 事前 ECE | 事後 ECE | Brier Score | Top-1 精度 | 較正状態 |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )

        buckets = eval_summary.get("buckets", {})
        for b_key, b_info in buckets.items():
            q_type = b_info.get("question_type", b_key)
            b_name = b_info.get("bucket", "")
            samples = b_info.get("samples", 0)
            tau = b_info.get("temperature", 1.0)
            pre_e = b_info.get("pre_calibration", {})
            post_e = b_info.get("post_calibration", {})

            pre_ece = pre_e.get("ece", 0.0)
            post_ece = post_e.get("ece", 0.0)
            brier = post_e.get("brier_score", 0.0)
            acc = post_e.get("accuracy", 0.0)
            status_text = b_info.get("reason", "最適化完了")

            lines.append(
                f"| {q_type.upper()} | `{b_name}` | {samples} | `{tau:.4f}` | {pre_ece:.4f} | **{post_ece:.4f}** | {brier:.4f} | {acc:.4f} | {status_text} |"
            )

        # 3. 視覚的信頼性ダイアグラムのリンク
        if diagram_paths:
            lines.extend(
                [
                    "",
                    "## 3. 信頼性ダイアグラム (Reliability Diagrams)",
                    "",
                ]
            )

            if "overall_comparison" in diagram_paths:
                rel_p = Path(diagram_paths["overall_comparison"]).name
                lines.extend(
                    [
                        "### 全体較正比較ダイアグラム (Pre vs Post)",
                        f"![Overall Reliability Diagram]({rel_p})",
                        "",
                    ]
                )

            for key, p in diagram_paths.items():
                if key == "overall_comparison":
                    continue
                img_name = Path(p).name
                lines.extend(
                    [
                        f"### {key.upper()} ダイアグラム",
                        f"![{key} Diagram]({img_name})",
                        "",
                    ]
                )

        lines.extend(
            [
                "## 4. 結論と次のステップ",
                "",
                "本レポートは `evaluator.py`, `diagram.py`, `reporter.py` により自動生成されました。",
                "合否判定が PASS の場合、導出された最適パラメータを本番モデルへ反映可能です。",
            ]
        )

        return "\n".join(lines) + "\n"
