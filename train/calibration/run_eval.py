"""キャリブレーション信頼性評価実行 CLI (run_eval.py)。

モデル推論結果または保存済み評価メトリクスから、
信頼性ダイアグラムの自動描画、Exit Criteria の合否判定、
および総合レポート (JSON / Markdown) の出力を一括実行する。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# パッケージパスを解決
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from calibration.diagram import export_all_diagrams
from calibration.reporter import CalibrationReporter, evaluate_exit_criteria

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("run_eval")


def run_evaluation_from_metrics(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> int:
    """保存済み calibration_metrics.json を読み込み、ダイアグラム描画とレポート生成を実行する。

    Args:
        metrics_path (str | Path): 評価メトリクス JSON ファイルパス。
        output_dir (str | Path): 出力ディレクトリ。

    Returns:
        int: 終了コード (0: Exit Criteria PASS, 1: FAIL)。
    """
    m_path = Path(metrics_path)
    if not m_path.exists():
        logger.error("メトリクスファイルが見つかりません: %s.", m_path)
        return 1

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_summary: dict[str, Any] = json.loads(m_path.read_text(encoding="utf-8"))
    plots_dir = out_path / "plots"

    logger.info("信頼性ダイアグラムを描画中...")
    diagram_files = export_all_diagrams(metrics_summary, plots_dir)
    logger.info(
        "%d 個のダイアグラム画像を保存しました: %s.", len(diagram_files), plots_dir
    )

    logger.info("Exit Criteria を判定中...")
    reporter = CalibrationReporter(out_path)
    json_path, md_path = reporter.generate_report(
        metrics_summary, diagram_paths=diagram_files
    )
    logger.info("レポートを保存しました: %s, %s.", json_path, md_path)

    criteria = evaluate_exit_criteria(metrics_summary)
    is_pass = criteria.get("overall_status") == "PASS"

    print("\n" + "=" * 60)
    print(
        f"📊 キャリブレーション信頼性評価 判定結果: {'✅ PASS' if is_pass else '❌ FAIL'}"
    )
    print("=" * 60)
    for item in criteria.get("items", []):
        mark = "✅" if item.get("passed") else "❌"
        print(
            f"  {mark} {item.get('name')}: {item.get('measured')} (目標: {item.get('target')})"
        )
    print("=" * 60 + "\n")

    return 0 if is_pass else 1


def main() -> None:
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="Local-Jev キャリブレーション信頼性評価 CLI"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="評価メトリクス JSON ファイル (calibration_metrics.json) のパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/calibration/eval_report",
        help="成果物出力先ディレクトリ",
    )

    args = parser.parse_args()

    if not args.metrics:
        logger.error("--metrics 引数が指定されていません。")
        parser.print_help()
        sys.exit(1)

    exit_code = run_evaluation_from_metrics(args.metrics, args.output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
