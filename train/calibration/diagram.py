"""信頼性ダイアグラム (Reliability Diagram) 自動描画モジュール。

予測確信度と実際の正答率の統計的一致度を視覚化する 2 段構成ダイアグラム
（上段: 信頼性曲線 & 較正ギャップ、下段: 確信度サンプル分布ヒストグラム）、
および事前 (Pre) vs 事後 (Post) 較正比較プロットの生成と画像保存を提供する。
ヘッドレス環境 (CI / Docker) に完全対応し、Marimo ノートブックとの親和性を持つ。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

# ヘッドレス環境での GUI クラッシュを防止
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_reliability_diagram(
    diagram_data: Sequence[dict[str, Any]],
    title: str = "Reliability Diagram",
    ece: float | None = None,
    brier_score: float | None = None,
    accuracy: float | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.5, 7.5),
    close_fig: bool = False,
) -> Figure:
    """単一モデルの信頼性ダイアグラムおよび確信度度数分布を描画する。

    数理仕様:
    上段に各ビンの平均確信度 vs 実測正解率を棒グラフで配置し、完全較正線 $y = x$ からの乖離
    （過信: 赤系、過小評価: 青系）を塗り分ける。
    下段に各ビンのサンプル件数分布を表示し、モデルの予測尖鋭度 (Sharpness) を可視化する。

    Args:
        diagram_data (Sequence[dict[str, Any]]): ビンごとの統計辞書リスト。
        title (str): グラフ上部に表示するタイトル。
        ece (float | None): 期待較正誤差 (表示用)。
        brier_score (float | None): Brier スコア (表示用)。
        accuracy (float | None): Top-1 正解率 (表示用)。
        save_path (str | Path | None): 画像保存先パス (None の場合は保存なし)。
        figsize (tuple[float, float]): 図の全体サイズ。

    Returns:
        Figure: 生成された Matplotlib Figure オブジェクト。
    """
    fig, (ax_rel, ax_hist) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        height_ratios=[3, 1],
        sharex=True,
    )

    if not diagram_data:
        ax_rel.text(
            0.5,
            0.5,
            "No Data Available",
            ha="center",
            va="center",
            transform=ax_rel.transAxes,
        )
        return fig

    # ビン統計量の抽出
    bin_lowers = [b["bin_lower"] for b in diagram_data]
    bin_uppers = [b["bin_upper"] for b in diagram_data]
    bin_centers = [(l + u) / 2.0 for l, u in zip(bin_lowers, bin_uppers, strict=False)]
    bin_widths = [u - l for l, u in zip(bin_lowers, bin_uppers, strict=False)]
    accuracies = [b["accuracy"] for b in diagram_data]
    confidences = [b["confidence"] for b in diagram_data]
    counts = [b["count"] for b in diagram_data]

    # 上段: 信頼性ダイアグラム
    # 完全較正対角線
    ax_rel.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Perfect Calibration ($y = x$)",
        zorder=2,
    )

    for i in range(len(diagram_data)):
        w = bin_widths[i] * 0.92
        c = bin_centers[i]
        acc = accuracies[i]
        conf = confidences[i]
        cnt = counts[i]

        if cnt == 0:
            continue

        # 実績正解率の主バー
        ax_rel.bar(
            c,
            acc,
            width=w,
            color="#3b82f6",
            alpha=0.75,
            edgecolor="#1d4ed8",
            linewidth=1.0,
            zorder=3,
        )

        # 較正ギャップの塗り分け (過信: 赤, 過小評価: シアン/青)
        gap = acc - conf
        if gap < 0:
            # 過信 (Confidence > Accuracy)
            ax_rel.bar(
                c,
                -gap,
                bottom=acc,
                width=w,
                color="#ef4444",
                alpha=0.45,
                hatch="//",
                edgecolor="#b91c1c",
                linewidth=0.8,
                zorder=3,
            )
        elif gap > 0:
            # 過小評価 (Accuracy > Confidence)
            ax_rel.bar(
                c,
                gap,
                bottom=conf,
                width=w,
                color="#06b6d4",
                alpha=0.45,
                hatch="\\\\",
                edgecolor="#0e7490",
                linewidth=0.8,
                zorder=3,
            )

        # Wilson 信頼区間エラーバー (存在する場合)
        if "ci_lower" in diagram_data[i] and "ci_upper" in diagram_data[i]:
            ci_l = diagram_data[i]["ci_lower"]
            ci_u = diagram_data[i]["ci_upper"]
            err_l = max(0.0, acc - ci_l)
            err_u = max(0.0, ci_u - acc)
            ax_rel.errorbar(
                c,
                acc,
                yerr=[[err_l], [err_u]],
                fmt="none",
                ecolor="#1e293b",
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
                zorder=4,
            )

    # 上段装飾
    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.05)
    ax_rel.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax_rel.grid(True, linestyle=":", alpha=0.6, zorder=1)

    # 指標テキスト注釈
    metric_texts = []
    if ece is not None:
        metric_texts.append(f"ECE: {ece:.4f}")
    if brier_score is not None:
        metric_texts.append(f"Brier: {brier_score:.4f}")
    if accuracy is not None:
        metric_texts.append(f"Acc: {accuracy:.4f}")

    if metric_texts:
        info_box = "\n".join(metric_texts)
        ax_rel.text(
            0.05,
            0.92,
            info_box,
            transform=ax_rel.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.9,
            },
            zorder=5,
        )

    # 凡例ダミー要素
    ax_rel.bar(
        [0],
        [0],
        color="#3b82f6",
        alpha=0.75,
        edgecolor="#1d4ed8",
        label="Accuracy",
    )
    ax_rel.bar(
        [0],
        [0],
        color="#ef4444",
        alpha=0.45,
        hatch="//",
        edgecolor="#b91c1c",
        label="Gap (Overconfidence)",
    )
    ax_rel.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax_rel.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # 下段: 確信度ヒストグラム
    ax_hist.bar(
        bin_centers,
        counts,
        width=[w * 0.92 for w in bin_widths],
        color="#64748b",
        alpha=0.8,
        edgecolor="#334155",
        linewidth=0.8,
        zorder=3,
    )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Confidence", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax_hist.grid(True, linestyle=":", alpha=0.6, zorder=1)

    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    if close_fig:
        plt.close(fig)

    return fig


def plot_calibration_comparison(
    pre_data: dict[str, Any],
    post_data: dict[str, Any],
    title: str = "Calibration Comparison (Pre-SFT vs Post-Calibration)",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 7.5),
    close_fig: bool = False,
) -> Figure:
    """較正前 (Pre) と較正後 (Post) の信頼性ダイアグラムを左右に並列描画する。

    Args:
        pre_data (dict[str, Any]): 較正前の評価メトリクス辞書。
        post_data (dict[str, Any]): 較正後の評価メトリクス辞書。
        title (str): 全体タイトル。
        save_path (str | Path | None): 画像保存先パス。
        figsize (tuple[float, float]): 全体図サイズ。

    Returns:
        Figure: 比較プロットの Matplotlib Figure。
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        height_ratios=[3, 1],
        sharex=True,
    )

    col_configs = [
        (
            axes[0, 0],
            axes[1, 0],
            pre_data,
            f"Pre-Calibration (tau={pre_data.get('temperature', 1.0):.2f})",
        ),
        (
            axes[0, 1],
            axes[1, 1],
            post_data,
            f"Post-Calibration (tau={post_data.get('temperature', 1.0):.2f})",
        ),
    ]

    for ax_rel, ax_hist, data, sub_title in col_configs:
        diagram_data = data.get("reliability_diagram", [])
        ece = data.get("ece")
        brier = data.get("brier_score")
        acc = data.get("accuracy")

        # 対角線
        ax_rel.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color="gray",
            linewidth=1.2,
            zorder=2,
        )

        if diagram_data:
            bin_lowers = [b["bin_lower"] for b in diagram_data]
            bin_uppers = [b["bin_upper"] for b in diagram_data]
            bin_centers = [
                (l + u) / 2.0 for l, u in zip(bin_lowers, bin_uppers, strict=False)
            ]
            bin_widths = [u - l for l, u in zip(bin_lowers, bin_uppers, strict=False)]
            accuracies = [b["accuracy"] for b in diagram_data]
            confidences = [b["confidence"] for b in diagram_data]
            counts = [b["count"] for b in diagram_data]

            for i in range(len(diagram_data)):
                w = bin_widths[i] * 0.92
                c = bin_centers[i]
                b_acc = accuracies[i]
                b_conf = confidences[i]
                cnt = counts[i]

                if cnt == 0:
                    continue

                ax_rel.bar(
                    c,
                    b_acc,
                    width=w,
                    color="#3b82f6",
                    alpha=0.75,
                    edgecolor="#1d4ed8",
                    linewidth=0.8,
                    zorder=3,
                )

                gap = b_acc - b_conf
                if gap < 0:
                    ax_rel.bar(
                        c,
                        -gap,
                        bottom=b_acc,
                        width=w,
                        color="#ef4444",
                        alpha=0.45,
                        hatch="//",
                        edgecolor="#b91c1c",
                        linewidth=0.8,
                        zorder=3,
                    )
                elif gap > 0:
                    ax_rel.bar(
                        c,
                        gap,
                        bottom=b_conf,
                        width=w,
                        color="#06b6d4",
                        alpha=0.45,
                        hatch="\\\\",
                        edgecolor="#0e7490",
                        linewidth=0.8,
                        zorder=3,
                    )

            # 下段ヒストグラム
            ax_hist.bar(
                bin_centers,
                counts,
                width=[w * 0.92 for w in bin_widths],
                color="#64748b",
                alpha=0.8,
                edgecolor="#334155",
                linewidth=0.8,
                zorder=3,
            )

        ax_rel.set_xlim(0.0, 1.0)
        ax_rel.set_ylim(0.0, 1.05)
        ax_rel.set_title(sub_title, fontsize=11, fontweight="bold")
        ax_rel.grid(True, linestyle=":", alpha=0.5, zorder=1)

        # 指標ボックス
        m_texts = []
        if ece is not None:
            m_texts.append(f"ECE: {ece:.4f}")
        if brier is not None:
            m_texts.append(f"Brier: {brier:.4f}")
        if acc is not None:
            m_texts.append(f"Acc: {acc:.4f}")

        if m_texts:
            ax_rel.text(
                0.05,
                0.92,
                "\n".join(m_texts),
                transform=ax_rel.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
                zorder=5,
            )

        ax_hist.set_xlim(0.0, 1.0)
        ax_hist.set_xlabel("Confidence", fontsize=10, fontweight="bold")
        ax_hist.grid(True, linestyle=":", alpha=0.5, zorder=1)

    axes[0, 0].set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Count", fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    if close_fig:
        plt.close(fig)

    return fig


def export_all_diagrams(
    eval_results: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """各プリミティブおよび全体のダイアグラム画像をまとめて生成・保存する。

    Args:
        eval_results (dict[str, Any]): 評価メトリクス集計結果辞書。
        output_dir (str | Path): 保存先親ディレクトリ。

    Returns:
        dict[str, str]: 生成された画像ファイルパスの対応辞書。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: dict[str, str] = {}

    # 1. 全体比較ダイアグラム (存在する場合)
    if "pre_calibration" in eval_results and "post_calibration" in eval_results:
        cmp_path = out_dir / "reliability_comparison_overall.png"
        plot_calibration_comparison(
            eval_results["pre_calibration"],
            eval_results["post_calibration"],
            title="Overall Reliability Diagram (Pre vs Post Calibration)",
            save_path=cmp_path,
            close_fig=True,
        )
        generated_files["overall_comparison"] = str(cmp_path)

    # 2. バケット別・プリミティブ別ダイアグラム
    buckets = eval_results.get("buckets", {})
    for b_key, b_info in buckets.items():
        post_calib = b_info.get("post_calibration", {})
        diagram_data = post_calib.get("reliability_diagram", [])
        if not diagram_data:
            continue

        q_type = b_info.get("question_type", b_key)
        b_name = b_info.get("bucket", "")
        img_name = f"reliability_diagram_{b_key}.png"
        img_path = out_dir / img_name

        plot_reliability_diagram(
            diagram_data=diagram_data,
            title=f"Reliability Diagram: {q_type.upper()} ({b_name})",
            ece=post_calib.get("ece"),
            brier_score=post_calib.get("brier_score"),
            accuracy=post_calib.get("accuracy"),
            save_path=img_path,
            close_fig=True,
        )
        generated_files[b_key] = str(img_path)

        # Noul 型の二値絶対確率ダイアグラム
        if q_type == "noul" and "binary_reliability_diagram" in post_calib:
            binary_data = post_calib["binary_reliability_diagram"]
            noul_img_name = "reliability_diagram_noul_binary.png"
            noul_img_path = out_dir / noul_img_name
            plot_reliability_diagram(
                diagram_data=binary_data,
                title="Reliability Diagram: NOUL (P(True) Calibration)",
                ece=post_calib.get("binary_ece"),
                brier_score=post_calib.get("brier_score"),
                accuracy=post_calib.get("accuracy"),
                save_path=noul_img_path,
                close_fig=True,
            )
            generated_files["noul_binary"] = str(noul_img_path)

    return generated_files
