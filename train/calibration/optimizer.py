"""事後温度最適化 (Temperature Optimizer) モジュール。

ホールドアウト検証データからロジットおよび有効候補マスクをキャッシュし、
質問プリミティブ別・候補数バケット別に 1 変数凸最適化 (Bounded Brent 法) を用いて
負の対数尤度 (NLL) を最小化する最適温度係数 tau* を同定する。
"""

import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import Tensor, nn
from torch.utils.data import DataLoader

from calibration.config import (
    CHOICE_BUCKETS,
    NOUL_BUCKET,
    SCORE_BUCKETS,
    CalibrationRunConfig,
    matches_bucket_expr,
)
from calibration.evaluator import (
    CalibrationEvaluator,
    compute_accuracy,
    compute_masked_nll,
)
from contract import CalibrationConfig, TemperatureMap
from data.schema import QuestionType

logger = logging.getLogger(__name__)


@dataclass
class LogitCache:
    """検証データから収集した推論結果のインメモリキャッシュ構造体。

    Attributes:
        logits (Tensor): 未マスクの生ロジットテンソル `[N, max_options]` (CPU)。
        op_mask (Tensor): 有効候補マスクテンソル `[N, max_options]` (CPU)。
        labels (Tensor): 正解インデックス `[N]` (CPU)。
        question_types (list[str]): 各サンプルの質問タイプ文字列。
        candidate_counts (list[int]): 各サンプルの有効候補数。
    """

    logits: Tensor
    op_mask: Tensor
    labels: Tensor
    question_types: list[str]
    candidate_counts: list[int]

    @property
    def total_samples(self) -> int:
        """キャッシュ内の総サンプル数を取得する。"""
        return int(self.logits.size(0))

    def filter_by_bucket(
        self,
        target_type: str | QuestionType,
        bucket_expr: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """質問タイプおよび候補数バケットに一致するサンプルをスライス抽出する。

        Args:
            target_type (str | QuestionType): 抽出対象の質問タイプ。
            bucket_expr (str | None): 候補数バケット文字列 (None の場合はバケット条件不問)。

        Returns:
            tuple[Tensor, Tensor, Tensor]: 抽出された (logits, labels, op_mask)。
        """
        type_str = (
            target_type.value
            if isinstance(target_type, QuestionType)
            else str(target_type)
        )

        indices: list[int] = []
        for i in range(self.total_samples):
            if self.question_types[i] != type_str:
                continue

            if bucket_expr is not None and not matches_bucket_expr(
                bucket_expr, self.candidate_counts[i]
            ):
                continue

            indices.append(i)

        if not indices:
            empty_k = self.logits.size(-1) if self.total_samples > 0 else 0
            return (
                torch.empty(0, empty_k, dtype=self.logits.dtype),
                torch.empty(0, dtype=self.labels.dtype),
                torch.empty(0, empty_k, dtype=self.op_mask.dtype),
            )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return (
            self.logits[idx_tensor],
            self.labels[idx_tensor],
            self.op_mask[idx_tensor],
        )


def collect_env_metadata(config: CalibrationRunConfig) -> dict[str, Any]:
    """実行環境および Git 追跡情報を収集する。

    Args:
        config (CalibrationRunConfig): 較正設定。

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
        logger.warning("Git 情報の取得に失敗しました: %s.", e)

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
        "device_name": device_name,
        "config": config.to_dict(),
    }


class TemperatureOptimizer:
    """質問タイプ別・バケット別に最適温度係数を探索・適用するオプティマイザ。"""

    def __init__(self, config: CalibrationRunConfig | None = None) -> None:
        """オプティマイザを初期化する。

        Args:
            config (CalibrationRunConfig | None): 較正設定。未指定時はデフォルト設定を利用。
        """
        self.config = config or CalibrationRunConfig()
        self.evaluator = CalibrationEvaluator(num_bins=self.config.num_bins)

    def collect_logits(
        self,
        model: nn.Module,
        dataloader: DataLoader[dict[str, Any]],
        device: torch.device,
        max_samples: int = 0,
    ) -> LogitCache:
        """検証データローダーに対して単一フォワードパス推論を実行し、ロジット群をキャッシュする。

        Args:
            model (nn.Module): 評価対象モデル。
            dataloader (DataLoader[dict[str, Any]]): 検証 DataLoader。
            device (torch.device): 推論実行デバイス。
            max_samples (int): 収集サンプル数の上限 (0 で全件)。

        Returns:
            LogitCache: 収集されたインメモリキャッシュ。
        """
        model.eval()

        logits_list: list[Tensor] = []
        op_mask_list: list[Tensor] = []
        labels_list: list[Tensor] = []
        qtypes_list: list[str] = []
        counts_list: list[int] = []

        collected_count = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                op_indices = batch["op_indices"].to(device)
                op_mask = batch["op_mask"].to(device)
                labels = batch["labels"].to(device)

                batch_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    op_indices=op_indices,
                    op_mask=op_mask,
                )

                batch_qtypes = batch["question_types"]
                batch_counts = op_mask.sum(dim=-1).cpu().tolist()

                logits_list.append(batch_logits.cpu())
                op_mask_list.append(op_mask.cpu())
                labels_list.append(labels.cpu())

                for q in batch_qtypes:
                    q_str = q.value if isinstance(q, QuestionType) else str(q)
                    qtypes_list.append(q_str)

                counts_list.extend([int(c) for c in batch_counts])

                collected_count += int(input_ids.size(0))
                if max_samples > 0 and collected_count >= max_samples:
                    break

        all_logits = torch.cat(logits_list, dim=0)
        all_op_mask = torch.cat(op_mask_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        if max_samples > 0 and all_logits.size(0) > max_samples:
            all_logits = all_logits[:max_samples]
            all_op_mask = all_op_mask[:max_samples]
            all_labels = all_labels[:max_samples]
            qtypes_list = qtypes_list[:max_samples]
            counts_list = counts_list[:max_samples]

        return LogitCache(
            logits=all_logits,
            op_mask=all_op_mask,
            labels=all_labels,
            question_types=qtypes_list,
            candidate_counts=counts_list,
        )

    def optimize_scalar(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        tau_min: float | None = None,
        tau_max: float | None = None,
    ) -> float:
        """単一バケットのロジットに対して 1 変数凸最適化を行い、最適温度 tau* を算出する。

        目的関数:
        ホールドアウトデータに対する負の対数尤度 (Masked NLL) の最小化。
        最適化前後で精度不変性 (Accuracy Invariance) が保たれていることを自動検証する。

        Args:
            logits (Tensor): 未マスクの生ロジットテンソル `[N, K]`。
            labels (Tensor): 正解インデックス `[N]`。
            op_mask (Tensor): 有効候補マスク `[N, K]`。
            tau_min (float | None): 探索下限 (None 時は config 値)。
            tau_max (float | None): 探索上限 (None 時は config 値)。

        Returns:
            float: 最適温度係数 tau*。
        """
        min_tau = tau_min if tau_min is not None else self.config.tau_min
        max_tau = tau_max if tau_max is not None else self.config.tau_max

        if logits.size(0) == 0:
            return self.config.default_temperature

        def objective(tau: float) -> float:
            return compute_masked_nll(logits, labels, op_mask, temperature=tau)

        res = minimize_scalar(
            objective,
            bounds=(min_tau, max_tau),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 100},
        )

        opt_tau = float(res.x)

        # 精度不変性の検証
        acc_before = compute_accuracy(logits, labels, op_mask)
        acc_after = compute_accuracy(logits / opt_tau, labels, op_mask)
        if not np.isclose(acc_before, acc_after, atol=1e-4):
            logger.warning(
                "温度スケーリング後に精度不変性が破られました (前: %.4f, 後: %.4f)。",
                acc_before,
                acc_after,
            )

        return float(np.clip(opt_tau, min_tau, max_tau))

    def calibrate(
        self,
        cache: LogitCache,
    ) -> tuple[CalibrationConfig, dict[str, Any]]:
        """キャッシュされたロジットを用いて全バケットの温度探索を実行する。

        Choice, Score, Noul 各バケットについて独立に最適温度を導出し、
        最適化前後の較正メトリクス (ECE, NLL, Brier, Accuracy) の比較を生成する。

        Args:
            cache (LogitCache): 検証データのロジットキャッシュ。

        Returns:
            tuple[CalibrationConfig, dict[str, Any]]: (Rust 連携用契約設定, 評価レポート辞書)。
        """
        choice_temps: dict[str, float] = {}
        score_temps: dict[str, float] = {}
        noul_temp = self.config.default_temperature

        bucket_metrics: dict[str, Any] = {}

        # 1. Choice 型のバケット別探索
        for b_expr in CHOICE_BUCKETS:
            b_logits, b_labels, b_op_mask = cache.filter_by_bucket(
                QuestionType.CHOICE, b_expr
            )
            sample_count = int(b_logits.size(0))

            if sample_count >= self.config.min_samples_per_bucket:
                tau = self.optimize_scalar(b_logits, b_labels, b_op_mask)
                is_fallback = False
            else:
                tau = self.config.default_temperature
                is_fallback = True

            choice_temps[b_expr] = round(tau, 4)

            # メトリクス比較
            eval_pre = self.evaluator.evaluate(
                b_logits, b_labels, b_op_mask, temperature=1.0
            )
            eval_post = self.evaluator.evaluate(
                b_logits, b_labels, b_op_mask, temperature=tau
            )
            bucket_metrics[f"choice_{b_expr}"] = {
                "question_type": "choice",
                "bucket": b_expr,
                "samples": sample_count,
                "is_fallback": is_fallback,
                "temperature": round(tau, 4),
                "pre_calibration": eval_pre,
                "post_calibration": eval_post,
                "ece_delta": round(eval_post["ece"] - eval_pre["ece"], 4),
                "nll_delta": round(eval_post["nll"] - eval_pre["nll"], 4),
            }

        # 2. Score 型のバケット別探索
        for b_expr in SCORE_BUCKETS:
            b_logits, b_labels, b_op_mask = cache.filter_by_bucket(
                QuestionType.SCORE, b_expr
            )
            sample_count = int(b_logits.size(0))

            if sample_count >= self.config.min_samples_per_bucket:
                tau = self.optimize_scalar(b_logits, b_labels, b_op_mask)
                is_fallback = False
            else:
                tau = self.config.default_temperature
                is_fallback = True

            score_temps[b_expr] = round(tau, 4)

            eval_pre = self.evaluator.evaluate(
                b_logits, b_labels, b_op_mask, temperature=1.0
            )
            eval_post = self.evaluator.evaluate(
                b_logits, b_labels, b_op_mask, temperature=tau
            )
            bucket_metrics[f"score_{b_expr}"] = {
                "question_type": "score",
                "bucket": b_expr,
                "samples": sample_count,
                "is_fallback": is_fallback,
                "temperature": round(tau, 4),
                "pre_calibration": eval_pre,
                "post_calibration": eval_post,
                "ece_delta": round(eval_post["ece"] - eval_pre["ece"], 4),
                "nll_delta": round(eval_post["nll"] - eval_pre["nll"], 4),
            }

        # 3. Noul 型の探索
        n_logits, n_labels, n_op_mask = cache.filter_by_bucket(QuestionType.NOUL)
        sample_count = int(n_logits.size(0))
        if sample_count >= self.config.min_samples_per_bucket:
            noul_temp = self.optimize_scalar(n_logits, n_labels, n_op_mask)
            is_fallback = False
        else:
            noul_temp = self.config.default_temperature
            is_fallback = True

        noul_temp = round(noul_temp, 4)
        eval_pre = self.evaluator.evaluate(
            n_logits, n_labels, n_op_mask, temperature=1.0
        )
        eval_post = self.evaluator.evaluate(
            n_logits, n_labels, n_op_mask, temperature=noul_temp
        )
        bucket_metrics["noul"] = {
            "question_type": "noul",
            "bucket": NOUL_BUCKET,
            "samples": sample_count,
            "is_fallback": is_fallback,
            "temperature": noul_temp,
            "pre_calibration": eval_pre,
            "post_calibration": eval_post,
            "ece_delta": round(eval_post["ece"] - eval_pre["ece"], 4),
            "nll_delta": round(eval_post["nll"] - eval_pre["nll"], 4),
        }

        # 4. 全体加重集計
        total_pre_ece = 0.0
        total_post_ece = 0.0
        total_pre_nll = 0.0
        total_post_nll = 0.0
        total_eval_samples = 0

        for m in bucket_metrics.values():
            s = m["samples"]
            if s > 0:
                total_pre_ece += m["pre_calibration"]["ece"] * s
                total_post_ece += m["post_calibration"]["ece"] * s
                total_pre_nll += m["pre_calibration"]["nll"] * s
                total_post_nll += m["post_calibration"]["nll"] * s
                total_eval_samples += s

        overall_pre_ece = (
            round(total_pre_ece / total_eval_samples, 4)
            if total_eval_samples > 0
            else 0.0
        )
        overall_post_ece = (
            round(total_post_ece / total_eval_samples, 4)
            if total_eval_samples > 0
            else 0.0
        )
        overall_pre_nll = (
            round(total_pre_nll / total_eval_samples, 4)
            if total_eval_samples > 0
            else 0.0
        )
        overall_post_nll = (
            round(total_post_nll / total_eval_samples, 4)
            if total_eval_samples > 0
            else 0.0
        )

        overall_summary = {
            "total_samples": total_eval_samples,
            "pre_calibration": {
                "ece": overall_pre_ece,
                "nll": overall_pre_nll,
            },
            "post_calibration": {
                "ece": overall_post_ece,
                "nll": overall_post_nll,
            },
            "ece_improvement": round(overall_pre_ece - overall_post_ece, 4),
            "target_ece_met": bool(overall_post_ece < 0.10),
            "buckets": bucket_metrics,
        }

        # 5. CalibrationConfig インスタンス構築
        calib_config = CalibrationConfig(
            version="1.0",
            default_temperature=self.config.default_temperature,
            temperature_map=TemperatureMap(
                choice=choice_temps,
                score=score_temps,
                noul=noul_temp,
            ),
        )

        return calib_config, overall_summary

    def save_run_artifacts(
        self,
        output_dir: Path | str,
        calib_config: CalibrationConfig,
        metrics_summary: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """較正結果の永続化成果物一式を指定ディレクトリへ保存する。

        保存物:
        1. calibration.json (Rust 側契約ファイル)
        2. calibration_run_config.yaml (実行設定)
        3. calibration_metrics.json (最適化前後の性能メトリクス)
        4. run_metadata.json (環境情報・Git 情報)
        5. summary.md (人間向け確認レポート)

        Args:
            output_dir (Path | str): 保存先ディレクトリ。
            calib_config (CalibrationConfig): Rust 契約用キャリブレーション設定。
            metrics_summary (dict[str, Any]): 最適化前後の性能比較。
            metadata (dict[str, Any] | None): 実行メタデータ。

        Returns:
            Path: 生成された成果物ディレクトリパス。
        """
        run_path = Path(output_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        # 1. calibration.json
        calib_config.save(run_path / "calibration.json")

        # 2. calibration_run_config.yaml
        self.config.save(run_path / "calibration_run_config.yaml")

        # 3. calibration_metrics.json
        import json

        (run_path / "calibration_metrics.json").write_text(
            json.dumps(metrics_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # 4. run_metadata.json
        run_metadata = metadata or collect_env_metadata(self.config)
        (run_path / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # 5. summary.md
        summary_md = self._generate_summary_markdown(calib_config, metrics_summary)
        (run_path / "summary.md").write_text(summary_md, encoding="utf-8")

        logger.info("キャリブレーション成果物を保存しました: %s.", run_path)
        return run_path

    def _generate_summary_markdown(
        self,
        calib_config: CalibrationConfig,
        metrics_summary: dict[str, Any],
    ) -> str:
        """人手確認用のサマリーレポート Markdown を生成する。

        Args:
            calib_config (CalibrationConfig): 較正設定。
            metrics_summary (dict[str, Any]): メトリクス集計結果。

        Returns:
            str: 整形された Markdown 文字列。
        """
        pre_ece = metrics_summary["pre_calibration"]["ece"]
        post_ece = metrics_summary["post_calibration"]["ece"]
        pre_nll = metrics_summary["pre_calibration"]["nll"]
        post_nll = metrics_summary["post_calibration"]["nll"]
        status = "達成" if metrics_summary["target_ece_met"] else "未達"

        lines = [
            "# 📊 Local-Jev: 事後温度較正 (Post-hoc Calibration) サマリーレポート",
            "",
            f"- **実行日時**: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **目標 ECE (< 0.10)**: **{status}** (事後 ECE: `{post_ece:.4f}`, 事前: `{pre_ece:.4f}`)",
            f"- **平均 NLL 改善**: `{pre_nll:.4f}` → `{post_nll:.4f}` (Δ: `{post_nll - pre_nll:+.4f}`)",
            f"- **総検証サンプル数**: {metrics_summary['total_samples']} 件",
            "",
            "## 1. 導出された最適温度マップ (`calibration.json`)",
            "",
            "```json",
            calib_config.to_json(),
            "```",
            "",
            "## 2. バケット別較正性能の比較",
            "",
            "| 質問タイプ | バケット | サンプル数 | 最適温度 $\\tau^*$ | 事前 ECE | 事後 ECE | ECE 改善幅 | 状態 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for m in metrics_summary["buckets"].values():
            q_type = m["question_type"]
            bucket = m["bucket"]
            samples = m["samples"]
            tau = m["temperature"]
            b_pre_ece = m["pre_calibration"]["ece"]
            b_post_ece = m["post_calibration"]["ece"]
            delta = m["ece_delta"]
            fallback_label = "フォールバック" if m["is_fallback"] else "最適化完了"
            lines.append(
                f"| {q_type} | `{bucket}` | {samples} | **{tau:.4f}** | {b_pre_ece:.4f} | {b_post_ece:.4f} | {delta:+.4f} | {fallback_label} |"
            )

        lines.extend(
            [
                "",
                "## 3. Rust ランタイム (`crates/local-jev-core`) への反映方法",
                "",
                "生成された `calibration.json` は、Rust ランタイムの `CalibrationConfig` スキーマと完全互換です。",
                "モデル成果物ディレクトリへ配置して直接利用できます:",
                "```bash",
                "cp calibration.json ../models/calibration.json",
                "```",
            ]
        )

        return "\n".join(lines) + "\n"
