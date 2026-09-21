import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌡️ Local-Jev: 事後温度較正 (Post-hoc Temperature Calibration) ノートブック

    本ノートブックは、TypeSafe AI アーキテクチャに準拠した決定モデル「Jev」の
    **質問プリミティブ別・候補数バケット別の事後温度スケーリング ($\tau^*$ 算出)** を対話的に実行・検証し、
    Rust 推論ランタイム (`local-jev-core` / `local-jev-runtime`) 向けに `calibration.json` を出力するための環境です。

    ### 📌 事後温度スケーリングの目的と数理背景
    1. **過剰な自信 (Overconfidence) の是正**:
       SFT や強化学習を経たディープモデルは、Top-1 予測精度が高い一方で、予測確率(確信度)が過大になる傾向があります。
    2. **精度不変性 (Accuracy Invariance)**:
       温度スケーリング $\tilde{p}_i = \text{Softmax}(z_i / \tau)$ はロジットの単調変換であるため、
       各サンプルの $\arg\max$ は一切変化せず、元のモデル精度を 100% 保持したまま確率較正度のみを改善します。
    3. **期待較正誤差 (ECE) < 0.10 の達成**:
       ホールドアウト検証セット上の負の対数尤度 (Masked NLL) を最小化する温度 $\tau^*$ を同定することで、
       信頼できる確率的決定(確信度ゲーティング)を実現します。
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    flowchart LR
        subgraph Data ["1. ホールドアウト検証データ & ロジット収集"]
            M["学習済みモデル (SFT / RLCD)"] --> FWD["推論実行 (torch.no_grad)"]
            V["検証データセット (Validation Set)"] --> FWD
            FWD --> Cache["LogitCache<br>(未スケーリング生ロジット, op_mask, labels, タイプ, 候補数)"]
        end

        subgraph Split ["2. バケット別スライシング"]
            Cache --> C["Choice: '2', '3-5', '6-10', '11+'"]
            Cache --> S["Score: '2-5', '6-10'"]
            Cache --> N["Noul: 真偽判定"]
        end

        subgraph Optimize ["3. 1変数スカラー最適化 (Bounded Brent 法)"]
            C & S & N --> Opt["scipy.optimize.minimize_scalar<br>tau* = argmin NLL(tau)"]
            Opt --> Fallback["スパースバケット安全フォールバック<br>(サンプル数 < 閾値 時は default_temp)"]
        end

        subgraph Evaluation ["4. 成果物永続化 (Rust 契約互換)"]
            Fallback --> CJ["calibration.json (Rust local-jev-core 契約)"]
            Fallback --> REP["calibration_metrics.json / summary.md / config.yaml"]
        end

        Data --> Split --> Optimize --> Evaluation
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. ライブラリおよびモジュールの読み込み
    キャリブレーションエンジン (`train/calibration/`)、モデル定義 (`train/models/`)、
    および成果物契約スキーマ (`train/contract.py`) から必要なコンポーネントをインポートします。
    """)


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    # プロジェクトルート (train/) を検索パスに追加
    notebook_dir = Path.cwd()
    if notebook_dir.name == "notebooks":
        train_root = notebook_dir.parent
    else:
        train_root = (
            notebook_dir / "train"
            if (notebook_dir / "train").exists()
            else notebook_dir
        )

    if str(train_root) not in sys.path:
        sys.path.insert(0, str(train_root))

    import torch

    from calibration.config import (
        CalibrationRunConfig,
    )
    from calibration.optimizer import (
        LogitCache,
        TemperatureOptimizer,
    )
    from data.schema import QuestionType

    return (
        CalibrationRunConfig,
        LogitCache,
        Path,
        QuestionType,
        TemperatureOptimizer,
        torch,
        train_root,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. チェックポイントの探索とハイパーパラメータ UI
    保存済みの SFT / RLCD チェックポイントを自動走査します。
    探索範囲、最小サンプル数閾値、検証サンプル数上限を対話的に調整可能です。
    """)


@app.cell(hide_code=True)
def _(mo, train_root):
    # 保存済みチェックポイントの自動探索
    runs_dir = train_root / "runs"
    checkpoint_candidates: dict[str, str] = {
        "✨ 合成シミュレーションデータ (高速デモモード)": "synthetic",
    }

    if runs_dir.exists():
        for ckpt in sorted(runs_dir.glob("*/*/best_checkpoint")):
            label = f"📁 {ckpt.parent.name} ({ckpt.name})"
            checkpoint_candidates[label] = str(ckpt)
        for ckpt in sorted(runs_dir.glob("*/interactive_run/best_checkpoint")):
            label = f"📁 {ckpt.parent.name} (interactive)"
            checkpoint_candidates[label] = str(ckpt)

    ckpt_dropdown = mo.ui.dropdown(
        options=checkpoint_candidates,
        value=next(iter(checkpoint_candidates.keys())),
        label="評価対象チェックポイント",
    )

    tau_range_slider = mo.ui.range_slider(
        start=0.05,
        stop=5.0,
        step=0.05,
        value=[0.05, 5.0],
        label="温度探索範囲 [tau_min, tau_max]",
    )

    min_samples_input = mo.ui.number(
        start=1,
        stop=100,
        step=5,
        value=15,
        label="バケット別最小サンプル数 (不足時はフォールバック)",
    )

    max_samples_input = mo.ui.number(
        start=0,
        stop=5000,
        step=50,
        value=0,
        label="最大検証サンプル数 (0=全件)",
    )

    load_config_input = mo.ui.text(
        label="過去の設定 YAML から読み込み (任意)",
        placeholder="runs/calibration/calib_.../calibration_run_config.yaml",
    )

    run_button = mo.ui.run_button(label="🚀 温度較正を実行する")

    ui_panel = mo.vstack(
        [
            mo.md("### ⚙️ キャリブレーション探索パラメータ"),
            ckpt_dropdown,
            tau_range_slider,
            mo.hstack([min_samples_input, max_samples_input]),
            load_config_input,
            run_button,
        ]
    )
    return (
        ckpt_dropdown,
        load_config_input,
        max_samples_input,
        min_samples_input,
        run_button,
        tau_range_slider,
        ui_panel,
    )


@app.cell
def _(ui_panel):
    ui_panel


@app.cell(hide_code=True)
def _(mo, run_button):
    # 実行ボタンが押下されるまで後続の処理を停止する
    mo.stop(
        not run_button.value,
        mo.md("💡 **「温度較正を実行する」ボタンを押すと最適化が開始されます。**"),
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ロジット収集と最適化の実行
    検証データまたはシミュレーションデータを構築し、バケット別温度探索を実行します。
    """)


@app.cell(hide_code=True)
def _(
    CalibrationRunConfig,
    LogitCache,
    Path,
    QuestionType,
    TemperatureOptimizer,
    ckpt_dropdown,
    load_config_input,
    max_samples_input,
    min_samples_input,
    mo,
    tau_range_slider,
    torch,
    train_root,
):
    # 設定の復元または UI 入力からの構築
    cfg_path_str = load_config_input.value.strip()
    if cfg_path_str and Path(cfg_path_str).is_file():
        active_config = CalibrationRunConfig.load(cfg_path_str)
        mo.output.append(
            mo.md(
                f"ℹ️ 設定ファイル `{cfg_path_str}` からハイパーパラメータを読み込みました。"
            )
        )
    else:
        selected_ckpt = ckpt_dropdown.value
        active_config = CalibrationRunConfig(
            checkpoint_path=selected_ckpt if selected_ckpt != "synthetic" else "",
            tau_min=float(tau_range_slider.value[0]),
            tau_max=float(tau_range_slider.value[1]),
            min_samples_per_bucket=int(min_samples_input.value),
            max_val_samples=int(max_samples_input.value),
            output_dir=str(train_root / "runs" / "calibration"),
        )

    # ロジットキャッシュの取得 (実モデルまたは合成シミュレーションデータ)
    if active_config.checkpoint_path and Path(active_config.checkpoint_path).exists():
        mo.output.append(
            mo.md(
                f"🔍 チェックポイント `{active_config.checkpoint_path}` をロードして推論中..."
            )
        )
        # 実際のチェックポイントからのロード
        import json

        from data.builders import UnifiedDatasetBuilder
        from models.decision_head import JevDecisionModel
        from training.trainer import (
            SFTDataset,
            prepare_backbone_and_tokenizer,
            sft_collate_fn,
        )

        ckpt_dir = Path(active_config.checkpoint_path)
        base_model_id = "answerdotai/ModernBERT-base"
        sft_cfg_path = ckpt_dir.parent / "config.json"
        if sft_cfg_path.exists():
            try:
                sft_data = json.loads(sft_cfg_path.read_text(encoding="utf-8"))
                base_model_id = sft_data.get("model_name_or_path", base_model_id)
            except (json.JSONDecodeError, OSError):
                pass

        backbone, tokenizer, op_token_id = prepare_backbone_and_tokenizer(base_model_id)

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        model = JevDecisionModel(backbone=backbone)
        model_weights = torch.load(
            str(ckpt_dir / "model.pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(model_weights)
        model.to(device)

        builder = UnifiedDatasetBuilder(seed=active_config.seed)
        val_samples = list(
            builder.stream_samples(
                dataset_names=["banking77", "mnli_choice", "sst5", "mnli_noul"],
                split="validation",
                max_samples_per_dataset=active_config.max_val_samples or None,
            )
        )
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        val_dataset = SFTDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            op_token_id=op_token_id,
            max_length=512,
            is_train=False,
            base_seed=active_config.seed,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=active_config.batch_size,
            shuffle=False,
            collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
        )

        optimizer = TemperatureOptimizer(config=active_config)
        logit_cache = optimizer.collect_logits(
            model=model,
            dataloader=val_loader,
            device=device,
            max_samples=active_config.max_val_samples,
        )
    else:
        # 合成シミュレーションデータ (過信ロジットの生成)
        torch.manual_seed(active_config.seed)
        n_samples = 300
        max_options = 12

        synth_logits = torch.randn(n_samples, max_options) * 0.8
        synth_labels = torch.zeros(n_samples, dtype=torch.long)
        synth_op_mask = torch.zeros(n_samples, max_options, dtype=torch.bool)
        synth_types: list[str] = []
        synth_counts: list[int] = []

        for i in range(n_samples):
            # プリミティブの割り当て
            if i < 150:
                qtype = QuestionType.CHOICE.value
                _k = 2 if i < 40 else (4 if i < 80 else (8 if i < 120 else 12))
            elif i < 240:
                qtype = QuestionType.SCORE.value
                _k = 5 if i < 195 else 7
            else:
                qtype = QuestionType.NOUL.value
                _k = 2

            synth_types.append(qtype)
            synth_counts.append(_k)
            synth_op_mask[i, :_k] = True

            lbl = int(torch.randint(0, _k, (1,)).item())
            synth_labels[i] = lbl

            # 70% の確率で過剰な自信 (ロジット +4.5) を注入
            if i % 10 < 7:
                synth_logits[i, lbl] += 4.5
            else:
                wrong_idx = (lbl + 1) % _k
                synth_logits[i, wrong_idx] += 4.5

        logit_cache = LogitCache(
            logits=synth_logits,
            op_mask=synth_op_mask,
            labels=synth_labels,
            question_types=synth_types,
            candidate_counts=synth_counts,
        )
        optimizer = TemperatureOptimizer(config=active_config)

    # 全バケットの較正実行
    calib_contract, metrics_summary = optimizer.calibrate(logit_cache)

    # 成果物の保存
    from datetime import UTC, datetime

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_save_dir = Path(active_config.output_dir) / f"calib_{ts}"
    saved_path = optimizer.save_run_artifacts(
        output_dir=run_save_dir,
        calib_config=calib_contract,
        metrics_summary=metrics_summary,
    )
    return calib_contract, metrics_summary, saved_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 較正性能の改善結果 (Evaluation Results)
    事前(スケーリングなし: $\tau = 1.0$)と事後(最適温度: $\tau^*$)の
    期待較正誤差 (ECE) および負の対数尤度 (NLL) を比較します。
    """)


@app.cell(hide_code=True)
def _(calib_contract, metrics_summary, mo, saved_path):
    pre_ece = metrics_summary["pre_calibration"]["ece"]
    post_ece = metrics_summary["post_calibration"]["ece"]
    pre_nll = metrics_summary["pre_calibration"]["nll"]
    post_nll = metrics_summary["post_calibration"]["nll"]
    ece_met = metrics_summary["target_ece_met"]

    status_badge = "✅ **達成 (ECE < 0.10)**" if ece_met else "⚠️ **未達 (要確認)**"

    summary_card = mo.stat(
        value=f"{post_ece:.4f}",
        caption=f"事前 ECE: {pre_ece:.4f} (改善幅: {pre_ece - post_ece:+.4f})",
        bordered=True,
    )
    nll_card = mo.stat(
        value=f"{post_nll:.4f}",
        caption=f"事前 NLL: {pre_nll:.4f} (改善幅: {post_nll - pre_nll:+.4f})",
        bordered=True,
    )

    # テーブル用 Markdown
    rows = []
    for _b in metrics_summary["buckets"].values():
        fb_str = "⚠️ フォールバック" if _b["is_fallback"] else "最適化完了"
        rows.append(
            f"| `{_b['question_type']}` | `{_b['bucket']}` | {_b['samples']} | **{_b['temperature']:.4f}** | "
            f"{_b['pre_calibration']['ece']:.4f} | {_b['post_calibration']['ece']:.4f} | "
            f"{_b['ece_delta']:+.4f} | {_b['pre_calibration']['accuracy']:.2%} | {fb_str} |"
        )
    table_md = "\n".join(
        [
            "| 質問タイプ | バケット | サンプル数 | 最適温度 $\\tau^*$ | 事前 ECE | 事後 ECE | ECE 改善幅 | 精度 (不変) | 状態 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            *rows,
        ]
    )

    mo.vstack(
        [
            mo.hstack([summary_card, nll_card]),
            mo.md(f"### 🎯 目標基準判定: {status_badge}"),
            mo.md("### 📋 バケット別詳細メトリクス"),
            mo.md(table_md),
            mo.md("### 📦 出力された Rust 契約設定 (`calibration.json`)"),
            mo.md(f"保存先: `{saved_path / 'calibration.json'}`"),
            mo.md(f"```json\n{calib_contract.to_json()}\n```"),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 信頼性ダイアグラム (Reliability Diagram)
    各バケットにおける予測確信度と正解率のギャップ(較正誤差)を確認します。
    対角線に近づくほど、モデルが自身の確からしさを正確に認識できていることを示します。
    """)


@app.cell(hide_code=True)
def _(metrics_summary, mo):
    # バケット選択用の簡易表示
    views = []
    for _k, _b in metrics_summary["buckets"].items():
        if _b["samples"] == 0:
            continue
        diagram_pre = _b["pre_calibration"]["reliability_diagram"]
        diagram_post = _b["post_calibration"]["reliability_diagram"]

        diag_rows = []
        for d_pre, d_post in zip(diagram_pre, diagram_post, strict=False):
            if d_pre["count"] == 0 and d_post["count"] == 0:
                continue
            diag_rows.append(
                f"| [{d_pre['bin_lower']:.2f}, {d_pre['bin_upper']:.2f}] | {d_pre['count']} | "
                f"{d_pre['confidence']:.3f} | {d_pre['accuracy']:.3f} | {d_pre['gap']:.3f} | "
                f"{d_post['confidence']:.3f} | {d_post['accuracy']:.3f} | **{d_post['gap']:.3f}** |"
            )

        if diag_rows:
            tbl = "\n".join(
                [
                    f"#### 🏷️ バケット: `{_k}` (サンプル数: {_b['samples']})",
                    "| 確信度区間 | 件数 | 事前 Conf | 事前 Acc | 事前 Gap | 事後 Conf | 事後 Acc | 事後 Gap |",
                    "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    *diag_rows,
                ]
            )
            views.append(mo.md(tbl))

    mo.vstack(views) if views else mo.md("有効なダイアグラムデータがありません。")


if __name__ == "__main__":
    app.run()
