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
    # 🚀 Local-Jev: SFT (教師あり指示学習) ノートブック

    本ノートブックは、TypeSafe AI アーキテクチャに準拠した非自己回帰型モデル「Jev」の
    SFT 学習ループを対話的に実行・検証・管理するための環境です。

    ### 📌 アーキテクチャとデータフロー

    文章生成を一切行わず、入力された `State + Instructions + Criteria` から、
    各候補アンカー `[OP]` の隠れベクトルを抽出し、動的スキーマ決定ヘッドによって各候補のロジットを単一フォワードパスで算出します。
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    flowchart LR
        subgraph Input ["入力系列"]
            S["State (文脈)"]
            I["Instructions (指示)"]
            C["Criteria ([OP] 候補群)"]
        end

        subgraph Model ["Jev アーキテクチャ"]
            BB["ModernBERT / mmBERT<br>双方向エンコーダ"]
            G["OptionGatherLayer<br>[OP] 埋め込み抽出"]
            H["DecisionHead<br>2層 MLP 射影"]
        end

        subgraph Output ["型安全決定"]
            L["Masked Logits<br>(op_mask で無効候補遮断)"]
            D["確信度 & Softmax 確率"]
        end

        Input --> BB --> G --> H --> L --> D
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. ライブラリおよびモジュールの読み込み
    学習エンジン (`train/training/`) およびデータパイプライン (`train/data/`) から
    設定、損失関数、評価指標、Trainer をインポートします。
    """)


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    # プロジェクトルート (train/) をモジュール検索パスに追加
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

    from models.backbone import DEFAULT_MMBERT_MODEL_ID, DEFAULT_MODERNBERT_MODEL_ID
    from training.config import SFTConfig
    from training.trainer import SFTTrainer

    return (
        DEFAULT_MMBERT_MODEL_ID,
        DEFAULT_MODERNBERT_MODEL_ID,
        Path,
        SFTConfig,
        SFTTrainer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 対話的ハイパーパラメータ設定 UI

    モデル構成、データセット、最適化パラメータ、および再現性設定を調整します。
    """)


@app.cell(hide_code=True)
def _(DEFAULT_MMBERT_MODEL_ID, DEFAULT_MODERNBERT_MODEL_ID, mo):
    # モデル設定
    model_selector = mo.ui.dropdown(
        options={
            "ModernBERT-base (英語標準)": DEFAULT_MODERNBERT_MODEL_ID,
            "mmBERT-base (多言語標準)": DEFAULT_MMBERT_MODEL_ID,
        },
        value="ModernBERT-base (英語標準)",
        label="バックボーンモデル",
    )

    # データセット選択
    dataset_options = [
        "banking77",
        "clinc150",
        "mnli_choice",
        "mnli_noul",
        "sst5",
    ]
    dataset_multiselect = mo.ui.multiselect(
        options=dataset_options,
        value=dataset_options,
        label="学習対象コーパス",
    )

    negative_ratio_slider = mo.ui.slider(
        start=0.0,
        stop=0.30,
        step=0.05,
        value=0.15,
        label="合成ネガティブ比率 (該当なし混入)",
    )

    max_samples_input = mo.ui.number(
        start=10,
        stop=10000,
        step=50,
        value=100,
        label="最大サンプル数/データセット (高速検証用, 0=全件)",
    )

    # 最適化設定
    batch_size_slider = mo.ui.slider(
        start=2,
        stop=64,
        step=2,
        value=8,
        label="バッチサイズ",
    )

    epochs_slider = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=2,
        label="エポック数",
    )

    lr_backbone_dropdown = mo.ui.dropdown(
        options=["1e-5", "2e-5", "3e-5", "5e-5"],
        value="2e-5",
        label="バックボーン学習率",
    )

    lr_head_dropdown = mo.ui.dropdown(
        options=["5e-5", "1e-4", "2e-4", "3e-4"],
        value="1e-4",
        label="デシジョンヘッド学習率",
    )

    mixed_precision_dropdown = mo.ui.dropdown(
        options=["no", "fp16", "bf16"],
        value="no",
        label="混合精度 (Mixed Precision)",
    )

    # 再現性設定
    seed_input = mo.ui.number(
        start=1,
        stop=999999,
        step=1,
        value=42,
        label="乱数シード",
    )

    output_dir_input = mo.ui.text(
        value="runs/sft",
        label="成果物出力ディレクトリ",
    )

    load_config_path_input = mo.ui.text(
        value="",
        placeholder="例: runs/sft/sft_20260922_010000_ModernBERT-base/config.yaml",
        label="過去の設定ファイルパスから読込 (任意)",
    )
    return (
        batch_size_slider,
        dataset_multiselect,
        epochs_slider,
        load_config_path_input,
        lr_backbone_dropdown,
        lr_head_dropdown,
        max_samples_input,
        mixed_precision_dropdown,
        model_selector,
        negative_ratio_slider,
        output_dir_input,
        seed_input,
    )


@app.cell(hide_code=True)
def _(
    batch_size_slider,
    dataset_multiselect,
    epochs_slider,
    load_config_path_input,
    lr_backbone_dropdown,
    lr_head_dropdown,
    max_samples_input,
    mixed_precision_dropdown,
    mo,
    model_selector,
    negative_ratio_slider,
    output_dir_input,
    seed_input,
):
    ui_controls = mo.vstack(
        [
            mo.md("### ⚙️ パラメータ調整パネル"),
            mo.hstack([model_selector, mixed_precision_dropdown]),
            mo.hstack([dataset_multiselect, negative_ratio_slider, max_samples_input]),
            mo.hstack(
                [
                    batch_size_slider,
                    epochs_slider,
                    lr_backbone_dropdown,
                    lr_head_dropdown,
                ]
            ),
            mo.hstack([seed_input, output_dir_input]),
            mo.md("---"),
            mo.md("### 📂 設定ファイルのインポート"),
            load_config_path_input,
        ]
    )
    ui_controls


@app.cell(hide_code=True)
def _(
    Path,
    SFTConfig,
    batch_size_slider,
    dataset_multiselect,
    epochs_slider,
    load_config_path_input,
    lr_backbone_dropdown,
    lr_head_dropdown,
    max_samples_input,
    mixed_precision_dropdown,
    mo,
    model_selector,
    negative_ratio_slider,
    output_dir_input,
    seed_input,
):
    # 外部設定ファイルが指定されており存在する場合はそちらを優先復元
    config_file_str = load_config_path_input.value.strip()
    if config_file_str and Path(config_file_str).is_file():
        active_config = SFTConfig.from_yaml(config_file_str)
        config_source_msg = (
            f"✅ 指定された外部設定 `{config_file_str}` から読み込みました。"
        )
    else:
        sample_limit = (
            int(max_samples_input.value) if int(max_samples_input.value) > 0 else None
        )
        active_config = SFTConfig(
            model_name_or_path=model_selector.value,
            dataset_names=list(dataset_multiselect.value),
            negative_ratio=float(negative_ratio_slider.value),
            max_samples_per_dataset=sample_limit,
            batch_size=int(batch_size_slider.value),
            num_epochs=int(epochs_slider.value),
            learning_rate_backbone=float(lr_backbone_dropdown.value),
            learning_rate_head=float(lr_head_dropdown.value),
            mixed_precision=mixed_precision_dropdown.value,
            seed=int(seed_input.value),
            output_dir=output_dir_input.value.strip() or "runs/sft",
        )
        config_source_msg = "🎛️ UI スライダー・セレクタの入力値から設定を生成しました。"

    preview_md = mo.md(f"""
    #### 📋 現在のアクティブ設定プレビュー
    {config_source_msg}

    ```yaml
    {active_config.to_yaml()}
    ```
    """)
    preview_md
    return (active_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 学習実行トリガー

    > [!IMPORTANT]
    > パラメータの変更による意図しない自動再学習を防ぐため、実行ガード (`mo.stop`) を設置しています。
    > 準備ができたら以下の「学習開始」ボタンをクリックしてください。
    """)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(
        label="🚀 SFT 学習開始 (Start Training)",
    )
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(SFTTrainer, active_config, mo, run_button):
    # 実行ガード: ボタンが押されるまでこれ以降のセルを実行しない
    mo.stop(
        not run_button.value,
        mo.md(
            "💡 上の「**🚀 SFT 学習開始**」ボタンを押すと、SFT 学習ループが開始されます。"
        ),
    )

    with mo.status.spinner(title="SFT 学習を実行中...") as _spinner:
        _spinner.update("Trainer を初期化中...")
        trainer = SFTTrainer(config=active_config)

        def _on_progress(event: dict[str, object]) -> None:
            stage = str(event.get("stage", ""))
            if stage == "train":
                ep = event.get("epoch")
                st = event.get("step")
                tot = event.get("total_steps")
                ls = event.get("loss")
                ac = event.get("accuracy")
                _spinner.update(
                    f"Epoch {ep} [Step {st}/{tot}] Loss: {ls:.4f}, Acc: {ac:.4f}"
                )
            elif stage == "epoch_end":
                ep = event.get("epoch")
                vm = event.get("val_metrics", {})
                if isinstance(vm, dict):
                    vacc = vm.get("accuracy", 0.0)
                    vloss = vm.get("loss", 0.0)
                    _spinner.update(
                        f"Epoch {ep} 完了 - Val Acc: {vacc:.4f}, Val Loss: {vloss:.4f}"
                    )

        training_summary = trainer.train(progress_callback=_on_progress)
        _spinner.update("学習が正常に完了しました。")

    result_banner = mo.md(f"""
    ### 🎉 SFT 学習完了！
    - **成果物保存先**: `{training_summary["run_dir"]}`
    - **初期ゼロショット精度**: `{training_summary["initial_accuracy"]:.4f}`
    - **最良検証精度**: `{training_summary["best_accuracy"]:.4f}` (Epoch {training_summary["best_epoch"]})
    """)
    result_banner
    return (training_summary,)


@app.cell(hide_code=True)
def _(mo, run_button):
    mo.stop(not run_button.value)

    mo.md(r"""
    ## 4. 学習結果とメトリクス履歴の可視化
    エポックごとの損失および各プリミティブ・候補数バケット別の精度サマリーを表示します。
    """)


@app.cell(hide_code=True)
def _(mo, run_button, training_summary):
    mo.stop(not run_button.value)

    history = training_summary.get("metrics_history", [])

    rows = []
    for item in history:
        ep = item.get("epoch")
        if ep == 0:
            vm = item.get("metrics", {})
            rows.append(
                {
                    "Epoch": "0 (Zero-Shot)",
                    "Train Loss": "-",
                    "Val Loss": vm.get("loss", "-"),
                    "Val Acc": vm.get("accuracy", "-"),
                    "Choice Acc": vm.get("choice_accuracy", "-"),
                    "Score Acc": vm.get("score_accuracy", "-"),
                    "Score MAE": vm.get("score_mae", "-"),
                    "Noul Acc": vm.get("noul_accuracy", "-"),
                    "Neg F1": vm.get("negative_f1", "-"),
                }
            )
        else:
            tm = item.get("train", {})
            vm = item.get("val", {})
            rows.append(
                {
                    "Epoch": str(ep),
                    "Train Loss": tm.get("loss", "-"),
                    "Val Loss": vm.get("loss", "-"),
                    "Val Acc": vm.get("accuracy", "-"),
                    "Choice Acc": vm.get("choice_accuracy", "-"),
                    "Score Acc": vm.get("score_accuracy", "-"),
                    "Score MAE": vm.get("score_mae", "-"),
                    "Noul Acc": vm.get("noul_accuracy", "-"),
                    "Neg F1": vm.get("negative_f1", "-"),
                }
            )

    table_widget = mo.ui.table(data=rows, label="エポック別検証結果")
    table_widget


@app.cell(hide_code=True)
def _(Path, mo, run_button, training_summary):
    mo.stop(not run_button.value)

    run_path = Path(str(training_summary["run_dir"]))
    files = sorted([f.name for f in run_path.iterdir() if f.is_file()])
    dirs = sorted([d.name for d in run_path.iterdir() if d.is_dir()])

    file_list_str = "\n".join([f"- `{f}`" for f in files])
    dir_list_str = "\n".join([f"- `{d}/`" for d in dirs])

    saved_artifacts_md = mo.md(f"""
    ### 📁 出力成果物ファイル一覧
    再現性を担保するため、以下の構成ファイルおよびチェックポイントが生成されました：

    **ファイル**:
    {file_list_str}

    **ディレクトリ**:
    {dir_list_str}
    """)
    saved_artifacts_md


if __name__ == "__main__":
    app.run()
