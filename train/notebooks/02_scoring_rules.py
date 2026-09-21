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
    # 🎯 Local-Jev: 厳密適格スコアリング規則 (Strictly Proper Scoring Rules) 評価ノートブック

    本ノートブックは、TypeSafe AI アーキテクチャに準拠した決定モデル「Jev」の
    厳密適格スコアリング規則の実装 における数理挙動を対話的に評価・可視化し、
    そのハイパーパラメータ設定と評価メトリクスを完全な再現性をもって永続化するための環境です。

    ### 📌 なぜ厳密適格スコアリング規則(PSR)が必要なのか？
    通常のクロスエントロピーのみで学習されたモデルは、往々にして「過剰な自信(Overconfidence)」に陥ります。
    厳密適格スコアリング規則は、**「モデルが真の信念(事後確率分布)をそのまま申告したときにのみ期待報酬が最大化される」** という数学的性質を持ち、過信を抑制して確信度と実際の正解率を一致させる較正(Calibration)を実現します。
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    flowchart LR
        subgraph Forward ["モデル予測"]
            Z["Logits z"] --> T["温度 tau 適用 (z / tau)"]
            T --> SM["Softmax (op_mask 完全遮断)"]
            SM --> P["正規化確率分布 p"]
        end

        subgraph PSR ["厳密適格スコアリング規則"]
            P --> S_log["有界対数スコア (S_log)<br>ln(max(p_y, eps)) 正規化"]
            P --> S_sph["球面スコア (S_sph)<br>p_y / ||p||_2 有界アンカー"]
            P --> S_rps["順位確率スコア (S_rps)<br>Score型順序尺度のCDF累積誤差"]
        end

        subgraph Composite ["動的ルーティング"]
            S_log & S_sph --> R1["Choice / Noul: alpha * S_log + (1-alpha) * S_sph"]
            S_rps & S_log --> R2["Score: beta * S_rps + (1-beta) * S_log"]
        end

        subgraph Persistence ["再現性保存 (runs/scoring/eval_.../)"]
            CFG["config.yaml / config.json"]
            RES["score_metrics.json / summary.md"]
            META["run_metadata.json"]
        end

        Forward --> PSR --> Composite --> Persistence
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. ライブラリおよびモジュールの読み込み
    学習エンジン (`train/training/`)、モデルアーキテクチャ (`train/models/`)、およびデータパイプライン (`train/data/`) から
    スコアリング数理、設定管理、および評価モジュールをインポートします。
    """)


@app.cell(hide_code=True)
def _():
    import json
    import platform
    import subprocess
    import sys
    from datetime import datetime
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

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from data.builders import UnifiedDatasetBuilder
    from models.backbone import prepare_backbone_and_tokenizer
    from models.decision_head import JevDecisionModel
    from training.scoring import ProperScoringEvaluator
    from training.scoring_config import ScoringConfig
    from training.trainer import SFTDataset, sft_collate_fn

    # 既存 SFT チェックポイントの探索
    sft_runs_dir = train_root / "runs" / "sft"
    available_checkpoints: dict[str, str] = {}
    if sft_runs_dir.exists():
        for run_dir in sorted(sft_runs_dir.iterdir(), reverse=True):
            best_ckpt = run_dir / "best_checkpoint"
            if best_ckpt.is_dir() and (best_ckpt / "model.pt").exists():
                available_checkpoints[f"{run_dir.name} (best)"] = str(best_ckpt)
            latest_ckpt = run_dir / "latest_checkpoint"
            if latest_ckpt.is_dir() and (latest_ckpt / "model.pt").exists():
                available_checkpoints[f"{run_dir.name} (latest)"] = str(latest_ckpt)

    if not available_checkpoints:
        available_checkpoints["(チェックポイントが見つかりません)"] = ""
    return (
        AutoTokenizer,
        DataLoader,
        JevDecisionModel,
        Path,
        ProperScoringEvaluator,
        SFTDataset,
        ScoringConfig,
        UnifiedDatasetBuilder,
        available_checkpoints,
        datetime,
        json,
        platform,
        prepare_backbone_and_tokenizer,
        sft_collate_fn,
        subprocess,
        sys,
        torch,
        train_root,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 対話的パラメータ調整 UI

    スコアリング規則の配分比率 ($\alpha, \beta$)、クリッピング下限 ($\epsilon$)、温度 ($\tau$)、
    および評価対象のチェックポイントを選択します。
    外部の `config.yaml` パスを指定することで、以前の設定をそのまま復元することも可能です。
    """)


@app.cell(hide_code=True)
def _(available_checkpoints: dict[str, str], mo):
    first_ckpt_key = next(iter(available_checkpoints.keys()))

    checkpoint_selector = mo.ui.dropdown(
        options=available_checkpoints,
        value=first_ckpt_key,
        label="評価対象チェックポイント",
    )

    alpha_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.5,
        label="alpha: Choice/Noul における S_log 比率 (1-alpha が S_sph 比率)",
    )

    beta_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.7,
        label="beta: Score における S_rps 比率 (1-beta が S_log 比率)",
    )

    eps_dropdown = mo.ui.dropdown(
        options=["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"],
        value="1e-6",
        label="eps: 対数スコアのクリッピング下限",
    )

    temperature_slider = mo.ui.slider(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=1.0,
        label="tau: ロジット温度パラメータ",
    )

    max_samples_input = mo.ui.number(
        start=10,
        stop=5000,
        step=50,
        value=100,
        label="最大評価サンプル数/データセット (0=全件)",
    )

    output_dir_input = mo.ui.text(
        value="runs/scoring",
        label="成果物保存先ベースディレクトリ",
    )

    load_config_path_input = mo.ui.text(
        value="",
        placeholder="例: runs/scoring/eval_20260922_120000/config.yaml",
        label="過去の設定 YAML から読み込み (任意)",
    )
    return (
        alpha_slider,
        beta_slider,
        checkpoint_selector,
        eps_dropdown,
        load_config_path_input,
        max_samples_input,
        output_dir_input,
        temperature_slider,
    )


@app.cell(hide_code=True)
def _(
    alpha_slider,
    beta_slider,
    checkpoint_selector,
    eps_dropdown,
    load_config_path_input,
    max_samples_input,
    mo,
    output_dir_input,
    temperature_slider,
):
    ui_panel = mo.vstack(
        [
            mo.md("### ⚙️ パラメータ設定"),
            checkpoint_selector,
            mo.hstack([alpha_slider, beta_slider]),
            mo.hstack([eps_dropdown, temperature_slider]),
            mo.hstack([max_samples_input, output_dir_input]),
            mo.md("---"),
            mo.md("### 📂 設定ファイルのインポート (双方向同期)"),
            load_config_path_input,
        ]
    )
    ui_panel


@app.cell(hide_code=True)
def _(
    Path,
    ScoringConfig,
    alpha_slider,
    beta_slider,
    checkpoint_selector,
    eps_dropdown,
    load_config_path_input,
    mo,
    output_dir_input,
    temperature_slider,
):
    # 外部設定ファイルが指定されており実在する場合はそちらを復元
    config_file_str = load_config_path_input.value.strip()
    if config_file_str and Path(config_file_str).is_file():
        active_config = ScoringConfig.from_yaml(config_file_str)
        config_source_msg = (
            f"✅ 指定された外部設定 `{config_file_str}` から設定を復元しました。"
        )
    else:
        active_config = ScoringConfig(
            alpha=float(alpha_slider.value),
            beta=float(beta_slider.value),
            eps=float(eps_dropdown.value),
            temperature=float(temperature_slider.value),
            output_dir=output_dir_input.value.strip() or "runs/scoring",
        )
        config_source_msg = "🎛️ UI スライダー・セレクタの入力値から設定を生成しました。"

    selected_ckpt_path = checkpoint_selector.value

    config_preview_md = mo.md(f"""
    #### 📋 現在のアクティブ設定プレビュー
    {config_source_msg}
    - **選択チェックポイント**: `{selected_ckpt_path or "(未選択)"}`

    ```yaml
    {active_config.to_yaml()}
    ```
    """)
    config_preview_md
    return active_config, selected_ckpt_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. スコアリング評価実行トリガー

    > [!IMPORTANT]
    > パラメータ変更のたびに重い評価処理が走るのを防ぐため、実行ガード (`mo.stop`) を設置しています。
    > 以下のボタンをクリックして評価と永続化を開始してください。
    """)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(
        label="📊 スコアリング規則を評価 & 保存 (Run Evaluation)",
    )
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(
    AutoTokenizer,
    DataLoader,
    JevDecisionModel,
    Path,
    ProperScoringEvaluator,
    SFTDataset,
    UnifiedDatasetBuilder,
    active_config,
    datetime,
    json,
    max_samples_input,
    mo,
    platform,
    prepare_backbone_and_tokenizer,
    run_button,
    selected_ckpt_path,
    sft_collate_fn,
    subprocess,
    sys,
    torch,
    train_root,
):
    mo.stop(
        not run_button.value,
        mo.md(
            "💡 上の「**📊 スコアリング規則を評価 & 保存**」ボタンを押すと、チェックポイントを読み込んで評価を実行します。"
        ),
    )

    mo.stop(
        not selected_ckpt_path or not Path(selected_ckpt_path).exists(),
        mo.md(
            "❌ 有効なチェックポイントが選択されていません。先に SFT 学習を実行してください。"
        ),
    )

    _ckpt_dir = Path(selected_ckpt_path)
    _model_pt_path = _ckpt_dir / "model.pt"

    with mo.status.spinner(title="スコアリング評価を実行中...") as _spinner:
        _spinner.update("モデル重みとトークナイザーを読み込み中...")
        _device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        _state_dict = torch.load(
            _model_pt_path, map_location=_device, weights_only=True
        )
        # SFT 設定からモデル名を取得
        _sft_cfg_path = _ckpt_dir.parent / "config.json"
        _base_model_id = "answerdotai/ModernBERT-base"
        if _sft_cfg_path.is_file():
            try:
                _sft_data = json.loads(_sft_cfg_path.read_text(encoding="utf-8"))
                _base_model_id = _sft_data.get("model_name_or_path", _base_model_id)
            except (json.JSONDecodeError, OSError):
                pass

        _tokenizer_dir = _ckpt_dir / "tokenizer"
        if _tokenizer_dir.exists():
            _tokenizer = AutoTokenizer.from_pretrained(str(_tokenizer_dir))
            _op_token_id = int(_tokenizer.convert_tokens_to_ids("[OP]"))
            _backbone, _, _ = prepare_backbone_and_tokenizer(_base_model_id)
        else:
            _backbone, _tokenizer, _op_token_id = prepare_backbone_and_tokenizer(
                _base_model_id
            )

        _model = JevDecisionModel(
            backbone=_backbone,
        )
        _model.load_state_dict(_state_dict)
        _model.to(_device)
        _model.eval()

        _spinner.update("評価用検証データセットを構築中...")
        _limit = (
            int(max_samples_input.value) if int(max_samples_input.value) > 0 else None
        )
        _builder = UnifiedDatasetBuilder()
        _val_samples = list(
            _builder.stream_samples(
                dataset_names=[
                    "banking77",
                    "clinc150",
                    "mnli_choice",
                    "mnli_noul",
                    "sst5",
                ],
                split="validation",
                max_samples_per_dataset=_limit,
            )
        )

        _pad_id = _tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else 0
        _max_seq_len = 2048
        if _sft_cfg_path.is_file():
            try:
                _sft_data = json.loads(_sft_cfg_path.read_text(encoding="utf-8"))
                _max_seq_len = int(_sft_data.get("max_sequence_length", 2048))
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        _val_dataset = SFTDataset(
            samples=_val_samples,
            tokenizer=_tokenizer,
            op_token_id=_op_token_id,
            max_length=_max_seq_len,
            is_train=False,
            base_seed=42,
        )

        _val_loader = DataLoader(
            _val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda b: sft_collate_fn(b, pad_token_id=_pad_id),
        )

        _spinner.update("厳密適格スコアをバッチ集計中...")
        _evaluator = ProperScoringEvaluator(config=active_config)

        with torch.no_grad():
            for _batch in _val_loader:
                _input_ids = _batch["input_ids"].to(_device)
                _attention_mask = _batch["attention_mask"].to(_device)
                _op_indices = _batch["op_indices"].to(_device)
                _op_mask = _batch["op_mask"].to(_device)
                _labels = _batch["labels"].to(_device)
                _q_types = _batch["question_types"]

                _logits = _model(
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                    op_indices=_op_indices,
                )

                _evaluator.update(
                    logits=_logits,
                    labels=_labels,
                    op_mask=_op_mask,
                    question_types=_q_types,
                )

        eval_summary = _evaluator.compute()

        # 成果物ディレクトリの作成と完全永続化
        _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _out_base = train_root / active_config.output_dir
        eval_run_dir = _out_base / f"eval_{_timestamp}"
        eval_run_dir.mkdir(parents=True, exist_ok=True)

        # 1. config.yaml
        active_config.save_yaml(eval_run_dir / "config.yaml")
        # 2. config.json
        active_config.save_json(eval_run_dir / "config.json")
        # 3. score_metrics.json
        (eval_run_dir / "score_metrics.json").write_text(
            json.dumps(eval_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 4. run_metadata.json
        _git_commit = "unknown"
        try:
            _git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=train_root)
                .decode()
                .strip()
            )
        except (subprocess.SubprocessError, OSError):
            pass

        _meta = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_evaluated": str(selected_ckpt_path),
            "git_commit": _git_commit,
            "device": str(_device),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        }
        (eval_run_dir / "run_metadata.json").write_text(
            json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 5. summary.md
        _md_content = f"""# 厳密適格スコアリング評価サマリー

- **実行日時**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **評価対象チェックポイント**: `{selected_ckpt_path}`
- **総評価サンプル数**: {eval_summary["total_samples"]}
- **平均複合スコア (Composite)**: {eval_summary["mean_composite"]}
- **平均有界対数スコア (S_log)**: {eval_summary["mean_log"]}
- **平均球面スコア (S_sph)**: {eval_summary["mean_sph"]}
- **平均順位確率スコア (S_rps)**: {eval_summary["mean_rps"]}

## 質問タイプ別スコア
| タイプ | サンプル数 | 複合スコア | 対数スコア (S_log) | 球面/順位スコア |
|---|---|---|---|---|
"""
        for _t_name, _t_val in eval_summary.get("by_type", {}).items():
            _aux_score = _t_val.get("mean_rps", _t_val.get("mean_sph", "-"))
            _md_content += f"| {_t_name} | {_t_val['count']} | {_t_val['mean_composite']} | {_t_val['mean_log']} | {_aux_score} |\n"

        _md_content += """
## 確信度帯別スコア
| 確信度帯 | サンプル数 | 比率 | 複合スコア | 正解率 |
|---|---|---|---|---|
"""
        for _c_name, _c_val in eval_summary.get("by_confidence", {}).items():
            _md_content += f"| {_c_name} | {_c_val['count']} | {_c_val['ratio']:.1%} | {_c_val['mean_composite']} | {_c_val['accuracy']:.1%} |\n"

        (eval_run_dir / "summary.md").write_text(_md_content, encoding="utf-8")
        _spinner.update("評価と成果物の保存が完了しました。")
    return eval_run_dir, eval_summary


@app.cell(hide_code=True)
def _(eval_run_dir, eval_summary, mo, run_button):
    mo.stop(not run_button.value)

    banner_md = mo.md(f"""
    ### 🎉 スコアリング評価完了！
    - **成果物保存先**: `{eval_run_dir}`
    - **総評価サンプル数**: `{eval_summary["total_samples"]}`
    - **全体平均複合スコア**: `{eval_summary["mean_composite"]:.4f}`
    - **平均対数スコア (S_log)**: `{eval_summary["mean_log"]:.4f}` | **平均球面スコア (S_sph)**: `{eval_summary["mean_sph"]:.4f}` | **平均順位スコア (S_rps)**: `{eval_summary["mean_rps"]:.4f}`
    """)
    banner_md


@app.cell(hide_code=True)
def _(mo, run_button):
    mo.stop(not run_button.value)

    mo.md(r"""
    ## 4. プリミティブ別および確信度帯別の詳細集計
    各決定プリミティブ(Choice, Score, Noul)における各規則のスコア、および確信度帯別の較正性を表示します。
    """)


@app.cell(hide_code=True)
def _(eval_summary, mo, run_button):
    mo.stop(not run_button.value)

    _type_rows = []
    for _type_name, _stat in eval_summary.get("by_type", {}).items():
        _type_rows.append(
            {
                "質問プリミティブ": _type_name,
                "サンプル数": _stat.get("count", 0),
                "複合スコア (Composite)": _stat.get("mean_composite", 0.0),
                "有界対数スコア (S_log)": _stat.get("mean_log", 0.0),
                "球面スコア (S_sph)": _stat.get("mean_sph", "-"),
                "順位確率スコア (S_rps)": _stat.get("mean_rps", "-"),
            }
        )

    _conf_rows = []
    for _conf_tier, _conf_stat in eval_summary.get("by_confidence", {}).items():
        _conf_rows.append(
            {
                "確信度帯 (Confidence)": _conf_tier,
                "サンプル数": _conf_stat.get("count", 0),
                "構成比率": f"{_conf_stat.get('ratio', 0.0):.1%}",
                "平均複合スコア": _conf_stat.get("mean_composite", 0.0),
                "実績分類精度 (Acc)": f"{_conf_stat.get('accuracy', 0.0):.1%}",
            }
        )

    _type_table = mo.ui.table(data=_type_rows, label="決定プリミティブ別スコア一覧")
    _conf_table = mo.ui.table(data=_conf_rows, label="確信度帯別較正性サマリー")

    tables_view = mo.vstack(
        [
            mo.md("#### 決定プリミティブ別スコア一覧"),
            _type_table,
            mo.md("#### 確信度帯別較正性サマリー"),
            _conf_table,
        ]
    )
    tables_view


@app.cell(hide_code=True)
def _(eval_run_dir, mo, run_button):
    mo.stop(not run_button.value)

    saved_files = sorted([f.name for f in eval_run_dir.iterdir() if f.is_file()])
    file_items = "\n".join([f"- `{f}`" for f in saved_files])

    artifacts_md = mo.md(f"""
    ### 📁 出力成果物ファイル一覧
    再現性を担保するため、以下のファイル群が `{eval_run_dir.name}/` に保存されました：

    {file_items}

    > [!TIP]
    > ノートブック上部の「過去の設定 YAML から読み込み」に `{eval_run_dir}/config.yaml` を入力することで、
    > 今回と全く同一のハイパーパラメータ設定でいつでも再評価を行うことができます。
    """)
    artifacts_md


if __name__ == "__main__":
    app.run()
