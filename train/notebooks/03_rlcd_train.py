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
    # 🎯 Local-Jev: RLCD (較正強化学習) ポリシー更新ノートブック

    本ノートブックは、TypeSafe AI アーキテクチャに準拠した決定モデル「Jev」の
    **RLCD (Reinforcement Learning from Calibrated Decisions) によるポリシー最適化** を
    対話的にシミュレーション・検証し、ハイパーパラメータ設定と実行結果を YAML または JSON として永続化するための環境です。

    ### 📌 なぜ SFT の後に RLCD が必要なのか？
    1. **過剰な確信度 (Overconfidence) の是正**:
       SFT (クロスエントロピー損失) は正解確率を $1.0$ に近づけることのみを目的とするため、誤答時にも高い確信度を出力する傾向があります。
    2. **厳密適格スコア複合報酬 (Strictly Proper Scoring Rules)**:
       対数スコア ($S_{\log}$)、球面スコア ($S_{\text{sph}}$)、および順位確率スコア ($S_{\text{rps}}$) を報酬関数とすることで、モデルが真の事後確率を申告したときにのみ期待報酬を最大化させます。
    3. **非自己回帰型決定モデルのためのバンディット型 GRPO**:
       文章生成を行わない Jev の特性に合わせ、**「ロジット空間でのグループ摂動サンプリング」** と **「サンプル内相対アドバンテージによるクリップ付きサロゲート最適化」** を行います。
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    flowchart TD
        subgraph Forward ["フォワードパス & ロジット取得"]
            X["入力 (State + Instructions + Criteria)"] --> P["Policy モデル (pi_theta)"]
            X --> R["Frozen Reference モデル (pi_ref)"]
            P --> Z["決定ロジット z [batch, K]"]
            R --> Z_ref["参照ロジット z_ref [batch, K]"]
        end

        subgraph Sampling ["ロジット摂動サンプリング (G 系統)"]
            Z --> P1["z^(1) = z + eps_1"]
            Z --> P2["z^(2) = z + eps_2"]
            Z --> PG["z^(G) = z + eps_G"]
            P1 & P2 & PG --> SM["Softmax (op_mask 完全維持)"]
            SM --> Probs["摂動確率分布群 {p^(g)}"]
        end

        subgraph RewardAdvantage ["複合報酬 & 相対アドバンテージ"]
            Probs --> PSR["厳密適格スコア算出<br>(Choice/Noul: S_log, S_sph / Score: S_rps)"]
            PSR --> REW["報酬群 {R_g}"]
            REW --> ADV["サンプル内相対アドバンテージ<br>A_g = (R_g - mu) / (sigma + eps)"]
        end

        subgraph Update ["GRPO ポリシー更新"]
            ADV & Probs --> SURR["PPO クリップサロゲート損失 L_policy"]
            Z & Z_ref --> KL["マスク付き KL ペナルティ D_KL(p_theta || p_ref)"]
            Z --> ENT["過信抑制エントロピーボーナス H(p_theta)"]
            SURR & KL & ENT --> TOTAL["総合損失 = L_policy + beta_KL * L_KL - beta_ent * H"]
        end

        Forward --> Sampling --> RewardAdvantage --> Update
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. ライブラリおよびモジュールの読み込み
    学習エンジン (`train/training/`) およびデータスキーマ (`train/data/`) から、
    RLCD 設定、損失関数、サンプリング数理、および較正度評価モジュールをインポートします。
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
    from torch import nn

    from data.schema import QuestionType
    from training.loss import DEFAULT_MASK_VALUE
    from training.rlcd_config import RLCDConfig
    from training.rlcd_loss import (
        RLCDLoss,
        compute_entropy,
        compute_group_advantages,
        compute_masked_kl_divergence,
        sample_perturbed_logits,
    )
    from training.rlcd_trainer import (
        RLCDTrainer,
    )
    from training.scoring import (
        compute_composite_scores,
        get_normalized_probabilities,
    )
    from training.scoring_config import ScoringConfig

    return (
        DEFAULT_MASK_VALUE,
        Path,
        QuestionType,
        RLCDConfig,
        RLCDLoss,
        RLCDTrainer,
        ScoringConfig,
        compute_composite_scores,
        compute_entropy,
        compute_group_advantages,
        compute_masked_kl_divergence,
        get_normalized_probabilities,
        nn,
        sample_perturbed_logits,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. RLCD ハイパーパラメータ対話コントロール
    以下のスライダーと入力フォームを用いて、ロジット摂動サンプリング数、クリップ率、
    KL ペナルティ係数、学習率などを対話的に調整できます。
    設定は自動的に検証され、再現性確保のためにファイルへ保存できます。
    """)


@app.cell(hide_code=True)
def _(mo):
    # 対話的 UI ウィジェットの構築
    ui_num_generations = mo.ui.slider(
        start=2,
        stop=8,
        step=2,
        value=4,
        label="グループサイズ G (num_generations)",
    )

    ui_perturbation_std = mo.ui.slider(
        start=0.01,
        stop=0.40,
        step=0.01,
        value=0.10,
        label="ロジット摂動標準偏差 sigma (perturbation_std)",
    )

    ui_clip_range = mo.ui.slider(
        start=0.05,
        stop=0.40,
        step=0.05,
        value=0.20,
        label="サロゲートクリップ範囲 epsilon (clip_range)",
    )

    ui_kl_coeff = mo.ui.slider(
        start=0.01,
        stop=0.30,
        step=0.01,
        value=0.05,
        label="参照モデル KL ペナルティ係数 beta_KL (kl_coeff)",
    )

    ui_entropy_coeff = mo.ui.slider(
        start=0.0,
        stop=0.05,
        step=0.005,
        value=0.01,
        label="過信抑制エントロピー係数 beta_ent (entropy_coeff)",
    )

    ui_temperature = mo.ui.slider(
        start=0.5,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="サンプリング温度 tau (sampling_temperature)",
    )

    ui_learning_rate = mo.ui.dropdown(
        options={
            "1e-5 (微調整・安全重視)": 1e-5,
            "2e-5 (標準推奨値)": 2e-5,
            "5e-5 (積極更新)": 5e-5,
            "1e-4 (実験用)": 1e-4,
        },
        value="2e-5 (標準推奨値)",
        label="ポリシー更新学習率 (learning_rate)",
    )

    ui_output_dir = mo.ui.text(
        value="runs/rlcd/interactive_run",
        label="アーティファクト出力ディレクトリ",
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    ui_num_generations,
                    ui_perturbation_std,
                    ui_clip_range,
                    ui_temperature,
                ]
            ),
            mo.vstack(
                [
                    ui_kl_coeff,
                    ui_entropy_coeff,
                    ui_learning_rate,
                    ui_output_dir,
                ]
            ),
        ]
    )
    return (
        ui_clip_range,
        ui_entropy_coeff,
        ui_kl_coeff,
        ui_learning_rate,
        ui_num_generations,
        ui_output_dir,
        ui_perturbation_std,
        ui_temperature,
    )


@app.cell(hide_code=True)
def _(
    Path,
    RLCDConfig,
    ScoringConfig,
    mo,
    ui_clip_range,
    ui_entropy_coeff,
    ui_kl_coeff,
    ui_learning_rate,
    ui_num_generations,
    ui_output_dir,
    ui_perturbation_std,
    ui_temperature,
):
    # UI 設定に基づく RLCDConfig インスタンス化
    rlcd_config = RLCDConfig(
        num_generations=ui_num_generations.value,
        perturbation_std=ui_perturbation_std.value,
        clip_range=ui_clip_range.value,
        kl_coeff=ui_kl_coeff.value,
        entropy_coeff=ui_entropy_coeff.value,
        sampling_temperature=ui_temperature.value,
        learning_rate=ui_learning_rate.value,
        output_dir=ui_output_dir.value,
        scoring_config=ScoringConfig(),
    )

    # 出力ディレクトリへの自動保存 (YAML と JSON の両方を出力して再現性を担保)
    out_dir = Path(rlcd_config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = rlcd_config.save_yaml(out_dir / "rlcd_config.yaml")
    json_path = rlcd_config.save_json(out_dir / "rlcd_config.json")
    content_preview = rlcd_config.to_yaml()

    mo.md(f"""
    ### 💾 設定ファイルの自動保存と再現性メタデータ
    現在の設定が保存されました:
    - **`{yaml_path}`**
    - **`{json_path}`**

    ```yaml
    {content_preview}
    ```
    """)
    return out_dir, rlcd_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ロジット摂動とグループ相対アドバンテージの数理シミュレーション
    ここでは、3 つの決定プリミティブ (Choice 3択, Score 5段階, Noul 2択) に対するサンプルロジットを用いて、
    $G$ 系統のガウシアン摂動注入、厳密適格スコア複合報酬、およびサンプル内アドバンテージ算出の数理挙動を詳細に可視化します。
    """)


@app.cell
def _(
    DEFAULT_MASK_VALUE,
    QuestionType,
    compute_composite_scores,
    compute_group_advantages,
    get_normalized_probabilities,
    mo,
    rlcd_config,
    sample_perturbed_logits,
    torch,
):
    # シミュレーション用ダミーバッチデータ
    # サンプル 0: Choice (3択, 有効候補 [A, B, C], パディング 2)
    # サンプル 1: Score (5段階, 有効候補 5)
    # サンプル 2: Noul (2択, 有効候補 [True, False], パディング 3)
    sim_logits = torch.tensor(
        [
            [2.5, 1.2, 0.1, -10.0, -10.0],
            [0.2, 1.1, 2.8, 1.5, 0.4],
            [3.1, 0.4, -10.0, -10.0, -10.0],
        ]
    )
    sim_op_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )
    sim_labels = torch.tensor([0, 2, 0])
    sim_types = [QuestionType.CHOICE, QuestionType.SCORE, QuestionType.NOUL]

    # 1. ロジット摂動サンプリング
    sim_perturbed = sample_perturbed_logits(
        logits=sim_logits,
        op_mask=sim_op_mask,
        num_generations=rlcd_config.num_generations,
        perturbation_std=rlcd_config.perturbation_std,
        mask_value=DEFAULT_MASK_VALUE,
    )

    # 2. 摂動確率分布の導出
    b, g, k = sim_perturbed.shape
    sim_mask_gen = sim_op_mask.unsqueeze(1).expand(b, g, k)
    _sim_p_gen = get_normalized_probabilities(
        logits=sim_perturbed.reshape(-1, k),
        op_mask=sim_mask_gen.reshape(-1, k),
        temperature=rlcd_config.sampling_temperature,
    ).reshape(b, g, k)

    # 3. 報酬計算 (一括並列処理)
    flat_labels = sim_labels.unsqueeze(1).expand(b, g).reshape(-1)
    flat_types = []
    for qt in sim_types:
        flat_types.extend([qt] * g)

    sim_reward_dict = compute_composite_scores(
        logits=sim_perturbed.reshape(-1, k),
        labels=flat_labels,
        op_mask=sim_mask_gen.reshape(-1, k),
        question_types=flat_types,
        config=rlcd_config.scoring_config,
    )
    sim_rewards = sim_reward_dict["composite"].reshape(b, g)

    # 4. グループ相対アドバンテージ算出
    sim_adv, sim_mean_r, sim_std_r = compute_group_advantages(sim_rewards)

    # 結果テーブルの生成
    rows_md = []
    for i, q_t in enumerate(["Choice (3択)", "Score (5段階)", "Noul (2択)"]):
        r_list = [f"{r:.3f}" for r in sim_rewards[i].tolist()]
        a_list = [f"{a:.3f}" for a in sim_adv[i].tolist()]
        rows_md.append(
            f"| Sample {i}: {q_t} | {sim_labels[i].item()} | `[{', '.join(r_list)}]` | {sim_mean_r[i, 0].item():.3f} | {sim_std_r[i, 0].item():.3f} | `[{', '.join(a_list)}]` |"
        )

    table_md = "\n".join(rows_md)

    mo.md(rf"""
    ### 📊 グループ摂動サンプリングとサンプル内アドバンテージ算出結果
    | サンプル (プリミティブ) | 正解ラベル | G 系統の報酬 $R_g$ | 平均 $\mu_R$ | 標準偏差 $\sigma_R$ | 相対アドバンテージ $\hat{{A}}_g$ |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    {table_md}

    > [!NOTE]
    > **サンプル内正規化の重要性**:
    > タスクごとに報酬のスケール(Choice の理論値と Score の理論値)が異なりますが、
    > サンプル内 ($G$ 世代間) で標準化することで、どのタスクでも公正にアドバンテージが $[-2, +2]$ 付近に正規化されます。
    """)
    return sim_labels, sim_logits, sim_op_mask, sim_types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 損失関数と正則化ペナルティ (KL & エントロピー) の挙動
    ポリシーモデルと凍結参照モデル (SFT 初期値) の出力確率から、
    **マスク付き KL ダイバージェンス** と **過信抑制エントロピー** を計算し、
    総合損失 $\mathcal{L}_{\text{total}}$ を算出します。
    """)


@app.cell
def _(
    RLCDLoss,
    compute_entropy,
    compute_masked_kl_divergence,
    get_normalized_probabilities,
    mo,
    rlcd_config,
    sim_labels,
    sim_logits,
    sim_op_mask,
    sim_types,
    torch,
):
    # 参照モデルのロジット (SFT 固定)
    sim_ref_logits = sim_logits.clone() + 0.1 * torch.randn_like(sim_logits)

    loss_fn = RLCDLoss(config=rlcd_config)
    sim_logits.requires_grad_(True)

    total_loss, metrics = loss_fn(
        policy_logits=sim_logits,
        ref_logits=sim_ref_logits,
        labels=sim_labels,
        op_mask=sim_op_mask,
        question_types=sim_types,
    )

    # 詳細分解
    p_theta = get_normalized_probabilities(
        sim_logits, sim_op_mask, rlcd_config.sampling_temperature
    )
    p_ref = get_normalized_probabilities(
        sim_ref_logits, sim_op_mask, rlcd_config.sampling_temperature
    )
    _kl_vals = compute_masked_kl_divergence(p_theta, p_ref, sim_op_mask)
    _ent_vals = compute_entropy(p_theta, sim_op_mask)

    mo.md(f"""
    ### 🔬 損失項の内訳と勾配健全性
    - **総合損失 $\\mathcal{{L}}_{{\\text{{total}}}}$**: `{total_loss.item():.4f}`
    - **サロゲートポリシー損失 $\\mathcal{{L}}_{{\\text{{policy}}}}$**: `{metrics["loss_policy"].item():.4f}`
    - **マスク付き KL 損失 $\\mathcal{{L}}_{{\\text{{KL}}}}$**: `{metrics["loss_kl"].item():.4f}` (重み $\\beta_{{\\text{{KL}}}} = {rlcd_config.kl_coeff}$)
    - **エントロピーボーナス $\\mathcal{{L}}_{{\\text{{ent}}}}$**: `{metrics["loss_entropy"].item():.4f}` (重み $\\beta_{{\\text{{ent}}}} = {rlcd_config.entropy_coeff}$)
    - **平均報酬 Mean Reward**: `{metrics["mean_reward"].item():.4f}`
    - **平均正解確率 $p_\\theta(y)$**: `{metrics["p_target_mean"].item():.4f}`
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ミニ学習ステップのシミュレーションと較正度 (ECE) 推移
    最後に、軽量ダミーモデルを用いて RLCDTrainer を 2 エポック実行し、
    学習に伴う **平均報酬、総合損失、分類精度、および期待較正誤差 (ECE)** の推移を確認します。
    """)


@app.cell
def _(
    RLCDConfig,
    RLCDTrainer,
    ScoringConfig,
    mo,
    nn,
    out_dir,
    rlcd_config,
    torch,
):
    # テスト用軽量ダミーモデルの構築
    class MiniBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)

        def forward(self, input_ids, attention_mask=None):
            class Out:
                last_hidden_state = self.embed(input_ids)

            return Out()

    class MiniHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 1)

        def forward(self, gathered):
            return self.fc(gathered).squeeze(-1)

    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = MiniBackbone()
            self.decision_head = MiniHead()

        def forward(self, input_ids, attention_mask, op_indices):
            h = self.backbone(input_ids, attention_mask).last_hidden_state
            expanded_idx = op_indices.unsqueeze(-1).expand(-1, -1, h.size(-1))
            gathered = torch.gather(h, dim=1, index=expanded_idx)
            return self.decision_head(gathered)

    # ダミーデータローダー
    mini_batches = [
        {
            "input_ids": torch.randint(0, 50, (4, 12)),
            "attention_mask": torch.ones(4, 12, dtype=torch.long),
            "op_indices": torch.tensor(
                [[1, 3, 5, 0], [2, 4, 6, 0], [1, 2, 0, 0], [3, 5, 7, 9]]
            ),
            "op_mask": torch.tensor(
                [
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, False, False],
                    [True, True, True, True],
                ]
            ),
            "labels": torch.tensor([1, 0, 0, 2]),
            "question_types": ["choice", "choice", "noul", "score"],
            "is_negatives": [False, False, False, False],
        }
    ]

    from torch.utils.data import DataLoader, Dataset

    class MiniDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    mini_train_loader = DataLoader(MiniDataset(mini_batches), batch_size=None)
    mini_val_loader = DataLoader(MiniDataset(mini_batches), batch_size=None)

    demo_config = RLCDConfig(
        epochs=2,
        num_generations=rlcd_config.num_generations,
        perturbation_std=rlcd_config.perturbation_std,
        learning_rate=1e-3,
        output_dir=str(out_dir / "mini_trainer_run"),
        scoring_config=ScoringConfig(),
    )

    mini_policy = MiniModel()
    mini_ref = MiniModel()

    demo_trainer = RLCDTrainer(
        config=demo_config,
        policy_model=mini_policy,
        ref_model=mini_ref,
        train_dataloader=mini_train_loader,
        val_dataloader=mini_val_loader,
    )

    train_res = demo_trainer.train()

    hist_rows = []
    for h in train_res["history"]:
        ep = h["epoch"]
        t_loss = h["train"].get("loss_total", 0.0)
        t_rew = h["train"].get("mean_reward", 0.0)
        v_comp = h["val"].get("composite_score", 0.0)
        v_ece = h["val"].get("ece", 0.0)
        v_acc = h["val"].get("accuracy", 0.0)
        hist_rows.append(
            f"| Epoch {ep} | `{t_loss:.4f}` | `{t_rew:.4f}` | `{v_comp:.4f}` | `{v_ece:.4f}` | `{v_acc:.1%}` |"
        )

    hist_table_md = "\n".join(hist_rows)

    mo.md(f"""
    ### 📈 RLCD 学習ループ実行履歴
    | エポック | 訓練損失 (Total Loss) | 平均報酬 (Reward) | 検証複合スコア (Composite) | 検証 ECE (較正誤差) | 検証 Top-1 精度 |
    | :---: | :---: | :---: | :---: | :---: | :---: |
    {hist_table_md}

    - **最良複合スコア**: `{train_res["best_composite_score"]:.4f}`
    - **チェックポイント保存場所**: `{demo_config.output_dir}/best_checkpoint/model.pt`
    """)


if __name__ == "__main__":
    app.run()
