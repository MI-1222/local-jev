"""SFT 学習モジュール (config, loss, metrics, trainer) の単体テスト。

設定シリアライズ、マスク付き損失計算の数値妥当性、
多面的メトリクス集計、および小規模学習ループの統合動作を検証する。
"""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from data.schema import QuestionType, UnifiedSample
from models.decision_head import JevDecisionModel
from training.config import SFTConfig
from training.loss import MaskedCrossEntropyLoss
from training.metrics import MetricsTracker
from training.trainer import (
    SFTDataset,
    SFTTrainer,
    get_optimizer_grouped_parameters,
)


def test_sft_config_yaml_json_serialization(tmp_path: Path) -> None:
    """SFTConfig の YAML および JSON 双方向シリアライズの検証。"""
    config = SFTConfig(
        model_name_or_path="test/model-id",
        batch_size=8,
        learning_rate_backbone=1e-5,
        learning_rate_embed=3e-5,
        learning_rate_head=5e-4,
        num_epochs=5,
        early_stopping_patience=2,
        eval_metric="choice_accuracy",
        seed=123,
        output_dir=str(tmp_path / "runs"),
    )

    # 1. YAML の検証
    yaml_path = tmp_path / "config.yaml"
    config.save_yaml(yaml_path)
    loaded_yaml = SFTConfig.from_yaml(yaml_path)
    assert loaded_yaml.model_name_or_path == "test/model-id"
    assert loaded_yaml.batch_size == 8
    assert loaded_yaml.learning_rate_embed == 3e-5
    assert loaded_yaml.learning_rate_head == 5e-4
    assert loaded_yaml.early_stopping_patience == 2
    assert loaded_yaml.eval_metric == "choice_accuracy"
    assert loaded_yaml.seed == 123

    # 2. JSON の検証
    json_path = tmp_path / "config.json"
    config.save_json(json_path)
    loaded_json = SFTConfig.from_json(json_path)
    assert loaded_json.num_epochs == 5
    assert loaded_json.learning_rate_embed == 3e-5
    assert loaded_json.output_dir == str(tmp_path / "runs")

    # 3. 自動判別 save() の検証
    auto_yaml = tmp_path / "auto.yml"
    config.save(auto_yaml)
    assert auto_yaml.exists()
    assert SFTConfig.from_yaml(auto_yaml).batch_size == 8


def test_get_optimizer_grouped_parameters_differential_lr() -> None:
    """3系統 Differential LR (Backbone, Embed, Head) および Weight Decay 分離の検証。"""

    class MockEmbeddings(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(50, 16)

    class MockLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(16, 16)
            self.norm = nn.LayerNorm(16)

    class MockBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = DummyBackboneConfig()
            self.embeddings = MockEmbeddings()
            self.layer = MockLayer()

    backbone = MockBackbone()
    model = JevDecisionModel(backbone=backbone, mlp_hidden_size=16)  # type: ignore[arg-type]

    lr_backbone = 2e-5
    lr_embed = 5e-5
    lr_head = 2e-4
    weight_decay = 0.01
    head_weight_decay = 0.001

    groups = get_optimizer_grouped_parameters(
        model=model,
        lr_backbone=lr_backbone,
        lr_embed=lr_embed,
        lr_head=lr_head,
        weight_decay=weight_decay,
        head_weight_decay=head_weight_decay,
    )

    # 少なくとも backbone, embed, head の各グループが存在すること
    lrs = {g["lr"] for g in groups}
    assert lrs == {lr_backbone, lr_embed, lr_head}, (
        f"期待される学習率 {lr_backbone, lr_embed, lr_head} に対し、{lrs} が返されました。"
    )

    # weight_decay の値が 0.0 または指定された減衰率であること
    for g in groups:
        assert g["weight_decay"] in {0.0, weight_decay, head_weight_decay}
        assert len(g["params"]) > 0


def test_masked_cross_entropy_loss_padding_immunity() -> None:
    """パディング候補のロジットが損失計算に影響を与えないことの検証。

    有効候補のみの損失値と、パディング候補に極端な値(巨大な正の値)を
    代入したマスク付き損失値が一致することを確認する。
    """
    # サンプル0は2択、サンプル1は3択
    op_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    # 基準ロジット
    logits_base = torch.tensor(
        [
            [1.0, 2.0, -100.0, -100.0],
            [3.0, 1.0, 0.5, -100.0],
        ],
        dtype=torch.float32,
    )

    # 無効候補にノイズや極端に大きなロジットを入れたテンソル
    logits_noisy = logits_base.clone()
    logits_noisy[0, 2:] = 999.0  # 本来なら Softmax を歪ませる巨大な値
    logits_noisy[1, 3] = 999.0

    loss_fn = MaskedCrossEntropyLoss(mask_value=-1e4)

    loss_base = loss_fn(logits_base, labels, op_mask)
    loss_noisy = loss_fn(logits_noisy, labels, op_mask)

    # マスクにより無効候補の値が遮断され、損失が完全に一致すること
    assert torch.isclose(loss_base, loss_noisy, atol=1e-5), (
        f"損失が一致しません: base={loss_base.item()}, noisy={loss_noisy.item()}。"
    )


def test_metrics_tracker_aggregation() -> None:
    """MetricsTracker の多面的集計(Choice, Score MAE, Noul, 合成ネガティブ)の検証。"""
    tracker = MetricsTracker()

    logits = torch.tensor(
        [
            [2.0, 0.5, -10.0],  # sample 0: pred=0, label=0 (正解), 2択
            [0.1, 1.8, 0.2],  # sample 1: pred=1, label=2 (不正解), 3択
            [3.0, 0.1, -10.0],  # sample 2: pred=0 (true), label=0, 2択
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2, 0], dtype=torch.long)
    op_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, True, False],
        ],
        dtype=torch.bool,
    )
    question_types = [
        QuestionType.CHOICE.value,
        QuestionType.SCORE.value,
        QuestionType.NOUL.value,
    ]
    is_negatives = [False, True, False]

    tracker.update(
        loss=0.5,
        logits=logits,
        labels=labels,
        op_mask=op_mask,
        question_types=question_types,
        is_negatives=is_negatives,
    )

    metrics = tracker.compute()

    # 全体精度: 3サンプル中2正解 -> 2/3 = 0.6667
    assert metrics["total_samples"] == 3
    assert metrics["accuracy"] == pytest.approx(0.6667, abs=1e-3)

    # タイプ別
    assert metrics["choice_accuracy"] == 1.0
    assert metrics["score_accuracy"] == 0.0
    assert "score_mae" in metrics
    assert metrics["noul_accuracy"] == 1.0

    # バケット別 (2択が2件、3択が1件)
    assert metrics["bucket_2_total"] == 2
    assert metrics["bucket_2_accuracy"] == 1.0
    assert metrics["bucket_3_5_total"] == 1
    assert metrics["bucket_3_5_accuracy"] == 0.0


class DummyBackboneConfig:
    """ダミーのバックボーン設定。"""

    hidden_size: int = 16


class DummyBackbone(nn.Module):
    """単体テスト用の軽量ダミーバックボーン。"""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = DummyBackboneConfig()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        class Output:
            def __init__(self, hidden_state: torch.Tensor) -> None:
                self.last_hidden_state = hidden_state

        emb = self.embedding(input_ids)
        return Output(emb)


class DummyTokenizer:
    """単体テスト用のモックトークナイザー。"""

    pad_token_id: int = 0

    def save_pretrained(self, output_dir: Path | str) -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_sft_trainer_dry_run(tmp_path: Path) -> None:
    """SFTTrainer のドライランと成果物保存(config, metadata, metrics)の検証。"""
    config = SFTConfig(
        num_epochs=1,
        batch_size=2,
        seed=42,
        output_dir=str(tmp_path / "runs"),
    )

    dummy_backbone = DummyBackbone(vocab_size=50, hidden_size=16)
    dummy_model = JevDecisionModel(
        backbone=dummy_backbone,  # type: ignore[arg-type]
        mlp_hidden_size=16,
    )
    dummy_tokenizer = DummyTokenizer()

    trainer = SFTTrainer(
        config=config,
        model=dummy_model,
        tokenizer=cast(PreTrainedTokenizerFast, dummy_tokenizer),
        op_token_id=99,
    )

    # 成果物ディレクトリが生成されていること
    assert trainer.run_dir.exists()

    # モック DataLoader の用意
    batch = {
        "input_ids": torch.randint(1, 40, (2, 8), dtype=torch.long),
        "attention_mask": torch.ones((2, 8), dtype=torch.long),
        "op_indices": torch.tensor([[2, 5], [1, 4]], dtype=torch.long),
        "op_mask": torch.tensor([[True, True], [True, True]], dtype=torch.bool),
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "question_types": ["choice", "choice"],
        "is_negatives": torch.tensor([False, False], dtype=torch.bool),
    }

    # evaluate の動作確認
    eval_metrics = trainer.evaluate(
        dummy_model, cast(DataLoader[dict[str, Any]], [batch])
    )
    assert "accuracy" in eval_metrics
    assert "loss" in eval_metrics

    # チェックポイント保存の動作確認
    ckpt_dir = trainer.save_checkpoint(dummy_model, eval_metrics, is_best=True)
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "tokenizer" / "tokenizer.json").exists()
    assert (ckpt_dir / "metrics.json").exists()

    # train() 全体フローの動作確認 (モックデータローダーを使用)
    sample_dataset = SFTDataset(
        samples=[],
        tokenizer=cast(PreTrainedTokenizerFast, dummy_tokenizer),
        op_token_id=99,
    )
    mock_loader = [batch]
    loader_typed = cast(DataLoader[dict[str, Any]], mock_loader)

    class DummyTrainer(SFTTrainer):
        """テスト用のモックデータ供給 Trainer。"""

        def prepare_data(
            self,
        ) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], SFTDataset]:
            return loader_typed, loader_typed, sample_dataset

    test_trainer = DummyTrainer(
        config=config,
        model=dummy_model,
        tokenizer=cast(PreTrainedTokenizerFast, dummy_tokenizer),
        op_token_id=99,
    )

    summary = test_trainer.train()
    assert "best_accuracy" in summary
    assert "run_dir" in summary
    assert (test_trainer.run_dir / "config.yaml").exists()
    assert (test_trainer.run_dir / "config.json").exists()
    assert (test_trainer.run_dir / "run_metadata.json").exists()
    assert (test_trainer.run_dir / "metrics_history.json").exists()
    assert (test_trainer.run_dir / "final_summary.md").exists()


def test_run_sft_cli_argument_parsing(tmp_path: Path) -> None:
    """run_sft の CLI 引数解析および設定上書きの動作検証。"""
    from training.run_sft import build_config_from_args, parse_args

    # 1. デフォルト設定の検証
    args = parse_args([])
    config = build_config_from_args(args)
    assert config.learning_rate_backbone == 2e-5
    assert config.learning_rate_embed == 5e-5
    assert config.learning_rate_head == 2e-4
    assert config.num_epochs == 5

    # 2. CLI 引数による上書きの検証
    cli_args = [
        "--model-name-or-path",
        "custom/model",
        "--dataset-names",
        "jglue_jnli",
        "jglue_jsts",
        "--lr-backbone",
        "1e-5",
        "--lr-embed",
        "3e-5",
        "--lr-head",
        "1e-4",
        "--num-epochs",
        "10",
        "--batch-size",
        "32",
        "--output-dir",
        str(tmp_path / "custom_run"),
    ]
    parsed = parse_args(cli_args)
    custom_config = build_config_from_args(parsed)
    assert custom_config.model_name_or_path == "custom/model"
    assert custom_config.dataset_names == ["jglue_jnli", "jglue_jsts"]
    assert custom_config.learning_rate_backbone == 1e-5
    assert custom_config.learning_rate_embed == 3e-5
    assert custom_config.learning_rate_head == 1e-4
    assert custom_config.num_epochs == 10
    assert custom_config.batch_size == 32
    assert custom_config.output_dir == str(tmp_path / "custom_run")


def test_composite_metric_calculation() -> None:
    """MetricsTracker における複合評価指標 (composite_metric) 算出の検証。"""
    tracker = MetricsTracker()

    # Choice, Score, Noul の3サンプルを投入
    # sample 0: Choice 正解 (acc=1.0)
    # sample 1: Score 誤差 0.5 (mae=0.5 -> score_component = 1.0 - 0.5 = 0.5)
    # sample 2: Noul 正解 (acc=1.0)
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    op_mask = torch.tensor([[True, True], [True, True], [True, True]], dtype=torch.bool)
    question_types = [
        QuestionType.CHOICE.value,
        QuestionType.SCORE.value,
        QuestionType.NOUL.value,
    ]

    tracker.update(
        loss=0.3,
        logits=logits,
        labels=labels,
        op_mask=op_mask,
        question_types=question_types,
    )

    metrics = tracker.compute()
    assert "composite_metric" in metrics
    # composite_score = 0.50 * 1.0 + 0.25 * (1.0 - 0.5) + 0.25 * 1.0
    # 期待値: 0.50 + 0.125 + 0.25 = 0.875 -> 0.875
    expected_composite = (
        0.50 * 1.0 + 0.25 * (1.0 - min(metrics["score_mae"], 1.0)) + 0.25 * 1.0
    )
    assert metrics["composite_metric"] == pytest.approx(
        round(expected_composite, 4), abs=1e-3
    )


def test_optimizer_parameter_assertion() -> None:
    """get_optimizer_grouped_parameters で全可変パラメータが分類されることの検証。"""

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_layer = nn.Linear(4, 4)
            self.embeddings = nn.Embedding(10, 4)
            self.decision_head = nn.Linear(4, 1)

    model = DummyModel()
    groups = get_optimizer_grouped_parameters(
        model=cast(JevDecisionModel, model),
        lr_backbone=1e-5,
        lr_embed=3e-5,
        lr_head=1e-4,
        weight_decay=0.01,
    )
    total_grouped = sum(len(g["params"]) for g in groups)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert total_grouped == len(trainable_params)


def test_evaluate_position_bias(tmp_path: Path) -> None:
    """SFTTrainer.evaluate_position_bias による選択肢順序シャッフル不変性の検証。"""
    config = SFTConfig(
        output_dir=str(tmp_path / "runs"),
        evaluate_position_bias=True,
    )

    class DummyModelWithForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy_param = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            op_indices: torch.Tensor,
        ) -> torch.Tensor:
            # 常に第1候補(インデックス0)に高いロジットを出力するモデル (強い位置バイアスを持つ)
            batch_size, num_options = op_indices.shape
            logits = torch.zeros((batch_size, num_options))
            logits[:, 0] = 5.0
            return logits

    from models.backbone import (
        DEFAULT_MODERNBERT_MODEL_ID,
        prepare_backbone_and_tokenizer,
    )

    _, tokenizer, op_token_id = prepare_backbone_and_tokenizer(
        DEFAULT_MODERNBERT_MODEL_ID
    )

    trainer = SFTTrainer(
        config=config,
        model=cast(JevDecisionModel, DummyModelWithForward()),
        tokenizer=tokenizer,
        op_token_id=op_token_id,
    )

    sample = UnifiedSample(
        sample_id="choice_01",
        dataset_name="test_dataset",
        question_type=QuestionType.CHOICE,
        state="ユーザーの問い合わせです。",
        instructions="カテゴリを選択してください。",
        criteria={"hw": "ハードウェア障害", "sw": "ソフトウェア障害"},
        target="hw",
    )

    bias_result = trainer.evaluate_position_bias(
        model=trainer.model,
        val_samples=[sample],
        threshold=0.03,
    )

    assert "position_bias_max_delta" in bias_result
    assert "position_bias_mean_delta" in bias_result
    assert "position_bias_pass_rate" in bias_result
    assert bias_result["position_bias_evaluated_samples"] == 1


def test_save_checkpoint_includes_backbone_config(tmp_path: Path) -> None:
    """save_checkpoint がバックボーンの Hugging Face 設定を保存することの検証。"""
    config = SFTConfig(
        output_dir=str(tmp_path / "runs"),
    )

    class MockConfig:
        hidden_size: int = 4

        def save_pretrained(self, save_dir: Path | str) -> None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "config.json").write_text(
                '{"model_type": "modernbert"}', encoding="utf-8"
            )

    class MockBackboneWithConfig(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = MockConfig()
            self.linear = nn.Linear(4, 4)

    dummy_model = JevDecisionModel(
        backbone=cast(Any, MockBackboneWithConfig()),
        mlp_hidden_size=4,
    )
    dummy_tokenizer = DummyTokenizer()

    trainer = SFTTrainer(
        config=config,
        model=dummy_model,
        tokenizer=cast(PreTrainedTokenizerFast, dummy_tokenizer),
        op_token_id=99,
    )

    ckpt_dir = trainer.save_checkpoint(dummy_model, {"loss": 0.5}, is_best=True)
    assert (ckpt_dir / "config.json").exists()
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "tokenizer" / "tokenizer.json").exists()
