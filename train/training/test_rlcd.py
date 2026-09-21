"""RLCD (Reinforcement Learning from Calibrated Decisions) 単体テストスイート。

設定の双方向シリアライズ、ロジット摂動サンプリングとマスク完全性、
サンプル内アドバンテージ算出とゼロ分散ガード、マスク付き KL ダイバージェンス、
RLCDLoss の勾配逆伝播、およびミニ学習ループの動作を検証する。
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    compute_expected_calibration_error,
)
from training.scoring_config import ScoringConfig


def test_rlcd_config_serialization() -> None:
    """RLCDConfig の YAML/JSON シリアライズおよび復元が完全であるかを検証する。"""
    config = RLCDConfig(
        num_generations=8,
        perturbation_std=0.15,
        clip_range=0.25,
        kl_coeff=0.08,
        entropy_coeff=0.02,
        sampling_temperature=1.2,
        learning_rate=3e-5,
        scoring_config=ScoringConfig(alpha=0.6, beta=0.8),
    )

    # 辞書シリアライズ
    d = config.to_dict()
    restored_from_dict = RLCDConfig.from_dict(d)
    assert restored_from_dict.num_generations == 8
    assert restored_from_dict.perturbation_std == pytest.approx(0.15)
    assert restored_from_dict.scoring_config.alpha == pytest.approx(0.6)

    # YAML シリアライズ
    yaml_str = config.to_yaml()
    restored_from_yaml = RLCDConfig.from_yaml(yaml_str)
    assert restored_from_yaml.num_generations == 8
    assert restored_from_yaml.kl_coeff == pytest.approx(0.08)

    # JSON シリアライズ
    json_str = config.to_json()
    restored_from_json = RLCDConfig.from_json(json_str)
    assert restored_from_json.clip_range == pytest.approx(0.25)
    assert restored_from_json.scoring_config.beta == pytest.approx(0.8)


def test_rlcd_config_file_io() -> None:
    """RLCDConfig のファイル保存および自動読み込みを検証する。"""
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = RLCDConfig(num_generations=6, learning_rate=5e-5)

        # YAML 保存
        yaml_file = tmp_path / "config.yaml"
        config.save(yaml_file)
        assert yaml_file.exists()
        loaded_yaml = RLCDConfig.from_yaml(yaml_file)
        assert loaded_yaml.num_generations == 6

        # JSON 保存
        json_file = tmp_path / "config.json"
        config.save(json_file)
        assert json_file.exists()
        loaded_json = RLCDConfig.from_json(json_file)
        assert loaded_json.learning_rate == pytest.approx(5e-5)


def test_sample_perturbed_logits() -> None:
    """ロジット摂動サンプリングの形状およびパディング無効マスクの完全性を検証する。"""
    batch_size, num_options, num_gen = 3, 5, 4
    logits = torch.randn(batch_size, num_options)
    op_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )

    perturbed = sample_perturbed_logits(
        logits=logits,
        op_mask=op_mask,
        num_generations=num_gen,
        perturbation_std=0.2,
    )

    assert perturbed.shape == (batch_size, num_gen, num_options)

    # パディング候補位置が厳密に DEFAULT_MASK_VALUE (-1e4) で維持されているか
    for b in range(batch_size):
        for g in range(num_gen):
            for k in range(num_options):
                if not op_mask[b, k]:
                    assert perturbed[b, g, k].item() == pytest.approx(
                        DEFAULT_MASK_VALUE
                    )

    # ノイズゼロの場合は元ロジットの複製と完全一致するか
    zero_noise_perturbed = sample_perturbed_logits(
        logits=logits,
        op_mask=op_mask,
        num_generations=num_gen,
        perturbation_std=0.0,
    )
    for b in range(batch_size):
        for g in range(num_gen):
            for k in range(num_options):
                if op_mask[b, k]:
                    assert zero_noise_perturbed[b, g, k].item() == pytest.approx(
                        logits[b, k].item(), abs=1e-5
                    )


def test_compute_group_advantages() -> None:
    """サンプル内アドバンテージ正規化およびゼロ分散ガードを検証する。"""
    # 2 サンプル、4 世代
    rewards = torch.tensor(
        [
            [0.2, 0.4, 0.6, 0.8],  # 平均 0.5、分散あり
            [1.0, 1.0, 1.0, 1.0],  # 全員同一正解 (分散 0)
        ]
    )

    advantages, mean_r, std_r = compute_group_advantages(rewards)

    assert advantages.shape == (2, 4)
    assert mean_r.shape == (2, 1)
    assert std_r.shape == (2, 1)

    # サンプル 0: 平均は 0、分散は 1 に正規化
    assert mean_r[0, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert advantages[0].mean().item() == pytest.approx(0.0, abs=1e-4)

    # サンプル 1: 分散ゼロガードによりアドバンテージがすべて 0 になること
    assert std_r[1, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.all(advantages[1] == 0.0)


def test_compute_masked_kl_divergence() -> None:
    """有効候補空間上での順方向 KL ダイバージェンスの数理特性を検証する。"""
    # 2 サンプル、4 候補
    op_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    p_theta = torch.tensor(
        [
            [0.5, 0.3, 0.2, 0.0],
            [0.7, 0.3, 0.0, 0.0],
        ]
    )
    p_ref = torch.tensor(
        [
            [0.5, 0.3, 0.2, 0.0],  # p_theta と完全一致
            [0.3, 0.7, 0.0, 0.0],  # 差異あり
        ]
    )

    kl = compute_masked_kl_divergence(p_theta, p_ref, op_mask)
    assert kl.shape == (2,)

    # サンプル 0: 完全一致のため KL はほぼ 0
    assert kl[0].item() == pytest.approx(0.0, abs=1e-5)

    # サンプル 1: 差異があるため正の値
    assert kl[1].item() > 0.0


def test_compute_entropy() -> None:
    """有効候補マスクを考慮したシャノンエントロピー計算を検証する。"""
    op_mask = torch.tensor(
        [
            [True, True, False],  # 2 択
            [True, True, False],  # 2 択
        ]
    )
    probs = torch.tensor(
        [
            [0.5, 0.5, 0.0],  # 一様分布 -> エントロピー ln(2)
            [1.0, 0.0, 0.0],  # ワンホット -> エントロピー 0
        ]
    )

    ent = compute_entropy(probs, op_mask)
    import math

    assert ent[0].item() == pytest.approx(math.log(2.0), abs=1e-4)
    assert ent[1].item() == pytest.approx(0.0, abs=1e-4)


def test_compute_expected_calibration_error() -> None:
    """期待較正誤差 (ECE) の計算を検証する。"""
    probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    op_mask = torch.ones_like(probs, dtype=torch.bool)
    labels = torch.tensor([0, 0, 1, 1])  # すべて完全正解かつ高確信度

    ece = compute_expected_calibration_error(probs, labels, op_mask, num_bins=5)
    # 正解率 1.0 に対し、平均確信度約 0.85 -> ECE は約 0.15
    assert 0.0 <= ece <= 1.0


def test_rlcd_loss_forward_backward() -> None:
    """RLCDLoss の順伝播、逆伝播、勾配健全性、およびメトリクス集計を検証する。"""
    torch.manual_seed(42)
    batch_size = 4
    num_options = 5

    policy_logits = torch.randn(batch_size, num_options, requires_grad=True)
    ref_logits = torch.randn(batch_size, num_options)
    labels = torch.tensor([1, 0, 2, 0])
    op_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    q_types = [
        QuestionType.CHOICE,
        QuestionType.SCORE,
        QuestionType.NOUL,
        QuestionType.CHOICE,
    ]

    config = RLCDConfig(num_generations=4, perturbation_std=0.1)
    loss_fn = RLCDLoss(config=config)

    loss, metrics = loss_fn(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        labels=labels,
        op_mask=op_mask,
        question_types=q_types,
    )

    assert torch.is_tensor(loss)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # メトリクスの存在チェック
    expected_keys = [
        "loss_total",
        "loss_policy",
        "loss_kl",
        "loss_entropy",
        "mean_reward",
        "std_reward",
        "mean_advantage",
        "p_target_mean",
    ]
    for key in expected_keys:
        assert key in metrics

    # 逆伝播テスト
    loss.backward()
    assert policy_logits.grad is not None
    assert not torch.isnan(policy_logits.grad).any()
    assert not torch.isinf(policy_logits.grad).any()


class DummyBackbone(nn.Module):
    """単体テスト用の軽量ダミーバックボーン。"""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Any:
        h = self.embed(input_ids)

        class Output:
            last_hidden_state = h

        return Output()


class DummyHead(nn.Module):
    """単体テスト用の軽量ダミーヘッド。"""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class DummyJevModel(nn.Module):
    """単体テスト用のダミー Jev モデル。"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DummyBackbone()
        self.decision_head = DummyHead()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids, attention_mask)
        h = out.last_hidden_state
        expanded_idx = op_indices.unsqueeze(-1).expand(-1, -1, h.size(-1))
        gathered = torch.gather(h, dim=1, index=expanded_idx)
        return self.decision_head(gathered)


class DummyDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """単体テスト用のインメモリーデータセットラッパー。"""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def test_rlcd_trainer_step() -> None:
    """RLCDTrainer による 1 エポックの学習実行とチェックポイント生成を検証する。"""
    torch.manual_seed(42)

    policy_model = DummyJevModel()
    ref_model = DummyJevModel()

    # ダミーバッチデータ
    batch_size = 2
    seq_len = 10

    batches = [
        {
            "input_ids": torch.randint(0, 50, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "op_indices": torch.tensor([[1, 3, 5], [2, 4, 6]]),
            "op_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "labels": torch.tensor([0, 1]),
            "question_types": ["choice", "choice"],
            "is_negatives": [False, False],
        }
    ]

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        DummyDataset(batches), batch_size=None
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        DummyDataset(batches), batch_size=None
    )

    with TemporaryDirectory() as tmp_dir:
        config = RLCDConfig(
            epochs=1,
            num_generations=2,
            perturbation_std=0.05,
            learning_rate=1e-3,
            output_dir=tmp_dir,
        )

        trainer = RLCDTrainer(
            config=config,
            policy_model=policy_model,
            ref_model=ref_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

        result = trainer.train()

        assert "best_composite_score" in result
        assert len(result["history"]) == 1
        assert (Path(tmp_dir) / "rlcd_config.yaml").exists()
        assert (Path(tmp_dir) / "rlcd_config.json").exists()
        assert (Path(tmp_dir) / "run_metadata.json").exists()
        assert (Path(tmp_dir) / "history.json").exists()
        assert (Path(tmp_dir) / "best_checkpoint" / "model.pt").exists()
