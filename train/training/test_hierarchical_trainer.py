"""粗密マルチタスク SFT トレーナーおよび評価の単体テスト。"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoTokenizer

from data.hierarchical import HierarchicalMapping
from data.hierarchical_dataset import (
    HierarchicalDatasetGenerator,
)
from data.schema import QuestionType, UnifiedSample
from training.hierarchical_config import HierarchicalSFTConfig
from training.hierarchical_trainer import (
    HierarchicalSFTTrainer,
    compute_expected_calibration_error,
)


class MockDecisionModel(nn.Module):
    """テスト用の軽量決定モデルスタブ。"""

    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_indices: torch.Tensor,
        op_mask: torch.Tensor | None = None,
    ) -> Any:
        batch_size, num_options = op_indices.shape
        logits = (
            torch.randn(batch_size, num_options, device=input_ids.device)
            + self.dummy_param * 0.0
        )

        class Output:
            logits: torch.Tensor

        out = Output()
        out.logits = logits
        return out


def test_ece_computation() -> None:
    """ECE (Expected Calibration Error) 計算関数が正しく動作することを検証する。"""
    # 完全に較正された予測
    confidences = np.array([0.9, 0.9, 0.1, 0.1])
    corrects = np.array([1.0, 1.0, 0.0, 0.0])
    ece = compute_expected_calibration_error(confidences, corrects, num_bins=5)
    assert ece <= 0.15

    # 空配列
    assert compute_expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_config_serialization(tmp_path: Path) -> None:
    """HierarchicalSFTConfig の JSON / YAML 保存と復元を検証する。"""
    cfg = HierarchicalSFTConfig(
        model_name_or_path="test-model",
        beta=1.2,
        gamma=0.25,
        num_epochs=3,
    )

    json_file = tmp_path / "config.json"
    cfg.to_json(json_file)
    restored_json = HierarchicalSFTConfig.from_json(json_file)
    assert restored_json.model_name_or_path == "test-model"
    assert restored_json.beta == pytest.approx(1.2)
    assert restored_json.gamma == pytest.approx(0.25)

    yaml_file = tmp_path / "config.yaml"
    cfg.to_yaml(yaml_file)
    restored_yaml = HierarchicalSFTConfig.from_yaml(yaml_file)
    assert restored_yaml.num_epochs == 3


def test_hierarchical_trainer_step_and_evaluate(tmp_path: Path) -> None:
    """HierarchicalSFTTrainer が 1 エポックの学習および評価シミュレーションを正常に完了することを検証する。"""
    mapping = HierarchicalMapping(
        name="SmallTest",
        coarse_categories={"c1": "大分類1", "c2": "大分類2"},
        fine_criteria={
            "f1": "細分類1",
            "f2": "細分類2",
            "f3": "細分類3",
            "f4": "細分類4",
        },
        coarse_to_fine={"c1": ["f1", "f2"], "c2": ["f3", "f4"]},
        fine_to_coarse={"f1": "c1", "f2": "c1", "f3": "c2", "f4": "c2"},
        all_coarse_keys=["c1", "c2"],
        all_fine_keys=["f1", "f2", "f3", "f4"],
    )

    from typing import cast

    from transformers import PreTrainedTokenizerFast

    from models.decision_head import JevDecisionModel

    raw_tokenizer = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-130m")
    tokenizer = cast(PreTrainedTokenizerFast, raw_tokenizer)
    if "[OP]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[OP]"]})

    gen = HierarchicalDatasetGenerator(mapping=mapping, soft_beam_ratio=0.5)
    samples = [
        UnifiedSample(
            dataset_name="SmallTest",
            sample_id=f"sample_{i}",
            question_type=QuestionType.CHOICE,
            state=f"状態テキスト {i}",
            instructions="指示",
            criteria=dict(mapping.fine_criteria),
            target="f1" if i % 2 == 0 else "f3",
        )
        for i in range(4)
    ]
    pairs = gen.generate_pairs(samples)

    config = HierarchicalSFTConfig(
        output_dir=str(tmp_path / "runs"),
        batch_size=2,
        num_epochs=1,
        gamma_warmup_epochs=1,
    )

    model = MockDecisionModel()

    trainer = HierarchicalSFTTrainer(
        config=config,
        model=cast(JevDecisionModel, model),
        tokenizer=tokenizer,
        mapping=mapping,
        train_pairs=pairs,
        eval_pairs=pairs,
        device=torch.device("cpu"),
    )

    # 1 エポック訓練の実行
    train_res = trainer.train_epoch(0)
    assert "train_loss" in train_res
    assert "train_loss_coarse" in train_res
    assert "train_loss_fine" in train_res
    assert "train_loss_consistency" in train_res

    # 検証および二段階推論シミュレーションの実行
    eval_res = trainer.evaluate()
    assert "eval_coarse_acc" in eval_res
    assert "eval_fine_acc" in eval_res
    assert "eval_e2e_acc" in eval_res
    assert "eval_soft_beam_rate" in eval_res
    assert "eval_ece" in eval_res

    # 一括 train() の実行
    summary = trainer.train()
    assert "best_e2e_acc" in summary
    assert (tmp_path / "runs" / "hierarchical_mapping.json").exists()
    assert (tmp_path / "runs" / "hierarchical_config.json").exists()
    assert (tmp_path / "runs" / "training_summary.json").exists()
