"""粗密マルチタスク複合損失および階層整合性正則化の単体テスト。"""

import pytest
import torch

from training.hierarchical_loss import (
    HierarchicalConsistencyLoss,
    HierarchicalMultiTaskLoss,
)


def test_consistency_loss_stop_gradient() -> None:
    """階層整合性損失において、大分類ロジットに勾配が逆流しない (Stop-Gradient) ことを検証する。"""
    loss_fn = HierarchicalConsistencyLoss(
        margin=1.0,
        use_marginal=True,
        use_suppression=True,
        suppression_weight=0.5,
    )

    batch_size = 2
    num_coarse = 3
    num_fine = 6

    coarse_logits = torch.randn(batch_size, num_coarse, requires_grad=True)
    fine_logits = torch.randn(batch_size, num_fine, requires_grad=True)

    coarse_op_mask = torch.ones(batch_size, num_coarse, dtype=torch.bool)
    fine_op_mask = torch.ones(batch_size, num_fine, dtype=torch.bool)

    # 各細分類スロットの親クラスタ ID: [0, 0, 1, 1, 2, 2]
    fine_parent_indices = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]], dtype=torch.long
    )
    coarse_labels = torch.tensor([0, 1], dtype=torch.long)

    loss = loss_fn(
        coarse_logits=coarse_logits,
        fine_logits=fine_logits,
        coarse_op_mask=coarse_op_mask,
        fine_op_mask=fine_op_mask,
        fine_parent_indices=fine_parent_indices,
        coarse_labels=coarse_labels,
    )

    assert loss.item() >= 0.0
    loss.backward()

    # 大分類ロジットには勾配が流れてはならない (Stop-Gradient)
    assert coarse_logits.grad is None

    # 細分類ロジットには正常にアラインメント勾配が流れていること
    assert fine_logits.grad is not None
    assert torch.any(fine_logits.grad != 0.0)


def test_multitask_loss_weighting_and_dict() -> None:
    """HierarchicalMultiTaskLoss の加重合成およびメトリクス辞書出力を検証する。"""
    loss_fn = HierarchicalMultiTaskLoss(
        label_smoothing=0.05,
        beta=1.5,
        gamma=0.2,
    )

    batch_size = 2
    num_coarse = 4
    num_fine = 8

    coarse_logits = torch.randn(batch_size, num_coarse, requires_grad=True)
    fine_logits = torch.randn(batch_size, num_fine, requires_grad=True)

    coarse_labels = torch.tensor([1, 2], dtype=torch.long)
    fine_labels = torch.tensor([3, 5], dtype=torch.long)

    coarse_op_mask = torch.ones(batch_size, num_coarse, dtype=torch.bool)
    fine_op_mask = torch.ones(batch_size, num_fine, dtype=torch.bool)
    fine_parent_indices = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long
    )

    total_loss, loss_dict = loss_fn(
        coarse_logits=coarse_logits,
        coarse_labels=coarse_labels,
        coarse_op_mask=coarse_op_mask,
        fine_logits=fine_logits,
        fine_labels=fine_labels,
        fine_op_mask=fine_op_mask,
        fine_parent_indices=fine_parent_indices,
        return_dict=True,
    )

    assert total_loss.item() > 0.0
    assert "loss" in loss_dict
    assert "loss_coarse" in loss_dict
    assert "loss_fine" in loss_dict
    assert "loss_consistency" in loss_dict
    assert loss_dict["gamma"] == pytest.approx(0.2)

    expected_loss = (
        loss_dict["loss_coarse"]
        + 1.5 * loss_dict["loss_fine"]
        + 0.2 * loss_dict["loss_consistency"]
    )
    assert loss_dict["loss"] == pytest.approx(expected_loss, rel=1e-5)

    # gamma 更新 (ウォームアップ) の検証
    loss_fn.set_consistency_weight(0.5)
    assert loss_fn.gamma == pytest.approx(0.5)
