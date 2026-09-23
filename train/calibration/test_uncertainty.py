"""不確実性・確信度数理テストモジュール。

正規化シャノンエントロピー H_norm、Top-Margin M(p)、複合確信度スコア S_confidence、
およびバッチテンソル実装の整合性と境界値を検証する。
"""

import math
from pathlib import Path

import pytest
import torch

from calibration.evaluator import (
    compute_batch_composite_confidence,
    compute_batch_normalized_entropy,
    compute_batch_top_margin,
)
from contract import (
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_MARGIN_THRESHOLD,
    CalibrationConfig,
    GatingThresholds,
    compute_composite_confidence,
    compute_normalized_entropy,
    compute_top_margin,
)


def test_contract_constants() -> None:
    """不確実性・ゲーティング契約定数のデフォルト値を検証する。"""
    assert DEFAULT_HIGH_CONFIDENCE_THRESHOLD == 0.85
    assert DEFAULT_LOW_CONFIDENCE_THRESHOLD == 0.50
    assert DEFAULT_TOP_MARGIN_THRESHOLD == 0.15


def test_normalized_entropy() -> None:
    """正規化シャノンエントロピーの計算と境界値を検証する。"""
    # 1. K=1 の特異点: 決定的なので 0.0
    assert compute_normalized_entropy([1.0]) == 0.0

    # 2. 完全一様分布: H_norm == 1.0
    assert abs(compute_normalized_entropy([0.5, 0.5]) - 1.0) < 1e-6
    assert abs(compute_normalized_entropy([0.25, 0.25, 0.25, 0.25]) - 1.0) < 1e-6

    # 3. ワンホット極限: H_norm == 0.0
    assert compute_normalized_entropy([1.0, 0.0, 0.0, 0.0]) < 1e-6

    # 4. 一般分布 (K=3)
    p3 = [0.7, 0.2, 0.1]
    expected_h = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
    expected_norm = expected_h / math.log(3)
    assert abs(compute_normalized_entropy(p3) - expected_norm) < 1e-6

    # 5. 異常系
    with pytest.raises(ValueError):
        compute_normalized_entropy([])
    with pytest.raises(ValueError):
        compute_normalized_entropy([-0.1, 1.1])
    with pytest.raises(ValueError):
        compute_normalized_entropy([float("nan"), 0.5])


def test_top_margin() -> None:
    """Top-Margin の計算と境界値を検証する。"""
    # 1. K=1 の特異点: 競合なしなので 1.0
    assert compute_top_margin([1.0]) == 1.0

    # 2. 明確な差
    assert abs(compute_top_margin([0.7, 0.2, 0.1]) - 0.5) < 1e-6

    # 3. タイブレーク (同率1位)
    assert compute_top_margin([0.4, 0.4, 0.2]) < 1e-6
    assert compute_top_margin([0.25, 0.25, 0.25, 0.25]) < 1e-6

    # 4. ワンホット極限: マージン 1.0
    assert abs(compute_top_margin([1.0, 0.0, 0.0]) - 1.0) < 1e-6

    # 5. 最大値が末尾
    assert abs(compute_top_margin([0.1, 0.2, 0.7]) - 0.5) < 1e-6

    # 6. 異常系
    with pytest.raises(ValueError):
        compute_top_margin([])
    with pytest.raises(ValueError):
        compute_top_margin([-0.1, 1.1])


def test_composite_confidence() -> None:
    """複合確信度スコアの計算と境界値を検証する。"""
    # 1. K=1 の特異点: 完全に決定しているので 1.0
    assert compute_composite_confidence([1.0]) == 1.0

    # 2. ワンホット極限: S_conf == 1.0
    assert abs(compute_composite_confidence([1.0, 0.0, 0.0]) - 1.0) < 1e-6

    # 3. 完全一様分布極限: S_conf == 0.0
    assert compute_composite_confidence([0.25, 0.25, 0.25, 0.25]) < 1e-6

    # 4. Margin Collapse (激しい拮抗): Margin == 0.0 なので S_conf == 0.0
    assert compute_composite_confidence([0.49, 0.49, 0.02]) < 1e-6

    # 5. 高確信判定ケース: 1 - H_norm と Margin の積
    p_high = [0.85, 0.10, 0.05]
    s_conf = compute_composite_confidence(p_high)
    h_norm = compute_normalized_entropy(p_high)
    margin = compute_top_margin(p_high)
    assert abs(s_conf - (1.0 - h_norm) * margin) < 1e-6
    assert 0.35 < s_conf < 0.45

    # 6. 圧倒的確信ケース
    p_extreme = [0.98, 0.01, 0.01]
    assert compute_composite_confidence(p_extreme) > 0.80


def test_batch_tensor_parity() -> None:
    """バッチテンソル計算関数と純粋 Python 計算関数の一致を検証する。"""
    probs_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.49, 0.49, 0.02, 0.0],
        [0.70, 0.20, 0.10, 0.0],
        [0.98, 0.01, 0.01, 0.0],
    ]
    # 最後の 2 サンプルは 3 候補 (末尾要素はパディング無効)
    mask_list = [
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
    ]

    probs = torch.tensor(probs_list, dtype=torch.float32)
    op_mask = torch.tensor(mask_list, dtype=torch.bool)

    batch_h = compute_batch_normalized_entropy(probs, op_mask)
    batch_m = compute_batch_top_margin(probs, op_mask)
    batch_s = compute_batch_composite_confidence(probs, op_mask)

    for i in range(len(probs_list)):
        valid_probs = [
            probs_list[i][j] for j in range(len(probs_list[i])) if mask_list[i][j]
        ]
        py_h = compute_normalized_entropy(valid_probs)
        py_m = compute_top_margin(valid_probs)
        py_s = compute_composite_confidence(valid_probs)

        assert abs(batch_h[i].item() - py_h) < 1e-5
        assert abs(batch_m[i].item() - py_m) < 1e-5
        assert abs(batch_s[i].item() - py_s) < 1e-5


def test_calibration_config_gating_thresholds(tmp_path: Path) -> None:
    """CalibrationConfig の JSON ラウンドトリップで gating_thresholds が維持されることを検証する。"""
    config = CalibrationConfig(
        version="1.0",
        default_temperature=1.0,
        gating_thresholds=GatingThresholds(
            high_threshold=0.80,
            low_threshold=0.45,
            top_margin_threshold=0.12,
        ),
    )
    save_path = tmp_path / "calibration.json"
    config.save(save_path)

    loaded = CalibrationConfig.load(save_path)
    assert loaded.gating_thresholds.high_threshold == 0.80
    assert loaded.gating_thresholds.low_threshold == 0.45
    assert loaded.gating_thresholds.top_margin_threshold == 0.12
