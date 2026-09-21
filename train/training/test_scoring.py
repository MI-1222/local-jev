"""厳密適格スコアリング規則モジュールの網羅的単体テスト。

ワンホット時の上限値、一様分布時の挙動、マスク不変性、RPSの順序距離感応度、
数値安定性、タイプ別ルーティング、勾配逆伝播、および設定シリアライズを検証する。
"""

import math
from pathlib import Path

import pytest
import torch

from data.schema import QuestionType
from training.scoring import (
    ProperScoringEvaluator,
    ProperScoringLoss,
    compute_bounded_log_score,
    compute_composite_scores,
    compute_ranked_probability_score,
    compute_spherical_score,
    get_normalized_probabilities,
)
from training.scoring_config import ScoringConfig


def test_one_hot_predictions() -> None:
    """ワンホット完全正解予測において全スコアが上限 1.0 に達することを検証する。"""
    # 4候補、正解は 2
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.long)
    op_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    s_log = compute_bounded_log_score(probs, labels, op_mask, normalize=True)
    s_sph = compute_spherical_score(probs, labels, op_mask)
    s_rps = compute_ranked_probability_score(probs, labels, op_mask)

    assert pytest.approx(s_log.item(), rel=1e-5) == 1.0
    assert pytest.approx(s_sph.item(), rel=1e-5) == 1.0
    assert pytest.approx(s_rps.item(), rel=1e-5) == 1.0


def test_uniform_predictions() -> None:
    """一様分布予測における各スコアの数理的整合性を検証する。"""
    # 4候補、一様分布
    k = 4
    probs = torch.full((1, k), 1.0 / k, dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)
    op_mask = torch.ones((1, k), dtype=torch.bool)

    s_log = compute_bounded_log_score(probs, labels, op_mask, eps=1e-6, normalize=True)
    s_sph = compute_spherical_score(probs, labels, op_mask)
    s_rps = compute_ranked_probability_score(probs, labels, op_mask)

    # 球面スコアは 1 / sqrt(k) = 0.5
    expected_sph = 1.0 / math.sqrt(k)
    assert pytest.approx(s_sph.item(), rel=1e-5) == expected_sph

    # 対数スコアおよび RPS は 0 より大きく 1 未満
    assert 0.0 < s_log.item() < 1.0
    assert 0.0 < s_rps.item() < 1.0


def test_mask_invariance() -> None:
    """パディング候補のロジットが変化しても全スコアが完全に不変であることを検証する。"""
    # 候補数 5、有効候補数 3、パディング 2
    op_mask = torch.tensor([[True, True, True, False, False]], dtype=torch.bool)
    labels = torch.tensor([1], dtype=torch.long)

    # 有効部分のロジットは同一、パディング部分のロジットのみ大幅に変更
    logits_a = torch.tensor([[2.0, 5.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    logits_b = torch.tensor([[2.0, 5.0, 1.0, 99.0, -99.0]], dtype=torch.float32)

    probs_a = get_normalized_probabilities(logits_a, op_mask)
    probs_b = get_normalized_probabilities(logits_b, op_mask)

    # 有効部分の確率が完全一致すること
    assert torch.allclose(probs_a, probs_b, atol=1e-6)
    assert probs_a[0, 3].item() == 0.0
    assert probs_a[0, 4].item() == 0.0

    # 各スコアが完全一致すること
    s_log_a = compute_bounded_log_score(probs_a, labels, op_mask)
    s_log_b = compute_bounded_log_score(probs_b, labels, op_mask)
    assert pytest.approx(s_log_a.item(), abs=1e-6) == s_log_b.item()

    s_sph_a = compute_spherical_score(probs_a, labels, op_mask)
    s_sph_b = compute_spherical_score(probs_b, labels, op_mask)
    assert pytest.approx(s_sph_a.item(), abs=1e-6) == s_sph_b.item()

    s_rps_a = compute_ranked_probability_score(probs_a, labels, op_mask)
    s_rps_b = compute_ranked_probability_score(probs_b, labels, op_mask)
    assert pytest.approx(s_rps_a.item(), abs=1e-6) == s_rps_b.item()


def test_rps_distance_sensitivity() -> None:
    """Score 型において正解に近い誤答ほど高い RPS を獲得することを検証する。"""
    # 5段階評価 (0, 1, 2, 3, 4), 正解は 2
    labels = torch.tensor([2], dtype=torch.long)
    op_mask = torch.ones((1, 5), dtype=torch.bool)

    # 予測1: 完全正解 (2 にワンホット)
    p_exact = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    # 予測2: 1段階外し (1 にワンホット)
    p_near = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    # 予測3: 2段階外し (0 にワンホット)
    p_far = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

    score_exact = compute_ranked_probability_score(p_exact, labels, op_mask).item()
    score_near = compute_ranked_probability_score(p_near, labels, op_mask).item()
    score_far = compute_ranked_probability_score(p_far, labels, op_mask).item()

    # 正解 > 1段階外し > 2段階外し
    assert score_exact > score_near > score_far
    assert pytest.approx(score_exact, abs=1e-5) == 1.0
    # K=5 のとき、K-1=4
    # p_near (y=2 vs pred=1): step m=1 のみ差 1 -> sum=1 -> loss=1/4=0.25 -> score=0.75
    assert pytest.approx(score_near, abs=1e-5) == 0.75
    # p_far (y=2 vs pred=0): step m=0, 1 で差 1 -> sum=2 -> loss=2/4=0.50 -> score=0.50
    assert pytest.approx(score_far, abs=1e-5) == 0.50


def test_numerical_stability() -> None:
    """極小予測確率 (p -> 0) 時にも NaN や -inf を出さず安定に計算されることを検証する。"""
    # 正解 0 に対して予測確率 0.0
    probs = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    op_mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    s_log = compute_bounded_log_score(probs, labels, op_mask, eps=1e-6, normalize=True)
    s_sph = compute_spherical_score(probs, labels, op_mask)
    s_rps = compute_ranked_probability_score(probs, labels, op_mask)

    assert not torch.isnan(s_log).any()
    assert not torch.isinf(s_log).any()
    assert pytest.approx(s_log.item(), abs=1e-6) == 0.0

    assert not torch.isnan(s_sph).any()
    assert not torch.isinf(s_sph).any()
    assert pytest.approx(s_sph.item(), abs=1e-6) == 0.0

    assert not torch.isnan(s_rps).any()
    assert not torch.isinf(s_rps).any()
    assert 0.0 <= s_rps.item() <= 1.0


def test_composite_routing() -> None:
    """質問タイプに応じた動的ルーティング (Choice vs Score) を検証する。"""
    # バッチサイズ 2: [0] は Choice, [1] は Score
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5],
            [0.5, 3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1], dtype=torch.long)
    op_mask = torch.ones((2, 3), dtype=torch.bool)
    q_types = [QuestionType.CHOICE, QuestionType.SCORE]

    cfg = ScoringConfig(alpha=0.6, beta=0.8)
    res = compute_composite_scores(
        logits=logits,
        labels=labels,
        op_mask=op_mask,
        question_types=q_types,
        config=cfg,
    )

    comp = res["composite"]
    s_log = res["s_log"]
    s_sph = res["s_sph"]
    s_rps = res["s_rps"]

    # サンプル 0 (Choice): alpha * s_log + (1 - alpha) * s_sph
    expected_0 = 0.6 * s_log[0] + 0.4 * s_sph[0]
    assert pytest.approx(comp[0].item(), abs=1e-5) == expected_0.item()

    # サンプル 1 (Score): beta * s_rps + (1 - beta) * s_log
    expected_1 = 0.8 * s_rps[1] + 0.2 * s_log[1]
    assert pytest.approx(comp[1].item(), abs=1e-5) == expected_1.item()


def test_proper_scoring_loss_backward() -> None:
    """ProperScoringLoss を通じて勾配が正常に逆伝播することを検証する。"""
    logits = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([1], dtype=torch.long)
    op_mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    loss_fn = ProperScoringLoss(config=ScoringConfig())
    loss = loss_fn(logits, labels, op_mask, question_types=[QuestionType.CHOICE])

    assert loss.requires_grad
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.shape == logits.shape


def test_scoring_config_serialization(tmp_path: Path) -> None:
    """ScoringConfig の YAML / JSON シリアライズと完全復元を検証する。"""
    original = ScoringConfig(
        alpha=0.45,
        beta=0.85,
        eps=1e-5,
        temperature=1.2,
        output_dir="test_runs/scoring",
    )

    yaml_path = tmp_path / "config.yaml"
    json_path = tmp_path / "config.json"

    original.save_yaml(yaml_path)
    original.save_json(json_path)

    loaded_yaml = ScoringConfig.from_yaml(yaml_path)
    loaded_json = ScoringConfig.from_json(json_path)

    assert loaded_yaml == original
    assert loaded_json == original


def test_evaluator_aggregation() -> None:
    """ProperScoringEvaluator による多面的集計が正常に機能することを検証する。"""
    evaluator = ProperScoringEvaluator(config=ScoringConfig(alpha=0.5, beta=0.7))

    logits = torch.tensor(
        [
            [10.0, -10.0],  # 高確信度
            [0.1, 0.0],  # 中〜低確信度
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0], dtype=torch.long)
    op_mask = torch.ones((2, 2), dtype=torch.bool)
    q_types = [QuestionType.CHOICE, QuestionType.NOUL]

    evaluator.update(logits, labels, op_mask, question_types=q_types)
    summary = evaluator.compute()

    assert summary["total_samples"] == 2
    assert "mean_composite" in summary
    assert "by_type" in summary
    assert "choice" in summary["by_type"]
    assert "noul" in summary["by_type"]
    assert "by_confidence" in summary
