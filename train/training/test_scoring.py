"""厳密適格スコアリング規則モジュールの網羅的単体テスト。

ワンホット時の上限値、一様分布時の挙動、マスク不変性、RPSの順序距離感応度、
数値安定性、厳密適格性の期待値不変性、Brierスコア・報酬整合性、
タイプ別動的ルーティング、勾配逆伝播、および設定シリアライズを網羅的に検証する。
"""

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from data.schema import QuestionType
from training.scoring import (
    ProperScoringEngine,
    ProperScoringEvaluator,
    ProperScoringLoss,
    compute_bounded_log_score,
    compute_brier_reward,
    compute_brier_score,
    compute_composite_scores,
    compute_ranked_probability_score,
    compute_rps_loss,
    compute_rps_reward,
    compute_spherical_score,
    get_normalized_probabilities,
)
from training.scoring_config import ScoringConfig


def test_one_hot_predictions() -> None:
    """ワンホット完全正解予測において全スコアが上限 1.0 に達することを検証する。"""
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.long)
    op_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    s_log = compute_bounded_log_score(probs, labels, op_mask, normalize=True)
    s_sph = compute_spherical_score(probs, labels, op_mask)
    s_rps = compute_ranked_probability_score(probs, labels, op_mask)
    s_brier_loss = compute_brier_score(probs, labels, op_mask)
    s_brier_rew_norm = compute_brier_reward(probs, labels, op_mask, normalize=True)
    s_brier_rew_raw = compute_brier_reward(probs, labels, op_mask, normalize=False)

    assert pytest.approx(s_log.item(), rel=1e-5) == 1.0
    assert pytest.approx(s_sph.item(), rel=1e-5) == 1.0
    assert pytest.approx(s_rps.item(), rel=1e-5) == 1.0
    assert pytest.approx(s_brier_loss.item(), abs=1e-5) == 0.0
    assert pytest.approx(s_brier_rew_norm.item(), rel=1e-5) == 1.0
    assert pytest.approx(s_brier_rew_raw.item(), abs=1e-5) == 0.0


def test_uniform_predictions() -> None:
    """一様分布予測における各スコアの数理的整合性を検証する。"""
    k = 4
    probs = torch.full((1, k), 1.0 / k, dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)
    op_mask = torch.ones((1, k), dtype=torch.bool)

    s_log = compute_bounded_log_score(probs, labels, op_mask, eps=1e-6, normalize=True)
    s_sph = compute_spherical_score(probs, labels, op_mask)
    s_rps = compute_ranked_probability_score(probs, labels, op_mask)
    s_brier_loss = compute_brier_score(probs, labels, op_mask)

    expected_sph = 1.0 / math.sqrt(k)
    assert pytest.approx(s_sph.item(), rel=1e-5) == expected_sph

    assert 0.0 < s_log.item() < 1.0
    assert 0.0 < s_rps.item() < 1.0

    # Brier 損失: (1/4 - 1)^2 + 3 * (1/4 - 0)^2 = 9/16 + 3/16 = 12/16 = 0.75
    assert pytest.approx(s_brier_loss.item(), rel=1e-5) == 0.75


def test_mask_invariance() -> None:
    """パディング候補のロジットが変化しても全スコアが完全に不変であることを検証する。"""
    op_mask = torch.tensor([[True, True, True, False, False]], dtype=torch.bool)
    labels = torch.tensor([1], dtype=torch.long)

    logits_a = torch.tensor([[2.0, 5.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    logits_b = torch.tensor([[2.0, 5.0, 1.0, 99.0, -99.0]], dtype=torch.float32)

    probs_a = get_normalized_probabilities(logits_a, op_mask)
    probs_b = get_normalized_probabilities(logits_b, op_mask)

    assert torch.allclose(probs_a, probs_b, atol=1e-6)
    assert probs_a[0, 3].item() == 0.0
    assert probs_a[0, 4].item() == 0.0

    s_log_a = compute_bounded_log_score(probs_a, labels, op_mask)
    s_log_b = compute_bounded_log_score(probs_b, labels, op_mask)
    assert pytest.approx(s_log_a.item(), abs=1e-6) == s_log_b.item()

    s_sph_a = compute_spherical_score(probs_a, labels, op_mask)
    s_sph_b = compute_spherical_score(probs_b, labels, op_mask)
    assert pytest.approx(s_sph_a.item(), abs=1e-6) == s_sph_b.item()

    s_rps_a = compute_ranked_probability_score(probs_a, labels, op_mask)
    s_rps_b = compute_ranked_probability_score(probs_b, labels, op_mask)
    assert pytest.approx(s_rps_a.item(), abs=1e-6) == s_rps_b.item()

    s_brier_a = compute_brier_score(probs_a, labels, op_mask)
    s_brier_b = compute_brier_score(probs_b, labels, op_mask)
    assert pytest.approx(s_brier_a.item(), abs=1e-6) == s_brier_b.item()


def test_rps_distance_sensitivity() -> None:
    """Score 型において正解に近い誤答ほど高い RPS を獲得することを検証する。"""
    labels = torch.tensor([2], dtype=torch.long)
    op_mask = torch.ones((1, 5), dtype=torch.bool)

    p_exact = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    p_near = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    p_far = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

    score_exact = compute_ranked_probability_score(p_exact, labels, op_mask).item()
    score_near = compute_ranked_probability_score(p_near, labels, op_mask).item()
    score_far = compute_ranked_probability_score(p_far, labels, op_mask).item()

    assert score_exact > score_near > score_far
    assert pytest.approx(score_exact, abs=1e-5) == 1.0
    assert pytest.approx(score_near, abs=1e-5) == 0.75
    assert pytest.approx(score_far, abs=1e-5) == 0.50

    loss_exact = compute_rps_loss(p_exact, labels, op_mask).item()
    loss_near = compute_rps_loss(p_near, labels, op_mask).item()
    loss_far = compute_rps_loss(p_far, labels, op_mask).item()
    assert loss_exact < loss_near < loss_far
    assert pytest.approx(loss_exact, abs=1e-5) == 0.0
    assert pytest.approx(loss_near, abs=1e-5) == 0.25
    assert pytest.approx(loss_far, abs=1e-5) == 0.50


def test_numerical_stability() -> None:
    """極小予測確率 (p -> 0) 時にも NaN や -inf を出さず安定に計算されることを検証する。"""
    probs = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    op_mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    s_log = compute_bounded_log_score(
        probs, labels, op_mask, min_log_score=-10.0, normalize=True
    )
    s_sph = compute_spherical_score(probs, labels, op_mask)
    s_rps = compute_ranked_probability_score(probs, labels, op_mask)
    s_brier = compute_brier_score(probs, labels, op_mask)

    assert not torch.isnan(s_log).any()
    assert not torch.isinf(s_log).any()
    assert pytest.approx(s_log.item(), abs=1e-6) == 0.0

    assert not torch.isnan(s_sph).any()
    assert not torch.isinf(s_sph).any()
    assert pytest.approx(s_sph.item(), abs=1e-6) == 0.0

    assert not torch.isnan(s_rps).any()
    assert not torch.isinf(s_rps).any()
    assert 0.0 <= s_rps.item() <= 1.0

    assert not torch.isnan(s_brier).any()
    assert not torch.isinf(s_brier).any()
    assert pytest.approx(s_brier.item(), abs=1e-6) == 2.0


def test_bounded_log_score_unnormalized() -> None:
    """非正規化有界対数スコアが下限 -10.0 で厳密にクリッピングされることを検証する。"""
    probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    op_mask = torch.tensor([[True, True]], dtype=torch.bool)

    log_score_min = compute_bounded_log_score(
        probs, labels, op_mask, min_log_score=-10.0, normalize=False
    )
    assert pytest.approx(log_score_min.item(), abs=1e-5) == -10.0

    labels_hit = torch.tensor([1], dtype=torch.long)
    log_score_max = compute_bounded_log_score(
        probs, labels_hit, op_mask, min_log_score=-10.0, normalize=False
    )
    assert pytest.approx(log_score_max.item(), abs=1e-5) == 0.0


def test_rps_cdf_consistency() -> None:
    """RPS 計算においてクラス確率と累積確率 (CDF) の入力結果が完全一致することを検証する。"""
    probs = torch.tensor([[0.1, 0.4, 0.3, 0.2]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.long)
    op_mask = torch.ones((1, 4), dtype=torch.bool)

    cum_probs = torch.cumsum(probs, dim=-1)

    loss_pdf = compute_rps_loss(probs, labels, op_mask, is_cdf=False)
    loss_cdf = compute_rps_loss(cum_probs, labels, op_mask, is_cdf=True)
    assert pytest.approx(loss_pdf.item(), abs=1e-6) == loss_cdf.item()

    reward_pdf = compute_rps_reward(probs, labels, op_mask, is_cdf=False)
    reward_cdf = compute_rps_reward(cum_probs, labels, op_mask, is_cdf=True)
    assert pytest.approx(reward_pdf.item(), abs=1e-6) == reward_cdf.item()


def test_strictly_proper_scoring_mathematical_property() -> None:
    """真の事後確率 q を報告したときのみ期待スコアが最大化される厳密適格性を数値実証する。"""
    # 真の事象分布 q = [0.70, 0.20, 0.10]
    q = torch.tensor([0.70, 0.20, 0.10], dtype=torch.float32)
    num_classes = 3
    op_mask = torch.ones((1, num_classes), dtype=torch.bool)

    # 1. 正直な申告 (p = q)
    p_honest = q.unsqueeze(0)
    # 2. 過信申告 (特定クラスを過剰に誇張)
    p_over = torch.tensor([[0.95, 0.05, 0.00]], dtype=torch.float32)
    # 3. 過小評価 / 無情報申告 (一様分布)
    p_under = torch.full((1, num_classes), 1.0 / num_classes, dtype=torch.float32)

    def calc_expected_score(p_cand: torch.Tensor, score_fn: Any) -> float:
        exp_val = 0.0
        for y_idx in range(num_classes):
            lbl = torch.tensor([y_idx], dtype=torch.long)
            s_val = score_fn(p_cand, lbl, op_mask).item()
            exp_val += float(q[y_idx].item()) * s_val
        return exp_val

    # Log Score (非正規化報酬)
    exp_log_honest = calc_expected_score(
        p_honest, lambda p, y, m: compute_bounded_log_score(p, y, m, normalize=False)
    )
    exp_log_over = calc_expected_score(
        p_over, lambda p, y, m: compute_bounded_log_score(p, y, m, normalize=False)
    )
    exp_log_under = calc_expected_score(
        p_under, lambda p, y, m: compute_bounded_log_score(p, y, m, normalize=False)
    )
    assert exp_log_honest > exp_log_over
    assert exp_log_honest > exp_log_under

    # Spherical Score
    exp_sph_honest = calc_expected_score(p_honest, compute_spherical_score)
    exp_sph_over = calc_expected_score(p_over, compute_spherical_score)
    exp_sph_under = calc_expected_score(p_under, compute_spherical_score)
    assert exp_sph_honest > exp_sph_over
    assert exp_sph_honest > exp_sph_under

    # Brier Reward (非正規化報酬)
    exp_brier_honest = calc_expected_score(
        p_honest, lambda p, y, m: compute_brier_reward(p, y, m, normalize=False)
    )
    exp_brier_over = calc_expected_score(
        p_over, lambda p, y, m: compute_brier_reward(p, y, m, normalize=False)
    )
    exp_brier_under = calc_expected_score(
        p_under, lambda p, y, m: compute_brier_reward(p, y, m, normalize=False)
    )
    assert exp_brier_honest > exp_brier_over
    assert exp_brier_honest > exp_brier_under

    # RPS Reward
    exp_rps_honest = calc_expected_score(p_honest, compute_rps_reward)
    exp_rps_over = calc_expected_score(p_over, compute_rps_reward)
    exp_rps_under = calc_expected_score(p_under, compute_rps_reward)
    assert exp_rps_honest > exp_rps_over
    assert exp_rps_honest > exp_rps_under


def test_composite_routing() -> None:
    """質問タイプに応じた動的ルーティング (Choice vs Score) を検証する。"""
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

    expected_0 = 0.6 * s_log[0] + 0.4 * s_sph[0]
    assert pytest.approx(comp[0].item(), abs=1e-5) == expected_0.item()

    expected_1 = 0.8 * s_rps[1] + 0.2 * s_log[1]
    assert pytest.approx(comp[1].item(), abs=1e-5) == expected_1.item()


def test_proper_scoring_engine_dispatch() -> None:
    """ProperScoringEngine のマルチタスク動的ディスパッチを検証する。"""
    probs = torch.tensor(
        [
            [0.1, 0.8, 0.1],  # Choice
            [0.2, 0.6, 0.2],  # Score
            [0.05, 0.95, 0.0],  # Noul
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 1], dtype=torch.long)
    op_mask = torch.ones((3, 3), dtype=torch.bool)
    q_types = [QuestionType.CHOICE, QuestionType.SCORE, QuestionType.NOUL]

    cfg = ScoringConfig(alpha=0.5, beta=0.7, brier_weight=0.2)
    engine = ProperScoringEngine(cfg)
    rewards = engine.compute_reward(probs, labels, op_mask, question_types=q_types)

    assert "composite" in rewards
    assert "s_brier" in rewards
    assert rewards["composite"].shape == (3,)
    assert 0.0 <= rewards["composite"][0].item() <= 1.0
    assert 0.0 <= rewards["composite"][1].item() <= 1.0
    assert 0.0 <= rewards["composite"][2].item() <= 1.0


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
        brier_weight=0.15,
        min_log_score=-10.0,
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
            [10.0, -10.0],
            [0.1, 0.0],
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
    assert "mean_brier" in summary
    assert "by_type" in summary
    assert "choice" in summary["by_type"]
    assert "noul" in summary["by_type"]
    assert "by_confidence" in summary
