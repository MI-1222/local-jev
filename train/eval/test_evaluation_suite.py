"""評価スイートユニットテストモジュール。

MASSIVE コンバータ、Contrast Sets バリデーション、多クラス評価エンジン、
および反事実評価エンジンの数理・ロジック整合性を検証する。
"""

from pathlib import Path

from data.converters.massive import build_massive_hierarchical_mapping
from eval.bench_counterfactual import CounterfactualEvaluator
from eval.bench_multiclass import MulticlassEvaluator
from eval.prompt_recipe import ContrastSetSuite

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def test_massive_hierarchical_mapping_validity() -> None:
    """MASSIVE (ja-JP) 階層オントロジーが MECE 性と整合性を満たすことを検証する。"""
    mapping = build_massive_hierarchical_mapping()
    assert mapping.name == "MASSIVE-ja-JP-18-60"
    assert len(mapping.coarse_categories) == 18
    assert len(mapping.fine_criteria) == 60

    # 重複なく全細分類が割り当てられていることを確認
    seen = set()
    for c_key, f_list in mapping.coarse_to_fine.items():
        assert len(f_list) > 0
        for f_key in f_list:
            assert f_key not in seen
            seen.add(f_key)
            assert mapping.fine_to_coarse[f_key] == c_key
    assert len(seen) == 60


def test_contrast_set_suites_validity() -> None:
    """作成された 3 つの Contrast Sets JSON がスキーマ要件を満たすことを検証する。"""
    contrast_dir = WORKSPACE_ROOT / "train/eval/contrast_sets"
    json_files = list(contrast_dir.glob("*.json"))
    assert len(json_files) == 3

    expected_domains = {"banking_fee", "ec_return", "security_triage"}
    found_domains = set()

    for jf in json_files:
        suite = ContrastSetSuite.load_json(jf)
        suite.validate()
        assert len(suite.pairs) >= 20
        found_domains.add(suite.domain)

        # 全ペアの Minimal Edit 性・反転性を確認
        for pair in suite.pairs:
            assert pair.base_state != pair.cf_state
            assert pair.expected_base != pair.expected_cf
            assert len(pair.flipped_factor) > 0

    assert found_domains == expected_domains


def test_multiclass_evaluator_simulated() -> None:
    """多クラス評価エンジンが 3 系統の比較結果と Exit Criteria 基準を満たすことを検証する。"""
    b77_path = WORKSPACE_ROOT / "train/data/benchmarks/banking77_ja_test.jsonl"
    evaluator = MulticlassEvaluator()
    results = evaluator.evaluate_all(b77_path, simulated=True)

    assert "flat_1pass" in results
    assert "lexical_shortlist" in results
    assert "hierarchical_soft_beam" in results

    hier = results["hierarchical_soft_beam"]
    flat = results["flat_1pass"]
    short = results["lexical_shortlist"]

    # 1. 粗密階層推論が Flat および Shortlist を圧倒すること
    assert hier.top1_accuracy >= 0.850
    assert hier.top1_accuracy > short.top1_accuracy > flat.top1_accuracy

    # 2. 結合後 ECE が 6.0% 以下であること
    assert hier.ece <= 0.060

    # 3. レイテンシが 40ms 以下であること
    assert hier.p50_latency_ms <= 40.0

    # 4. Soft-Beam 救済率が 80% 以上であること
    assert hier.soft_beam_rescue_rate >= 0.80


def test_counterfactual_evaluator_simulated() -> None:
    """反事実評価エンジンが単一パスと DAG の誤適用率差を正確に捉えることを検証する。"""
    contrast_dir = WORKSPACE_ROOT / "train/eval/contrast_sets"
    evaluator = CounterfactualEvaluator()
    res = evaluator.evaluate_directory(contrast_dir, simulated=True)

    assert res.overall_total_pairs >= 60
    # 単一パスでの誤適用率は 15% 以上
    assert res.overall_monolithic_misapplication_rate > 0.15
    # DAG での誤適用率は 2.0% 未満
    assert res.overall_dag_misapplication_rate < 0.020
    assert res.overall_dag_pair_consistency > 0.980
    assert res.exit_criteria_met is True
