//! # 不確実性・確信度数理のパリティテスト (`uncertainty_parity.rs`)
//!
//! ロードマップ 3.1「候補数非依存の不確実性数理の導入」に基づき、
//! Rust 実装 (`math.rs`) と Python 実装 (`contract.py` / `evaluator.py`) の
//! 数値計算パリティ（絶対誤差 Atol <= 1e-6）および特異点・境界値の整合性を検証する。

use local_jev_core::math::{composite_confidence, normalized_entropy, top_margin};

/// 確率分布に対する期待値定義。
struct TestCase {
    name: &'static str,
    probs: Vec<f64>,
    expected_h_norm: f64,
    expected_margin: f64,
    expected_composite: f64,
}

#[test]
fn test_uncertainty_parity_across_candidate_scales() {
    let cases = vec![
        // 1. K=1 特異点
        TestCase {
            name: "k1_singularity",
            probs: vec![1.0],
            expected_h_norm: 0.0,
            expected_margin: 1.0,
            expected_composite: 1.0,
        },
        // 2. K=2 一様分布
        TestCase {
            name: "k2_uniform",
            probs: vec![0.5, 0.5],
            expected_h_norm: 1.0,
            expected_margin: 0.0,
            expected_composite: 0.0,
        },
        // 3. K=2 偏り分布
        // p = [0.8, 0.2]
        // H(p) = -(0.8*ln 0.8 + 0.2*ln 0.2) ≈ 0.5004024235
        // H_norm = 0.5004024235 / ln(2) ≈ 0.7219280949
        // Margin = 0.8 - 0.2 = 0.6
        // S_conf = (1 - 0.7219280949) * 0.6 ≈ 0.166843143
        TestCase {
            name: "k2_biased",
            probs: vec![0.8, 0.2],
            expected_h_norm: 0.7219280948873623,
            expected_margin: 0.6,
            expected_composite: (1.0 - 0.7219280948873623) * 0.6,
        },
        // 4. K=3 Margin Collapse (激しい拮抗)
        // p = [0.49, 0.49, 0.02]
        // H(p) = -(2 * 0.49 * ln 0.49 + 0.02 * ln 0.02) ≈ 0.77661664
        // H_norm = 0.77661664 / ln(3) ≈ 0.70689408
        // Margin = 0.49 - 0.49 = 0.0
        // S_conf = 0.0
        TestCase {
            name: "k3_margin_collapse",
            probs: vec![0.49, 0.49, 0.02],
            expected_h_norm: (-2.0 * 0.49 * (0.49_f64).ln() - 0.02 * (0.02_f64).ln())
                / (3.0_f64).ln(),
            expected_margin: 0.0,
            expected_composite: 0.0,
        },
        // 5. K=5 高確信分布
        // p = [0.88, 0.04, 0.03, 0.03, 0.02]
        TestCase {
            name: "k5_high_confidence",
            probs: vec![0.88, 0.04, 0.03, 0.03, 0.02],
            expected_h_norm: (-(0.88 * (0.88_f64).ln()
                + 0.04 * (0.04_f64).ln()
                + 0.03 * (0.03_f64).ln()
                + 0.03 * (0.03_f64).ln()
                + 0.02 * (0.02_f64).ln()))
                / (5.0_f64).ln(),
            expected_margin: 0.88 - 0.04,
            expected_composite: (1.0
                - (-(0.88 * (0.88_f64).ln()
                    + 0.04 * (0.04_f64).ln()
                    + 0.03 * (0.03_f64).ln()
                    + 0.03 * (0.03_f64).ln()
                    + 0.02 * (0.02_f64).ln()))
                    / (5.0_f64).ln())
                * (0.88 - 0.04),
        },
        // 6. K=10 ワンホット極限
        TestCase {
            name: "k10_one_hot",
            probs: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            expected_h_norm: 0.0,
            expected_margin: 1.0,
            expected_composite: 1.0,
        },
        // 7. K=50 大規模候補数での一様分布
        TestCase {
            name: "k50_uniform",
            probs: vec![0.02; 50],
            expected_h_norm: 1.0,
            expected_margin: 0.0,
            expected_composite: 0.0,
        },
    ];

    for tc in cases {
        let actual_h = normalized_entropy(&tc.probs).expect(tc.name);
        let actual_m = top_margin(&tc.probs).expect(tc.name);
        let actual_s = composite_confidence(&tc.probs).expect(tc.name);

        assert!(
            (actual_h - tc.expected_h_norm).abs() < 1e-6,
            "[{}] H_norm mismatch: actual={}, expected={}",
            tc.name,
            actual_h,
            tc.expected_h_norm
        );
        assert!(
            (actual_m - tc.expected_margin).abs() < 1e-6,
            "[{}] Margin mismatch: actual={}, expected={}",
            tc.name,
            actual_m,
            tc.expected_margin
        );
        assert!(
            (actual_s - tc.expected_composite).abs() < 1e-6,
            "[{}] Composite confidence mismatch: actual={}, expected={}",
            tc.name,
            actual_s,
            tc.expected_composite
        );
    }
}
