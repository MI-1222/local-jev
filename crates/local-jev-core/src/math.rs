//! # 決定プリミティブ数理モジュール
//!
//! 確率較正、ソフトマックス、エントロピー確信度、およびスコア期待値の計算ユーティリティを提供する。

use crate::error::{CoreError, Result};

/// ロジット列から温度付きソフトマックス確率分布を計算する。
pub fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if logits.is_empty() {
        return Err(CoreError::MathError {
            message: "ロジット配列が空です。".to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の実数である必要があります: {temperature}。"),
        });
    }

    // 数値安定性のための最大値引き算(Max-subtraction)。
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scaled_exps: Vec<f64> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .collect();

    let sum_exp: f64 = scaled_exps.iter().sum();
    if sum_exp <= 0.0 || sum_exp.is_nan() {
        return Err(CoreError::MathError {
            message: "ソフトマックスの分母が不正な値になりました。".to_string(),
        });
    }

    Ok(scaled_exps.iter().map(|&e| e / sum_exp).collect())
}

/// シャノンエントロピーに基づき、0.0〜1.0 の正規化確信度(Confidence)を算出する。
///
/// 候補数 $K=1$ の場合は完全に確信しているため 1.0 を返す。
/// 一様分布のとき 0.0、単一の候補に確率が集中しているとき 1.0 となる。
pub fn normalized_entropy_confidence(probabilities: &[f64]) -> Result<f64> {
    let k = probabilities.len();
    if k == 0 {
        return Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        });
    }
    if k == 1 {
        return Ok(1.0);
    }

    // シャノンエントロピー H(p) = - Σ p_i * ln(p_i)
    let mut entropy = 0.0;
    for &p in probabilities {
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    let max_entropy = (k as f64).ln();
    let confidence = 1.0 - (entropy / max_entropy);

    // 数値誤差を考慮して 0.0〜1.0 にクランプする。
    Ok(confidence.clamp(0.0, 1.0))
}

/// Score 型の確率分布から加重平均スコア実数値を算出する。
///
/// 各段階レベル $k \in \{0, \dots, M-1\}$ に対して $\text{Score} = \sum k \cdot p_k$ を計算する。
pub fn expected_score(probabilities: &[f64]) -> Result<f64> {
    if probabilities.len() < 2 {
        return Err(CoreError::MathError {
            message: "Score の確率分布は 2 段階以上必要です。".to_string(),
        });
    }

    let mut score = 0.0;
    for (k, &p) in probabilities.iter().enumerate() {
        score += (k as f64) * p;
    }

    Ok(score)
}

/// Noul 型のシグモイド真実確率値を算出する。
///
/// 真と偽のロジット差分に基づき、$P(\text{true}) = \sigma((z_{\text{true}} - z_{\text{false}}) / \tau)$ を計算する。
pub fn noul_probability(logit_true: f64, logit_false: f64, temperature: f64) -> Result<f64> {
    if temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の実数である必要があります: {temperature}。"),
        });
    }

    let diff = (logit_true - logit_false) / temperature;
    let prob = 1.0 / (1.0 + (-diff).exp());
    Ok(prob.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![2.0, 1.0, 0.1];
        let probs = softmax(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_normalized_entropy_confidence() {
        // 完全な一様分布 -> 確信度は 0.0
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let conf_uniform = normalized_entropy_confidence(&uniform).unwrap();
        assert!(conf_uniform < 1e-6);

        // 完全にピークした分布 -> 確信度は 1.0
        let peaked = vec![1.0, 0.0, 0.0, 0.0];
        let conf_peaked = normalized_entropy_confidence(&peaked).unwrap();
        assert!((conf_peaked - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_expected_score() {
        let probs = vec![0.1, 0.2, 0.7]; // k=0, 1, 2
        let score = expected_score(&probs).unwrap();
        let expected = 0.0 * 0.1 + 1.0 * 0.2 + 2.0 * 0.7;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_noul_probability() {
        let prob_equal = noul_probability(1.0, 1.0, 1.0).unwrap();
        assert!((prob_equal - 0.5).abs() < 1e-6);

        let prob_high = noul_probability(10.0, 0.0, 1.0).unwrap();
        assert!(prob_high > 0.99);
    }
}
