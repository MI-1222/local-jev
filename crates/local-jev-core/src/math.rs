//! # 決定プリミティブ数理モジュール
//!
//! 確率較正、ソフトマックス、エントロピー確信度、およびスコア期待値の計算ユーティリティを提供する。

use crate::error::{CoreError, Result};

/// ロジット列から温度付きソフトマックス確率分布を計算し、指定されたバッファへ書き込む。
///
/// # 概要
/// - 推論ランタイム等の高スループット処理向けに、ヒープ割り当てを行わずに確率分布を算出する。
/// - 数値安定性を担保するため、最大値減算(Max-subtraction)および温度除算を適用する。
/// - 算出された確率値は浮動小数点演算の累積丸め誤差を考慮し、`[0.0, 1.0]` の閉区間にクランプされる。
///
/// ```mermaid
/// flowchart TD
///     Start["入力: logits, out, temperature"] --> Val{"バリデーション<br>(空配列 / 長さ不一致 / 非有限値 / tau <= 0)"}
///     Val -- 不正 --> Err["CoreError::MathError 返却"]
///     Val -- 正常 --> KCheck{"候補数 K == 1 ?"}
///     KCheck -- Yes --> Single["out[0] = 1.0 (ショートサーキット)"]
///     KCheck -- No --> MaxFind["Pass 1: 最大値 z_max 探索<br>(同時 NaN/Inf 検出)"]
///     MaxFind --> ExpSum["Pass 2: 指数計算 & 累積和<br>exp((z_i - z_max) / tau_eff)"]
///     ExpSum --> Norm["Pass 3: 分母除算正規化 & clamp(0.0, 1.0)"]
///     Single --> Done["Ok(()) 返却"]
///     Norm --> Done
/// ```
///
/// # 数理仕様
/// $$p_i = \frac{\exp\left(\frac{z_i - z_{\max}}{\tau_{\text{eff}}}\right)}{\sum_{j=1}^K \exp\left(\frac{z_j - z_{\max}}{\tau_{\text{eff}}}\right)}, \quad z_{\max} = \max_{1 \le j \le K} z_j$$
/// ここで、有効温度 $\tau_{\text{eff}} = \max(\tau, 10^{-4})$ である。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `out`: 算出された確率値を書き込む出力先バッファスライス(`f64`)。`logits` と同一の要素数が必要。
/// - `temperature`: 較正温度パラメータ $\tau$。正の有限実数。
///
/// # 戻り値
/// - 成功時は `Ok(())`、入力異常時は `CoreError::MathError` を返却する。
#[aquamarine::aquamarine]
pub fn softmax_into(logits: &[f64], out: &mut [f64], temperature: f64) -> Result<()> {
    if logits.is_empty() {
        return Err(CoreError::MathError {
            message: "ロジット配列が空です。".to_string(),
        });
    }

    if logits.len() != out.len() {
        return Err(CoreError::MathError {
            message: format!(
                "出力バッファ長({out_len})がロジット長({logits_len})と一致しません。",
                out_len = out.len(),
                logits_len = logits.len()
            ),
        });
    }

    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の有限実数である必要があります: {temperature}。"),
        });
    }

    // 単一候補(K=1)のエッジケースは指数計算をスキップして即座に 1.0 を設定する。
    if logits.len() == 1 {
        if !logits[0].is_finite() {
            return Err(CoreError::MathError {
                message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        out[0] = 1.0;
        return Ok(());
    }

    // Python 側の較正評価器(evaluator.py)と平仄を合わせ、極小温度によるオーバーフローを防ぐ安全下限クランプ。
    let effective_tau = temperature.max(1e-4);

    // Pass 1: 最大値探索および非有限値(NaN, Inf)の厳密な検出。
    // IEEE 754 の f64::max による NaN すり抜けを防ぐため、明示的な走査を行う。
    let mut max_logit = f64::NEG_INFINITY;
    for &logit in logits {
        if !logit.is_finite() {
            return Err(CoreError::MathError {
                message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if logit > max_logit {
            max_logit = logit;
        }
    }

    // Pass 2: 最大値減算後の温度スケーリング、指数計算、および総和累積。
    // (logit - max_logit) <= 0.0 であるため、指数関数の引数は常に (-inf, 0.0] となりオーバーフローしない。
    // かつ最大値要素自身は exp(0.0) = 1.0 となるため、sum_exp >= 1.0 が数学的に保証される。
    let mut sum_exp = 0.0;
    for (i, &logit) in logits.iter().enumerate() {
        let scaled = (logit - max_logit) / effective_tau;
        let exp_val = scaled.exp();
        out[i] = exp_val;
        sum_exp += exp_val;
    }

    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return Err(CoreError::MathError {
            message: "ソフトマックスの分母が不正な値になりました。".to_string(),
        });
    }

    // Pass 3: 分母除算による正規化と丸め誤差対策のクランプ。
    for val in out.iter_mut() {
        *val = (*val / sum_exp).clamp(0.0, 1.0);
    }

    Ok(())
}

/// ロジット列から温度付きソフトマックス確率分布を計算し、新しいベクタとして返却する。
///
/// # 概要
/// - 内部でゼロアロケーション版関数 [`softmax_into`] を呼び出す。
/// - 返却される確率ベクタの総和はほぼ 1.0 となり、各要素は `[0.0, 1.0]` に収まる。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `temperature`: 較正温度パラメータ $\tau$。正の有限実数。
///
/// # 戻り値
/// - 成功時は計算された確率分布ベクタ(`Vec<f64>`)、異常時は `CoreError::MathError`。
#[aquamarine::aquamarine]
pub fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    let mut probs = vec![0.0; logits.len()];
    softmax_into(logits, &mut probs, temperature)?;
    Ok(probs)
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
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の実数である必要があります: {temperature}。"),
        });
    }

    let effective_tau = temperature.max(1e-4);
    let diff = (logit_true - logit_false) / effective_tau;
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
    fn test_softmax_into_basic() {
        let logits = vec![2.0, 1.0, 0.1];
        let mut probs = vec![0.0; 3];
        softmax_into(&logits, &mut probs, 1.0).unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_extreme_logits() {
        // 巨大なロジット差でもオーバーフロー・アンダーフローでクラッシュしないことを検証する。
        let logits = vec![10000.0, 0.0, -10000.0];
        let probs = softmax(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 1.0).abs() < 1e-6);
        assert_eq!(probs[1], 0.0);
        assert_eq!(probs[2], 0.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_identical_logits() {
        // 全要素が同一ロジットの場合、厳密に一様分布となることを検証する。
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let probs = softmax(&logits, 1.0).unwrap();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_single_candidate() {
        // K=1 のエッジケースにおいて、即座に 1.0 が返却されることを検証する。
        let logits = vec![std::f64::consts::PI];
        let probs = softmax(&logits, 0.5).unwrap();
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0], 1.0);

        let mut out = vec![0.0];
        softmax_into(&logits, &mut out, 2.0).unwrap();
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn test_softmax_temperature_limits() {
        let logits = vec![2.0, 1.0, 0.0];

        // 高温 (tau -> inf): 一様分布へ平滑化される。
        let probs_high = softmax(&logits, 1000.0).unwrap();
        for &p in &probs_high {
            assert!((p - 1.0 / 3.0).abs() < 1e-2);
        }

        // 低温 (tau -> 0): 最大値へ確率が 1.0 集中する (One-hot)。
        let probs_low = softmax(&logits, 1e-4).unwrap();
        assert!((probs_low[0] - 1.0).abs() < 1e-6);
        assert!(probs_low[1] < 1e-6);
        assert!(probs_low[2] < 1e-6);

        // 安全下限クランプ (tau = 1e-9): ゼロ除算・オーバーフローなく計算できる。
        let probs_micro = softmax(&logits, 1e-9).unwrap();
        assert!((probs_micro[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_errors() {
        // 空ロジット
        let empty_logits: Vec<f64> = vec![];
        assert!(softmax(&empty_logits, 1.0).is_err());

        // 出力バッファ長不一致
        let logits = vec![1.0, 2.0];
        let mut short_out = vec![0.0];
        assert!(softmax_into(&logits, &mut short_out, 1.0).is_err());

        // 不正な温度 (tau <= 0, NaN, Inf)
        assert!(softmax(&logits, 0.0).is_err());
        assert!(softmax(&logits, -1.0).is_err());
        assert!(softmax(&logits, f64::NAN).is_err());
        assert!(softmax(&logits, f64::INFINITY).is_err());

        // ロジットに NaN または Inf
        let nan_logits = vec![1.0, f64::NAN, 3.0];
        assert!(softmax(&nan_logits, 1.0).is_err());
        let inf_logits = vec![1.0, f64::INFINITY, 3.0];
        assert!(softmax(&inf_logits, 1.0).is_err());
        let neg_inf_logits = vec![f64::NEG_INFINITY, 1.0];
        assert!(softmax(&neg_inf_logits, 1.0).is_err());
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

        // 不正な温度
        assert!(noul_probability(1.0, 0.0, 0.0).is_err());
        assert!(noul_probability(1.0, 0.0, -1.0).is_err());
        assert!(noul_probability(1.0, 0.0, f64::NAN).is_err());
    }
}
