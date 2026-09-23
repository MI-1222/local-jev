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
pub fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    let mut probs: Vec<f64> = vec![0.0; logits.len()];
    softmax_into(logits, &mut probs, temperature)?;
    Ok(probs)
}

/// 確率分布から候補数 $K$ に非依存な正規化シャノンエントロピー $H_{\text{norm}}(p) \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$H_{\text{norm}}(p) = \begin{cases} 0.0 & (K = 1) \\ \frac{-\sum_{k=1}^K p_k \ln p_k}{\ln K} & (K \ge 2) \end{cases}$$
///
/// - $p_k \le 0.0$ の要素は情報量 $0 \ln 0 = 0$ として対数計算をスキップする。
/// - $K=1$ の特異点ではゼロ除算を回避し、決定的な $0.0$(不確実性なし)を即座に返却する。
/// - 戻り値は浮動小数点誤差を考慮して `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `probabilities`: 各候補の確率分布スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた正規化エントロピー、不正時は `CoreError::MathError`。
pub fn normalized_entropy(probabilities: &[f64]) -> Result<f64> {
    let k = probabilities.len();
    if k == 0 {
        return Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        });
    }
    if k == 1 {
        if !probabilities[0].is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if probabilities[0] < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {}。", probabilities[0]),
            });
        }
        return Ok(0.0);
    }

    let mut entropy = 0.0;
    for &p in probabilities {
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
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
    if max_entropy <= 0.0 {
        return Ok(0.0);
    }

    let h_norm = entropy / max_entropy;
    Ok(h_norm.clamp(0.0, 1.0))
}

/// 確率分布から上位2候補の確率差(Top-Margin) $M(p) \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$M(p) = \begin{cases} 1.0 & (K = 1) \\ p_{(1)} - p_{(2)} & (K \ge 2) \end{cases}$$
///
/// - $O(K)$ 単一パス走査により、ヒープメモリ割り当て(アロケーション)ゼロで計算する。
/// - $K=1$ の特異点では $1.0$(他候補なし・完全なマージン)を即座に返却する。
/// - 同率1位(タイ)の場合は $M(p) = 0.0$ となる。
/// - 戻り値は `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `probabilities`: 各候補の確率分布スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた Top-Margin、不正時は `CoreError::MathError`。
pub fn top_margin(probabilities: &[f64]) -> Result<f64> {
    let k = probabilities.len();
    if k == 0 {
        return Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        });
    }
    if k == 1 {
        if !probabilities[0].is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if probabilities[0] < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {}。", probabilities[0]),
            });
        }
        return Ok(1.0);
    }

    let mut max1 = f64::NEG_INFINITY;
    let mut max2 = f64::NEG_INFINITY;

    for &p in probabilities {
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }

        if p > max1 {
            max2 = max1;
            max1 = p;
        } else if p > max2 {
            max2 = p;
        }
    }

    let margin = max1 - max2;
    Ok(margin.clamp(0.0, 1.0))
}

/// 確率分布から、正規化エントロピーと Top-Margin を統合した複合確信度スコア $S_{\text{confidence}} \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$S_{\text{confidence}} = (1.0 - H_{\text{norm}}(p)) \times M(p)$$
///
/// - 分布全体が鋭利であり(平坦度 $H_{\text{norm}}$ が小)、かつ第1位と第2位の差分(マージン $M(p)$)が明瞭な場合にのみ
///   高い確信度を出力する。
/// - $K=1$ の特異点では $1.0$ を返却する。
/// - 同率タイブレーク時はマージンが $0.0$ となるため、複合確信度も厳密に $0.0$ に収束する。
/// - 戻り値は `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `probabilities`: 各候補の確率分布スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた複合確信度スコア、不正時は `CoreError::MathError`。
pub fn composite_confidence(probabilities: &[f64]) -> Result<f64> {
    let k = probabilities.len();
    if k == 0 {
        return Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        });
    }
    if k == 1 {
        if !probabilities[0].is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if probabilities[0] < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {}。", probabilities[0]),
            });
        }
        return Ok(1.0);
    }

    let h_norm = normalized_entropy(probabilities)?;
    let margin = top_margin(probabilities)?;

    let score = (1.0 - h_norm) * margin;
    Ok(score.clamp(0.0, 1.0))
}

/// シャノンエントロピーに基づき、0.0〜1.0 の正規化確信度(Confidence)を算出する。
///
/// 候補数 $K=1$ の場合は完全に確信しているため 1.0 を返す。
/// 一様分布のとき 0.0、単一の候補に確率が集中しているとき 1.0 となる。
///
/// # 備考
/// 本関数は後方互換性のために維持されており、$1.0 - H_{\text{norm}}(p)$ を返却する。
pub fn normalized_entropy_confidence(probabilities: &[f64]) -> Result<f64> {
    let h_norm = normalized_entropy(probabilities)?;
    Ok((1.0 - h_norm).clamp(0.0, 1.0))
}

/// スライス内から最大値を持つ要素のインデックスを取得する。
///
/// # 概要
/// - 決定論的タイブレーク規則として、同一の最大値が複数存在する場合は「最小のインデックス(先勝ち)」を確定的に採択する。
/// - 配列内に `NaN` が含まれている場合は `CoreError::MathError` を返却する。
///
/// # 引数
/// - `values`: 探索対象の実数値スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は最大値のインデックス(`usize`)、配列が空または非有限値を含む場合は `CoreError::MathError`。
pub fn argmax(values: &[f64]) -> Result<usize> {
    if values.is_empty() {
        return Err(CoreError::MathError {
            message: "探索対象のスライスが空です。".to_string(),
        });
    }

    let mut best_index = 0;
    let mut max_val = f64::NEG_INFINITY;

    for (i, &val) in values.iter().enumerate() {
        if !val.is_finite() {
            return Err(CoreError::MathError {
                message: "スライス内に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        // 厳密な不等号(val > max_val)を用いることで、同率値の場合は先頭インデックスを維持する。
        if val > max_val {
            max_val = val;
            best_index = i;
        }
    }

    Ok(best_index)
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
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }
        score += (k as f64) * p;
    }

    Ok(score)
}

/// Score 型の確率分布からスコアの事後分散 $\text{Var}\[S\]$ を算出する。
///
/// 段階レベル $k \in \{0, \dots, M-1\}$ に対し、加重平均期待値 $\hat{s} = E\[S\]$ を用いて
/// $\text{Var}\[S\] = \sum_{k=0}^{M-1} (k - \hat{s})^2 p_k$ を計算する。
pub fn score_variance(probabilities: &[f64]) -> Result<f64> {
    let mean = expected_score(probabilities)?;

    let mut variance = 0.0;
    for (k, &p) in probabilities.iter().enumerate() {
        let diff = (k as f64) - mean;
        variance += diff * diff * p;
    }

    // 浮動小数点誤差による負数の発生を防止する。
    Ok(variance.max(0.0))
}

/// Score 型の確率分布から、最大可能分散で正規化した確信度 $C_{\text{var}} \in [0.0, 1.0]$ を算出する。
///
/// # 概要
/// - 順序尺度(Score)においてカテゴリカルエントロピーを用いると、両極端への分裂(高分散・バイモーダル)と
///   隣接段階への分裂(低分散)を区別できず、誤った過信を招く問題がある。
/// - 本関数では事後分散 $\text{Var}\[S\]$ を計算し、$M$ 段階における理論上の最大分散
///   $V_{\max} = \frac{(M-1)^2}{4}$(両極端に 0.5 ずつ配分された二峰性分布)で正規化する。
///
/// # 数理仕様
/// $$C_{\text{var}} = 1.0 - \frac{\text{Var}\[S\]}{V_{\max}} = 1.0 - \frac{4 \cdot \text{Var}\[S\]}{(M-1)^2}$$
///
/// # 引数
/// - `probabilities`: 各段階レベルの確率分布スライス($M \ge 2$)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` の閉区間にクランプされた正規化確信度、不正時は `CoreError::MathError`。
pub fn normalized_variance_confidence(probabilities: &[f64]) -> Result<f64> {
    let m = probabilities.len();
    if m < 2 {
        return Err(CoreError::MathError {
            message: "Score の確信度計算には 2 段階以上の確率分布が必要です。".to_string(),
        });
    }

    let var = score_variance(probabilities)?;
    let max_var = ((m - 1) as f64).powi(2) / 4.0;

    if max_var <= 0.0 {
        return Ok(1.0);
    }

    let confidence = 1.0 - (var / max_var);
    Ok(confidence.clamp(0.0, 1.0))
}

/// Noul 型の言明真実確率値 $P(\text{true}) \in [0.0, 1.0]$ を算出する。
///
/// # 概要
/// - ロジット差分 $x = (z_{\text{true}} - z_{\text{false}}) / \tau_{\text{eff}}$ に対し、
///   符号分岐型シグモイド $\sigma(x)$ を用いることで指数オーバーフローおよびアンダーフローを完全に排除する。
/// - $x \ge 0$ のときは $\frac{1}{1 + \exp(-x)}$、$x < 0$ のときは $\frac{\exp(x)}{1 + \exp(x)}$ を計算する。
///
/// # 引数
/// - `logit_true`: 真(True)候補の生ロジット。
/// - `logit_false`: 偽(False)候補の生ロジット。
/// - `temperature`: 較正温度パラメータ $\tau$。正の有限実数。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた真実確率、不正時は `CoreError::MathError`。
pub fn noul_probability(logit_true: f64, logit_false: f64, temperature: f64) -> Result<f64> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の実数である必要があります: {temperature}。"),
        });
    }

    if !logit_true.is_finite() || !logit_false.is_finite() {
        return Err(CoreError::MathError {
            message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
        });
    }

    let effective_tau = temperature.max(1e-4);
    let diff = (logit_true - logit_false) / effective_tau;

    let prob = if diff >= 0.0 {
        1.0 / (1.0 + (-diff).exp())
    } else {
        let exp_diff = diff.exp();
        exp_diff / (1.0 + exp_diff)
    };

    Ok(prob.clamp(0.0, 1.0))
}

/// Noul 型の真実確率から、決定の尖り度に基づく正規化確信度 $C_{\text{noul}} \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$C_{\text{noul}} = |2 \cdot P(\text{true}) - 1.0|$$
/// - $P(\text{true}) = 0.5$(完全な迷い)のとき 0.0。
/// - $P(\text{true}) = 1.0$ または $0.0$(完全な確信)のとき 1.0。
pub fn noul_confidence(probability: f64) -> Result<f64> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(CoreError::MathError {
            message: format!("確率は [0.0, 1.0] の有限実数である必要があります: {probability}。"),
        });
    }

    let conf = (2.0 * probability - 1.0).abs();
    Ok(conf.clamp(0.0, 1.0))
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

    #[test]
    fn test_argmax() {
        // 基本的な最大値探索
        let values = vec![0.1, 0.8, 0.4];
        assert_eq!(argmax(&values).unwrap(), 1);

        // 決定論的タイブレーク(同率最大値の場合は先頭インデックスを採択)
        let tied = vec![0.5, 0.9, 0.9, 0.2];
        assert_eq!(argmax(&tied).unwrap(), 1);

        // 先頭が最大値の場合
        let first_max = vec![10.0, 5.0, 2.0];
        assert_eq!(argmax(&first_max).unwrap(), 0);

        // 末尾が最大値の場合
        let last_max = vec![1.0, 2.0, 3.0];
        assert_eq!(argmax(&last_max).unwrap(), 2);

        // 異常系: 空スライス
        let empty: Vec<f64> = vec![];
        assert!(argmax(&empty).is_err());

        // 異常系: NaN または Inf
        let nan_vals = vec![1.0, f64::NAN, 2.0];
        assert!(argmax(&nan_vals).is_err());
        let inf_vals = vec![1.0, f64::INFINITY];
        assert!(argmax(&inf_vals).is_err());
    }

    #[test]
    fn test_score_variance_and_confidence() {
        // 1. ワンホット分布(単一レベルに集中): 分散 0.0、確信度 1.0
        let one_hot = vec![0.0, 0.0, 1.0, 0.0];
        let var_one_hot = score_variance(&one_hot).unwrap();
        let conf_one_hot = normalized_variance_confidence(&one_hot).unwrap();
        assert!(var_one_hot < 1e-6);
        assert!((conf_one_hot - 1.0).abs() < 1e-6);

        // 2. 両極端に分裂した二峰性分布 [0.5, 0.0, 0.0, 0.5] (M=4)
        // 最大可能分散 V_max = (4-1)^2 / 4 = 2.25
        // Mean = 0 * 0.5 + 3 * 0.5 = 1.5
        // Var = (0 - 1.5)^2 * 0.5 + (3 - 1.5)^2 * 0.5 = 2.25
        // 確信度 C_var = 1.0 - 2.25 / 2.25 = 0.0
        let bimodal = vec![0.5, 0.0, 0.0, 0.5];
        let var_bimodal = score_variance(&bimodal).unwrap();
        let conf_bimodal = normalized_variance_confidence(&bimodal).unwrap();
        assert!((var_bimodal - 2.25).abs() < 1e-6);
        assert!(conf_bimodal < 1e-6);

        // 3. 隣接段階に分裂した低分散分布 [0.0, 0.5, 0.5, 0.0] (M=4)
        // Mean = 1 * 0.5 + 2 * 0.5 = 1.5
        // Var = (1 - 1.5)^2 * 0.5 + (2 - 1.5)^2 * 0.5 = 0.25
        // 確信度 C_var = 1.0 - 0.25 / 2.25 = 1.0 - 1/9 ≈ 0.888889
        let adjacent = vec![0.0, 0.5, 0.5, 0.0];
        let var_adjacent = score_variance(&adjacent).unwrap();
        let conf_adjacent = normalized_variance_confidence(&adjacent).unwrap();
        assert!((var_adjacent - 0.25).abs() < 1e-6);
        assert!((conf_adjacent - (8.0 / 9.0)).abs() < 1e-6);

        // エントロピー確信度との対比検証:
        // bimodal と adjacent はカテゴリカルエントロピーが完全に同一だが、
        // 分散確信度では bimodal(0.0) と adjacent(0.889) で明確に区別される。
        let ent_bimodal = normalized_entropy_confidence(&bimodal).unwrap();
        let ent_adjacent = normalized_entropy_confidence(&adjacent).unwrap();
        assert!((ent_bimodal - ent_adjacent).abs() < 1e-6);
        assert!(conf_adjacent > conf_bimodal);

        // 異常系: 段階数が 2 未満
        let single_level = vec![1.0];
        assert!(score_variance(&single_level).is_err());
        assert!(normalized_variance_confidence(&single_level).is_err());
    }

    #[test]
    fn test_noul_stability_and_confidence() {
        // 極端なロジット差でもオーバーフロー・アンダーフローせず安定計算できることの検証
        let prob_huge_pos = noul_probability(10000.0, 0.0, 1.0).unwrap();
        assert!((prob_huge_pos - 1.0).abs() < 1e-6);

        let prob_huge_neg = noul_probability(-10000.0, 0.0, 1.0).unwrap();
        assert!(prob_huge_neg < 1e-6);

        // 確信度計算の検証
        // P = 0.5 -> 確信度 0.0
        assert!((noul_confidence(0.5).unwrap() - 0.0).abs() < 1e-6);
        // P = 1.0 -> 確信度 1.0
        assert!((noul_confidence(1.0).unwrap() - 1.0).abs() < 1e-6);
        // P = 0.0 -> 確信度 1.0
        assert!((noul_confidence(0.0).unwrap() - 1.0).abs() < 1e-6);
        // P = 0.8 -> 確信度 0.6
        assert!((noul_confidence(0.8).unwrap() - 0.6).abs() < 1e-6);

        // 異常系: 範囲外または非有限値
        assert!(noul_confidence(-0.1).is_err());
        assert!(noul_confidence(1.1).is_err());
        assert!(noul_confidence(f64::NAN).is_err());
    }

    #[test]
    fn test_softmax_non_finite_positions() {
        // 先頭、中間、末尾の各位置に非有限値(NaN, +Inf, -Inf)が注入された場合の捕捉を検証する。
        let test_cases = vec![
            vec![f64::NAN, 1.0, 2.0],
            vec![1.0, f64::NAN, 2.0],
            vec![1.0, 2.0, f64::NAN],
            vec![f64::INFINITY, 1.0, 2.0],
            vec![1.0, f64::INFINITY, 2.0],
            vec![1.0, 2.0, f64::INFINITY],
            vec![f64::NEG_INFINITY, 1.0, 2.0],
            vec![1.0, f64::NEG_INFINITY, 2.0],
            vec![1.0, 2.0, f64::NEG_INFINITY],
        ];

        for (i, case) in test_cases.iter().enumerate() {
            let res = softmax(case, 1.0);
            assert!(
                res.is_err(),
                "ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
        }

        // K=1 における非有限値の遮断を検証する。
        assert!(softmax(&[f64::NAN], 1.0).is_err());
        assert!(softmax(&[f64::INFINITY], 1.0).is_err());
        assert!(softmax(&[f64::NEG_INFINITY], 1.0).is_err());

        // 全要素が -Inf の場合、指数計算で分母が 0 になるため安全にエラー捕捉されることを検証する。
        let all_neg_inf = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert!(softmax(&all_neg_inf, 1.0).is_err());
    }

    #[test]
    fn test_softmax_mask_value_convergence() {
        // Python 側のパディングマスク値 -1e4 が混入した場合の挙動を検証する。
        // 有効ロジットに対してマスクされた位置の確率が厳密に 0.0 となり、
        // かつ有効要素間の確率比率が保たれることを確認する。
        let logits = vec![3.0, 1.0, -10000.0];
        let probs = softmax(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);

        // マスク位置は 0.0 にアンダーフローして安全に収束する。
        assert_eq!(probs[2], 0.0);

        // 有効要素の和が 1.0 になる。
        let valid_sum = probs[0] + probs[1];
        assert!((valid_sum - 1.0).abs() < 1e-6);

        // 3.0 と 1.0 の二値ソフトマックス理論値と一致する。
        let expected_ratio = (2.0_f64).exp(); // exp(3 - 1)
        let actual_ratio = probs[0] / probs[1];
        assert!((actual_ratio - expected_ratio).abs() < 1e-4);
    }

    #[test]
    fn test_argmax_non_finite_positions() {
        // argmax において先頭、中間、末尾に非有限値が含まれる場合にエラーとなることを検証する。
        let test_cases = vec![
            vec![f64::NAN, 0.5, 0.2],
            vec![0.5, f64::NAN, 0.2],
            vec![0.5, 0.2, f64::NAN],
            vec![f64::INFINITY, 0.5, 0.2],
            vec![0.5, f64::INFINITY, 0.2],
            vec![0.5, 0.2, f64::INFINITY],
            vec![f64::NEG_INFINITY, 0.5, 0.2],
            vec![0.5, f64::NEG_INFINITY, 0.2],
            vec![0.5, 0.2, f64::NEG_INFINITY],
        ];

        for (i, case) in test_cases.iter().enumerate() {
            assert!(
                argmax(case).is_err(),
                "argmax ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
        }
    }

    #[test]
    fn test_noul_non_finite_inputs() {
        // noul_probability の真偽ロジットに非有限値が渡された場合の拒絶を検証する。
        assert!(noul_probability(f64::NAN, 0.0, 1.0).is_err());
        assert!(noul_probability(0.0, f64::NAN, 1.0).is_err());
        assert!(noul_probability(f64::INFINITY, 0.0, 1.0).is_err());
        assert!(noul_probability(0.0, f64::INFINITY, 1.0).is_err());
        assert!(noul_probability(f64::NEG_INFINITY, 0.0, 1.0).is_err());
        assert!(noul_probability(0.0, f64::NEG_INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_normalized_entropy() {
        // 1. K=1 の特異点: 不確実性ゼロのため 0.0 を返却する。
        assert_eq!(normalized_entropy(&[1.0]).unwrap(), 0.0);

        // 2. 完全な一様分布: 理論値 H(p) = ln(K) により正規化エントロピーは厳密に 1.0 となる。
        let uniform2 = vec![0.5, 0.5];
        assert!((normalized_entropy(&uniform2).unwrap() - 1.0).abs() < 1e-6);

        let uniform4 = vec![0.25, 0.25, 0.25, 0.25];
        assert!((normalized_entropy(&uniform4).unwrap() - 1.0).abs() < 1e-6);

        // 3. ワンホット(完全確信)分布: H(p) = 0.0 により正規化エントロピーは 0.0 となる。
        let peaked = vec![1.0, 0.0, 0.0, 0.0];
        assert!(normalized_entropy(&peaked).unwrap() < 1e-6);

        // 4. 一般分布 (K=3)
        // p = [0.7, 0.2, 0.1]
        // H(p) = -(0.7 * ln 0.7 + 0.2 * ln 0.2 + 0.1 * ln 0.1) ≈ 0.80181855
        // H_norm = 0.80181855 / ln(3) ≈ 0.80181855 / 1.09861229 ≈ 0.7298466
        let p3 = vec![0.7, 0.2, 0.1];
        let h_norm3 = normalized_entropy(&p3).unwrap();
        assert!((h_norm3 - 0.7298466).abs() < 1e-5);

        // 5. 異常系
        assert!(normalized_entropy(&[]).is_err());
        assert!(normalized_entropy(&[-0.1, 1.1]).is_err());
        assert!(normalized_entropy(&[0.5, f64::NAN]).is_err());
        assert!(normalized_entropy(&[f64::INFINITY, 0.5]).is_err());
        assert!(normalized_entropy(&[f64::NAN]).is_err());
    }

    #[test]
    fn test_top_margin() {
        // 1. K=1 の特異点: 競合候補が存在しないため 1.0 を返却する。
        assert_eq!(top_margin(&[1.0]).unwrap(), 1.0);

        // 2. 明確な差がある場合
        let p_clear = vec![0.7, 0.2, 0.1];
        assert!((top_margin(&p_clear).unwrap() - 0.5).abs() < 1e-6);

        // 3. 同率1位(タイ)の場合: マージンは 0.0 となる。
        let p_tied = vec![0.4, 0.4, 0.2];
        assert!(top_margin(&p_tied).unwrap() < 1e-6);

        let p_all_equal = vec![0.25, 0.25, 0.25, 0.25];
        assert!(top_margin(&p_all_equal).unwrap() < 1e-6);

        // 4. ワンホット極限: マージンは 1.0 となる。
        let p_onehot = vec![1.0, 0.0, 0.0];
        assert!((top_margin(&p_onehot).unwrap() - 1.0).abs() < 1e-6);

        // 5. 最大値が末尾にある場合
        let p_last = vec![0.1, 0.2, 0.7];
        assert!((top_margin(&p_last).unwrap() - 0.5).abs() < 1e-6);

        // 6. 異常系
        assert!(top_margin(&[]).is_err());
        assert!(top_margin(&[-0.5, 1.5]).is_err());
        assert!(top_margin(&[0.5, f64::NAN]).is_err());
        assert!(top_margin(&[f64::INFINITY, 0.0]).is_err());
        assert!(top_margin(&[f64::NAN]).is_err());
    }

    #[test]
    fn test_composite_confidence() {
        // 1. K=1 の特異点: 完全に決定しているため 1.0 を返却する。
        assert_eq!(composite_confidence(&[1.0]).unwrap(), 1.0);

        // 2. ワンホット極限: H_norm = 0.0, Margin = 1.0 -> S_conf = 1.0
        let p_onehot = vec![1.0, 0.0, 0.0];
        assert!((composite_confidence(&p_onehot).unwrap() - 1.0).abs() < 1e-6);

        // 3. 完全一様分布極限: H_norm = 1.0, Margin = 0.0 -> S_conf = 0.0
        let p_uniform = vec![0.25, 0.25, 0.25, 0.25];
        assert!(composite_confidence(&p_uniform).unwrap() < 1e-6);

        // 4. Margin Collapse (上位2候補激突):
        // p = [0.49, 0.49, 0.02]
        // 分布全体としてはそこそこ尖っているが、Margin = 0.0 のため S_conf は 0.0 となる。
        let p_collapse = vec![0.49, 0.49, 0.02];
        assert!(composite_confidence(&p_collapse).unwrap() < 1e-6);

        // 5. 高確信判定ケース:
        // p = [0.85, 0.10, 0.05]
        // H(p) = -(0.85*ln 0.85 + 0.1*ln 0.1 + 0.05*ln 0.05) ≈ 0.518
        // H_norm = 0.518 / ln(3) ≈ 0.518 / 1.0986 ≈ 0.4715
        // Margin = 0.85 - 0.10 = 0.75
        // S_conf = (1.0 - 0.4715) * 0.75 ≈ 0.396
        let p_high = vec![0.85, 0.10, 0.05];
        let s_conf = composite_confidence(&p_high).unwrap();
        assert!(s_conf > 0.35 && s_conf < 0.45);

        // 6. 圧倒的確信ケース:
        // p = [0.98, 0.01, 0.01]
        // Margin = 0.97
        // 1 - H_norm ≈ 0.89
        // S_conf ≈ 0.86
        let p_extreme = vec![0.98, 0.01, 0.01];
        let s_extreme = composite_confidence(&p_extreme).unwrap();
        assert!(s_extreme > 0.80);

        // 7. 異常系
        assert!(composite_confidence(&[]).is_err());
        assert!(composite_confidence(&[-0.1, 1.1]).is_err());
        assert!(composite_confidence(&[0.5, f64::NAN]).is_err());
    }
}
