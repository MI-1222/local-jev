//! # 階層較正・確率結合モジュール
//!
//! 大分類エントロピー連動の温度スケーリング、ODIR 正則化 Dirichlet 較正、
//! および全候補空間への確率復元・周辺化結合を提供する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{Result, RuntimeError};

/// 確率計算および対数変換時のアンダーフロー防止用微小量。
pub const EPSILON: f64 = 1e-12;

/// デフォルトのエントロピー連動温度係数 ($\gamma$)。
pub const DEFAULT_ENTROPY_TEMP_GAMMA: f64 = 0.5;

/// デフォルトの大分類確率減衰べき乗指数 ($\alpha$)。
pub const DEFAULT_COARSE_POWER_ALPHA: f64 = 0.85;

/// ディリクレ較正器 (Dirichlet Calibrator)。
///
/// Kull et al. (NeurIPS 2019) の Dirichlet 較正に基づく。
/// 対数確率ベクトル $\ln \mathbf{p}$ に対するアフィン変換 $\mathbf{z} = W \ln \mathbf{p} + \mathbf{b}$ を行い、
/// Softmax を適用して適正較正された確率分布を出力する。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletCalibrator {
    /// クラス数 $K$。
    pub num_classes: usize,
    /// 重み行列 $W \in \mathbb{R}^{K \times K}$ (行優先フラット配列)。
    pub weights: Vec<f64>,
    /// バイアスベクトル $\mathbf{b} \in \mathbb{R}^K$。
    pub bias: Vec<f64>,
}

impl DirichletCalibrator {
    /// 恒等変換を行う較正器 (未較正時) を生成する。
    ///
    /// $W = I, \mathbf{b} = \mathbf{0}$ となり、入力確率分布がそのまま維持される。
    pub fn identity(num_classes: usize) -> Self {
        let mut weights = vec![0.0; num_classes * num_classes];
        for i in 0..num_classes {
            weights[i * num_classes + i] = 1.0;
        }
        let bias = vec![0.0; num_classes];
        Self {
            num_classes,
            weights,
            bias,
        }
    }

    /// 事前学習されたパラメータを指定して較正器を構築する。
    ///
    /// # 引数
    /// - `num_classes`: 分類クラス数 $K$。
    /// - `weights`: $K \times K$ の行優先重み行列。
    /// - `bias`: 長さ $K$ のバイアスベクトル。
    pub fn new(num_classes: usize, weights: Vec<f64>, bias: Vec<f64>) -> Result<Self> {
        if weights.len() != num_classes * num_classes {
            return Err(RuntimeError::InvalidQuestion(format!(
                "Dirichlet較正重み行列のサイズ ({}) が想定 ({}x{}={}) と一致しません。",
                weights.len(),
                num_classes,
                num_classes,
                num_classes * num_classes
            )));
        }
        if bias.len() != num_classes {
            return Err(RuntimeError::InvalidQuestion(format!(
                "Dirichlet較正バイアスベクトルの長さ ({}) が想定 ({}) と一致しません。",
                bias.len(),
                num_classes
            )));
        }
        Ok(Self {
            num_classes,
            weights,
            bias,
        })
    }

    /// 未較正の確率ベクトルを較正済み確率ベクトルへ変換する。
    ///
    /// 入力確率値は $\epsilon$ でクリップされ、対数空間でのアフィン変換後に Softmax で正規化される。
    pub fn calibrate(&self, probs: &[f64]) -> Result<Vec<f64>> {
        if probs.len() != self.num_classes {
            return Err(RuntimeError::InvalidQuestion(format!(
                "入力確率ベクトルの長さ ({}) が較正器のクラス数 ({}) と一致しません。",
                probs.len(),
                self.num_classes
            )));
        }

        let k = self.num_classes;
        let mut log_p = Vec::with_capacity(k);
        for &p in probs {
            let clamped = p.max(EPSILON);
            log_p.push(clamped.ln());
        }

        let mut logits = vec![0.0; k];
        for (i, logit) in logits.iter_mut().enumerate() {
            let mut sum = self.bias[i];
            let row_offset = i * k;
            for (j, &lp) in log_p.iter().enumerate() {
                sum += self.weights[row_offset + j] * lp;
            }
            *logit = sum;
        }

        Ok(softmax_f64(&logits))
    }
}

/// 確率分布からシャノンエントロピーおよび正規化エントロピー $\tilde{H} \in [0, 1]$ を計算する。
///
/// # 引数
/// - `probs`: 確率スライス (総和がおおむね 1.0 であること)。
///
/// # 戻り値
/// `(shannon_entropy, normalized_entropy)` のタプル。
pub fn compute_normalized_entropy(probs: &[f64]) -> (f64, f64) {
    let k = probs.len();
    if k <= 1 {
        return (0.0, 0.0);
    }

    let mut entropy = 0.0;
    for &p in probs {
        if p > EPSILON {
            entropy -= p * p.ln();
        }
    }

    let max_entropy = (k as f64).ln();
    let normalized = if max_entropy > EPSILON {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (entropy, normalized)
}

/// 大分類の正規化エントロピーに基づいて、第 2 パス (細分類) に適用する動的温度を算出する。
///
/// 数式:
/// $$T_{\text{fine}} = 1.0 + \gamma \cdot \tilde{H}(\mathbf{P}_C)$$
///
/// 大分類で迷いが生じている (エントロピーが高い) ほど温度を引き上げ、細分類の過信 Softmax を平滑化する。
pub fn compute_fine_temperature(normalized_entropy: f64, gamma: f64) -> f64 {
    (1.0 + gamma * normalized_entropy).max(1.0)
}

/// 細分類のロジットに温度スケーリングを適用した上で Softmax を計算する。
///
/// # 引数
/// - `logits`: 第 2 パスで得られた未正規化ロジット。
/// - `temperature`: スケーリング温度 ($T \ge 1.0$)。
pub fn temperature_scaled_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let temp = temperature.max(0.01);
    let mut scaled = Vec::with_capacity(logits.len());
    for &l in logits {
        scaled.push(l / temp);
    }
    softmax_f64(&scaled)
}

/// ロジット列に対する数値的に安定な Softmax 計算。
pub fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| if a > b { a } else { b });

    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &l in logits {
        let e = (l - max_logit).exp();
        exps.push(e);
        sum += e;
    }

    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0 / (logits.len() as f64);
        return vec![uniform; logits.len()];
    }

    for e in &mut exps {
        *e /= sum;
    }
    exps
}

/// 第 1 パス (大分類) 確率と第 2 パス (細分類) 確率を減衰周辺化結合し、全候補空間の確率分布を復元する。
///
/// # アルゴリズム
/// 1. 各細分類候補 $F_k$ に対し、所属する親大分類 $C_j$ の確率 $P(C_j)$ を取得する。
/// 2. 減衰べき乗指数 $\alpha$ を用いて不確実性を伝播させた結合スコアを算出:
///    $$S(F_k) = P(C_j)^\alpha \cdot P(F_k \mid C_j)$$
/// 3. 第 2 パスに含まれる全候補で正規化 (周辺化)。
/// 4. 評価対象外となった除外クラスタの候補には `0.0` を設定し、全体の確率和を 1.0 に正規化する。
///
/// # 引数
/// - `all_fine_keys`: 全細分類キーの完全リスト (順序維持)。
/// - `fine_to_coarse`: 細分類キーから親大分類キーへの逆引きマップ。
/// - `coarse_probs`: 大分類キーとその確率のマッピング。
/// - `evaluated_fine_probs`: 第 2 パスで実際に評価された細分類キーとその確率 (温度補正後)。
/// - `coarse_alpha`: 大分類確率の減衰指数 ($\alpha \in [0.8, 1.0]$)。
///
/// # 戻り値
/// 全細分類キーに対応する確率マップ (`IndexMap<String, f64>`)。総和は厳密に 1.0 となる。
pub fn combine_and_reconstruct_probabilities(
    all_fine_keys: &[String],
    fine_to_coarse: &IndexMap<String, String>,
    coarse_probs: &IndexMap<String, f64>,
    evaluated_fine_probs: &IndexMap<String, f64>,
    coarse_alpha: f64,
) -> Result<IndexMap<String, f64>> {
    if all_fine_keys.is_empty() {
        return Err(RuntimeError::InvalidQuestion(
            "all_fine_keys が空です。".to_string(),
        ));
    }
    if evaluated_fine_probs.is_empty() {
        return Err(RuntimeError::InvalidQuestion(
            "evaluated_fine_probs が空です。".to_string(),
        ));
    }

    // 1. 第 2 パス評価対象候補の結合未正規化スコアを計算
    let mut raw_scores = IndexMap::with_capacity(evaluated_fine_probs.len());
    let mut total_score = 0.0;

    for (fine_key, &fine_prob) in evaluated_fine_probs {
        let coarse_key = fine_to_coarse.get(fine_key).ok_or_else(|| {
            RuntimeError::InvalidQuestion(format!(
                "細分類 `{}` の親大分類が見つかりません。",
                fine_key
            ))
        })?;

        let coarse_prob = coarse_probs.get(coarse_key).copied().unwrap_or(0.0);
        let attenuated_coarse = coarse_prob.max(EPSILON).powf(coarse_alpha);

        let combined = attenuated_coarse * fine_prob.max(0.0);
        raw_scores.insert(fine_key.clone(), combined);
        total_score += combined;
    }

    // 2. 正規化
    let mut normalized_evaluated = IndexMap::with_capacity(evaluated_fine_probs.len());
    if total_score > EPSILON && total_score.is_finite() {
        for (k, s) in raw_scores {
            normalized_evaluated.insert(k, s / total_score);
        }
    } else {
        // フォールバック: 一様分布
        let uniform = 1.0 / (evaluated_fine_probs.len() as f64);
        for k in evaluated_fine_probs.keys() {
            normalized_evaluated.insert(k.clone(), uniform);
        }
    }

    // 3. 全候補空間への再構成 (除外候補には 0.0 を設定)
    let mut full_probs = IndexMap::with_capacity(all_fine_keys.len());
    let mut final_sum = 0.0;

    for key in all_fine_keys {
        let prob = normalized_evaluated.get(key).copied().unwrap_or(0.0);
        full_probs.insert(key.clone(), prob);
        final_sum += prob;
    }

    // 厳密な 1.0 への再正規化
    if (final_sum - 1.0).abs() > 1e-6 && final_sum > 0.0 {
        for p in full_probs.values_mut() {
            *p /= final_sum;
        }
    }

    Ok(full_probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_calibrator() {
        let cal = DirichletCalibrator::identity(3);
        let probs = vec![0.6, 0.3, 0.1];
        let calibrated = cal.calibrate(&probs).unwrap();

        for (a, b) in probs.iter().zip(calibrated.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_entropy_computation() {
        // 確信分布 (エントロピー最小)
        let certain = vec![1.0, 0.0, 0.0];
        let (_, norm_certain) = compute_normalized_entropy(&certain);
        assert!(norm_certain < 1e-5);

        // 一様分布 (エントロピー最大)
        let uniform = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let (_, norm_uniform) = compute_normalized_entropy(&uniform);
        assert!((norm_uniform - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_fine_temperature() {
        let temp_certain = compute_fine_temperature(0.0, 0.5);
        assert_eq!(temp_certain, 1.0);

        let temp_uncertain = compute_fine_temperature(1.0, 0.5);
        assert_eq!(temp_uncertain, 1.5);
    }

    #[test]
    fn test_combine_and_reconstruct() {
        let all_keys = vec![
            "c1_f1".to_string(),
            "c1_f2".to_string(),
            "c2_f1".to_string(),
            "c2_f2".to_string(),
            "c3_f1".to_string(),
        ];

        let mut fine_to_coarse = IndexMap::new();
        fine_to_coarse.insert("c1_f1".to_string(), "c1".to_string());
        fine_to_coarse.insert("c1_f2".to_string(), "c1".to_string());
        fine_to_coarse.insert("c2_f1".to_string(), "c2".to_string());
        fine_to_coarse.insert("c2_f2".to_string(), "c2".to_string());
        fine_to_coarse.insert("c3_f1".to_string(), "c3".to_string());

        let mut coarse_probs = IndexMap::new();
        coarse_probs.insert("c1".to_string(), 0.7);
        coarse_probs.insert("c2".to_string(), 0.25);
        coarse_probs.insert("c3".to_string(), 0.05);

        // 第 2 パスでは c1_f1 と c1_f2 のみが評価されたケース
        let mut evaluated = IndexMap::new();
        evaluated.insert("c1_f1".to_string(), 0.8);
        evaluated.insert("c1_f2".to_string(), 0.2);

        let full = combine_and_reconstruct_probabilities(
            &all_keys,
            &fine_to_coarse,
            &coarse_probs,
            &evaluated,
            0.85,
        )
        .unwrap();

        assert_eq!(full.len(), 5);
        // 評価外は 0.0
        assert_eq!(full.get("c2_f1").copied().unwrap(), 0.0);
        assert_eq!(full.get("c2_f2").copied().unwrap(), 0.0);
        assert_eq!(full.get("c3_f1").copied().unwrap(), 0.0);

        // 評価対象は正の値
        assert!(full.get("c1_f1").copied().unwrap() > full.get("c1_f2").copied().unwrap());

        // 総和が 1.0
        let sum: f64 = full.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
