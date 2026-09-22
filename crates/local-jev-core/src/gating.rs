//! # 確信度ゲーティングおよび3系統ルーティングモジュール
//!
//! 事後較正(RLCD + 最適温度スケーリング $\tau^*$)を経た実効確信度に基づき、
//! 決定論的に 3 系統のアクションへ分岐する安全制御ロジックを提供する。
//!
//! ## 3系統ルーティングの判定基準
//! - **`AutoExecute` ($\ge 0.85$ & Margin $\ge 0.15$)**: 自動実行(高信頼・人手や大型LLMの介在なし)。
//! - **`ConfirmOrEscalate` ($0.50 \dots 0.85$ または Margin $< 0.15$)**: 確認/二次検証要求(System 2 LLMへの委託または人手確認)。
//! - **`Fallback` ($< 0.50$)**: 安全弁フォールバック(情報不足・一様迷い時のデフォルト動作・棄却)。

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::schema::{Answer, Question, QuestionType};

/// デフォルトの高確信度閾値 (自動実行境界)。
pub const DEFAULT_HIGH_CONFIDENCE_THRESHOLD: f64 = 0.85;

/// デフォルトの中確信度下限閾値 (確認・二次検証要求境界)。
pub const DEFAULT_LOW_CONFIDENCE_THRESHOLD: f64 = 0.50;

/// デフォルトの上位 2 候補確率マージン閾値。
pub const DEFAULT_TOP_MARGIN_THRESHOLD: f64 = 0.15;

/// 浮動小数点比較時の微小許容誤差。
const EPSILON: f64 = 1e-9;

/// ゲーティング判定に基づく 3 系統のアクション種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "openapi", schema(rename_all = "snake_case"))]
pub enum DecisionRoute {
    /// 自動実行: モデルの確信度が極めて高く、人手介在なしに後続処理を実行する。
    AutoExecute,
    /// 確認/二次検証要求: 候補間で迷いが生じており、System 2 (大型LLM) やユーザー確認を求める。
    ConfirmOrEscalate,
    /// 安全弁フォールバック: 信頼性が著しく低く、安全なデフォルト値の採用や人手転送を行う。
    Fallback,
}

impl DecisionRoute {
    /// ルーティング文字列スライスを取得する。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AutoExecute => "auto_execute",
            Self::ConfirmOrEscalate => "confirm_or_escalate",
            Self::Fallback => "fallback",
        }
    }
}

impl fmt::Display for DecisionRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// ゲーティング判定の設定パラメータ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GatingConfig {
    /// ゲーティング処理を有効化するかどうか。
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// 自動実行 (AutoExecute) と判定するための確信度下限値 (デフォルト: 0.85)。
    #[serde(default = "default_high_threshold")]
    pub high_threshold: f64,

    /// 確認/エスカレーション (ConfirmOrEscalate) と判定するための確信度下限値 (デフォルト: 0.50)。
    #[serde(default = "default_low_threshold")]
    pub low_threshold: f64,

    /// 上位 2 候補の最小確率マージン閾値 (デフォルト: 0.15)。
    ///
    /// 確信度が `high_threshold` 以上であっても、1位と2位の確率差がこの値を下回る場合は
    /// 選択肢間で拮抗が生じていると判断し、強制的に `ConfirmOrEscalate` へ降格する。
    #[serde(default = "default_top_margin_threshold")]
    pub top_margin_threshold: f64,
}

fn default_true() -> bool {
    true
}

fn default_high_threshold() -> f64 {
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD
}

fn default_low_threshold() -> f64 {
    DEFAULT_LOW_CONFIDENCE_THRESHOLD
}

fn default_top_margin_threshold() -> f64 {
    DEFAULT_TOP_MARGIN_THRESHOLD
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            high_threshold: DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
            low_threshold: DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            top_margin_threshold: DEFAULT_TOP_MARGIN_THRESHOLD,
        }
    }
}

impl GatingConfig {
    /// 閾値設定値の数学的妥当性を検証する。
    pub fn validate(&self) -> Result<()> {
        if self.high_threshold < 0.0 || self.high_threshold > 1.0 {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "high_threshold ({}) は 0.0〜1.0 の範囲である必要があります。",
                    self.high_threshold
                ),
            });
        }
        if self.low_threshold < 0.0 || self.low_threshold > 1.0 {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "low_threshold ({}) は 0.0〜1.0 の範囲である必要があります。",
                    self.low_threshold
                ),
            });
        }
        if self.low_threshold > self.high_threshold {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "low_threshold ({}) は high_threshold ({}) 以下である必要があります。",
                    self.low_threshold, self.high_threshold
                ),
            });
        }
        if self.top_margin_threshold < 0.0 || self.top_margin_threshold > 1.0 {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "top_margin_threshold ({}) は 0.0〜1.0 の範囲である必要があります。",
                    self.top_margin_threshold
                ),
            });
        }
        Ok(())
    }
}

/// 候補とその予測確率値のペア。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CandidateProbability {
    /// 候補識別子ラベル。
    pub candidate: String,
    /// 較正済み確率値 (0.0 <= p <= 1.0)。
    pub probability: f64,
}

/// 大型自己回帰 LLM (System 2) へのエスカレーション用コンテキスト。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct EscalationContext {
    /// 確率上位の候補一覧 (降順、最大3候補)。
    pub top_candidates: Vec<CandidateProbability>,

    /// 1位候補と2位候補の確率差。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,

    /// 不確実性および迷いが生じた理由の説明文。
    pub uncertainty_reason: String,

    /// 大型 LLM への入力プロンプト案。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
}

/// 単一質問に対する確信度ゲーティング判定結果および監査メタデータ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GatingMetadata {
    /// 判定されたルーティングアクション種別。
    pub route: DecisionRoute,

    /// 判定に用いられた較正済み実効確信度スコア (0.0 <= C <= 1.0)。
    pub confidence: f64,

    /// 確率分布の不確実性・散らばり指標値。
    ///
    /// # 備考
    /// 質問タイプに応じて以下の正規化不確実性指標を格納する:
    /// - **Choice 型**: 逆算された正規化エントロピー指標 ($1 - C_{\text{entropy}}$)。
    /// - **Score 型**: 逆算された正規化分散指標 ($1 - C_{\text{var}}$)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy: Option<f64>,

    /// 上位 2 候補の確率マージン (Top-1 - Top-2)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,

    /// 判定根拠の説明文 (監査証跡用)。
    pub reason: String,

    /// 中確信度または低確信度時の大型 LLM 連携コンテキスト。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation: Option<EscalationContext>,
}

impl GatingMetadata {
    /// 確率分布の散らばり度合い(不確実性指標)を取得する。
    ///
    /// Choice 型のエントロピーおよび Score 型の分散指標を同一の不確実性スコアとして参照する。
    pub fn dispersion(&self) -> Option<f64> {
        self.entropy
    }
}

/// リクエスト全体の集約ルーティングサマリー。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SystemRoutingSummary {
    /// リクエスト全体の集約ルーティング (最悪値ルール)。
    pub aggregate_route: DecisionRoute,

    /// 自動実行 (AutoExecute) と判定された質問数。
    pub auto_execute_count: usize,

    /// 確認・二次検証要求 (ConfirmOrEscalate) と判定された質問数。
    pub confirm_count: usize,

    /// 安全弁フォールバック (Fallback) と判定された質問数。
    pub fallback_count: usize,

    /// System 2 へのエスカレーションまたは人手介入が必要であるかどうか。
    pub escalation_needed: bool,
}

/// 単一の `Answer` に対して確信度ゲーティング評価を実施し、メタデータを生成する。
///
/// # 引数
/// - `answer`: 事後較正済みの判定結果。
/// - `question_id`: 質問識別子(プロンプト構築用、任意)。
/// - `question`: 質問定義(指示文やタイプ取得用、任意)。
/// - `config`: ゲーティング設定。
///
/// # 戻り値
/// - 導出された `GatingMetadata`。
pub fn evaluate_answer_gating(
    answer: &Answer,
    question_id: Option<&str>,
    question: Option<&Question>,
    config: &GatingConfig,
) -> GatingMetadata {
    let conf = answer.effective_confidence().unwrap_or(0.0).clamp(0.0, 1.0);

    // 確率分布から上位候補のソートリストを構築
    let mut candidates: Vec<CandidateProbability> = Vec::new();
    if let Some(probs) = &answer.probabilities {
        candidates = probs
            .iter()
            .map(|(k, &p)| CandidateProbability {
                candidate: k.clone(),
                probability: p,
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else if let Some(p_true) = answer.noul {
        let p_false = (1.0 - p_true).clamp(0.0, 1.0);
        if p_true >= p_false {
            candidates.push(CandidateProbability {
                candidate: "true".to_string(),
                probability: p_true,
            });
            candidates.push(CandidateProbability {
                candidate: "false".to_string(),
                probability: p_false,
            });
        } else {
            candidates.push(CandidateProbability {
                candidate: "false".to_string(),
                probability: p_false,
            });
            candidates.push(CandidateProbability {
                candidate: "true".to_string(),
                probability: p_true,
            });
        }
    }

    // 上位 2 候補の確率差(Margin)を算出
    let margin = if candidates.len() >= 2 {
        Some(
            (candidates[0].probability - candidates[1].probability)
                .abs()
                .clamp(0.0, 1.0),
        )
    } else {
        None
    };

    // 分布エントロピー指標の取得
    let entropy = match question.map(|q| q.question_type) {
        Some(QuestionType::Choice) => {
            // Choice の場合、confidence = 1 - H/ln(K) であるため、逆算した正規化エントロピーを記録
            answer.confidence.map(|c| (1.0 - c).clamp(0.0, 1.0))
        }
        Some(QuestionType::Score) => {
            // Score の場合、confidence = 1 - Var/Var_max であるため、正規化分散指標を記録
            answer.confidence.map(|c| (1.0 - c).clamp(0.0, 1.0))
        }
        _ => None,
    };

    // 閾値判定とマージンガード
    let (route, reason) = if conf >= config.high_threshold - EPSILON {
        if let Some(m) = margin {
            if m < config.top_margin_threshold - EPSILON {
                (
                    DecisionRoute::ConfirmOrEscalate,
                    format!(
                        "実効確信度({:.3} >= {:.3})は高水準ですが、上位2候補の確率差({:.3} < {:.3})が僅差のため確認要求に降格しました。",
                        conf, config.high_threshold, m, config.top_margin_threshold
                    ),
                )
            } else {
                (
                    DecisionRoute::AutoExecute,
                    format!(
                        "実効確信度({:.3} >= {:.3})および上位候補マージン({:.3} >= {:.3})を満たすため自動実行が承認されました。",
                        conf, config.high_threshold, m, config.top_margin_threshold
                    ),
                )
            }
        } else {
            (
                DecisionRoute::AutoExecute,
                format!(
                    "実効確信度({:.3} >= {:.3})が高水準であるため自動実行が承認されました。",
                    conf, config.high_threshold
                ),
            )
        }
    } else if conf >= config.low_threshold - EPSILON {
        (
            DecisionRoute::ConfirmOrEscalate,
            format!(
                "実効確信度({:.3})が確認要求境界 [{:.3}, {:.3}) 内のため二次検証が必要です。",
                conf, config.low_threshold, config.high_threshold
            ),
        )
    } else {
        (
            DecisionRoute::Fallback,
            format!(
                "実効確信度({:.3} < {:.3})が低水準のため安全弁フォールバックを適用しました。",
                conf, config.low_threshold
            ),
        )
    };

    // エスカレーションコンテキストの生成 (ConfirmOrEscalate または Fallback 時)
    let escalation = if route != DecisionRoute::AutoExecute {
        let top_candidates = candidates.iter().take(3).cloned().collect();
        let prompt_template = build_escalation_prompt(question_id, question, &candidates, margin);

        Some(EscalationContext {
            top_candidates,
            margin,
            uncertainty_reason: reason.clone(),
            prompt_template: Some(prompt_template),
        })
    } else {
        None
    };

    GatingMetadata {
        route,
        confidence: conf,
        entropy,
        margin,
        reason,
        escalation,
    }
}

/// System 2 (大型自己回帰LLM) に向けた委託用プロンプト案を生成する。
fn build_escalation_prompt(
    question_id: Option<&str>,
    question: Option<&Question>,
    candidates: &[CandidateProbability],
    margin: Option<f64>,
) -> String {
    let q_name = question_id.unwrap_or("target_question");
    let instructions = question
        .map(|q| q.instructions.as_str())
        .unwrap_or("指示文なし");

    if candidates.len() >= 2 {
        let c1 = &candidates[0];
        let c2 = &candidates[1];
        let p1_str = format!("{:.1}", c1.probability * 100.0);
        let p2_str = format!("{:.1}", c2.probability * 100.0);
        let m_str = match margin {
            Some(m) => format!("(確率差: {:.1}%)", m * 100.0),
            None => String::new(),
        };

        let mut prompt = String::with_capacity(256);
        prompt.push_str("質問「");
        prompt.push_str(q_name);
        prompt.push_str("」(");
        prompt.push_str(instructions);
        prompt.push_str(") について、System 1 の判定では候補「");
        prompt.push_str(&c1.candidate);
        prompt.push_str("」(確信度 ");
        prompt.push_str(&p1_str);
        prompt.push_str("%) と「");
        prompt.push_str(&c2.candidate);
        prompt.push_str("」(確信度 ");
        prompt.push_str(&p2_str);
        prompt.push_str("%) の間で迷いが生じています ");
        prompt.push_str(&m_str);
        prompt.push_str("。State コンテキストを精読し、思考連鎖 (Chain-of-Thought) に基づき最適な決定を行ってください。");
        return prompt;
    }

    let mut prompt = String::with_capacity(128);
    prompt.push_str("質問「");
    prompt.push_str(q_name);
    prompt.push_str("」(");
    prompt.push_str(instructions);
    prompt.push_str(") について、System 1 の判定信頼性が低水準となっています。State コンテキストを精読し、論理的根拠に基づいて決定を下してください。");
    prompt
}

/// リクエストに含まれる全回答のゲーティング情報から集約サマリーを算出する。
///
/// 最悪値ルール(Conservative Policy)を適用する:
/// - 1 つでも `Fallback` が存在する場合 $\rightarrow$ 全体も `Fallback`
/// - そうでなく 1 つでも `ConfirmOrEscalate` が存在する場合 $\rightarrow$ 全体も `ConfirmOrEscalate`
/// - 全ての質問が `AutoExecute` である場合 $\rightarrow$ 全体も `AutoExecute`
pub fn evaluate_response_routing(answers: &IndexMap<String, Answer>) -> SystemRoutingSummary {
    let mut auto_execute_count = 0;
    let mut confirm_count = 0;
    let mut fallback_count = 0;

    for answer in answers.values() {
        match answer.route() {
            Some(DecisionRoute::AutoExecute) => auto_execute_count += 1,
            Some(DecisionRoute::ConfirmOrEscalate) => confirm_count += 1,
            Some(DecisionRoute::Fallback) => fallback_count += 1,
            None => {
                // ゲーティング未適用の場合は実効確信度から簡易分類
                let conf = answer.effective_confidence().unwrap_or(0.0);
                if conf >= DEFAULT_HIGH_CONFIDENCE_THRESHOLD {
                    auto_execute_count += 1;
                } else if conf >= DEFAULT_LOW_CONFIDENCE_THRESHOLD {
                    confirm_count += 1;
                } else {
                    fallback_count += 1;
                }
            }
        }
    }

    let aggregate_route = if fallback_count > 0 {
        DecisionRoute::Fallback
    } else if confirm_count > 0 {
        DecisionRoute::ConfirmOrEscalate
    } else {
        DecisionRoute::AutoExecute
    };

    let escalation_needed = confirm_count > 0 || fallback_count > 0;

    SystemRoutingSummary {
        aggregate_route,
        auto_execute_count,
        confirm_count,
        fallback_count,
        escalation_needed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_config_default_and_validation() {
        let config = GatingConfig::default();
        assert!(config.enabled);
        assert!((config.high_threshold - 0.85).abs() < EPSILON);
        assert!((config.low_threshold - 0.50).abs() < EPSILON);
        assert!((config.top_margin_threshold - 0.15).abs() < EPSILON);
        assert!(config.validate().is_ok());

        let invalid_config = GatingConfig {
            enabled: true,
            high_threshold: 0.40,
            low_threshold: 0.60,
            top_margin_threshold: 0.15,
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_evaluate_answer_gating_auto_execute() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.90);
        probs.insert("opt_b".to_string(), 0.10);
        let answer = Answer::choice("opt_a", probs, 0.92);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::AutoExecute);
        assert!((meta.confidence - 0.92).abs() < EPSILON);
        assert!((meta.margin.unwrap() - 0.80).abs() < EPSILON);
        assert!(meta.escalation.is_none());
    }

    #[test]
    fn test_evaluate_answer_gating_margin_guard_demote_to_confirm() {
        // 確信度は 0.88 と高水準だが、Top-1 と Top-2 が僅差 (0.48 vs 0.44 -> margin 0.04 < 0.15) のケース
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.48);
        probs.insert("opt_b".to_string(), 0.44);
        probs.insert("opt_c".to_string(), 0.08);
        let answer = Answer::choice("opt_a", probs, 0.88);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.reason.contains("上位2候補の確率差"));
        assert!(meta.escalation.is_some());
        let esc = meta.escalation.unwrap();
        assert_eq!(esc.top_candidates.len(), 3);
        assert_eq!(esc.top_candidates[0].candidate, "opt_a");
        assert_eq!(esc.top_candidates[1].candidate, "opt_b");
    }

    #[test]
    fn test_evaluate_answer_gating_medium_confirm() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.60);
        probs.insert("opt_b".to_string(), 0.40);
        let answer = Answer::choice("opt_a", probs, 0.65);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.escalation.is_some());
    }

    #[test]
    fn test_evaluate_answer_gating_low_fallback() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.34);
        probs.insert("opt_b".to_string(), 0.33);
        probs.insert("opt_c".to_string(), 0.33);
        let answer = Answer::choice("opt_a", probs, 0.10);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::Fallback);
        assert!(meta.escalation.is_some());
    }

    #[test]
    fn test_evaluate_answer_gating_noul_high_and_low() {
        let config = GatingConfig::default();

        // High: P=0.96 -> C=|2*0.96 - 1| = 0.92 >= 0.85
        let ans_high = Answer::noul(0.96);
        let meta_high = evaluate_answer_gating(&ans_high, Some("noul_q"), None, &config);
        assert_eq!(meta_high.route, DecisionRoute::AutoExecute);

        // Low: P=0.52 -> C=|2*0.52 - 1| = 0.04 < 0.50
        let ans_low = Answer::noul(0.52);
        let meta_low = evaluate_answer_gating(&ans_low, Some("noul_q"), None, &config);
        assert_eq!(meta_low.route, DecisionRoute::Fallback);
        assert!(meta_low.escalation.is_some());
    }

    #[test]
    fn test_evaluate_response_routing_aggregation() {
        let mut answers = IndexMap::new();

        let ans1 = Answer::choice("a", IndexMap::new(), 0.95).with_gating(GatingMetadata {
            route: DecisionRoute::AutoExecute,
            confidence: 0.95,
            entropy: None,
            margin: None,
            reason: String::new(),
            escalation: None,
        });

        let ans2 = Answer::choice("b", IndexMap::new(), 0.70).with_gating(GatingMetadata {
            route: DecisionRoute::ConfirmOrEscalate,
            confidence: 0.70,
            entropy: None,
            margin: None,
            reason: String::new(),
            escalation: None,
        });

        answers.insert("q1".to_string(), ans1);
        answers.insert("q2".to_string(), ans2);

        let summary = evaluate_response_routing(&answers);
        assert_eq!(summary.aggregate_route, DecisionRoute::ConfirmOrEscalate);
        assert_eq!(summary.auto_execute_count, 1);
        assert_eq!(summary.confirm_count, 1);
        assert_eq!(summary.fallback_count, 0);
        assert!(summary.escalation_needed);

        // Fallback が 1 つでも加わると全体が Fallback
        let ans3 = Answer::choice("c", IndexMap::new(), 0.20).with_gating(GatingMetadata {
            route: DecisionRoute::Fallback,
            confidence: 0.20,
            entropy: None,
            margin: None,
            reason: String::new(),
            escalation: None,
        });
        answers.insert("q3".to_string(), ans3);

        let summary_fallback = evaluate_response_routing(&answers);
        assert_eq!(summary_fallback.aggregate_route, DecisionRoute::Fallback);
        assert_eq!(summary_fallback.fallback_count, 1);
        assert!(summary_fallback.escalation_needed);
    }
}
