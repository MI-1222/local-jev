//! # 前処理ガードレール統括パイプラインモジュール
//!
//! リクエスト受信直後から推論エンジン投入までの間に、以下の一連のパイプラインを適用する。
//!
//! - 物理的 OOM 防壁 (Limits / Fast-Fail)
//! - 特殊トークン偽装・プロンプトインジェクション防壁 (Sanitizer)
//! - コンテキスト縮約・ノイズ除去 (Compactor / Anti-Context Rot)
//! - 相対日時絶対正規化 (Temporal Grounding)
//! - 算術・数え上げ事前集計 (Arithmetic Annotation)

pub mod arithmetic;
pub mod compactor;
pub mod limits;
pub mod sanitization;
pub mod temporal;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use local_jev_core::schema::{Criteria, Question, SystemOneRequest};
use serde_json::Value;

use crate::error::ServerError;
use crate::guardrails::arithmetic::{ArithmeticConfig, compute_arithmetic_summary};
use crate::guardrails::compactor::{CompactorConfig, compact_text};
use crate::guardrails::limits::{LimitsConfig, validate_limits};
use crate::guardrails::sanitization::{SanitizerConfig, sanitize_text};
use crate::guardrails::temporal::{
    TemporalConfig, append_reference_time_metadata, normalize_temporal,
};

/// ガードレール全体の統合設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailConfig {
    /// ガードレール全体の有効化フラグ。
    pub enabled: bool,
    /// 物理リソース制限設定。
    pub limits: LimitsConfig,
    /// サニタイズ設定。
    pub sanitizer: SanitizerConfig,
    /// コンテキスト縮約設定。
    pub compactor: CompactorConfig,
    /// 相対日時正規化設定。
    pub temporal: TemporalConfig,
    /// 算術事前集計設定。
    pub arithmetic: ArithmeticConfig,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            limits: LimitsConfig::default(),
            sanitizer: SanitizerConfig::default(),
            compactor: CompactorConfig::default(),
            temporal: TemporalConfig::default(),
            arithmetic: ArithmeticConfig::default(),
        }
    }
}

/// 前処理ガードレール実行結果レポート。
#[derive(Debug, Clone, Default)]
pub struct PreprocessReport {
    /// 事前見積もりされたトークン数。
    pub estimated_tokens: usize,
    /// 算術集計ブロックが付加されたか。
    pub added_arithmetic_summary: bool,
    /// 基準時刻メタデータが付加されたか。
    pub added_reference_time: bool,
}

/// 前処理ガードレールパイプライン。
#[derive(Clone)]
pub struct GuardrailPipeline {
    config: Arc<GuardrailConfig>,
}

impl GuardrailPipeline {
    /// 新規 `GuardrailPipeline` を生成する。
    pub fn new(config: GuardrailConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    /// 設定への参照を取得する。
    pub fn config(&self) -> &GuardrailConfig {
        &self.config
    }

    /// リクエストを検査および正規化する。
    ///
    /// # 引数
    /// - `req`: 可変推論リクエスト参照。
    /// - `ref_time`: 基準日時 (省略時は現在日時 `Utc::now()`)。
    ///
    /// # 戻り値
    /// 成功時は前処理レポートを返却し、物理上限超過時は `ServerError::PayloadTooLarge` を返却する。
    pub fn process(
        &self,
        req: &mut SystemOneRequest,
        ref_time: Option<DateTime<Utc>>,
    ) -> Result<PreprocessReport, ServerError> {
        if !self.config.enabled {
            return Ok(PreprocessReport::default());
        }

        let now = ref_time.unwrap_or_else(Utc::now);
        let mut report = PreprocessReport::default();

        // 1. 物理 OOM 防壁 (L1/L2 Fast-Fail)
        let (_, estimated_tokens) = validate_limits(req, &self.config.limits)?;
        report.estimated_tokens = estimated_tokens;

        // 2. State の正規化 (文字列への展開と前処理)
        let mut raw_state = match &req.state {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        // 算術集計サマリーの事前計算 (元の Value から集計)
        let arithmetic_summary = compute_arithmetic_summary(&req.state, &self.config.arithmetic);

        // State 文字列のサニタイズ、縮約、日時正規化
        let state_sanitized = sanitize_text(&raw_state, &self.config.sanitizer);
        let state_compacted = compact_text(&state_sanitized, &self.config.compactor);
        let state_temporal = normalize_temporal(&state_compacted, now, &self.config.temporal);

        raw_state = state_temporal.into_owned();

        // 算術集計サマリーの付加
        if let Some(summary) = arithmetic_summary {
            raw_state.push_str(&summary);
            report.added_arithmetic_summary = true;
        }

        // 基準時刻メタデータの付加
        if self.config.temporal.attach_reference_time {
            append_reference_time_metadata(
                &mut raw_state,
                now,
                self.config.temporal.timezone_offset_hours,
            );
            report.added_reference_time = true;
        }

        // 正規化後の State を req に再格納
        req.state = Value::String(raw_state);

        // 3. 各 Question (Instructions, Criteria) の正規化
        let mut normalized_questions = IndexMap::with_capacity(req.questions.len());

        for (q_key, question) in &req.questions {
            // Instructions の正規化
            let inst_sanitized = sanitize_text(&question.instructions, &self.config.sanitizer);
            let inst_compacted = compact_text(&inst_sanitized, &self.config.compactor);
            let inst_temporal = normalize_temporal(&inst_compacted, now, &self.config.temporal);
            let normalized_inst = inst_temporal.into_owned();

            // Criteria の正規化
            let normalized_criteria = match &question.criteria {
                Some(Criteria::Map(map)) => {
                    let mut new_map = IndexMap::with_capacity(map.len());
                    for (k, v) in map {
                        let k_sanitized = sanitize_text(k, &self.config.sanitizer);
                        let v_sanitized = sanitize_text(v, &self.config.sanitizer);
                        let v_compacted = compact_text(&v_sanitized, &self.config.compactor);
                        new_map.insert(k_sanitized.into_owned(), v_compacted.into_owned());
                    }
                    Some(Criteria::Map(new_map))
                }
                Some(Criteria::List(list)) => {
                    let mut new_list = Vec::with_capacity(list.len());
                    for item in list {
                        let item_sanitized = sanitize_text(item, &self.config.sanitizer);
                        let item_compacted = compact_text(&item_sanitized, &self.config.compactor);
                        new_list.push(item_compacted.into_owned());
                    }
                    Some(Criteria::List(new_list))
                }
                Some(Criteria::None) => Some(Criteria::None),
                None => None,
            };

            let new_question = Question {
                question_type: question.question_type,
                instructions: normalized_inst,
                criteria: normalized_criteria,
            };

            normalized_questions.insert(q_key.clone(), new_question);
        }

        req.questions = normalized_questions;

        Ok(report)
    }
}

impl Default for GuardrailPipeline {
    fn default() -> Self {
        Self::new(GuardrailConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    #[test]
    fn test_pipeline_end_to_end() {
        let pipeline = GuardrailPipeline::default();
        let ref_time = Utc.with_ymd_and_hms(2026, 9, 22, 10, 0, 0).unwrap();

        let mut questions = IndexMap::new();
        questions.insert(
            "q1".to_string(),
            Question::new_choice(
                "3日前の注文 [OP] を確認せよ。".to_string(),
                IndexMap::from([("yes".to_string(), "はい [OP]".to_string())]),
            ),
        );

        let mut req = SystemOneRequest::new(
            json!({
                "items": [
                    {"name": "apple", "amount": 100},
                    {"name": "banana", "amount": 200}
                ]
            }),
            questions,
        );

        let res = pipeline.process(&mut req, Some(ref_time));
        assert!(res.is_ok());

        let state_str = req.state.as_str().unwrap();
        // 特殊トークンは含まれていないが、算術集計と基準時刻が付与されていること
        assert!(state_str.contains("items_count=2"));
        assert!(state_str.contains("total_amount=300"));
        assert!(state_str.contains("[Reference Time: 2026-09-22T10:00:00Z]"));

        // Question 側の [OP] がサニタイズされ、相対日時が絶対化されていること
        let q1 = req.questions.get("q1").unwrap();
        assert!(q1.instructions.contains("3日前 (2026-09-19)"));
        assert!(q1.instructions.contains("[ OP ]"));

        let criteria_map = q1.criteria.as_ref().unwrap().as_map().unwrap();
        assert_eq!(criteria_map.get("yes").unwrap(), "はい [ OP ]");
    }
}
