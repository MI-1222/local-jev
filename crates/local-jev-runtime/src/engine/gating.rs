//! # 推論エンジン向け確信度ゲーティング統合モジュール
//!
//! モデル成果物の較正情報 (`CalibrationConfig.gating_thresholds`) または
//! リクエスト指定の `GatingConfig` を用いて、推論直後の判定結果 (`Answer`) に対して
//! 透過的に 3 系統ルーティング (`AutoExecute` / `ConfirmOrEscalate` / `Fallback`) を確定・付与する。

use indexmap::IndexMap;
use local_jev_core::contract::CalibrationConfig;
pub use local_jev_core::gating::DecisionRoute;
use local_jev_core::gating::{
    GatingConfig, SystemRoutingSummary, evaluate_answer_gating, evaluate_response_routing,
};
use local_jev_core::schema::{Answer, Question};

/// リクエスト固有設定とモデル較正設定から実効的なゲーティング設定をカスケード解決する。
///
/// # 解決優先順位
/// 1. `gating_config`: リクエストで明示された設定。
/// 2. `calib_config.gating_thresholds`: モデル成果物同梱の閾値設定。
///
/// # 引数
/// - `gating_config`: リクエストで指定された設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
///
/// # 戻り値
/// 解決された `GatingConfig`。
pub fn resolve_gating_config(
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> GatingConfig {
    match gating_config {
        Some(cfg) => cfg.clone(),
        None => calib_config.gating_config(),
    }
}

/// 単一の判定結果 (`Answer`) に対し、解決されたゲーティング設定を適用する。
///
/// 設定の `enabled` が `true` の場合、`evaluate_answer_gating` を適用して
/// `answer.gating` に監査メタデータを付与する。無効 (`enabled == false`) の場合は何もしない。
///
/// # 引数
/// - `answer`: 判定結果 (可変参照)。
/// - `question_id`: 質問識別子 (プロンプト構築用、任意)。
/// - `question`: 質問定義 (任意)。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
pub fn apply_gating_to_answer(
    answer: &mut Answer,
    question_id: Option<&str>,
    question: Option<&Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) {
    let config = resolve_gating_config(gating_config, calib_config);
    if config.enabled {
        let meta = evaluate_answer_gating(answer, question_id, question, &config);
        answer.gating = Some(meta);
    }
}

/// 複数質問の判定結果マップに対し、ゲーティング判定を一括適用し集約サマリーを算出する。
///
/// 各回答にゲーティングメタデータを付与し、有効な場合は `SystemRoutingSummary` を返却する。
///
/// # 引数
/// - `answers`: 判定結果マップ (可変参照)。
/// - `questions`: 質問定義マップ。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
///
/// # 戻り値
/// ゲーティングが有効な場合は `Some(SystemRoutingSummary)`、無効な場合は `None`。
pub fn apply_gating_to_answers(
    answers: &mut IndexMap<String, Answer>,
    questions: &IndexMap<String, Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> Option<SystemRoutingSummary> {
    let config = resolve_gating_config(gating_config, calib_config);
    if !config.enabled {
        return None;
    }

    for (qid, answer) in answers.iter_mut() {
        let q_def = questions.get(qid);
        let meta = evaluate_answer_gating(answer, Some(qid), q_def, &config);
        answer.gating = Some(meta);
    }

    Some(evaluate_response_routing(answers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_jev_core::contract::calibration::GatingThresholds;

    #[test]
    fn test_resolve_gating_config_cascade() {
        let calib = CalibrationConfig {
            gating_thresholds: GatingThresholds {
                high_threshold: 0.72,
                low_threshold: 0.38,
                top_margin_threshold: 0.18,
            },
            ..Default::default()
        };

        // 1. 指定なしの場合は calib_config から導出
        let resolved_default = resolve_gating_config(None, &calib);
        assert!(resolved_default.enabled);
        assert!((resolved_default.high_threshold - 0.72).abs() < 1e-9);

        // 2. 明示指定された場合はそれを優先
        let explicit = GatingConfig {
            enabled: false,
            high_threshold: 0.90,
            low_threshold: 0.40,
            top_margin_threshold: 0.20,
        };
        let resolved_explicit = resolve_gating_config(Some(&explicit), &calib);
        assert!(!resolved_explicit.enabled);
        assert!((resolved_explicit.high_threshold - 0.90).abs() < 1e-9);
    }

    #[test]
    fn test_apply_gating_to_answer() {
        let calib = CalibrationConfig::default();
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.95);
        probs.insert("opt_b".to_string(), 0.05);

        let mut ans = Answer::choice("opt_a", probs, 0.90);
        assert!(ans.gating.is_none());

        apply_gating_to_answer(&mut ans, Some("q1"), None, None, &calib);
        assert!(ans.gating.is_some());
        assert_eq!(
            ans.gating.as_ref().unwrap().route,
            DecisionRoute::AutoExecute
        );
    }

    #[test]
    fn test_apply_gating_to_answer_disabled() {
        let calib = CalibrationConfig::default();
        let disabled_cfg = GatingConfig {
            enabled: false,
            ..GatingConfig::default()
        };

        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.95);
        probs.insert("opt_b".to_string(), 0.05);

        let mut ans = Answer::choice("opt_a", probs, 0.90);
        apply_gating_to_answer(&mut ans, Some("q1"), None, Some(&disabled_cfg), &calib);
        assert!(ans.gating.is_none());
    }
}
