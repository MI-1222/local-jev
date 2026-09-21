//! # Jev 公式互換ペイロードのシリアライズ/デシリアライズ双方向結合テスト
//!
//! TypeSafe AI の `POST /v1/systemone` リクエスト・レスポンススキーマとの
//! 完全な互換性を検証する。

use indexmap::IndexMap;
use local_jev_core::{Answer, CoreError, QuestionType, SystemOneRequest, SystemOneResponse, Usage};
use serde_json::json;

/// 典型的な Choice, Score, Noul が混在するリクエスト JSON のデシリアライズテスト。
#[test]
fn test_deserialize_mixed_request() {
    let payload = json!({
        "model": "local-jev-large",
        "state": "ユーザーから注文 #12345 の配送遅延について苦情が届いた。",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "問い合わせの主な意図を分類せよ。",
                "criteria": {
                    "shipping_inquiry": "配送状況の確認",
                    "complaint": "対応への苦情・クレーム",
                    "cancellation": "注文のキャンセル要望"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "対応の緊急度を 3 段階で評価せよ。",
                "criteria": [
                    "通常対応(24時間以内)",
                    "優先対応(本日中)",
                    "至急対応(即時)"
                ]
            },
            "is_angry": {
                "type": "noul",
                "instructions": "顧客は強い怒りを示しているか。"
            }
        }
    });

    let json_str = serde_json::to_string(&payload).unwrap();
    let request: SystemOneRequest = serde_json::from_str(&json_str).unwrap();

    assert_eq!(request.model.as_deref(), Some("local-jev-large"));
    assert_eq!(request.questions.len(), 3);

    // バリデーションが正常に通ることを検証する。
    assert!(request.validate().is_ok());

    // 各質問のプロパティを検証する。
    let q_intent = request.questions.get("intent").unwrap();
    assert_eq!(q_intent.question_type, QuestionType::Choice);
    assert_eq!(q_intent.criteria.as_ref().unwrap().len(), 3);

    let q_urgency = request.questions.get("urgency").unwrap();
    assert_eq!(q_urgency.question_type, QuestionType::Score);
    assert_eq!(q_urgency.criteria.as_ref().unwrap().len(), 3);

    let q_angry = request.questions.get("is_angry").unwrap();
    assert_eq!(q_angry.question_type, QuestionType::Noul);
    assert!(q_angry.criteria.is_none() || q_angry.criteria.as_ref().unwrap().is_none());
}

/// 構造化オブジェクトや配列形式の State を受け入れるテスト。
#[test]
fn test_structured_state_deserialization() {
    let payload = json!({
        "state": {
            "user_id": 9999,
            "cart_total": 45000,
            "items": ["keyboard", "monitor"]
        },
        "questions": {
            "high_value": {
                "type": "noul",
                "instructions": "高額決済に該当するか。"
            }
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert!(request.validate().is_ok());
    assert!(request.state.is_object());
}

/// Jev 互換レスポンスのシリアライズおよびデシリアライズ双方向(Round-trip)テスト。
#[test]
fn test_response_round_trip() {
    let mut answers = IndexMap::new();

    let mut choice_probs = IndexMap::new();
    choice_probs.insert("refund".to_string(), 0.88);
    choice_probs.insert("other".to_string(), 0.12);
    answers.insert(
        "intent".to_string(),
        Answer::choice("refund", choice_probs, 0.91),
    );

    let mut score_probs = IndexMap::new();
    score_probs.insert("level_0".to_string(), 0.05);
    score_probs.insert("level_1".to_string(), 0.25);
    score_probs.insert("level_2".to_string(), 0.70);
    answers.insert(
        "severity".to_string(),
        Answer::score(1.65, score_probs, 0.84),
    );

    answers.insert("is_verified".to_string(), Answer::noul(0.96));

    let usage = Usage::new(142);
    let original_response = SystemOneResponse::new(answers, usage);

    // シリアライズして JSON 文字列化する。
    let serialized = serde_json::to_string_pretty(&original_response).unwrap();

    // 出力 JSON に completion_tokens: 0 が含まれていることを検証する。
    assert!(serialized.contains(r#""completion_tokens": 0"#));

    // デシリアライズして元の構造体と一致することを検証する。
    let deserialized: SystemOneResponse = serde_json::from_str(&serialized).unwrap();
    assert_eq!(original_response, deserialized);
}

/// バリデーション異常系: 空の questions。
#[test]
fn test_validation_empty_questions() {
    let payload = json!({
        "state": "テキスト",
        "questions": {}
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert_eq!(request.validate().unwrap_err(), CoreError::EmptyQuestions);
}

/// バリデーション異常系: Choice で criteria が欠落している場合。
#[test]
fn test_validation_choice_missing_criteria() {
    let payload = json!({
        "state": "テキスト",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "分類せよ。"
            }
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert_eq!(
        request.validate().unwrap_err(),
        CoreError::MissingCriteria {
            question_type: "choice".to_string(),
        }
    );
}

/// バリデーション異常系: Choice で criteria が配列(List)形式で渡された場合。
#[test]
fn test_validation_choice_invalid_criteria_type() {
    let payload = json!({
        "state": "テキスト",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "分類せよ。",
                "criteria": ["opt1", "opt2"]
            }
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert_eq!(
        request.validate().unwrap_err(),
        CoreError::InvalidCriteriaType {
            question_type: "choice".to_string(),
            expected: "Map (オブジェクト)",
            actual: "List (配列)",
        }
    );
}

/// バリデーション異常系: Score の段階数が 2 未満または 10 超過の場合。
#[test]
fn test_validation_score_invalid_count() {
    // 段階数が 1 の場合
    let payload_single = json!({
        "state": "テキスト",
        "questions": {
            "rating": {
                "type": "score",
                "instructions": "評価せよ。",
                "criteria": ["レベル1のみ"]
            }
        }
    });

    let req_single: SystemOneRequest = serde_json::from_value(payload_single).unwrap();
    assert_eq!(
        req_single.validate().unwrap_err(),
        CoreError::InvalidScoreLevelCount { count: 1 }
    );

    // 段階数が 11 の場合
    let payload_eleven = json!({
        "state": "テキスト",
        "questions": {
            "rating": {
                "type": "score",
                "instructions": "評価せよ。",
                "criteria": (0..11).map(|i| format!("レベル{i}")).collect::<Vec<_>>()
            }
        }
    });

    let req_eleven: SystemOneRequest = serde_json::from_value(payload_eleven).unwrap();
    assert_eq!(
        req_eleven.validate().unwrap_err(),
        CoreError::InvalidScoreLevelCount { count: 11 }
    );
}
