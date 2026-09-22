//! # OpenAPI スキーマ整合性・回帰テストスイート
//!
//! Serde によるシリアライズ/デシリアライズの実測結果と、
//! `utoipa` から自動導出された OpenAPI 3.1 仕様書のプロパティ整合性を検証する。

use indexmap::IndexMap;
use local_jev_core::schema::{
    Answer, Criteria, Question, QuestionType, SystemOneRequest, SystemOneResponse, Usage,
};
use local_jev_server::openapi::generate_openapi_spec;

#[test]
fn test_openapi_spec_generation_and_components() {
    let spec = generate_openapi_spec();

    // 仕様書の基本メタデータ検証
    assert_eq!(spec.info.title, "Local-Jev HTTP API");
    assert_eq!(spec.info.version, env!("CARGO_PKG_VERSION"));

    // コンポーネントスキーマの登録確認
    let components = spec.components.expect("components が定義されていること。");
    let schemas = components.schemas;

    let required_schemas = [
        "SystemOneRequest",
        "SystemOneResponse",
        "Question",
        "QuestionType",
        "Criteria",
        "Answer",
        "Usage",
        "DecisionRoute",
        "GatingConfig",
        "GatingMetadata",
        "CandidateProbability",
        "EscalationContext",
        "SystemRoutingSummary",
        "ErrorResponse",
        "ErrorDetail",
        "HealthStatusResponse",
    ];

    for schema_name in required_schemas {
        assert!(
            schemas.contains_key(schema_name),
            "スキーマ '{schema_name}' が OpenAPI 仕様書に含まれていません。"
        );
    }
}

#[test]
fn test_decision_route_enum_casing_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let route_schema = &spec_json["components"]["schemas"]["DecisionRoute"];
    let enum_values = route_schema["enum"]
        .as_array()
        .expect("enum 定義が存在すること。");

    let string_values: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();

    // serde(rename_all = "snake_case") と完全一致すること
    assert_eq!(
        string_values,
        vec!["auto_execute", "confirm_or_escalate", "fallback"]
    );

    use local_jev_core::gating::DecisionRoute;
    assert_eq!(
        serde_json::to_string(&DecisionRoute::AutoExecute).unwrap(),
        "\"auto_execute\""
    );
    assert_eq!(
        serde_json::to_string(&DecisionRoute::ConfirmOrEscalate).unwrap(),
        "\"confirm_or_escalate\""
    );
    assert_eq!(
        serde_json::to_string(&DecisionRoute::Fallback).unwrap(),
        "\"fallback\""
    );
}

#[test]
fn test_question_type_enum_casing_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    // QuestionType の enum 定義を抽出
    let question_type_schema = &spec_json["components"]["schemas"]["QuestionType"];
    let enum_values = question_type_schema["enum"]
        .as_array()
        .expect("enum 定義が存在すること。");

    let string_values: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();

    // serde(rename_all = "lowercase") と完全一致すること
    assert_eq!(string_values, vec!["choice", "score", "noul"]);

    // Serde 側の実測値
    let serde_choice = serde_json::to_string(&QuestionType::Choice).unwrap();
    let serde_score = serde_json::to_string(&QuestionType::Score).unwrap();
    let serde_noul = serde_json::to_string(&QuestionType::Noul).unwrap();

    assert_eq!(serde_choice, "\"choice\"");
    assert_eq!(serde_score, "\"score\"");
    assert_eq!(serde_noul, "\"noul\"");
}

#[test]
fn test_question_field_rename_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let question_schema = &spec_json["components"]["schemas"]["Question"];
    let properties = &question_schema["properties"];

    // `question_type` ではなく `type` としてエクスポートされていること
    assert!(
        properties.get("type").is_some(),
        "Question スキーマに 'type' プロパティが存在すること。"
    );
    assert!(
        properties.get("question_type").is_none(),
        "内部フィールド名 'question_type' は露出していないこと。"
    );

    // Serde シリアライズ結果との突合
    let mut criteria_map = IndexMap::new();
    criteria_map.insert("a".to_string(), "オプション A".to_string());
    let question = Question::new_choice("指示文", criteria_map);

    let json_val = serde_json::to_value(&question).unwrap();
    assert!(json_val.get("type").is_some());
    assert!(json_val.get("question_type").is_none());
}

#[test]
fn test_systemone_request_roundtrip_and_schema_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let req_schema = &spec_json["components"]["schemas"]["SystemOneRequest"];
    let properties = &req_schema["properties"];

    // 必須および主要フィールドの確認
    assert!(properties.get("state").is_some());
    assert!(properties.get("questions").is_some());
    assert!(properties.get("model").is_some());

    // 実リクエストペイロードとの互換性
    let mut questions = IndexMap::new();
    questions.insert("q1".to_string(), Question::new_noul("言明文"));

    let req = SystemOneRequest::new("共通コンテキスト", questions);
    let req_json = serde_json::to_value(&req).unwrap();

    assert_eq!(req_json["state"], "共通コンテキスト");
    assert!(req_json["questions"]["q1"].is_object());
    assert_eq!(req_json["questions"]["q1"]["type"], "noul");
}

#[test]
fn test_systemone_response_schema_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let resp_schema = &spec_json["components"]["schemas"]["SystemOneResponse"];
    let properties = &resp_schema["properties"];

    assert!(properties.get("answers").is_some());
    assert!(properties.get("usage").is_some());

    let mut answers = IndexMap::new();
    answers.insert(
        "q1".to_string(),
        Answer::choice("opt_a", IndexMap::new(), 0.95),
    );

    let resp = SystemOneResponse::new(answers, Usage::new(42));
    let resp_json = serde_json::to_value(&resp).unwrap();

    assert!(resp_json["answers"]["q1"].is_object());
    assert_eq!(resp_json["answers"]["q1"]["choice"], "opt_a");
    assert_eq!(resp_json["answers"]["q1"]["confidence"], 0.95);
    assert_eq!(resp_json["usage"]["prompt_tokens"], 42);
    assert_eq!(resp_json["usage"]["completion_tokens"], 0);
    assert_eq!(resp_json["usage"]["total_tokens"], 42);
}

#[test]
fn test_criteria_untagged_schema_compatibility() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let criteria_schema = &spec_json["components"]["schemas"]["Criteria"];
    assert!(
        criteria_schema.is_object(),
        "Criteria スキーマがオブジェクト定義として存在すること。"
    );

    // 空オブジェクトに縮退せず、oneOf としてバリアントが定義されていることを厳密に検証
    let one_of = criteria_schema["oneOf"]
        .as_array()
        .expect("Criteria は oneOf 配列として定義されている必要があります。");
    assert_eq!(
        one_of.len(),
        3,
        "Criteria は Map, List, None の 3 バリアントを持つ必要があります。"
    );

    // 1. Choice Map (object)
    assert!(one_of.iter().any(|s| s["type"] == "object"));
    // 2. Score List (array)
    assert!(one_of.iter().any(|s| s["type"] == "array"));
    // 3. Noul None (null)
    assert!(one_of.iter().any(|s| s["type"] == "null"));

    // Map 形式
    let mut map = IndexMap::new();
    map.insert("k1".to_string(), "説明 1".to_string());
    let c_map = Criteria::Map(map);
    let json_map = serde_json::to_value(&c_map).unwrap();
    assert!(json_map.is_object());

    // List 形式
    let c_list = Criteria::List(vec!["段階1".to_string(), "段階2".to_string()]);
    let json_list = serde_json::to_value(&c_list).unwrap();
    assert!(json_list.is_array());

    // None 形式
    let c_none = Criteria::None;
    let json_none = serde_json::to_value(&c_none).unwrap();
    assert!(json_none.is_null());
}
