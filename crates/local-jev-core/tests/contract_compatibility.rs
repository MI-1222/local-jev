//! # 成果物引き渡し契約の整合性テスト
//!
//! Python 側で出力される典型的な `calibration.json` および
//! ONNX テンソル仕様が正しく解決・検証できることをテストする。

use local_jev_core::{
    CalibrationConfig, ModelInputDimensions, QuestionType, TENSOR_ATTENTION_MASK, TENSOR_INPUT_IDS,
    TENSOR_LOGITS, TENSOR_OP_INDICES, TOKEN_OPTION_MARKER,
};
use serde_json::json;

/// Python の CalibrationConfig が出力する JSON 形式との互換性テスト。
#[test]
fn test_python_calibration_json_compatibility() {
    let raw_json = json!({
        "version": "1.0",
        "default_temperature": 1.0,
        "temperature_map": {
            "choice": {
                "2": 1.05,
                "3-5": 1.12,
                "6-10": 1.20,
                "11+": 1.35
            },
            "score": {
                "2-5": 1.00,
                "6-10": 1.08
            },
            "noul": 0.95
        }
    });

    let json_str = serde_json::to_string_pretty(&raw_json).unwrap();
    let config = CalibrationConfig::from_json_str(&json_str).unwrap();

    assert_eq!(config.version, "1.0");
    assert_eq!(config.default_temperature, 1.0);

    // バケット解決のテスト
    assert_eq!(config.get_temperature(QuestionType::Choice, 2), 1.05);
    assert_eq!(config.get_temperature(QuestionType::Choice, 3), 1.12);
    assert_eq!(config.get_temperature(QuestionType::Choice, 5), 1.12);
    assert_eq!(config.get_temperature(QuestionType::Choice, 7), 1.20);
    assert_eq!(config.get_temperature(QuestionType::Choice, 100), 1.35);

    assert_eq!(config.get_temperature(QuestionType::Score, 2), 1.00);
    assert_eq!(config.get_temperature(QuestionType::Score, 5), 1.00);
    assert_eq!(config.get_temperature(QuestionType::Score, 8), 1.08);

    assert_eq!(config.get_temperature(QuestionType::Noul, 1), 0.95);

    // 定義外の候補数に対するフォールバック
    assert_eq!(config.get_temperature(QuestionType::Score, 1), 1.0);
}

/// テンソル仕様定数と形状バリデーションのテスト。
#[test]
fn test_model_tensor_spec_constants_and_shapes() {
    assert_eq!(TENSOR_INPUT_IDS, "input_ids");
    assert_eq!(TENSOR_ATTENTION_MASK, "attention_mask");
    assert_eq!(TENSOR_OP_INDICES, "op_indices");
    assert_eq!(TENSOR_LOGITS, "logits");
    assert_eq!(TOKEN_OPTION_MARKER, "[OP]");

    // バッチ 4、系列長 512、候補数 16 の形状検証
    let input_ids = [4, 512];
    let attention_mask = [4, 512];
    let op_indices = [4, 16];

    let dims =
        ModelInputDimensions::validate_tensor_shapes(&input_ids, &attention_mask, &op_indices)
            .unwrap();

    assert_eq!(dims.batch_size, 4);
    assert_eq!(dims.sequence_length, 512);
    assert_eq!(dims.num_options, 16);

    // 出力 logits 形状の照合
    assert!(dims.validate_output_shape(&[4, 16]).is_ok());
    assert!(dims.validate_output_shape(&[4, 15]).is_err());
}
