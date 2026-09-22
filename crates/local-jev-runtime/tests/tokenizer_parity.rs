//! # Python-Rust トークンパリティ統合テスト
//!
//! Python 側の学習・前処理パイプライン (`train/data/formatter.py`) と
//! Rust 側のランタイムトークナイザー (`local-jev-runtime::JevTokenizer`) において、
//! 同一プロンプトに対する `input_ids` および `op_indices` が 1 トークン単位で
//! 完全に一致することを検証する。

use indexmap::IndexMap;
use local_jev_core::schema::Question;
use local_jev_runtime::JevTokenizer;
use std::path::PathBuf;

/// ワークスペース内の default tokenizer.json のパスを取得する。
fn default_tokenizer_path() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models")
        .join("default")
        .join("tokenizer.json");

    if !path.exists() {
        if std::env::var("CI").is_ok() {
            panic!(
                "CI 環境で必須トークナイザーファイルが見つかりません: {:?}",
                path
            );
        }
        eprintln!("スキップ: tokenizer.json が存在しません: {:?}", path);
        return None;
    }

    Some(path)
}

#[test]
fn test_python_parity_choice_question() {
    let Some(path) = default_tokenizer_path() else {
        return;
    };

    let tokenizer = JevTokenizer::from_file(&path).unwrap();

    let mut criteria = IndexMap::new();
    criteria.insert(
        "card_arrival".to_string(),
        "Card delivery status".to_string(),
    );
    criteria.insert("lost_card".to_string(), "Reporting lost card".to_string());
    criteria.insert("pin_reset".to_string(), "Resetting PIN code".to_string());

    let question = Question::new_choice("意図を分類せよ。", criteria);
    let state = "I lost my card yesterday.";

    let tokenized = tokenizer.encode_question(state, &question).unwrap();

    // Python 側 (train/data/test_formatter.py) で実測された期待値
    let expected_input_ids: Vec<i64> = vec![
        50281, 5443, 27, 209, 42, 3663, 619, 3120, 11066, 15, 187, 10548, 6477, 27, 209, 31129,
        10041, 113, 6449, 18434, 19780, 241, 24617, 15275, 4340, 187, 49489, 27, 209, 50368, 9858,
        6742, 3708, 209, 50368, 40283, 3663, 3120, 209, 50368, 2213, 33513, 39317, 2127, 50282,
    ];
    let expected_op_indices: Vec<i64> = vec![29, 34, 39];

    assert_eq!(tokenized.input_ids, expected_input_ids);
    assert_eq!(tokenized.op_indices, expected_op_indices);
    assert_eq!(
        tokenized.option_keys,
        vec!["card_arrival", "lost_card", "pin_reset"]
    );
}

#[test]
fn test_python_parity_noul_question() {
    let Some(path) = default_tokenizer_path() else {
        return;
    };

    let tokenizer = JevTokenizer::from_file(&path).unwrap();
    let question = Question::new_noul("言明「空は青い」の真偽を判定せよ。");
    let state = "The sky is blue.";

    let tokenized = tokenizer.encode_question(state, &question).unwrap();

    // Python 側で実測された期待値
    let expected_input_ids: Vec<i64> = vec![
        50281, 5443, 27, 209, 510, 8467, 310, 4797, 15, 187, 10548, 6477, 27, 209, 31982, 29687,
        13748, 45249, 6418, 19127, 229, 5151, 13752, 3917, 48561, 28915, 123, 6449, 6765, 99,
        17576, 24617, 15275, 4340, 187, 49489, 27, 209, 50368, 209, 48561, 313, 5088, 10, 209,
        50368, 209, 28915, 123, 313, 5653, 10, 50282,
    ];
    let expected_op_indices: Vec<i64> = vec![38, 45];

    assert_eq!(tokenized.input_ids, expected_input_ids);
    assert_eq!(tokenized.op_indices, expected_op_indices);
    assert_eq!(tokenized.option_keys, vec!["true", "false"]);
}

#[test]
fn test_python_parity_state_priority_truncation() {
    let Some(path) = default_tokenizer_path() else {
        return;
    };

    let tokenizer = JevTokenizer::from_file(&path).unwrap();

    let mut criteria = IndexMap::new();
    criteria.insert("opt_a".to_string(), "First option description".to_string());
    criteria.insert("opt_b".to_string(), "Second option description".to_string());
    criteria.insert("opt_c".to_string(), "Third option description".to_string());
    criteria.insert("opt_d".to_string(), "Fourth option description".to_string());

    let question = Question::new_choice("分類せよ。", criteria);
    let long_state =
        "This is an extremely long transaction log with hundreds of details. ".repeat(50);

    let max_len = 128;
    let tokenized = tokenizer
        .encode_question_with_max_len(&long_state, &question, max_len)
        .unwrap();

    // Python 側で実測された期待値
    let expected_input_ids: Vec<i64> = vec![
        50281, 5443, 27, 209, 1552, 310, 271, 6685, 1048, 9996, 2412, 342, 8307, 273, 4278, 15,
        831, 310, 271, 6685, 1048, 9996, 2412, 342, 8307, 273, 4278, 15, 831, 310, 271, 6685, 1048,
        9996, 2412, 342, 8307, 273, 4278, 15, 831, 310, 271, 6685, 1048, 9996, 2412, 342, 8307,
        273, 4278, 15, 831, 310, 271, 6685, 1048, 9996, 2412, 342, 8307, 273, 4278, 15, 831, 310,
        271, 6685, 1048, 9996, 2412, 342, 8307, 273, 4278, 15, 831, 310, 271, 6685, 1048, 9996,
        2412, 342, 8307, 273, 4278, 15, 831, 310, 271, 6685, 1048, 187, 10548, 6477, 27, 209,
        18434, 19780, 241, 24617, 15275, 4340, 187, 49489, 27, 209, 50368, 3973, 4500, 5740, 209,
        50368, 6347, 4500, 5740, 209, 50368, 12245, 4500, 5740, 209, 50368, 16650, 4500, 5740,
        50282,
    ];
    let expected_op_indices: Vec<i64> = vec![108, 113, 118, 123];

    assert_eq!(tokenized.input_ids, expected_input_ids);
    assert_eq!(tokenized.op_indices, expected_op_indices);
    assert_eq!(tokenized.input_ids.len(), max_len);
    assert_eq!(
        tokenized.option_keys,
        vec!["opt_a", "opt_b", "opt_c", "opt_d"]
    );
}
