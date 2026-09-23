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
        "hardware_defect".to_string(),
        "製品初期不良・無償交換対応".to_string(),
    );
    criteria.insert(
        "pairing_guide".to_string(),
        "ペアリング・接続設定手順の案内".to_string(),
    );
    criteria.insert(
        "driver_update".to_string(),
        "ファームウェア・ドライバ更新の案内".to_string(),
    );

    let question = Question::new_choice(
        "ユーザーの問い合わせ内容から最も適切なサポート対応カテゴリを選択せよ。",
        criteria,
    );
    let state = "昨日届いたキーボードのBluetoothが途切れて全く接続できません。";

    let tokenized = tokenizer.encode_question(state, &question).unwrap();

    // Python 側 (train/data/formatter.py) で実測された期待値。
    let expected_input_ids: Vec<i64> = vec![
        6, 1765, 287, 271, 5347, 21395, 11085, 291, 473, 10200, 83961, 573, 4383, 4697, 6839, 278,
        25, 26996, 496, 287, 271, 18675, 12514, 1864, 362, 4560, 8938, 2321, 1185, 9393, 12615,
        24835, 278, 25, 39806, 287, 271, 102400, 271, 2546, 76509, 312, 33362, 2611, 1185, 271,
        102400, 271, 47071, 312, 4697, 2049, 9451, 34379, 271, 102400, 271, 61605, 312, 32974,
        2802, 34379, 4,
    ];
    let expected_op_indices: Vec<i64> = vec![37, 46, 55];

    assert_eq!(tokenized.input_ids, expected_input_ids);
    assert_eq!(tokenized.op_indices, expected_op_indices);
    assert_eq!(
        tokenized.option_keys,
        vec!["hardware_defect", "pairing_guide", "driver_update"]
    );
}

#[test]
fn test_python_parity_noul_question() {
    let Some(path) = default_tokenizer_path() else {
        return;
    };

    let tokenizer = JevTokenizer::from_file(&path).unwrap();
    let question = Question::new_noul("言明「注文は正常に処理されている」の真偽を判定せよ。");
    let state = "本注文は決済が正常に完了しており、出荷準備段階に移行しています。";

    let tokenized = tokenizer.encode_question(state, &question).unwrap();

    // Python 側で実測された期待値。
    let expected_input_ids: Vec<i64> = vec![
        6, 1765, 287, 271, 686, 4772, 302, 6765, 75669, 7618, 3831, 275, 9362, 3823, 8162, 36512,
        835, 278, 25, 26996, 496, 287, 271, 7116, 2428, 313, 4772, 302, 45520, 3182, 1576, 316,
        291, 87485, 315, 12919, 24835, 278, 25, 39806, 287, 271, 102400, 271, 2023, 310, 3855, 293,
        271, 102400, 271, 19456, 310, 7752, 293, 4,
    ];
    let expected_op_indices: Vec<i64> = vec![42, 49];

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
    criteria.insert("opt_a".to_string(), "第1候補の説明".to_string());
    criteria.insert("opt_b".to_string(), "第2候補の説明".to_string());
    criteria.insert("opt_c".to_string(), "第3候補の説明".to_string());
    criteria.insert("opt_d".to_string(), "第4候補の説明".to_string());

    let question = Question::new_choice("カテゴリを選択せよ。", criteria);
    let long_state = "非常に長いシステムログログログ。".repeat(30);

    let max_len = 96;
    let tokenized = tokenizer
        .encode_question_with_max_len(&long_state, &question, max_len)
        .unwrap();

    // Python 側で実測された期待値。
    let expected_input_ids: Vec<i64> = vec![
        6, 1765, 287, 271, 3000, 5782, 1665, 12169, 12169, 12169, 278, 3000, 5782, 1665, 12169,
        12169, 12169, 278, 3000, 5782, 1665, 12169, 12169, 12169, 278, 3000, 5782, 1665, 12169,
        12169, 12169, 278, 3000, 5782, 1665, 12169, 12169, 12169, 278, 3000, 5782, 1665, 12169,
        12169, 12169, 278, 3000, 5782, 1665, 12169, 12169, 12169, 278, 3000, 5782, 25, 26996, 496,
        287, 271, 9393, 12615, 24835, 278, 25, 39806, 287, 271, 102400, 271, 614, 277, 9904, 11056,
        271, 102400, 271, 614, 279, 9904, 11056, 271, 102400, 271, 614, 289, 9904, 11056, 271,
        102400, 271, 614, 294, 9904, 11056, 4,
    ];
    let expected_op_indices: Vec<i64> = vec![68, 75, 82, 89];

    assert_eq!(tokenized.input_ids, expected_input_ids);
    assert_eq!(tokenized.op_indices, expected_op_indices);
    assert_eq!(tokenized.input_ids.len(), max_len);
    assert_eq!(
        tokenized.option_keys,
        vec!["opt_a", "opt_b", "opt_c", "opt_d"]
    );
}
