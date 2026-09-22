//! # CLI OpenAPI エクスポート統合テスト
//!
//! `local-jev export-openapi` による静的スキーマ生成と
//! ファイル出力の完全性を検証する。

use local_jev_server::openapi::export_openapi_json;
use serde_json::Value;
use std::fs;

#[test]
fn test_export_openapi_json_structure() {
    let json_str = export_openapi_json();
    assert!(!json_str.is_empty());

    let parsed: Value = serde_json::from_str(&json_str)
        .expect("エクスポートされた文字列が有効な JSON であること。");

    // OpenAPI 3.1 仕様書の必須フィールド
    assert!(parsed.get("openapi").is_some());
    assert!(parsed.get("info").is_some());
    assert!(parsed.get("paths").is_some());
    assert!(parsed.get("components").is_some());

    // エンドポイントおよび主要スキーマの存在確認
    assert!(parsed["paths"]["/v1/systemone"].is_object());
    assert!(parsed["paths"]["/health"].is_object());
    assert!(parsed["paths"]["/ready"].is_object());
    assert!(parsed["paths"]["/metrics"].is_object());
    assert!(parsed["components"]["schemas"]["SystemOneRequest"].is_object());
    assert!(parsed["components"]["schemas"]["SystemOneResponse"].is_object());
}

#[test]
fn test_export_openapi_file_output() {
    let json_str = export_openapi_json();

    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("local_jev_test_openapi.json");

    fs::write(&temp_file, &json_str).expect("一時ファイルへの書き込みに成功すること。");

    let read_content =
        fs::read_to_string(&temp_file).expect("書き込んだファイルが読み込めること。");
    assert_eq!(read_content, json_str);

    // テスト後クリーンアップ
    let _ = fs::remove_file(&temp_file);
}
