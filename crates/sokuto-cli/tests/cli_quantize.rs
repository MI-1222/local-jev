//! # CLI 量子化統合テスト
//!
//! `sokuto quantize` による動的 INT8 PTQ 量子化と
//! 成果物バンドルの生成、および Rust ランタイムでのロード可能性を検証する。

use std::fs;
use std::path::PathBuf;

use sokuto_cli::commands::quantize::{QuantizeArgs, run_quantize};
use sokuto_runtime::engine::{InferenceEngine, SessionConfig};

#[test]
fn test_run_quantize_bundle_generation() {
    let project_root = {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(parent) = manifest_dir.parent() {
            if let Some(root) = parent.parent() {
                root.to_path_buf()
            } else {
                manifest_dir.clone()
            }
        } else {
            manifest_dir.clone()
        }
    };
    let model_dir = if PathBuf::from("models/default").exists() {
        PathBuf::from("models/default")
    } else if project_root.join("models/default").exists() {
        project_root.join("models/default")
    } else {
        PathBuf::from("models/default")
    };

    if !model_dir.exists() || !model_dir.join("model.onnx").exists() {
        eprintln!(
            "models/default/model.onnx が存在しないためスキップします (検索パス: {})。",
            model_dir.display()
        );
        return;
    }

    let temp_output_dir = std::env::temp_dir().join("sokuto_test_quantized_bundle");
    if temp_output_dir.exists() {
        let _ = fs::remove_dir_all(&temp_output_dir);
    }

    let args = QuantizeArgs {
        model_dir,
        output_dir: temp_output_dir.clone(),
        per_channel: true,
        verify: true,
    };

    let result = run_quantize(args);
    assert!(result.is_ok(), "量子化処理が正常に完了すること。");

    // 成果物ファイルの存在検証
    let quant_model = temp_output_dir.join("model.onnx");
    let quant_tok = temp_output_dir.join("tokenizer.json");
    let quant_meta = temp_output_dir.join("quantize_metadata.json");

    assert!(quant_model.exists(), "量子化後 model.onnx が存在すること。");
    assert!(quant_tok.exists(), "tokenizer.json が複製されていること。");
    assert!(
        quant_meta.exists(),
        "quantize_metadata.json が生成されていること。"
    );

    // Rust 推論エンジンでのロード可能性スモークテスト
    let engine = InferenceEngine::new(&quant_model, SessionConfig::cpu_only());
    assert!(
        engine.is_ok(),
        "量子化モデルが InferenceEngine で初期化可能であること。"
    );

    // テスト後クリーンアップ (一時ディレクトリの削除)
    let _ = fs::remove_dir_all(&temp_output_dir);
}
