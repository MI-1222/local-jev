//! # local-jev-cli
//!
//! Local-Jev のコマンドライン運用・管理ツール。
//!
//! ## 概要
//! - OpenAPI 3.1 仕様書の静的エクスポート (`export-openapi`)。
//! - HTTP API サーバーの起動 (`serve`)。

use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use local_jev_core::contract::calibration::CalibrationConfig;
use local_jev_runtime::engine::{InferenceEngine, SessionConfig};
use local_jev_runtime::tokenizer::JevTokenizer;
use local_jev_server::metrics::setup_metrics_recorder;
use local_jev_server::openapi::export_openapi_json;
use local_jev_server::state::AppState;

/// Local-Jev CLI 管理ツールのコマンドライン引数定義。
#[derive(Debug, Parser)]
#[command(
    name = "local-jev",
    version,
    about = "TypeSafe AI Jev 互換の非自己回帰型判断特化モデル CLI ツール。"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// 利用可能なサブコマンド一覧。
#[derive(Debug, Subcommand)]
enum Commands {
    /// OpenAPI 3.1 仕様書を JSON 形式で出力する。
    ExportOpenapi {
        /// 出力先ファイルパス(省略時は標準出力)。
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Jev 互換 HTTP API サーバーを起動する。
    Serve {
        /// バインド先ホストアドレス。
        #[arg(short = 'H', long, default_value = "0.0.0.0")]
        host: String,

        /// バインド先ポート番号。
        #[arg(short, long, default_value_t = 3000)]
        port: u16,

        /// モデルファイル格納ディレクトリ。
        #[arg(short, long, default_value = "models/default")]
        model_dir: PathBuf,

        /// セッションプールサイズ。
        #[arg(long, default_value_t = 2)]
        pool_size: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ExportOpenapi { output } => {
            let json_str = export_openapi_json();
            if let Some(path) = output {
                fs::write(&path, &json_str)?;
                println!("OpenAPI 3.1 仕様書を正常に出力しました: {}", path.display());
            } else {
                println!("{json_str}");
            }
        }
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
        } => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "local_jev_server=info,tower_http=info".into()),
                )
                .init();

            tracing::info!(
                "local-jev サーバーを起動します (モデル: {}, アドレス: {}:{})...",
                model_dir.display(),
                host,
                port
            );

            let prometheus_handle = match setup_metrics_recorder() {
                Ok(handle) => Some(handle),
                Err(err) => {
                    tracing::warn!("Prometheus メトリクス初期化をスキップしました: {err}");
                    None
                }
            };

            let onnx_path = model_dir.join("model.onnx");
            let tok_path = model_dir.join("tokenizer.json");
            let calib_path = model_dir.join("calibration.json");

            let session_config = SessionConfig {
                pool_size: pool_size.max(1),
                ..Default::default()
            };

            let engine = Arc::new(InferenceEngine::new(&onnx_path, session_config)?);
            let tokenizer = Arc::new(JevTokenizer::from_file(&tok_path)?);

            let calib_config = if calib_path.exists() {
                let calib_json = fs::read_to_string(&calib_path)?;
                Arc::new(CalibrationConfig::from_json_str(&calib_json)?)
            } else {
                Arc::new(CalibrationConfig::default())
            };

            let state = Arc::new(AppState::new(engine, tokenizer, calib_config));
            let addr: SocketAddr = format!("{host}:{port}").parse()?;

            local_jev_server::run_server(addr, state, prometheus_handle).await?;
        }
    }

    Ok(())
}
