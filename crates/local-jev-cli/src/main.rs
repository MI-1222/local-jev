//! # local-jev-cli
//!
//! Local-Jev のコマンドライン運用・管理・性能検証ツール。
//!
//! ## 概要
//! - HTTP API サーバーの起動 (`serve`)。
//! - インプロセス推論レイテンシ・スループット測定 (`benchmark`)。
//! - ONNX INT8 PTQ (Post-Training Quantization) 量子化 (`quantize`)。
//! - OpenAPI 3.1 仕様書の静的エクスポート (`export-openapi`)。

use clap::Parser;
use local_jev_cli::commands::benchmark::{BenchmarkArgs, run_benchmark};
use local_jev_cli::commands::export_openapi::run_export_openapi;
use local_jev_cli::commands::healthcheck::{HealthcheckArgs, run_healthcheck};
use local_jev_cli::commands::quantize::{QuantizeArgs, run_quantize};
use local_jev_cli::commands::serve::{ServeArgs, run_serve};
use local_jev_cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ExportOpenapi { output } => {
            run_export_openapi(output)?;
        }
        Commands::Healthcheck { url, timeout_secs } => {
            let args = HealthcheckArgs { url, timeout_secs };
            run_healthcheck(args)?;
        }
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
            provider,
            intra_threads,
            inter_threads,
        } => {
            let args = ServeArgs {
                host,
                port,
                model_dir,
                pool_size,
                provider,
                intra_threads,
                inter_threads,
            };
            run_serve(args).await?;
        }
        Commands::Benchmark {
            model_dir,
            iterations,
            warmup,
            provider,
            scenario,
            json,
        } => {
            let args = BenchmarkArgs {
                model_dir,
                iterations,
                warmup,
                provider,
                scenario,
                json,
            };
            // CPU バウンドな計算ループをワーカースレッドのブロッキングから分離
            tokio::task::spawn_blocking(move || run_benchmark(args)).await??;
        }
        Commands::Quantize {
            model_dir,
            output_dir,
            per_channel,
            verify,
        } => {
            let args = QuantizeArgs {
                model_dir,
                output_dir,
                per_channel,
                verify,
            };
            tokio::task::spawn_blocking(move || run_quantize(args)).await??;
        }
    }

    Ok(())
}
