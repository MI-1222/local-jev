//! # CLI 引数パース統合テスト
//!
//! `clap` を用いた各サブコマンドの引数パース、
//! デフォルト値設定、およびオプション指定の正当性を検証する。

use std::path::PathBuf;

use clap::Parser;
use local_jev_cli::{Cli, Commands};

#[test]
fn test_parse_export_openapi_default() {
    let args = ["local-jev", "export-openapi"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::ExportOpenapi { output } => {
            assert_eq!(output, None);
        }
        _ => panic!("ExportOpenapi コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_export_openapi_with_output() {
    let args = ["local-jev", "export-openapi", "--output", "openapi.json"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::ExportOpenapi { output } => {
            assert_eq!(output, Some(PathBuf::from("openapi.json")));
        }
        _ => panic!("ExportOpenapi コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_serve_defaults() {
    let args = ["local-jev", "serve"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
            provider,
            intra_threads,
            inter_threads,
        } => {
            assert_eq!(host, "0.0.0.0");
            assert_eq!(port, 3000);
            assert_eq!(model_dir, PathBuf::from("models/default"));
            assert_eq!(pool_size, 2);
            assert_eq!(provider, None);
            assert_eq!(intra_threads, None);
            assert_eq!(inter_threads, None);
        }
        _ => panic!("Serve コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_serve_custom_options() {
    let args = [
        "local-jev",
        "serve",
        "-H",
        "127.0.0.1",
        "-p",
        "8080",
        "-m",
        "custom_models",
        "--pool-size",
        "4",
        "--provider",
        "cpu",
        "--intra-threads",
        "4",
        "--inter-threads",
        "2",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
            provider,
            intra_threads,
            inter_threads,
        } => {
            assert_eq!(host, "127.0.0.1");
            assert_eq!(port, 8080);
            assert_eq!(model_dir, PathBuf::from("custom_models"));
            assert_eq!(pool_size, 4);
            assert_eq!(provider, Some("cpu".to_string()));
            assert_eq!(intra_threads, Some(4));
            assert_eq!(inter_threads, Some(2));
        }
        _ => panic!("Serve コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_benchmark_defaults() {
    let args = ["local-jev", "benchmark"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Benchmark {
            model_dir,
            iterations,
            warmup,
            provider,
            scenario,
            json,
        } => {
            assert_eq!(model_dir, PathBuf::from("models/default"));
            assert_eq!(iterations, 50);
            assert_eq!(warmup, 10);
            assert_eq!(provider, None);
            assert_eq!(scenario, "all");
            assert!(!json);
        }
        _ => panic!("Benchmark コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_benchmark_custom_options() {
    let args = [
        "local-jev",
        "benchmark",
        "-m",
        "bench_model",
        "-n",
        "100",
        "-w",
        "20",
        "--provider",
        "coreml",
        "-s",
        "batch",
        "--json",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Benchmark {
            model_dir,
            iterations,
            warmup,
            provider,
            scenario,
            json,
        } => {
            assert_eq!(model_dir, PathBuf::from("bench_model"));
            assert_eq!(iterations, 100);
            assert_eq!(warmup, 20);
            assert_eq!(provider, Some("coreml".to_string()));
            assert_eq!(scenario, "batch");
            assert!(json);
        }
        _ => panic!("Benchmark コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_quantize_defaults() {
    let args = ["local-jev", "quantize"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Quantize {
            model_dir,
            output_dir,
            per_channel,
            verify,
        } => {
            assert_eq!(model_dir, PathBuf::from("models/default"));
            assert_eq!(output_dir, PathBuf::from("models/quantized"));
            assert!(per_channel);
            assert!(verify);
        }
        _ => panic!("Quantize コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_quantize_custom_options() {
    let args = [
        "local-jev",
        "quantize",
        "--model-dir",
        "input_dir",
        "--output-dir",
        "output_dir",
        "--per-channel=false",
        "--verify=false",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Quantize {
            model_dir,
            output_dir,
            per_channel,
            verify,
        } => {
            assert_eq!(model_dir, PathBuf::from("input_dir"));
            assert_eq!(output_dir, PathBuf::from("output_dir"));
            assert!(!per_channel);
            assert!(!verify);
        }
        _ => panic!("Quantize コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_invalid_command() {
    let args = ["local-jev", "unknown-command"];
    let result = Cli::try_parse_from(args);
    assert!(
        result.is_err(),
        "未定義のコマンドはパースエラーとなること。"
    );
}
