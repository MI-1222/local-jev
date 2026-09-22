//! # local-jev-cli ライブラリクレート
//!
//! Local-Jev CLI のコマンド実装および構成パラメータを提供する。

pub mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Local-Jev CLI 管理ツールのコマンドライン引数定義。
#[derive(Debug, Parser)]
#[command(
    name = "local-jev",
    version,
    about = "TypeSafe AI Jev 互換の非自己回帰型判断特化モデル CLI ツール。"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// 利用可能なサブコマンド一覧。
#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum Commands {
    /// OpenAPI 3.1 仕様書を JSON 形式で出力する。
    ExportOpenapi {
        /// 出力先ファイルパス (省略時は標準出力)。
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

        /// 優先 Execution Provider (auto, coreml, cuda, tensorrt, cpu)。
        #[arg(long)]
        provider: Option<String>,

        /// 単一オペレータ内の並列スレッド数 (intra-op)。
        #[arg(long)]
        intra_threads: Option<usize>,

        /// 複数オペレータ間の並列スレッド数 (inter-op)。
        #[arg(long)]
        inter_threads: Option<usize>,
    },

    /// インプロセスでの推論レイテンシおよびスループットを計測する。
    Benchmark {
        /// モデルファイル格納ディレクトリ。
        #[arg(short, long, default_value = "models/default")]
        model_dir: PathBuf,

        /// 測定反復回数。
        #[arg(short = 'n', long, default_value_t = 50)]
        iterations: usize,

        /// ウォームアップ反復回数。
        #[arg(short = 'w', long, default_value_t = 10)]
        warmup: usize,

        /// 優先 Execution Provider (auto, coreml, cuda, tensorrt, cpu)。
        #[arg(long)]
        provider: Option<String>,

        /// 実行対象シナリオ (all, single, batch, coarse, scratchpad)。
        #[arg(short = 's', long, default_value = "all")]
        scenario: String,

        /// CI 連携用 JSON 形式出力フラグ。
        #[arg(long)]
        json: bool,
    },

    /// FP32 ONNX モデルに対して動的 INT8 PTQ 量子化を実行する。
    Quantize {
        /// 元モデルファイル群が存在するディレクトリパス。
        #[arg(short, long, default_value = "models/default")]
        model_dir: PathBuf,

        /// 量子化後成果物バンドルの出力先ディレクトリパス。
        #[arg(short, long, default_value = "models/quantized")]
        output_dir: PathBuf,

        /// チャンネル単位で重みを量子化する (デフォルト: 有効)。
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        per_channel: bool,

        /// FP32 モデルとのパリティ検証を実行する (デフォルト: 有効)。
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        verify: bool,
    },
}
