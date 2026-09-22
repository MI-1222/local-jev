//! # 推論セッション設定モジュール
//!
//! ONNX Runtime セッションの Execution Provider 優先度、
//! スレッド並列数、グラフ最適化レベルなどの構成パラメータを定義する。

use ort::session::builder::GraphOptimizationLevel;

/// 利用可能な Execution Provider(実行バックエンド)の指定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// 自動選択(利用可能なアクセラレータを優先し、利用不可時は CPU へフォールバック)。
    #[default]
    Auto,
    /// Apple Silicon 向け CoreML バックエンド。
    CoreML,
    /// NVIDIA GPU 向け CUDA バックエンド。
    CUDA,
    /// NVIDIA GPU 向け TensorRT バックエンド。
    TensorRT,
    /// CPU バックエンド(高移植性・標準フォールバック)。
    CPU,
}

impl ExecutionProvider {
    /// バックエンドの表示名を取得する。
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::CoreML => "CoreML",
            Self::CUDA => "CUDA",
            Self::TensorRT => "TensorRT",
            Self::CPU => "CPU",
        }
    }
}

/// グラフ最適化レベル。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// 最適化を無効化。
    Disable,
    /// 基本的な最適化(ノード結合、定数畳み込み等)。
    Basic,
    /// 拡張最適化(レイヤー正規化融合、GELU融合等)。
    Extended,
    /// すべての最適化を適用(デフォルト)。
    #[default]
    All,
}

impl OptimizationLevel {
    /// `ort` クレートの `GraphOptimizationLevel` に変換する。
    pub fn to_ort_level(self) -> GraphOptimizationLevel {
        match self {
            Self::Disable => GraphOptimizationLevel::Disable,
            Self::Basic => GraphOptimizationLevel::Level1,
            Self::Extended => GraphOptimizationLevel::Level2,
            Self::All => GraphOptimizationLevel::Level3,
        }
    }
}

/// ONNX Runtime セッション構築設定。
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// 優先的に試行する Execution Provider の順序付きリスト。
    pub preferred_providers: Vec<ExecutionProvider>,
    /// 単一オペレータ内の並列スレッド数(intra_op)。None の場合は自動設定。
    pub intra_threads: Option<usize>,
    /// 複数オペレータ間の並列スレッド数(inter_op)。None の場合は自動設定。
    pub inter_threads: Option<usize>,
    /// グラフ最適化レベル。
    pub optimization_level: OptimizationLevel,
    /// CPU メモリアリーナの有効化。
    pub enable_mem_arena: bool,
    /// 並行推論用セッションプールサイズ(デフォルト: 1)。
    pub pool_size: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        let default_providers = vec![ExecutionProvider::CoreML, ExecutionProvider::CPU];

        #[cfg(not(target_os = "macos"))]
        let default_providers = vec![
            ExecutionProvider::TensorRT,
            ExecutionProvider::CUDA,
            ExecutionProvider::CPU,
        ];

        Self {
            preferred_providers: default_providers,
            intra_threads: None,
            inter_threads: Some(1),
            optimization_level: OptimizationLevel::All,
            enable_mem_arena: true,
            pool_size: 1,
        }
    }
}

impl SessionConfig {
    /// CPU 推論に特化した設定を生成する。
    pub fn cpu_only() -> Self {
        Self {
            preferred_providers: vec![ExecutionProvider::CPU],
            intra_threads: None,
            inter_threads: Some(1),
            optimization_level: OptimizationLevel::All,
            enable_mem_arena: true,
            pool_size: 1,
        }
    }

    /// スレッド数を明示指定したビルダーメソッド。
    pub fn with_threads(mut self, intra: Option<usize>, inter: Option<usize>) -> Self {
        self.intra_threads = intra;
        self.inter_threads = inter;
        self
    }

    /// 最適化レベルを指定したビルダーメソッド。
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// メモリアリーナ有効化フラグを指定したビルダーメソッド。
    pub fn with_memory_arena(mut self, enable: bool) -> Self {
        self.enable_mem_arena = enable;
        self
    }

    /// セッションプールサイズを指定したビルダーメソッド。
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size.max(1);
        self
    }
}
