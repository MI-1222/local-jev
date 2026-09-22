//! # 推論エンジンモジュール
//!
//! ONNX Runtime を用いた低遅延フォワードパス、セッション管理、
//! および Execution Provider 解決機構を提供する。

pub mod batch;
pub mod coarse;
pub mod config;
pub mod provider;
pub mod session;

pub use batch::{BatchScratchpad, DEFAULT_MAX_BATCH_CHUNK_SIZE};
pub use coarse::{
    CandidateEmbeddingCache, CoarseScorer, CoarseToFineConfig, DEFAULT_COARSE_THRESHOLD,
    DEFAULT_NEGATIVE_KEYS, DEFAULT_TOP_M_CANDIDATES, EmbeddingCoarseScorer, FilteredCandidates,
    LexicalCoarseScorer, filter_top_candidates, is_negative_candidate, reconstruct_probabilities,
};
pub use config::{ExecutionProvider, OptimizationLevel, SessionConfig};
pub use provider::register_execution_providers;
pub use session::InferenceEngine;
