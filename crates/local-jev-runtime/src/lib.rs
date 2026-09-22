//! # local-jev-runtime
//!
//! 高速トークナイズおよび ONNX Runtime による超低遅延推論パイプラインを提供するクレート。
//!
//! ## 概要
//! - Hugging Face `tokenizers` を用いたトークナイズ処理。
//! - ONNX Runtime (`ort`) を用いた単一フォワードパス決定ヘッド推論。
//! - 候補数が多い場合の粗密探索(Coarse-to-Fine)やバッチ推論。

pub use local_jev_core as core;

pub mod engine;
pub mod error;
pub mod tokenizer;

pub use engine::{
    BatchScratchpad, CandidateEmbeddingCache, CoarseScorer, CoarseToFineConfig,
    DEFAULT_COARSE_THRESHOLD, DEFAULT_MAX_BATCH_CHUNK_SIZE, DEFAULT_NEGATIVE_KEYS,
    DEFAULT_TOP_M_CANDIDATES, EmbeddingCoarseScorer, ExecutionProvider, FilteredCandidates,
    InferenceEngine, LexicalCoarseScorer, OptimizationLevel, SessionConfig, filter_top_candidates,
    is_negative_candidate, reconstruct_probabilities,
};
pub use error::{Result, RuntimeError};
pub use tokenizer::{BatchTokenizedQuestions, JevTokenizer, TokenizedQuestion};

/// ランタイムの初期化確認用関数。
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }
}
