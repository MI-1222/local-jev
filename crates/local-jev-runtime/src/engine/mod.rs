//! # 推論エンジンモジュール
//!
//! ONNX Runtime を用いた低遅延フォワードパス、セッション管理、
//! および Execution Provider 解決機構を提供する。

pub mod config;
pub mod provider;
pub mod session;

pub use config::{ExecutionProvider, OptimizationLevel, SessionConfig};
pub use provider::register_execution_providers;
pub use session::InferenceEngine;
