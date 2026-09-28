//! # エンドポイントハンドラ集約モジュール
//!
//! 推論 API および運用監視 API のハンドラを再エクスポートする。

pub mod ops;
pub mod systemone;

pub use ops::{HealthStatusResponse, health_handler, metrics_handler, ready_handler};
pub use systemone::system_one_handler;
