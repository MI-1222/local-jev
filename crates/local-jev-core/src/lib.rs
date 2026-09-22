//! # local-jev-core
//!
//! TypeSafe AI の Jev 互換データ型、決定プリミティブ数理、および Serde スキーマを提供するクレート。
//!
//! ## 概要
//! - Jev 互換の API スキーマ(`SystemOneRequest`, `SystemOneResponse`)の定義。
//! - 決定プリミティブ(`Choice`, `Score`, `Noul`)の基本型定義とバリデーション。
//! - 決定プリミティブ数理(ソフトマックス、エントロピー確信度、分散確信度、安定シグモイド)の提供。
//! - 決定解決層(`evaluate_choice`, `evaluate_score`, `evaluate_noul`, `evaluate_question`)の提供。
//! - 柔軟な `Criteria` 型(Map, List, None)のサポート。
//! - 成果物引き渡し契約(`calibration.json` スキーマ、ONNX テンソル仕様)の定義。

pub mod contract;
pub mod decision;
pub mod error;
pub mod gating;
pub mod math;
pub mod schema;

pub use contract::*;
pub use decision::*;
pub use error::{CoreError, Result};
pub use gating::*;
pub use math::*;
pub use schema::*;

/// ライブラリのバージョン文字列を取得する。
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}
