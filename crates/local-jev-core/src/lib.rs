//! # local-jev-core
//!
//! TypeSafe AI の Jev 互換データ型、決定プリミティブ数理、および Serde スキーマを提供するクレート。
//!
//! ## 概要
//! - Jev 互換の API スキーマ(`SystemOneRequest`, `SystemOneResponse`)の定義。
//! - 決定プリミティブ(`Choice`, `Score`, `Noul`)の基本型定義とバリデーション。
//! - 柔軟な `Criteria` 型(Map, List, None)のサポート。
//! - 確率較正・エントロピー確信度等の数理計算ユーティリティ。

pub mod error;
pub mod math;
pub mod schema;

pub use error::{CoreError, Result};
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
        assert_eq!(version(), "0.1.0");
    }
}
