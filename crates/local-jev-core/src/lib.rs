//! # local-jev-core
//!
//! TypeSafe AI の Jev 互換データ型、決定プリミティブ数理、および Serde スキーマを提供するクレート。
//!
//! ## 概要
//! - Jev 互換の API スキーマ(リクエスト・レスポンス型)の定義。
//! - 決定プリミティブ(Choice, Score, Noul)の基本型定義。
//! - 確率較正・エントロピー確信度等の数理ユーティリティ。

/// ライブラリの初期化確認用関数。
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
