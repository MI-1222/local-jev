//! # local-jev-server
//!
//! Axum をベースとした Jev 互換 HTTP API サーバーを提供するクレート。
//!
//! ## 概要
//! - `POST /v1/systemone` エンドポイントの実装。
//! - リクエスト検証および前処理ガードレール。
//! - 確信度ゲーティングによるルーティング制御。

pub use local_jev_core as core;
pub use local_jev_runtime as runtime;

/// サーバーの初期化確認用関数。
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
