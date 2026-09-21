//! # local-jev-cli
//!
//! Local-Jev のコマンドラインインターフェース。
//!
//! ## 概要
//! - モデル推論テスト、サーバー起動、ベンチマーク測定、量子化管理などのサブコマンドを提供。

fn main() {
    println!("local-jev-cli v{}", env!("CARGO_PKG_VERSION"));
    println!(
        "連携クレート: local-jev-core v{}",
        local_jev_core::version()
    );
    println!(
        "連携クレート: local-jev-runtime v{}",
        local_jev_runtime::version()
    );
    println!(
        "連携クレート: local-jev-server v{}",
        local_jev_server::version()
    );
}
