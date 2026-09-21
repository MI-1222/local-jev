//! # local-jev-server スタンドアロン実行バイナリ
//!
//! Jev 互換 HTTP API サーバーを直接起動するためのエントリーポイント。

fn main() {
    println!(
        "local-jev-server v{} が起動しました。",
        local_jev_server::version()
    );
}
