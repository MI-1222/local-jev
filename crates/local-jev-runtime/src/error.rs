//! # ランタイムエラー型定義モジュール
//!
//! トークナイザー処理および推論パイプラインで発生するエラーを定義する。

use thiserror::Error;

/// ランタイム推論エンジンおよびトークナイザーで発生するエラー型。
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// トークナイザー定義ファイルの読み込み失敗。
    #[error("トークナイザーファイルの読み込みに失敗しました: {0}。")]
    TokenizerLoadError(String),

    /// 必須特殊トークンが未定義。
    #[error("必須特殊トークン '{token}' がトークナイザー内に存在しません。")]
    MissingSpecialToken {
        /// 未定義の特殊トークン文字列。
        token: &'static str,
    },

    /// オプションマーカートークン [OP] が単一トークンとして分割されていない。
    #[error(
        "オプションマーカー [OP] の分割テストに失敗しました: 期待ID={expected}, 実際={actual:?}。"
    )]
    InvalidOptionMarkerEncoding {
        /// 期待される単一トークンID。
        expected: u32,
        /// 実際にエンコードされたトークンID列。
        actual: Vec<u32>,
    },

    /// トークナイズ処理時の内部エラー。
    #[error("トークナイズ処理中にエラーが発生しました: {0}。")]
    TokenizerEncodeError(String),

    /// プロンプトの固定部が最大許容系列長を超過している。
    #[error(
        "プロンプト固定長 ({fixed_len}) が最大系列長 ({max_len}) を超過しており、State を収容できません。"
    )]
    PromptExceedsMaxLength {
        /// 固定部分の合計トークン数。
        fixed_len: usize,
        /// 許容最大系列長。
        max_len: usize,
    },

    /// 検出された [OP] マーカー数と質問候補数の不一致。
    #[error("検出された [OP] マーカー数 ({detected}) が候補数 ({expected}) と一致しません。")]
    OptionMarkerCountMismatch {
        /// 検出されたマーカー数。
        detected: usize,
        /// 期待される候補数。
        expected: usize,
    },

    /// 不正な質問スキーマまたは評価基準。
    #[error("質問仕様が不正です: {0}。")]
    InvalidQuestion(String),

    /// コア数理・型スキーマクレート由来のエラー。
    #[error("コアエラー: {0}。")]
    Core(#[from] local_jev_core::error::CoreError),
}

/// ランタイム処理の標準結果型。
pub type Result<T> = std::result::Result<T, RuntimeError>;
