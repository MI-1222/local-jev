//! # エラー型モジュール
//!
//! `local-jev-core` クレート内で発生するドメインエラーおよび結果型を定義する。

use thiserror::Error;

/// `local-jev-core` の標準結果型。
pub type Result<T> = std::result::Result<T, CoreError>;

/// コアモジュールで発生するエラーの列挙型。
#[derive(Debug, Error, PartialEq)]
pub enum CoreError {
    /// 選択肢の数が制限範囲外である場合のエラー(Choice 型は 1〜255 個)。
    #[error("Choice 型の候補数は 1〜255 個である必要がありますが、{count} 個指定されました。")]
    InvalidChoiceCount {
        /// 指定された候補数。
        count: usize,
    },

    /// スコア評価尺度の段階数が範囲外である場合のエラー(Score 型は 2〜10 段階)。
    #[error("Score 型の評価尺度数は 2〜10 段階である必要がありますが、{count} 個指定されました。")]
    InvalidScoreLevelCount {
        /// 指定された段階数。
        count: usize,
    },

    /// 質問タイプに対する評価基準(Criteria)の形式が不一致である場合のエラー。
    #[error(
        "{question_type} 型に対して不正な Criteria 形式が指定されました。期待値: {expected}、実際: {actual}。"
    )]
    InvalidCriteriaType {
        /// 対象の質問タイプ名。
        question_type: String,
        /// 期待される形式名。
        expected: &'static str,
        /// 実際に指定された形式名。
        actual: &'static str,
    },

    /// 必須の評価基準(Criteria)が欠落している場合のエラー。
    #[error("{question_type} 型には Criteria の指定が必須です。")]
    MissingCriteria {
        /// 対象の質問タイプ名。
        question_type: String,
    },

    /// 質問リストが空である場合のエラー。
    #[error("リクエストに含まれる questions が空です。1 件以上の質問を指定してください。")]
    EmptyQuestions,

    /// 確率分布の計算やエントロピー算出における数値計算エラー。
    #[error("数値計算エラーが発生しました: {message}。")]
    MathError {
        /// エラー詳細メッセージ。
        message: String,
    },

    /// シリアライズまたはデシリアライズ失敗エラー。
    #[error("シリアライズ/デシリアライズに失敗しました: {message}。")]
    SerializationError {
        /// エラー詳細メッセージ。
        message: String,
    },
}
