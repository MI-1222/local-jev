//! # Jev 互換推論エンドポイントモジュール
//!
//! `POST /v1/systemone` に対するリクエスト検証、非同期ワーカースレッド隔離、
//! 粗密 2 段階探索連携、および型付き判定結果のシリアライズを提供する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::IntoResponse;
use local_jev_core::schema::{QuestionType, SystemOneRequest, SystemOneResponse, Usage};
use serde_json::Value;

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{
    record_confidence, record_http_duration, record_http_request, record_inference_metrics,
    record_question_type,
};
use crate::state::AppState;

/// TypeSafe AI Jev 互換の非自己回帰型判断推論エンドポイント。
///
/// 共通 State と複数質問群を受け取り、単一フォワードパスおよび
/// 決定プリミティブ数理に基づく型付き確率決定を一括返却する。
#[utoipa::path(
    post,
    path = "/v1/systemone",
    tag = "Inference",
    request_body = SystemOneRequest,
    responses(
        (status = 200, description = "推論および決定解決成功", body = SystemOneResponse),
        (status = 400, description = "スキーマまたはパラメータ検証違反", body = ErrorResponse),
        (status = 413, description = "質問数またはペイロード上限超過", body = ErrorResponse),
        (status = 500, description = "推論エンジンまたはサーバー内部障害", body = ErrorResponse)
    )
)]
pub async fn system_one_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<SystemOneRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start_time = Instant::now();

    // 1. 高速リクエスト検証 (Fast-Fail)
    req.validate()?;

    let question_count = req.questions.len();
    if question_count > state.max_questions_per_request {
        return Err(ServerError::PayloadTooLarge(format!(
            "リクエストに含まれる質問数 ({question_count}) が上限値 ({}) を超過しています。",
            state.max_questions_per_request
        )));
    }

    // 質問タイプ別メトリクスカウント
    for question in req.questions.values() {
        match question.question_type {
            QuestionType::Choice => record_question_type("choice"),
            QuestionType::Score => record_question_type("score"),
            QuestionType::Noul => record_question_type("noul"),
        }
    }

    // 2. State の文字列正規化 (文字列型はそのままクローン、それ以外は JSON 文字列化)
    let state_text = match &req.state {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    // 3. CPU/GPU バウンドな推論処理をブロッキングプールへオフロード
    let engine = Arc::clone(&state.engine);
    let tokenizer = Arc::clone(&state.tokenizer);
    let calib_config = Arc::clone(&state.calib_config);
    let coarse_config = state.coarse_config.clone();
    let questions = req.questions;

    let inference_start = Instant::now();
    let (answers, prompt_tokens) = tokio::task::spawn_blocking(move || {
        // ユーザー送信ペイロードの正味トークン数を正確に算出 (特殊トークンの重複混入を防止)
        let state_tokens = tokenizer
            .encode_state(&state_text)
            .map(|toks| toks.len())
            .unwrap_or(0);
        let questions_tokens: usize = questions
            .values()
            .map(|q| {
                local_jev_runtime::tokenizer::prompt::format_prompt(q)
                    .and_then(|f| {
                        tokenizer
                            .inner()
                            .encode(f.suffix.as_str(), false)
                            .map_err(|e| {
                                local_jev_runtime::error::RuntimeError::TokenizerEncodeError(
                                    e.to_string(),
                                )
                            })
                    })
                    .map(|toks| toks.len())
                    .unwrap_or(0)
            })
            .sum();
        let total_prompt_tokens = state_tokens + questions_tokens;

        let answers = engine.evaluate_batch_questions_coarse_to_fine(
            &tokenizer,
            &state_text,
            &questions,
            &calib_config,
            &coarse_config,
        )?;

        Ok::<_, ServerError>((answers, total_prompt_tokens))
    })
    .await
    .map_err(|join_err| {
        ServerError::Internal(format!(
            "ブロッキング推論タスクが異常終了しました: {join_err}"
        ))
    })??;

    let inference_duration = inference_start.elapsed().as_secs_f64();
    record_inference_metrics(inference_duration, question_count);

    // 確信度メトリクスの記録
    for answer in answers.values() {
        if let Some(conf) = answer.effective_confidence() {
            record_confidence(conf);
        }
    }

    // 4. トークン消費量のメタデータ付与
    let usage = Usage::new(prompt_tokens);

    let response = SystemOneResponse::new(answers, usage);

    let total_duration = start_time.elapsed().as_secs_f64();
    record_http_duration("/v1/systemone", total_duration);
    record_http_request("/v1/systemone", 200);

    Ok(axum::Json(response))
}
