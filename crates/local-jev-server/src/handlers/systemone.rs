//! # Jev 互換推論エンドポイントモジュール
//!
//! `POST /v1/systemone` に対するリクエスト検証、非同期ワーカースレッド隔離、
//! 粗密 2 段階探索連携、および型付き判定結果のシリアライズを提供する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::IntoResponse;
use local_jev_core::gating::evaluate_response_routing;
use local_jev_core::schema::{QuestionType, SystemOneRequest, SystemOneResponse, Usage};
use serde_json::Value;

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{
    record_confidence, record_gating_route, record_http_duration, record_http_request,
    record_inference_metrics, record_question_type,
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
    axum::Json(mut req): axum::Json<SystemOneRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start_time = Instant::now();

    // 1. 高速スキーマ検証
    req.validate()?;

    // 2. 前処理ガードレールパイプラインの実行 (物理OOM防壁、サニタイズ、縮約、相対日時、算術集計)
    state.guardrail_pipeline.process(&mut req, None)?;

    let question_count = req.questions.len();

    // 質問タイプ別メトリクスカウント
    for question in req.questions.values() {
        match question.question_type {
            QuestionType::Choice => record_question_type("choice"),
            QuestionType::Score => record_question_type("score"),
            QuestionType::Noul => record_question_type("noul"),
        }
    }

    // 3. 正規化済み State の文字列取得
    let state_text = match &req.state {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    // 4. CPU/GPU バウンドな推論処理およびゲーティング判定をブロッキングプールへオフロード
    let engine = Arc::clone(&state.engine);
    let tokenizer = Arc::clone(&state.tokenizer);
    let calib_config = Arc::clone(&state.calib_config);
    let coarse_config = state.coarse_config.clone();
    let chunk_size = state.chunk_size;
    let questions = req.questions;
    let gating_config = req.gating.unwrap_or_else(|| state.gating_config.clone());

    let inference_start = Instant::now();
    let (answers, prompt_tokens, routing_summary, route_records) =
        tokio::task::spawn_blocking(move || {
            // 正規化後ペイロードの正味トークン数を正確に算出
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

            // 粗密 2 段階探索とマイクロバッチチャンキングを統合した推論を実行 (ゲーティング判定も透過適用)
            let answers = engine.evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
                &tokenizer,
                &state_text,
                &questions,
                &calib_config,
                &coarse_config,
                chunk_size,
                Some(&gating_config),
            )?;

            // 確信度ゲーティング (3系統ルーティング) メトリクス収集および集約サマリー算出
            let (routing_summary, route_records) = if gating_config.enabled {
                let mut records = Vec::with_capacity(answers.len());
                for (qid, answer) in answers.iter() {
                    let q_def = questions.get(qid);
                    let q_type_str = q_def
                        .map(|q| match q.question_type {
                            QuestionType::Choice => "choice",
                            QuestionType::Score => "score",
                            QuestionType::Noul => "noul",
                        })
                        .unwrap_or("unknown");

                    if let Some(ref meta) = answer.gating {
                        records.push((
                            meta.route.as_str(),
                            q_type_str,
                            qid.clone(),
                            meta.route,
                            meta.confidence,
                            meta.reason.clone(),
                        ));
                    }
                }
                let summary = evaluate_response_routing(&answers);
                (Some(summary), records)
            } else {
                (None, Vec::new())
            };

            Ok::<_, ServerError>((answers, total_prompt_tokens, routing_summary, route_records))
        })
        .await
        .map_err(|join_err| {
            ServerError::Internal(format!(
                "ブロッキング推論タスクが異常終了しました: {join_err}。"
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

    // 5. ゲーティング判定ログおよびメトリクスの記録
    for (route_str, q_type_str, qid, route, confidence, reason) in route_records {
        record_gating_route(route_str, q_type_str);
        tracing::info!(
            question_id = %qid,
            route = %route,
            confidence = confidence,
            reason = %reason,
            "確信度ゲーティング判定を適用しました。"
        );
    }

    // 6. トークン消費量のメタデータ付与
    let usage = Usage::new(prompt_tokens);

    let mut response = SystemOneResponse::new(answers, usage);
    if let Some(summary) = routing_summary {
        response = response.with_routing(summary);
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    record_http_duration("/v1/systemone", total_duration);
    record_http_request("/v1/systemone", 200);

    Ok(axum::Json(response))
}
