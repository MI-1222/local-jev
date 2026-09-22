//! # 運用監視エンドポイントハンドラモジュール
//!
//! Liveness Probe (`GET /health`), Readiness Probe (`GET /ready`),
//! および Prometheus メトリクスエクスポート (`GET /metrics`) を提供する。

use std::sync::Arc;

use axum::Extension;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::Serialize;
use utoipa::ToSchema;

use crate::state::AppState;

/// ヘルスチェック・準備状態のステータスレスポンス。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, ToSchema)]
pub struct HealthStatusResponse {
    /// 稼働ステータス文字列(`ok`, `ready`, `not_ready`)。
    pub status: String,
}

/// Liveness Probe エンドポイント。
///
/// サーバープロセスが生存し、HTTP イベントループが正常に応答可能かを即座に判定する。
#[utoipa::path(
    get,
    path = "/health",
    tag = "Operations",
    responses(
        (status = 200, description = "プロセス正常稼働中", body = HealthStatusResponse)
    )
)]
pub async fn health_handler() -> impl IntoResponse {
    axum::Json(HealthStatusResponse {
        status: "ok".to_string(),
    })
}

/// Readiness Probe エンドポイント。
///
/// モデルおよびトークナイザーがロードされ、推論要求を安全に処理可能な状態にあるかを検証する。
#[utoipa::path(
    get,
    path = "/ready",
    tag = "Operations",
    responses(
        (status = 200, description = "推論準備完了", body = HealthStatusResponse),
        (status = 503, description = "推論エンジン未準備", body = HealthStatusResponse)
    )
)]
pub async fn ready_handler(State(state): State<Arc<AppState>>) -> Response {
    if state.is_ready() {
        (
            StatusCode::OK,
            axum::Json(HealthStatusResponse {
                status: "ready".to_string(),
            }),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(HealthStatusResponse {
                status: "not_ready".to_string(),
            }),
        )
            .into_response()
    }
}

/// Prometheus メトリクスエンドポイント。
///
/// 蓄積されたパフォーマンス指標およびリクエスト統計を Prometheus テキスト形式で返却する。
#[utoipa::path(
    get,
    path = "/metrics",
    tag = "Operations",
    responses(
        (status = 200, description = "Prometheus メトリクス出力", content_type = "text/plain; version=0.0.4")
    )
)]
pub async fn metrics_handler(
    Extension(prometheus_handle): Extension<Option<PrometheusHandle>>,
) -> impl IntoResponse {
    match prometheus_handle {
        Some(handle) => (
            StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
            handle.render(),
        )
            .into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            "Prometheus メトリクスレコーダーが有効化されていません。",
        )
            .into_response(),
    }
}
