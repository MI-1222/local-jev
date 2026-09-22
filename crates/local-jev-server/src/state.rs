//! # アプリケーション共有状態管理モジュール
//!
//! 推論エンジン、トークナイザー、較正設定、およびサーバー運用パラメータをスレッドセーフに保持する。

use std::sync::Arc;

use local_jev_core::contract::calibration::CalibrationConfig;
use local_jev_runtime::engine::{CoarseToFineConfig, InferenceEngine};
use local_jev_runtime::tokenizer::JevTokenizer;

use crate::guardrails::{GuardrailConfig, GuardrailPipeline};

/// デフォルトのリクエストあたり最大質問数。
pub const DEFAULT_MAX_QUESTIONS_PER_REQUEST: usize = 128;

/// デフォルトのバッチチャンクサイズ。
pub const DEFAULT_CHUNK_SIZE: usize = 16;

/// サーバー全体で共有されるアプリケーション状態。
#[derive(Clone)]
pub struct AppState {
    /// ONNX Runtime 推論エンジン(内部でセッションプールと同期制御を保持)。
    pub engine: Arc<InferenceEngine>,
    /// Jev 高速トークナイザー。
    pub tokenizer: Arc<JevTokenizer>,
    /// 事後較正温度設定。
    pub calib_config: Arc<CalibrationConfig>,
    /// 大規模候補数向けの粗密 2 段階探索設定。
    pub coarse_config: CoarseToFineConfig,
    /// 1 リクエストで許容される最大質問数。
    pub max_questions_per_request: usize,
    /// マイクロバッチ分割時のチャンクサイズ。
    pub chunk_size: usize,
    /// 前処理ガードレールパイプライン。
    pub guardrail_pipeline: GuardrailPipeline,
}

impl AppState {
    /// 新規 `AppState` を構築する。
    ///
    /// # 引数
    /// - `engine`: 推論エンジン。
    /// - `tokenizer`: 高速トークナイザー。
    /// - `calib_config`: 較正設定。
    pub fn new(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: Arc<CalibrationConfig>,
    ) -> Self {
        let mut guardrail_config = GuardrailConfig::default();
        guardrail_config.limits.max_questions = DEFAULT_MAX_QUESTIONS_PER_REQUEST;
        let guardrail_pipeline = GuardrailPipeline::new(guardrail_config);

        Self {
            engine,
            tokenizer,
            calib_config,
            coarse_config: CoarseToFineConfig::default(),
            max_questions_per_request: DEFAULT_MAX_QUESTIONS_PER_REQUEST,
            chunk_size: DEFAULT_CHUNK_SIZE,
            guardrail_pipeline,
        }
    }

    /// 設定値をカスタマイズして `AppState` を構築する。
    pub fn with_options(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: Arc<CalibrationConfig>,
        coarse_config: CoarseToFineConfig,
        max_questions_per_request: usize,
        chunk_size: usize,
    ) -> Self {
        let mut guardrail_config = GuardrailConfig::default();
        guardrail_config.limits.max_questions = max_questions_per_request;
        let guardrail_pipeline = GuardrailPipeline::new(guardrail_config);

        Self {
            engine,
            tokenizer,
            calib_config,
            coarse_config,
            max_questions_per_request,
            chunk_size: chunk_size.max(1),
            guardrail_pipeline,
        }
    }

    /// ガードレール設定も含めてフルカスタマイズした `AppState` を構築する。
    pub fn with_guardrails(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: Arc<CalibrationConfig>,
        coarse_config: CoarseToFineConfig,
        max_questions_per_request: usize,
        chunk_size: usize,
        guardrail_config: GuardrailConfig,
    ) -> Self {
        Self {
            engine,
            tokenizer,
            calib_config,
            coarse_config,
            max_questions_per_request,
            chunk_size: chunk_size.max(1),
            guardrail_pipeline: GuardrailPipeline::new(guardrail_config),
        }
    }

    /// サーバーがリクエストを安全に処理可能(Readiness)であるかを検査する。
    pub fn is_ready(&self) -> bool {
        self.engine.pool_size() > 0 && self.tokenizer.max_sequence_length() > 0
    }
}
