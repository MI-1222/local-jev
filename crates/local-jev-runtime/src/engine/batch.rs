//! # 複数質問一括バッチ推論モジュール
//!
//! トークンレベルのプレフィックス共有と動的バッチ Collate により、
//! 単一フォワードパスでの並列推論および決定プリミティブ解決を実現する。

use indexmap::IndexMap;
use local_jev_core::contract::CalibrationConfig;
use local_jev_core::decision::evaluate_question_with_buf;
use local_jev_core::schema::{Answer, Question};

use crate::engine::session::InferenceEngine;
use crate::error::Result;
use crate::tokenizer::{BatchTokenizedQuestions, JevTokenizer};

/// サーバー層との連携において推奨されるデフォルトの最大バッチチャンクサイズ。
///
/// 質問数がこの値を超える場合、OOM (Out Of Memory) を防止するため
/// マイクロバッチに分割して実行することが推奨される。
pub const DEFAULT_MAX_BATCH_CHUNK_SIZE: usize = 16;

/// バッチ推論および決定数理解決用のスクラッチパッドバッファ。
///
/// スレッドローカルまたはリクエストコンテキスト内で再利用することで、
/// 推論ごとのメモリアロケーションを根絶する。
#[derive(Debug, Default, Clone)]
pub struct BatchScratchpad {
    /// 平坦化生ロジット書き込み用バッファ (`[batch_size * num_options]`)。
    pub flat_logits: Vec<f64>,
    /// ソフトマックス確率分布算出用作業バッファ (`[max_options_per_q]`)。
    pub probs_buf: Vec<f64>,
}

impl BatchScratchpad {
    /// 指定されたキャパシティでスクラッチパッドを初期化する。
    pub fn with_capacity(total_options: usize, max_options_per_q: usize) -> Self {
        Self {
            flat_logits: Vec::with_capacity(total_options),
            probs_buf: Vec::with_capacity(max_options_per_q),
        }
    }

    /// 要求されるサイズに合わせてバッファ領域を確保・リサイズする。
    pub fn prepare(&mut self, total_options: usize, max_options_per_q: usize) {
        if self.flat_logits.len() < total_options {
            self.flat_logits.resize(total_options, 0.0);
        }
        if self.probs_buf.len() < max_options_per_q {
            self.probs_buf.resize(max_options_per_q, 0.0);
        }
    }
}

impl InferenceEngine {
    /// 事前パッキング済みバッチトークン列を受け取り、単一フォワードパスで各質問のロジットを算出する。
    ///
    /// 出力されたテンソルから各質問の真の候補数 $K_i$ 個のロジットのみをスライス抽出することで、
    /// パディングスロット由来のダミーロジットを完全に除外して返却する。
    ///
    /// # 引数
    /// - `batch`: パディングおよび平坦化済みのバッチトークナイズ結果。
    ///
    /// # 戻り値
    /// 質問キーと有効ロジット列 (`Vec<f64>`) の順序付きマップ。
    ///
    /// # エラー
    /// ONNX Runtime 推論処理またはテンソル検証に失敗した場合にエラーを返す。
    pub fn forward_batch_tokenized(
        &self,
        batch: &BatchTokenizedQuestions,
    ) -> Result<IndexMap<String, Vec<f64>>> {
        let total_logits = batch.dims.batch_size * batch.dims.num_options;
        let mut flat_logits = vec![0.0f64; total_logits];

        self.forward_batch_tokenized_into(batch, &mut flat_logits)?;

        let mut results = IndexMap::with_capacity(batch.dims.batch_size);
        let num_options = batch.dims.num_options;

        for (i, q_key) in batch.question_keys.iter().enumerate() {
            let k_i = batch.candidate_counts[i];
            let start_idx = i * num_options;
            let valid_logits = flat_logits[start_idx..start_idx + k_i].to_vec();
            results.insert(q_key.clone(), valid_logits);
        }

        Ok(results)
    }

    /// 事前パッキング済みバッチトークン列を受け取り、推論結果を指定されたスライスに書き込む。
    ///
    /// 借用ビュー (`TensorRef`) によりゼロコピーで ONNX Runtime に渡される。
    ///
    /// # 引数
    /// - `batch`: パディングおよび平坦化済みのバッチトークナイズ結果。
    /// - `out_logits`: 平坦化生ロジットの書き込み先バッファ (`[batch_size * num_options]`)。
    pub fn forward_batch_tokenized_into(
        &self,
        batch: &BatchTokenizedQuestions,
        out_logits: &mut [f64],
    ) -> Result<()> {
        self.forward_raw_into(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.op_indices,
            batch.dims,
            out_logits,
        )
    }

    /// 共通 State と複数質問群を受け取り、トークンプレフィックス共有バッチ推論を実行する。
    ///
    /// State のトークナイズを 1 度だけ実行し、各質問を単一のバッチ次元に集約して
    /// 1 回のフォワードパスで全質問の有効ロジットを返却する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    ///
    /// # 戻り値
    /// 質問キーと有効ロジット列 (`Vec<f64>`) の順序付きマップ。
    pub fn forward_batch_questions(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<IndexMap<String, Vec<f64>>> {
        let batch = tokenizer.encode_batch_questions(state, questions)?;
        self.forward_batch_tokenized(&batch)
    }

    /// 共通 State、複数質問群、および較正設定を受け取り、単一フォワードパスで判定結果を一括導出する。
    ///
    /// # 処理フロー
    /// 1. `tokenizer.encode_batch_questions` によるプレフィックス共有バッチ化。
    /// 2. `forward_batch_tokenized_into` による単一フォワードパス実行。
    /// 3. 各質問の真の候補数 $K_i$ によるスライス抽出とダミーロジット破棄。
    /// 4. `local_jev_core::decision::evaluate_question_with_buf` による決定論的プリミティブ解決。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    ///
    /// # 戻り値
    /// 質問キーと決定結果 (`Answer`) の順序付きマップ。
    pub fn evaluate_batch_questions(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
    ) -> Result<IndexMap<String, Answer>> {
        let batch = tokenizer.encode_batch_questions(state, questions)?;
        let total_logits = batch.dims.batch_size * batch.dims.num_options;
        let mut flat_logits = vec![0.0f64; total_logits];

        self.forward_batch_tokenized_into(&batch, &mut flat_logits)?;

        let mut answers = IndexMap::with_capacity(batch.dims.batch_size);
        let mut probs_buf = vec![0.0f64; batch.dims.num_options];
        let num_options = batch.dims.num_options;

        for (i, q_key) in batch.question_keys.iter().enumerate() {
            let k_i = batch.candidate_counts[i];
            let start_idx = i * num_options;
            let valid_logits = &flat_logits[start_idx..start_idx + k_i];
            let question = questions.get(q_key).expect("キーの一致が保証されている。");

            let answer = evaluate_question_with_buf(
                question,
                valid_logits,
                calib_config,
                &mut probs_buf[..k_i],
            )?;
            answers.insert(q_key.clone(), answer);
        }

        Ok(answers)
    }

    /// スクラッチパッドバッファを再利用し、共通 State と複数質問群の判定結果を一括導出する。
    ///
    /// アロケーションを最小化したい高頻度 API サーバー環境向けの高速推論インターフェース。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `scratchpad`: 再利用可能な作業用スクラッチパッドバッファ。
    pub fn evaluate_batch_questions_with_scratchpad(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        scratchpad: &mut BatchScratchpad,
    ) -> Result<IndexMap<String, Answer>> {
        let batch = tokenizer.encode_batch_questions(state, questions)?;
        let total_logits = batch.dims.batch_size * batch.dims.num_options;
        let num_options = batch.dims.num_options;

        scratchpad.prepare(total_logits, num_options);

        self.forward_batch_tokenized_into(&batch, &mut scratchpad.flat_logits[..total_logits])?;

        let mut answers = IndexMap::with_capacity(batch.dims.batch_size);

        for (i, q_key) in batch.question_keys.iter().enumerate() {
            let k_i = batch.candidate_counts[i];
            let start_idx = i * num_options;
            let valid_logits = &scratchpad.flat_logits[start_idx..start_idx + k_i];
            let question = questions.get(q_key).expect("キーの一致が保証されている。");

            let answer = evaluate_question_with_buf(
                question,
                valid_logits,
                calib_config,
                &mut scratchpad.probs_buf[..k_i],
            )?;
            answers.insert(q_key.clone(), answer);
        }

        Ok(answers)
    }

    /// 共通 State と複数質問群を指定されたチャンクサイズで分割実行し、有効ロジットを返却する。
    ///
    /// 質問数が極端に多いリクエストにおいて、メモリ使用量 ($N \times L_{\max}$) の急増を抑制し
    /// OOM (Out Of Memory) を防止するマイクロバッチ推論インターフェース。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    pub fn forward_batch_questions_chunked(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        chunk_size: usize,
    ) -> Result<IndexMap<String, Vec<f64>>> {
        let chunk_size = chunk_size.max(1);
        if questions.len() <= chunk_size {
            return self.forward_batch_questions(tokenizer, state, questions);
        }

        let state_ids = tokenizer.encode_state(state)?;
        let mut results = IndexMap::with_capacity(questions.len());

        for chunk_slice in questions.iter().collect::<Vec<_>>().chunks(chunk_size) {
            let mut sub_questions = IndexMap::with_capacity(chunk_slice.len());
            for (k, v) in chunk_slice {
                sub_questions.insert((*k).clone(), (*v).clone());
            }

            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                &sub_questions,
                tokenizer.max_sequence_length(),
            )?;
            let sub_results = self.forward_batch_tokenized(&batch)?;
            results.extend(sub_results);
        }

        Ok(results)
    }

    /// 共通 State と複数質問群を指定されたチャンクサイズで分割実行し、判定結果を一括導出する。
    ///
    /// サーバー層 (local-jev-server) のガードレール等と連携し、大規模質問リクエストを
    /// 安全な固定バッチサイズで順次推論・解決する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    pub fn evaluate_batch_questions_chunked(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        chunk_size: usize,
    ) -> Result<IndexMap<String, Answer>> {
        let chunk_size = chunk_size.max(1);
        if questions.len() <= chunk_size {
            return self.evaluate_batch_questions(tokenizer, state, questions, calib_config);
        }

        let state_ids = tokenizer.encode_state(state)?;
        let mut answers = IndexMap::with_capacity(questions.len());

        for chunk_slice in questions.iter().collect::<Vec<_>>().chunks(chunk_size) {
            let mut sub_questions = IndexMap::with_capacity(chunk_slice.len());
            for (k, v) in chunk_slice {
                sub_questions.insert((*k).clone(), (*v).clone());
            }

            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                &sub_questions,
                tokenizer.max_sequence_length(),
            )?;

            let total_logits = batch.dims.batch_size * batch.dims.num_options;
            let mut flat_logits = vec![0.0f64; total_logits];
            self.forward_batch_tokenized_into(&batch, &mut flat_logits)?;

            let mut probs_buf = vec![0.0f64; batch.dims.num_options];
            let num_options = batch.dims.num_options;

            for (i, q_key) in batch.question_keys.iter().enumerate() {
                let k_i = batch.candidate_counts[i];
                let start_idx = i * num_options;
                let valid_logits = &flat_logits[start_idx..start_idx + k_i];
                let question = sub_questions
                    .get(q_key)
                    .expect("キーの一致が保証されている。");

                let answer = evaluate_question_with_buf(
                    question,
                    valid_logits,
                    calib_config,
                    &mut probs_buf[..k_i],
                )?;
                answers.insert(q_key.clone(), answer);
            }
        }

        Ok(answers)
    }
}
