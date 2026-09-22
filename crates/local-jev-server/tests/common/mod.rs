//! # ベンチマーク共通ヘルパーモジュール
//!
//! レイテンシパーセンタイル (p50, p90, p95, p99) 集計、スループット (RPS, DPS) 計算、
//! テスト用サーバー起動、および代表ベンチマークシナリオのペイロード構築を提供する。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use local_jev_core::contract::calibration::CalibrationConfig;
use local_jev_core::schema::{Question, SystemOneRequest};
use local_jev_runtime::engine::{CoarseToFineConfig, InferenceEngine, SessionConfig};
use local_jev_runtime::tokenizer::JevTokenizer;
use local_jev_server::state::AppState;

/// ワークスペースのルートディレクトリを取得する。
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ワークスペースルートの解決に失敗しました。")
        .to_path_buf()
}

/// 配布モデルディレクトリを取得する。
pub fn default_model_dir() -> PathBuf {
    workspace_root().join("models").join("default")
}

/// テスト用 `AppState` を指定パラメータで初期化する。
pub fn init_benchmark_app_state(max_questions: usize, pool_size: usize) -> Option<Arc<AppState>> {
    let model_dir = default_model_dir();
    let onnx_path = model_dir.join("model.onnx");
    let tok_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !onnx_path.exists() || !tok_path.exists() {
        return None;
    }

    let session_config = SessionConfig {
        pool_size: pool_size.max(1),
        ..Default::default()
    };

    let engine = Arc::new(InferenceEngine::new(&onnx_path, session_config).ok()?);
    let tokenizer = Arc::new(JevTokenizer::from_file(&tok_path).ok()?);
    let calib_config = if calib_path.exists() {
        std::fs::read_to_string(&calib_path)
            .ok()
            .and_then(|s| CalibrationConfig::from_json_str(&s).ok())
            .unwrap_or_default()
    } else {
        CalibrationConfig::default()
    };

    let state = AppState::with_options(
        engine,
        tokenizer,
        Arc::new(calib_config),
        CoarseToFineConfig::default(),
        max_questions,
        16,
    );

    Some(Arc::new(state))
}

/// ベンチマーク統計量集計構造体。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// 測定シナリオ名。
    pub scenario_name: String,
    /// 完了した総リクエスト数。
    pub total_requests: usize,
    /// 下された総決定数。
    pub total_decisions: usize,
    /// 総計測経過時間。
    pub elapsed: Duration,
    /// 最小レイテンシ。
    pub min: Duration,
    /// 中央値レイテンシ (p50)。
    pub p50: Duration,
    /// 90パーセンタイルレイテンシ (p90)。
    pub p90: Duration,
    /// 95パーセンタイルレイテンシ (p95)。
    pub p95: Duration,
    /// 99パーセンタイルレイテンシ (p99)。
    pub p99: Duration,
    /// 最大レイテンシ。
    pub max: Duration,
    /// 平均レイテンシ。
    pub mean: Duration,
    /// Requests Per Second (RPS)。
    pub rps: f64,
    /// Decisions Per Second (DPS)。
    pub dps: f64,
}

impl BenchmarkStats {
    /// 生の所要時間リストおよび決定数から統計量を算出する。
    pub fn calculate(
        scenario_name: impl Into<String>,
        mut durations: Vec<Duration>,
        questions_per_req: usize,
        total_elapsed: Duration,
    ) -> Self {
        let total_requests = durations.len();
        assert!(total_requests > 0, "計測サンプル数が 0 件です。");

        durations.sort();

        let min = durations[0];
        let max = durations[total_requests - 1];

        let p50 = durations[(total_requests as f64 * 0.50) as usize];
        let p90 = durations[((total_requests as f64 * 0.90) as usize).min(total_requests - 1)];
        let p95 = durations[((total_requests as f64 * 0.95) as usize).min(total_requests - 1)];
        let p99 = durations[((total_requests as f64 * 0.99) as usize).min(total_requests - 1)];

        let sum_millis: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
        let mean = Duration::from_secs_f64((sum_millis / total_requests as f64) / 1000.0);

        let elapsed_secs = total_elapsed.as_secs_f64().max(1e-6);
        let rps = total_requests as f64 / elapsed_secs;
        let total_decisions = total_requests * questions_per_req;
        let dps = total_decisions as f64 / elapsed_secs;

        Self {
            scenario_name: scenario_name.into(),
            total_requests,
            total_decisions,
            elapsed: total_elapsed,
            min,
            p50,
            p90,
            p95,
            p99,
            max,
            mean,
            rps,
            dps,
        }
    }

    /// Markdown 表形式でサマリーを出力する。
    pub fn to_markdown_row(&self) -> String {
        format!(
            "| {} | {} | {:.1} ms | {:.1} ms | {:.1} ms | {:.1} ms | {:.2} req/s | **{:.2} dec/s** |",
            self.scenario_name,
            self.total_requests,
            self.min.as_secs_f64() * 1000.0,
            self.p50.as_secs_f64() * 1000.0,
            self.p95.as_secs_f64() * 1000.0,
            self.p99.as_secs_f64() * 1000.0,
            self.rps,
            self.dps,
        )
    }
}

/// 代表シナリオのペイロード構築ヘルパー。
pub struct ScenarioBuilder;

impl ScenarioBuilder {
    /// シナリオ A: 単一即応 (Lightweight: State 100 tok 程度, Choice 3 候補 × 1)。
    pub fn build_lightweight() -> SystemOneRequest {
        let state = "顧客からの問い合わせ: 先週注文した商品の配送状況を確認したいです。注文番号はORD-98765です。配送先住所の変更が可能かも知りたいです。".repeat(2);
        let mut questions = IndexMap::new();

        let mut criteria = IndexMap::new();
        criteria.insert("shipping".to_string(), "配送状況や追跡番号に関する問い合わせ".to_string());
        criteria.insert("address".to_string(), "配送先住所や宛先の変更に関する要望".to_string());
        criteria.insert("other".to_string(), "それ以外のその他の問い合わせ".to_string());

        questions.insert(
            "intent".to_string(),
            Question::new_choice("主な問い合わせ意図を選択してください。", criteria),
        );

        SystemOneRequest::new(state, questions)
    }

    /// シナリオ B: プレフィックス共有バッチ (State 500 tok 程度, 混合 8 問)。
    pub fn build_prefix_sharing_batch() -> SystemOneRequest {
        let state = "【契約規約抜粋】第1条（目的）本規約は、当社が提供するクラウドストレージサービスの利用条件を定めるものです。利用者は本規約に同意した上で利用するものとします。第2条（アカウント管理）利用者は自己の責任においてパスワードを管理し、第三者への譲渡または貸与を禁止します。第3条（料金と支払）利用料金は月額1,000円とし、翌月末までにクレジットカード決済にて支払うものとします。遅延損害金は年14.6%とします。第4条（契約解除）当社は、利用者が規約に違反した場合、事前の催告なく即座に利用契約を解除できるものとします。第5条（免責事項）天災地変等の不可抗力によりデータが消失した場合、当社は一切の損害賠償責任を負わないものとします。".repeat(2);

        let mut questions = IndexMap::new();

        // Q1: Choice 契約類型
        let mut c1 = IndexMap::new();
        c1.insert("cloud".to_string(), "クラウドサービス利用規約".to_string());
        c1.insert("sla".to_string(), "保守サービス品質合意書".to_string());
        c1.insert("nda".to_string(), "秘密保持契約書".to_string());
        questions.insert("contract_type".to_string(), Question::new_choice("契約類型を判定してください。", c1));

        // Q2: Noul 遅延損害金の有無
        questions.insert("has_penalty".to_string(), Question::new_noul("遅延損害金に関する規定が存在しますか？"));

        // Q3: Score 重要度
        let c3 = vec![
            "軽微な通知".to_string(),
            "一般的な契約約款".to_string(),
            "極めて厳格な法的拘束力".to_string(),
        ];
        questions.insert("importance_score".to_string(), Question::new_score("規約の重要度を1から3で評価してください。", c3));

        // Q4: Noul 免責事項の有無
        questions.insert("has_disclaimer".to_string(), Question::new_noul("天災地変に関する免責条項が含まれていますか？"));

        // Q5: Choice 支払方法
        let mut c5 = IndexMap::new();
        c5.insert("credit".to_string(), "クレジットカード決済".to_string());
        c5.insert("bank".to_string(), "銀行振込決済".to_string());
        c5.insert("cash".to_string(), "現金払い".to_string());
        questions.insert("payment_method".to_string(), Question::new_choice("指定されている支払方法を判定してください。", c5));

        // Q6: Noul 即時解除規定の有無
        questions.insert("can_terminate_immediately".to_string(), Question::new_noul("規約違反時の事前催告なき即時契約解除規定が存在しますか？"));

        // Q7: Score リスク度評価
        let c7 = vec![
            "リスク極小".to_string(),
            "中程度のリスク".to_string(),
            "重大な法的リスク".to_string(),
        ];
        questions.insert("risk_score".to_string(), Question::new_score("利用者の免責制限リスクを評価してください。", c7));

        // Q8: Choice 月額料金区分
        let mut c8 = IndexMap::new();
        c8.insert("low".to_string(), "月額3000円未満の低価格帯".to_string());
        c8.insert("mid".to_string(), "月額3000円以上10000円未満".to_string());
        c8.insert("high".to_string(), "月額10000円以上の高価格帯".to_string());
        questions.insert("price_tier".to_string(), Question::new_choice("月額利用料金の価格帯を分類してください。", c8));

        SystemOneRequest::new(state, questions)
    }

    /// シナリオ C: 粗密探索 (Coarse-to-Fine: State 200 tok 程度, Banking77 相当の 77 候補 Choice)。
    pub fn build_coarse_to_fine_77() -> SystemOneRequest {
        let state = "モバイルアプリから送金しようとしたところ、エラーコードE-401が表示されて送金が完了しませんでした。口座残高は十分にあるはずです。至急対応してください。";
        let mut questions = IndexMap::new();

        let mut criteria = IndexMap::new();
        for i in 1..=77 {
            let key = format!("intent_{i:02}");
            let desc = format!("銀行決済・送金・口座管理に関する詳細問い合わせカテゴリ {i}");
            criteria.insert(key, desc);
        }

        questions.insert(
            "banking_category".to_string(),
            Question::new_choice("77カテゴリから最も該当する問い合わせ分類を特定してください。", criteria),
        );

        SystemOneRequest::new(state, questions)
    }

    /// シナリオ D: マイクロバッチ・長文境界値 (Stress/Heavy: State 2000+ tok, 質問 16 問)。
    pub fn build_stress_heavy() -> SystemOneRequest {
        let paragraph = "大規模分散システムにおけるコンセンサスアルゴリズムとしてRaftが広く採用されている。リーダー選出、ログ複製、安全性の3つの独立したサブ問題に分割することで理解容易性を高めている。ノード障害時のスプリットブレインを防ぐため、過半数（Quorum）の合意を必須要件とする。ネットワーク分断発生時でも一貫性（Consistency）を優先するCP型システムとして動作する。";
        let state = paragraph.repeat(15);

        let mut questions = IndexMap::new();
        for i in 1..=16 {
            let qid = format!("consensus_q{i:02}");
            let q = if i % 2 == 0 {
                Question::new_noul(format!("設問{i}: 過半数のノード合意がコミット条件に含まれますか？"))
            } else {
                let mut c = IndexMap::new();
                c.insert("cp".to_string(), "一貫性と分断耐性を重視するCP型設計".to_string());
                c.insert("ap".to_string(), "可用性と分断耐性を重視するAP型設計".to_string());
                c.insert("ca".to_string(), "分断耐性を考慮しないCA型設計".to_string());
                Question::new_choice(format!("設問{i}: CAP定理におけるシステムの分類を選択してください。"), c)
            };
            questions.insert(qid, q);
        }

        SystemOneRequest::new(state, questions)
    }
}
