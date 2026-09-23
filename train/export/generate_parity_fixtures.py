"""Python 参照実装による E2E パリティテスト用ゴールデンフィクスチャ生成スクリプト。

models/default/ 配下の ONNX モデル、トークナイザー、および calibration.json を用いて、
代表的な質問プリミティブ (Choice, Score, Noul)、実務再現シナリオ (EC障害、決済、不正検知)、
および境界値ケースに対する決定論的な推論・較正・Gating 判定結果を JSON 形式で出力する。
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

# train ディレクトリを Python パスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from contract import (
    CalibrationConfig,
    compute_composite_confidence,
    compute_normalized_entropy,
    compute_top_margin,
)


def softmax(logits: list[float] | np.ndarray, temperature: float = 1.0) -> list[float]:
    """温度付きソフトマックスを計算する。

    Args:
        logits (list[float] | np.ndarray): 入力生ロジット配列。
        temperature (float): 温度パラメータ。

    Returns:
        list[float]: ソフトマックス正規化後の確率分布。
    """
    temp = max(float(temperature), 1.0e-4)
    z = np.array(logits, dtype=np.float64) / temp
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    probs = exp_z / np.sum(exp_z)
    return [float(p) for p in probs]


def sigmoid(diff: float, temperature: float = 1.0) -> float:
    """温度付きシグモイド関数を計算する。

    Args:
        diff (float): ロジット差分 (z_true - z_false)。
        temperature (float): 温度パラメータ。

    Returns:
        float: 真実確率値。
    """
    temp = max(float(temperature), 1.0e-4)
    scaled_diff = diff / temp
    if scaled_diff >= 40.0:
        return 1.0
    elif scaled_diff <= -40.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-scaled_diff)))


def compute_score_variance_confidence(probs: list[float]) -> float:
    """Score プリミティブ向けの正規化分散確信度 C_var を計算する。

    Args:
        probs (list[float]): 段階別確率分布。

    Returns:
        float: [0.0, 1.0] にクランプされた分散確信度。
    """
    m = len(probs)
    if m <= 1:
        return 1.0

    mean = sum(k * p for k, p in enumerate(probs))
    variance = sum(p * ((k - mean) ** 2) for k, p in enumerate(probs))
    max_variance = ((m - 1) ** 2) / 4.0
    if max_variance <= 0.0:
        return 1.0
    conf = 1.0 - (variance / max_variance)
    return max(0.0, min(1.0, float(conf)))


def decide_route(
    confidence: float,
    margin: float | None,
    high_threshold: float = 0.70,
    low_threshold: float = 0.35,
    top_margin_threshold: float = 0.15,
) -> tuple[str, str | None]:
    """確信度および Top-Margin に基づく 3 系統ルーティングを判定する。

    Args:
        confidence (float): 実効確信度。
        margin (float | None): 上位2候補の確率マージン (Choice 型のみ)。
        high_threshold (float): 高確信度下限閾値。
        low_threshold (float): 中確信度下限閾値。
        top_margin_threshold (float): 上位2候補マージン閾値。

    Returns:
        tuple[str, str | None]: (ルーティング文字列, 降格理由)。
    """
    eps = 1.0e-9
    if confidence >= high_threshold - eps:
        if margin is not None and margin < top_margin_threshold - eps:
            reason = (
                f"確信度は十分ですが ({confidence:.4f} >= {high_threshold:.2f})、"
                f"上位候補の確率差が小さいため確認を要求します ({margin:.4f} < {top_margin_threshold:.2f})。"
            )
            return "confirm_or_escalate", reason
        return "auto_execute", None
    elif confidence >= low_threshold - eps:
        reason = (
            f"確信度が中程度であるため確認を要求します "
            f"({low_threshold:.2f} <= {confidence:.4f} < {high_threshold:.2f})。"
        )
        return "confirm_or_escalate", reason
    else:
        reason = (
            f"確信度が低いため安全弁フォールバックを発動します "
            f"({confidence:.4f} < {low_threshold:.2f})。"
        )
        return "fallback", reason


def determine_aggregate_route(routes: list[str]) -> str:
    """最悪値ルール (Conservative Policy) に基づく集約ルーティングを決定する。

    Args:
        routes (list[str]): 各質問のルーティング一覧。

    Returns:
        str: 集約されたルーティング文字列。
    """
    if any(r == "fallback" for r in routes):
        return "fallback"
    if any(r == "confirm_or_escalate" for r in routes):
        return "confirm_or_escalate"
    return "auto_execute"


def matches_bucket(expr: str, count: int) -> bool:
    """バケット定義文字列 (例: '2', '3-5', '11+') と候補数のマッチ判定を行う。

    Args:
        expr (str): バケット表現。
        count (int): 候補数。

    Returns:
        bool: 一致するかどうか。
    """
    trimmed = expr.strip()
    if trimmed.isdigit():
        return int(trimmed) == count
    if trimmed.endswith("+"):
        prefix = trimmed[:-1].strip()
        if prefix.isdigit():
            return count >= int(prefix)
    if "-" in trimmed:
        parts = trimmed.split("-", 1)
        if parts[0].strip().isdigit() and parts[1].strip().isdigit():
            low = int(parts[0].strip())
            high = int(parts[1].strip())
            return low <= count <= high
    return False


def resolve_temperature(
    calib_config: CalibrationConfig, question_type: str, count: int
) -> float:
    """質問タイプと候補数に基づき較正温度を解決する。

    Args:
        calib_config (CalibrationConfig): 較正設定。
        question_type (str): 'choice' | 'score' | 'noul'。
        count (int): 候補数。

    Returns:
        float: 解決された温度パラメータ。
    """
    if question_type == "choice":
        for bucket, temp in calib_config.temperature_map.choice.items():
            if matches_bucket(bucket, count):
                return float(temp)
    elif question_type == "score":
        for bucket, temp in calib_config.temperature_map.score.items():
            if matches_bucket(bucket, count):
                return float(temp)
    elif question_type == "noul":
        return float(calib_config.temperature_map.noul)
    return float(calib_config.default_temperature)


def run_inference_for_question(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerFast,
    op_token_id: int,
    state: str,
    instructions: str,
    question_type: str,
    criteria_data: dict[str, str] | list[str] | None,
    calib_config: CalibrationConfig,
    gating_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """単一の質問に対してトークナイズ、ORT 推論、較正、および Gating を実行する。

    Args:
        session (ort.InferenceSession): ORT 推論セッション。
        tokenizer (PreTrainedTokenizerFast): トークナイザー。
        op_token_id (int): [OP] トークン ID。
        state (str): 文脈テキスト。
        instructions (str): 指示文。
        question_type (str): "choice" | "score" | "noul"。
        criteria_data (dict[str, str] | list[str] | None): Criteria 定義。
        calib_config (CalibrationConfig): 較正設定。
        gating_config (dict[str, Any] | None): リクエスト指定の Gating 設定。

    Returns:
        dict[str, Any]: 判定結果および期待値メタデータ。
    """
    if question_type == "choice":
        assert isinstance(criteria_data, dict)
        criteria_data = dict(sorted(criteria_data.items()))
        option_keys = list(criteria_data.keys())
        criteria_parts = [f"[OP] {criteria_data[k]}" for k in option_keys]
        temperature = resolve_temperature(calib_config, "choice", len(option_keys))
    elif question_type == "score":
        assert isinstance(criteria_data, list)
        option_keys = [str(i) for i in range(len(criteria_data))]
        criteria_parts = [f"[OP] {desc}" for desc in criteria_data]
        temperature = resolve_temperature(calib_config, "score", len(criteria_data))
    elif question_type == "noul":
        option_keys = ["true", "false"]
        criteria_parts = ["[OP] 真 (True)", "[OP] 偽 (False)"]
        temperature = resolve_temperature(calib_config, "noul", 2)
    else:
        raise ValueError(f"未知の質問タイプ: {question_type}。")

    criteria_text = " ".join(criteria_parts)
    prefix_text = "State: "
    suffix_text = f"\nInstructions: {instructions}\nCriteria: {criteria_text}"

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    state_ids = tokenizer.encode(state, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    leading_special = (
        [tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []
    )
    trailing_special = (
        [tokenizer.sep_token_id] if tokenizer.sep_token_id is not None else []
    )

    input_ids = leading_special + prefix_ids + state_ids + suffix_ids + trailing_special
    attention_mask = [1] * len(input_ids)

    # [OP] トークン位置の検出
    op_indices = [idx for idx, t_id in enumerate(input_ids) if t_id == op_token_id]
    assert len(op_indices) == len(option_keys), (
        f"[OP] 検出数 ({len(op_indices)}) と候補数 ({len(option_keys)}) が一致しません。"
    )

    # ORT 推論
    ort_inputs = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
        "op_indices": np.array([op_indices], dtype=np.int64),
    }
    ort_outputs = session.run(["logits"], ort_inputs)
    logits_arr = np.asarray(ort_outputs[0])
    logits = [float(val) for val in logits_arr[0]]

    # Gating 設定の解決
    high_th = calib_config.gating_thresholds.high_threshold
    low_th = calib_config.gating_thresholds.low_threshold
    margin_th = calib_config.gating_thresholds.top_margin_threshold
    if gating_config:
        high_th = gating_config.get("high_threshold", high_th)
        low_th = gating_config.get("low_threshold", low_th)
        margin_th = gating_config.get("top_margin_threshold", margin_th)

    if question_type == "choice":
        probs = softmax(logits, temperature)
        best_idx = int(np.argmax(probs))
        selected_key = option_keys[best_idx]
        h_norm = compute_normalized_entropy(probs)
        margin = compute_top_margin(probs)
        confidence = compute_composite_confidence(probs)
        route, reason = decide_route(confidence, margin, high_th, low_th, margin_th)

        prob_map = {k: probs[i] for i, k in enumerate(option_keys)}
        return {
            "choice": selected_key,
            "score": None,
            "noul": None,
            "probabilities": prob_map,
            "confidence": confidence,
            "logits": logits,
            "temperature": temperature,
            "gating": {
                "route": route,
                "confidence": confidence,
                "normalized_entropy": h_norm,
                "top_margin": margin,
                "degradation_reason": reason,
            },
        }

    elif question_type == "score":
        probs = softmax(logits, temperature)
        expected_score_val = sum(i * p for i, p in enumerate(probs))
        confidence = compute_score_variance_confidence(probs)
        h_norm = compute_normalized_entropy(probs)
        margin = compute_top_margin(probs)
        route, reason = decide_route(confidence, margin, high_th, low_th, margin_th)

        # ラベル重複検査
        criteria_list = criteria_data if isinstance(criteria_data, list) else []
        is_unique = len(set(criteria_list)) == len(criteria_list) and all(criteria_list)
        if is_unique:
            prob_map = {criteria_list[i]: probs[i] for i in range(len(probs))}
        else:
            prob_map = {str(i): probs[i] for i in range(len(probs))}

        return {
            "choice": None,
            "score": expected_score_val,
            "noul": None,
            "probabilities": prob_map,
            "confidence": confidence,
            "logits": logits,
            "temperature": temperature,
            "gating": {
                "route": route,
                "confidence": confidence,
                "normalized_entropy": h_norm,
                "top_margin": margin,
                "degradation_reason": reason,
            },
        }

    elif question_type == "noul":
        prob_true = sigmoid(logits[0] - logits[1], temperature)
        effective_conf = abs(2.0 * prob_true - 1.0)
        route, reason = decide_route(effective_conf, None, high_th, low_th, margin_th)

        return {
            "choice": None,
            "score": None,
            "noul": prob_true,
            "probabilities": None,
            "confidence": None,
            "logits": logits,
            "temperature": temperature,
            "gating": {
                "route": route,
                "confidence": effective_conf,
                "normalized_entropy": None,
                "top_margin": None,
                "degradation_reason": reason,
            },
        }

    raise ValueError(f"未知の質問タイプです: {question_type}。")


def generate_all_fixtures() -> dict[str, Any]:
    """全パリティ検証用ケースを構築し、Python 側の推論結果を集約する。

    Returns:
        dict[str, Any]: フィクスチャ全体の辞書構造。
    """
    workspace_dir = Path(__file__).resolve().parent.parent.parent
    model_dir = workspace_dir / "models" / "default"
    onnx_path = model_dir / "model.onnx"
    tok_path = model_dir / "tokenizer.json"
    calib_path = model_dir / "calibration.json"

    if not onnx_path.exists() or not tok_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_dir}。")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    raw_op_token_id = tokenizer.convert_tokens_to_ids("[OP]")
    assert isinstance(raw_op_token_id, int)
    op_token_id: int = raw_op_token_id
    calib_config = (
        CalibrationConfig.load(calib_path)
        if calib_path.exists()
        else CalibrationConfig()
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_scenarios: list[dict[str, Any]] = [
        # 1. Choice 最小構成 (K=2)
        {
            "id": "choice_k2_sentiment",
            "state": "商品が翌日午前中に届き、梱包も丁寧で大変満足しています。",
            "questions": {
                "sentiment": {
                    "type": "choice",
                    "instructions": "レビューの感情極性を選択せよ。",
                    "criteria": {
                        "positive": "肯定的な評価や満足の表明",
                        "negative": "否定的な評価や不満の表明",
                    },
                }
            },
            "gating": None,
        },
        # 2. Choice 中間構成 (K=3)
        {
            "id": "choice_k3_intent",
            "state": "先週注文した商品の配送状況を確認したいです。注文番号はORD-12345です。",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "顧客の問い合わせ意図を分類せよ。",
                    "criteria": {
                        "shipping": "配送状況や配達予定日の確認",
                        "refund": "注文キャンセルや返金の要求",
                        "specs": "商品の機能や仕様に関する問い合わせ",
                    },
                }
            },
            "gating": None,
        },
        # 3. Score 段階評価 (M=5)
        {
            "id": "score_m5_satisfaction",
            "state": "サポート窓口の対応が迅速で、問題がすぐに解決しました。",
            "questions": {
                "satisfaction_level": {
                    "type": "score",
                    "instructions": "サポート対応の満足度を1から5で評価せよ。",
                    "criteria": [
                        "非常に不満",
                        "不満",
                        "普通",
                        "満足",
                        "大変満足",
                    ],
                }
            },
            "gating": None,
        },
        # 4. Score 段階評価 (M=3)
        {
            "id": "score_m3_urgency",
            "state": "サーバーがダウンしており、業務が完全に停止しています。至急復旧してください。",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "インシデントの緊急度を評価せよ。",
                    "criteria": [
                        "低 (業務影響なし)",
                        "中 (一部業務に遅延)",
                        "高 (全社業務停止)",
                    ],
                }
            },
            "gating": None,
        },
        # 5. Noul 真偽判定
        {
            "id": "noul_statement_verification",
            "state": "契約第3条に基づき、解約手続きは月末の10日前までに申請する必要があります。",
            "questions": {
                "can_cancel_anytime": {
                    "type": "noul",
                    "instructions": "言明「利用者はいつでも即座にペナルティなしで解約できる」の真偽を判定せよ。",
                }
            },
            "gating": None,
        },
        # 6. 実務再現シナリオ A: EC・ハードウェアトラブル (Roadmap 4.2 シナリオ1)
        {
            "id": "scenario_ec_hardware_defect",
            "state": "昨日届いたキーボードのBluetoothが途切れて全く接続できません。交換できますか？",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "最適な担当部署を分類せよ。",
                    "criteria": {
                        "tech_support": "製品初期不良・ハードウェアトラブル対応",
                        "billing": "請求・クレジットカード決済窓口",
                        "shipping": "配送先変更および追跡調査",
                        "general": "その他一般的な問い合わせ",
                    },
                },
                "urgency_score": {
                    "type": "score",
                    "instructions": "対応の緊急度を1から5で判定せよ。",
                    "criteria": [
                        "緊急度1: 軽微な質問",
                        "緊急度2: 急ぎではない確認",
                        "緊急度3: 通常の対応要求",
                        "緊急度4: 早期対応が望ましい",
                        "緊急度5: 即時対応必須の重大インシデント",
                    ],
                },
                "is_hardware_defect": {
                    "type": "noul",
                    "instructions": "言明「ハードウェア初期不良の可能性が高い」の真偽を判定せよ。",
                },
            },
            "gating": None,
        },
        # 7. 実務再現シナリオ B: 決済インシデント (Roadmap 4.2 シナリオ2)
        {
            "id": "scenario_payment_incident_e403",
            "state": "カード決済でエラーコード E-403 が発生しました。",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "推奨される初期対応アクションを選択せよ。",
                    "criteria": {
                        "reauth": "カード会社への再認証要求",
                        "retry": "別決済手段への切り替え案内",
                        "contact_support": "カスタマーサポートへのエスカレーション",
                        "cancel": "注文の強制キャンセル",
                    },
                }
            },
            "gating": None,
        },
        # 8. 実務再現シナリオ C: 不正検知スクリーニング (Roadmap 4.2 シナリオ3)
        {
            "id": "scenario_fraud_detection_vip",
            "state": "高額VIP注文（過去利用実績10年以上、IPリスクスコア低、配送先一致）。",
            "questions": {
                "fraud_risk": {
                    "type": "noul",
                    "instructions": "言明「不正注文のリスクが高い」の真偽を判定せよ。",
                }
            },
            "gating": None,
        },
        # 9. Gating 境界値: 低閾値による AutoExecute 強制
        {
            "id": "gating_boundary_forced_auto",
            "state": "新規契約の申込みを行います。プランはスタンダードでお願いします。",
            "questions": {
                "plan": {
                    "type": "choice",
                    "instructions": "希望プランを選択せよ。",
                    "criteria": {
                        "free": "無料フリープラン",
                        "standard": "標準スタンダードプラン",
                        "enterprise": "大規模エンタープライズプラン",
                    },
                }
            },
            "gating": {
                "enabled": True,
                "high_threshold": 0.001,
                "low_threshold": 0.0005,
                "top_margin_threshold": 0.0001,
            },
        },
        # 10. Gating 境界値: 高閾値による Fallback 強制
        {
            "id": "gating_boundary_forced_fallback",
            "state": "何が言いたいのかよくわからない曖昧なテキストです。",
            "questions": {
                "ambiguity": {
                    "type": "choice",
                    "instructions": "分類せよ。",
                    "criteria": {
                        "opt1": "選択肢 1",
                        "opt2": "選択肢 2",
                        "opt3": "選択肢 3",
                    },
                }
            },
            "gating": {
                "enabled": True,
                "high_threshold": 0.9999,
                "low_threshold": 0.9998,
                "top_margin_threshold": 0.50,
            },
        },
        # 11. 大規模候補数 (K=16)
        {
            "id": "choice_k16_large_scale",
            "state": "東京都千代田区で発生した交通障害に関する報道です。",
            "questions": {
                "prefecture": {
                    "type": "choice",
                    "instructions": "都道府県・地域を分類せよ。",
                    "criteria": {
                        f"area_{i:02d}": f"地域分類コード {i:02d}" for i in range(1, 17)
                    },
                }
            },
            "gating": None,
        },
    ]

    fixture_cases = []

    for scenario in test_scenarios:
        s_id = scenario["id"]
        state = scenario["state"]
        questions_dict = scenario["questions"]
        gating_cfg = scenario["gating"]

        expected_answers: dict[str, Any] = {}
        routes: list[str] = []

        # 複数質問時のプロンプト順序を保証するためキー昇順で整列
        sorted_questions = dict(sorted(questions_dict.items()))
        for q_id, q_def in sorted_questions.items():
            q_type = q_def["type"]
            instructions = q_def["instructions"]
            criteria = q_def.get("criteria")
            if isinstance(criteria, dict):
                criteria = dict(sorted(criteria.items()))
                q_def["criteria"] = criteria

            ans = run_inference_for_question(
                session=session,
                tokenizer=tokenizer,
                op_token_id=op_token_id,
                state=state,
                instructions=instructions,
                question_type=q_type,
                criteria_data=criteria,
                calib_config=calib_config,
                gating_config=gating_cfg,
            )

            expected_answers[q_id] = ans
            routes.append(ans["gating"]["route"])

        agg_route = determine_aggregate_route(routes)

        fixture_cases.append(
            {
                "id": s_id,
                "state": state,
                "questions": sorted_questions,
                "gating_config": gating_cfg,
                "expected": {
                    "answers": expected_answers,
                    "routing": {
                        "aggregate_route": agg_route,
                    },
                },
            }
        )

    return {
        "version": "1.0",
        "description": "Python Reference vs Rust local-jev-server E2E Parity Fixtures",
        "generated_with": "sbintuitions/modernbert-ja-130m + models/default/model.onnx",
        "num_cases": len(fixture_cases),
        "cases": fixture_cases,
    }


def main() -> None:
    """メイン実行関数。"""
    workspace_dir = Path(__file__).resolve().parent.parent.parent
    fixtures_dir = workspace_dir / "crates" / "local-jev-server" / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_file = fixtures_dir / "e2e_parity_fixtures.json"

    print("[INFO] ゴールデンフィクスチャの生成を開始します...")
    fixtures = generate_all_fixtures()

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)

    print(
        f"[SUCCESS] {len(fixtures['cases'])} 件のフィクスチャを保存しました: {out_file}"
    )


if __name__ == "__main__":
    main()
