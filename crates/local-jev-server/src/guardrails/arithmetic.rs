//! # 算術・数え上げ事前集計ヘルパーモジュール (Arithmetic Annotation)
//!
//! 小規模エンコーダーモデルが原理的に苦手とする厳密な四則演算 (金額合算など) や
//! 配列要素のカウントを前処理で決定論的に代行し、State 末尾に集計メタデータを透過付加する。

use serde_json::Value;

/// 算術事前集計設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticConfig {
    /// 構造化 State に対する事前集計を有効にするか。
    pub enable_aggregation: bool,
    /// 集計対象とする代表的な数値フィールド名のリスト。
    pub target_numeric_keys: Vec<String>,
}

impl Default for ArithmeticConfig {
    fn default() -> Self {
        Self {
            enable_aggregation: true,
            target_numeric_keys: vec![
                "amount".to_string(),
                "total".to_string(),
                "total_amount".to_string(),
                "price".to_string(),
                "quantity".to_string(),
                "cost".to_string(),
            ],
        }
    }
}

/// 集計サマリー情報。
#[derive(Debug, Default, Clone)]
struct DerivedSummary {
    item_count: Option<usize>,
    key_count: Option<usize>,
    numeric_sums: Vec<(String, f64)>,
}

/// キー名が集計対象ターゲットに厳格に一致するか判定する。
#[inline]
fn matches_target_key(key: &str, target_keys: &[String]) -> bool {
    let lower = key.to_lowercase();
    target_keys.iter().any(|target| {
        lower == *target
            || lower.ends_with(&format!("_{target}"))
            || lower.starts_with(&format!("{target}_"))
    })
}

/// JSON 値から統計サマリーを再帰的または表層走査で集計する。
fn inspect_and_aggregate(val: &Value, target_keys: &[String]) -> Option<DerivedSummary> {
    let mut summary = DerivedSummary::default();

    match val {
        Value::Array(arr) => {
            summary.item_count = Some(arr.len());

            // 数値配列の場合の合計
            let mut num_sum = 0.0;
            let mut num_count = 0;
            for item in arr {
                if let Some(n) = item.as_f64() {
                    num_sum += n;
                    num_count += 1;
                }
            }
            if num_count > 0 && num_count == arr.len() {
                summary
                    .numeric_sums
                    .push(("elements_sum".to_string(), num_sum));
            }

            // オブジェクト配列の場合の指定キー合計
            for key in target_keys {
                let mut sum = 0.0;
                let mut match_count = 0;
                for item in arr {
                    if let Some(obj) = item.as_object()
                        && let Some(val_num) = obj.get(key).and_then(|v| v.as_f64())
                    {
                        sum += val_num;
                        match_count += 1;
                    }
                }
                if match_count > 0 {
                    summary.numeric_sums.push((format!("total_{key}"), sum));
                }
            }
        }
        Value::Object(obj) => {
            summary.key_count = Some(obj.len());

            for (k, v) in obj {
                // 配列フィールドのカウント
                if let Some(arr) = v.as_array() {
                    summary
                        .numeric_sums
                        .push((format!("{k}_count"), arr.len() as f64));

                    // 配列内のオブジェクトの数値キー合計
                    for key in target_keys {
                        let mut sum = 0.0;
                        let mut match_count = 0;
                        for item in arr {
                            if let Some(item_obj) = item.as_object()
                                && let Some(val_num) = item_obj.get(key).and_then(|v| v.as_f64())
                            {
                                sum += val_num;
                                match_count += 1;
                            }
                        }
                        if match_count > 0 {
                            summary.numeric_sums.push((format!("total_{key}"), sum));
                        }
                    }
                }

                // 特定数値フィールド
                if matches_target_key(k, target_keys)
                    && let Some(num) = v.as_f64()
                {
                    summary.numeric_sums.push((k.clone(), num));
                }
            }
        }
        _ => return None,
    }

    if summary.item_count.is_some() || summary.key_count.is_some() {
        Some(summary)
    } else {
        None
    }
}

/// State 文字列および JSON Value を検査し、事前集計サマリーブロックを生成する。
///
/// # 引数
/// - `state_val`: リクエストの State (`serde_json::Value`)。
/// - `config`: 算術設定。
///
/// # 戻り値
/// 集計可能な項目が存在した場合は `[Derived Stats: ...]` の注記文字列を返却する。
pub fn compute_arithmetic_summary(state_val: &Value, config: &ArithmeticConfig) -> Option<String> {
    if !config.enable_aggregation {
        return None;
    }

    // 文字列の場合は JSON パースを試行
    let parsed_val: Option<Value> = match state_val {
        Value::String(s) => serde_json::from_str(s).ok(),
        other => Some(other.clone()),
    };

    let target_val = parsed_val?;
    let summary = inspect_and_aggregate(&target_val, &config.target_numeric_keys)?;

    let mut parts = Vec::new();

    if let Some(items) = summary.item_count {
        parts.push(format!("item_count={items}"));
    }
    if let Some(keys) = summary.key_count {
        parts.push(format!("key_count={keys}"));
    }
    for (k, v) in summary.numeric_sums {
        if v.fract() == 0.0 {
            parts.push(format!("{k}={}", v as i64));
        } else {
            parts.push(format!("{k}={v:.2}"));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(format!("\n[Derived Stats: {}]", parts.join(", ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_compute_arithmetic_summary_array_objects() {
        let config = ArithmeticConfig::default();
        let val = json!([
            {"name": "item A", "amount": 1000},
            {"name": "item B", "amount": 2500},
            {"name": "item C", "amount": 1500}
        ]);

        let summary = compute_arithmetic_summary(&val, &config);
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert!(s.contains("item_count=3"));
        assert!(s.contains("total_amount=5000"));
    }

    #[test]
    fn test_compute_arithmetic_summary_numbers_array() {
        let config = ArithmeticConfig::default();
        let val = json!([10, 20, 30, 40]);

        let summary = compute_arithmetic_summary(&val, &config);
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert!(s.contains("item_count=4"));
        assert!(s.contains("elements_sum=100"));
    }
}
