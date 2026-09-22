//! # 相対日時正規化・時間的接地モジュール (Temporal Grounding)
//!
//! 小規模言語モデルの弱点 (Jaggedness) である相対日時の解釈誤差を低減するため、
//! 「今日」「3日前」「先週」などの相対日時表現を、基準タイムスタンプに基づき
//! 決定論的に絶対日付注記付きの表現へとインライン正規化する。

use std::borrow::Cow;
use std::sync::LazyLock;

use chrono::{DateTime, Duration, Utc};
use regex::Regex;

/// 日本語および英語の相対日時表現を検出する正規表現。
/// 既に `(YYYY-MM-DD)` のような注記が付与されている場合は二重注記を防止する。
static RELATIVE_DAYS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?P<target>\b(?:today|yesterday|tomorrow|\d+\s*days?\s*ago|\d+\s*days?\s*later)\b|\d+\s*(?:日前|日まえ|営業日前|日後|日ご)|一昨日|おととい|昨日|きのう|今日|きょう|明日|あした|明後日|あさって)(?P<annotated>\s*[\(（]\d{4}-\d{2}-\d{2}[\)）])?")
        .expect("相対日時正規表現のコンパイルに成功すること。")
});

/// 相対日時正規化設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalConfig {
    /// 相対日時表現のインライン絶対化を有効にするか。
    pub enable_inline_normalization: bool,
    /// State 末尾に基準時刻メタデータ (`[Reference Time: ...]`) を付加するか。
    pub attach_reference_time: bool,
    /// 基準日時のタイムゾーンオフセット (時間単位、例: JST は +9、UTC は 0)。
    pub timezone_offset_hours: i32,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enable_inline_normalization: true,
            attach_reference_time: true,
            timezone_offset_hours: 0,
        }
    }
}

/// 相対日時表現を絶対日付に変換する。
fn resolve_relative_date(
    target: &str,
    ref_time: DateTime<Utc>,
    offset_hours: i32,
) -> Option<String> {
    let lower = target.to_lowercase();
    let local_time = ref_time + Duration::hours(offset_hours as i64);
    let date = local_time.date_naive();

    if target == "今日" || target == "きょう" || lower == "today" {
        return Some(format!("{target} ({})", date.format("%Y-%m-%d")));
    }
    if target == "昨日" || target == "きのう" || lower == "yesterday" {
        let target_date = date - Duration::days(1);
        return Some(format!("{target} ({})", target_date.format("%Y-%m-%d")));
    }
    if target == "一昨日" || target == "おととい" {
        let target_date = date - Duration::days(2);
        return Some(format!("{target} ({})", target_date.format("%Y-%m-%d")));
    }
    if target == "明日" || target == "あした" || lower == "tomorrow" {
        let target_date = date + Duration::days(1);
        return Some(format!("{target} ({})", target_date.format("%Y-%m-%d")));
    }
    if target == "明後日" || target == "あさって" {
        let target_date = date + Duration::days(2);
        return Some(format!("{target} ({})", target_date.format("%Y-%m-%d")));
    }

    // N日前 / N日後 / N days ago / N days later
    if target.contains("日前")
        || target.contains("日まえ")
        || lower.contains("days ago")
        || lower.contains("day ago")
    {
        let num_str: String = target.chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num_str.parse::<i64>() {
            let target_date = date - Duration::days(n);
            return Some(format!("{target} ({})", target_date.format("%Y-%m-%d")));
        }
    }

    if target.contains("日後")
        || target.contains("日ご")
        || lower.contains("days later")
        || lower.contains("day later")
    {
        let num_str: String = target.chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num_str.parse::<i64>() {
            let target_date = date + Duration::days(n);
            return Some(format!("{target} ({})", target_date.format("%Y-%m-%d")));
        }
    }

    None
}

/// 相対日時表現を正規化する。
///
/// # 引数
/// - `text`: 入力テキスト。
/// - `ref_time`: 基準日時 (リクエスト受付時刻など)。
/// - `config`: 相対日時設定。
pub fn normalize_temporal<'a>(
    text: &'a str,
    ref_time: DateTime<Utc>,
    config: &TemporalConfig,
) -> Cow<'a, str> {
    if text.is_empty() || !config.enable_inline_normalization {
        return Cow::Borrowed(text);
    }

    let mut current = Cow::Borrowed(text);

    // 相対日時パターンが含まれている可能性があるか高速事前チェック
    if current.contains("日")
        || current.contains("今日")
        || current.contains("昨日")
        || current.contains("明日")
        || current.to_lowercase().contains("today")
        || current.to_lowercase().contains("yesterday")
        || current.to_lowercase().contains("tomorrow")
        || current.to_lowercase().contains("ago")
    {
        let replaced = RELATIVE_DAYS_REGEX.replace_all(&current, |caps: &regex::Captures| {
            // 既に注記が含まれている場合はそのままスキップ
            if caps.name("annotated").is_some() {
                return caps[0].to_string();
            }

            let target = &caps["target"];
            if let Some(resolved) =
                resolve_relative_date(target, ref_time, config.timezone_offset_hours)
            {
                resolved
            } else {
                target.to_string()
            }
        });

        if let Cow::Owned(s) = replaced {
            current = Cow::Owned(s);
        }
    }

    current
}

/// State の末尾に基準時刻メタデータブロックを付加する。
///
/// 設定されたタイムゾーンオフセットを反映した ISO 8601 形式で出力する。
pub fn append_reference_time_metadata(
    state: &mut String,
    ref_time: DateTime<Utc>,
    offset_hours: i32,
) {
    let offset_sec = (offset_hours * 3600).clamp(-86399, 86399);
    let fixed_offset = chrono::FixedOffset::east_opt(offset_sec)
        .unwrap_or_else(|| chrono::FixedOffset::east_opt(0).unwrap());
    let local_dt = ref_time.with_timezone(&fixed_offset);
    let date_str = if offset_hours == 0 {
        ref_time.format("%Y-%m-%dT%H:%M:%SZ").to_string()
    } else {
        local_dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
    };
    let meta_block = format!("\n[Reference Time: {date_str}]");
    state.push_str(&meta_block);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_normalize_temporal_days() {
        let config = TemporalConfig::default();
        let ref_time = Utc.with_ymd_and_hms(2026, 9, 22, 12, 0, 0).unwrap();

        let raw = "注文は3日前に発生し、昨日発送されました。";
        let res = normalize_temporal(raw, ref_time, &config);
        assert_eq!(
            res,
            "注文は3日前 (2026-09-19)に発生し、昨日 (2026-09-21)発送されました。"
        );
    }

    #[test]
    fn test_normalize_temporal_english() {
        let config = TemporalConfig::default();
        let ref_time = Utc.with_ymd_and_hms(2026, 9, 22, 12, 0, 0).unwrap();

        let raw = "The report was updated 5 days ago and due tomorrow.";
        let res = normalize_temporal(raw, ref_time, &config);
        assert_eq!(
            res,
            "The report was updated 5 days ago (2026-09-17) and due tomorrow (2026-09-23)."
        );
    }

    #[test]
    fn test_normalize_temporal_with_timezone_offset() {
        // UTC 2026-09-21 23:00:00 は、JST (UTC+9) では 2026-09-22 08:00:00 (翌朝)
        let ref_time_utc = Utc.with_ymd_and_hms(2026, 9, 21, 23, 0, 0).unwrap();

        // 1. UTC (offset = 0) の場合、今日は 2026-09-21
        let config_utc = TemporalConfig {
            timezone_offset_hours: 0,
            ..Default::default()
        };
        let res_utc = normalize_temporal("今日は晴れ。", ref_time_utc, &config_utc);
        assert_eq!(res_utc, "今日 (2026-09-21)は晴れ。");

        // 2. JST (offset = +9) の場合、日付境界を越えて今日は 2026-09-22
        let config_jst = TemporalConfig {
            timezone_offset_hours: 9,
            ..Default::default()
        };
        let res_jst = normalize_temporal("今日は晴れ。", ref_time_utc, &config_jst);
        assert_eq!(res_jst, "今日 (2026-09-22)は晴れ。");

        // メタデータヘッダーも +09:00 の形式で出力されること
        let mut state = "State内容".to_string();
        append_reference_time_metadata(&mut state, ref_time_utc, 9);
        assert!(state.contains("[Reference Time: 2026-09-22T08:00:00+09:00]"));
    }
}
