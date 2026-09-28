//! # コンテキスト縮約・ノイズ除去モジュール (Anti-Context Rot)
//!
//! 連続する過剰な空白・改行、不可視のゼロ幅スペース、同一ログ行の極端な繰り返しを圧縮し、
//! トークン長バジェット (最大系列長) を有効活用するためのテキスト正規化を提供する。

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

/// 3 回以上連続する改行を 2 回 (`\n\n`) に正規化する正規表現。
static EXCESSIVE_NEWLINES_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").expect("連続改行正規表現のコンパイルに成功すること。"));

/// 2 個以上連続する半角空白・タブを 1 つの空白に正規化する正規表現。
static EXCESSIVE_WHITESPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[^\S\r\n]{2,}").expect("連続空白正規表現のコンパイルに成功すること。")
});

/// コンテキスト縮約設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactorConfig {
    /// ゼロ幅スペース (`\u{200B}` 等) および不可視制御文字の除去を有効にするか。
    pub strip_invisible_chars: bool,
    /// 連続する過剰な空白・改行の縮約を有効にするか。
    pub collapse_whitespace: bool,
    /// 同一文字の極端な連続の圧縮を有効にするか。
    pub collapse_repeated_chars: bool,
    /// ログ等の同一行の極端な繰り返しの圧縮閾値 (0 の場合は無効化、推奨: 5)。
    pub max_identical_consecutive_lines: usize,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            strip_invisible_chars: true,
            collapse_whitespace: true,
            collapse_repeated_chars: true,
            max_identical_consecutive_lines: 5,
        }
    }
}

/// ゼロ幅文字や不可視制御文字を判定する。
#[inline]
fn is_invisible_or_control(c: char) -> bool {
    matches!(
        c,
        '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{00AD}'
    ) || (c.is_control() && c != '\n' && c != '\t' && c != '\r')
}

/// 同一テキスト行の過剰な連続繰り返しを検知し圧縮する。
fn collapse_consecutive_lines(text: &str, max_consecutive: usize) -> Option<String> {
    if max_consecutive == 0 || !text.contains('\n') {
        return None;
    }

    let mut result = String::with_capacity(text.len());
    let mut current_line: Option<&str> = None;
    let mut repeat_count = 0;
    let mut modified = false;

    let flush = |line: &str, count: usize, out: &mut String, modified: &mut bool| {
        if count <= max_consecutive {
            for _ in 0..count {
                out.push_str(line);
                out.push('\n');
            }
        } else {
            *modified = true;
            for _ in 0..max_consecutive {
                out.push_str(line);
                out.push('\n');
            }
            out.push_str(&format!(
                "[... 同一行がさらに {} 回繰り返されたため省略 ...]\n",
                count - max_consecutive
            ));
        }
    };

    for line in text.split('\n') {
        if line.is_empty() {
            if let Some(prev) = current_line.take() {
                flush(prev, repeat_count, &mut result, &mut modified);
                repeat_count = 0;
            }
            result.push('\n');
            continue;
        }

        match current_line {
            Some(prev) if prev == line => {
                repeat_count += 1;
            }
            Some(prev) => {
                flush(prev, repeat_count, &mut result, &mut modified);
                current_line = Some(line);
                repeat_count = 1;
            }
            None => {
                current_line = Some(line);
                repeat_count = 1;
            }
        }
    }

    if let Some(prev) = current_line {
        flush(prev, repeat_count, &mut result, &mut modified);
    }

    // 末尾の不要な改行調整 (元のテキストの末尾に改行がなかった場合)
    if !text.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    if modified { Some(result) } else { None }
}

/// 同一文字の極端な連続 (20 回以上) を検知して圧縮する。
fn collapse_repeated_chars_fast(text: &str) -> Option<String> {
    let mut chars = text.chars().peekable();
    let mut modified = false;
    let mut result = String::with_capacity(text.len());

    while let Some(c) = chars.next() {
        let mut count = 1;
        while let Some(&next_c) = chars.peek() {
            if next_c == c {
                count += 1;
                chars.next();
            } else {
                break;
            }
        }

        if count >= 20 {
            modified = true;
            for _ in 0..5 {
                result.push(c);
            }
            result.push_str("[...省略...]");
        } else {
            for _ in 0..count {
                result.push(c);
            }
        }
    }

    if modified { Some(result) } else { None }
}

/// テキストのコンテキスト縮約およびノイズ除去を実行する。
///
/// # 引数
/// - `text`: 正規化対象文字列。
/// - `config`: コンパクター設定。
pub fn compact_text<'a>(text: &'a str, config: &CompactorConfig) -> Cow<'a, str> {
    if text.is_empty() {
        return Cow::Borrowed(text);
    }

    let mut current = Cow::Borrowed(text);

    // 1. 不可視文字・ゼロ幅文字の除去
    if config.strip_invisible_chars && current.chars().any(is_invisible_or_control) {
        let cleaned: String = current
            .chars()
            .filter(|&c| !is_invisible_or_control(c))
            .collect();
        current = Cow::Owned(cleaned);
    }

    // 2. 同一行の繰り返し圧縮
    if config.max_identical_consecutive_lines > 0
        && let Some(collapsed) =
            collapse_consecutive_lines(&current, config.max_identical_consecutive_lines)
    {
        current = Cow::Owned(collapsed);
    }

    // 3. 同一文字の極端な連続の短縮
    if config.collapse_repeated_chars
        && let Some(collapsed) = collapse_repeated_chars_fast(&current)
    {
        current = Cow::Owned(collapsed);
    }

    // 4. 過剰な空白・改行の縮約
    if config.collapse_whitespace {
        let has_excessive_newlines = EXCESSIVE_NEWLINES_REGEX.is_match(&current);
        let has_excessive_whitespace = EXCESSIVE_WHITESPACE_REGEX.is_match(&current);

        if has_excessive_newlines || has_excessive_whitespace {
            let mut s = current.into_owned();
            if has_excessive_newlines {
                s = EXCESSIVE_NEWLINES_REGEX
                    .replace_all(&s, "\n\n")
                    .into_owned();
            }
            if has_excessive_whitespace {
                s = EXCESSIVE_WHITESPACE_REGEX.replace_all(&s, " ").into_owned();
            }
            current = Cow::Owned(s);
        }
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_invisible_chars() {
        let config = CompactorConfig {
            collapse_whitespace: false,
            collapse_repeated_chars: false,
            max_identical_consecutive_lines: 0,
            ..Default::default()
        };

        let raw = "こんにちは\u{200B}世界\u{FEFF}！";
        let res = compact_text(raw, &config);
        assert_eq!(res, "こんにちは世界！");
    }

    #[test]
    fn test_collapse_whitespace_and_newlines() {
        let config = CompactorConfig {
            strip_invisible_chars: false,
            collapse_repeated_chars: false,
            max_identical_consecutive_lines: 0,
            collapse_whitespace: true,
        };

        let raw = "行1\n\n\n\n\n行2   空白   テスト";
        let res = compact_text(raw, &config);
        assert_eq!(res, "行1\n\n行2 空白 テスト");
    }

    #[test]
    fn test_collapse_consecutive_lines() {
        let config = CompactorConfig {
            collapse_whitespace: false,
            strip_invisible_chars: false,
            collapse_repeated_chars: false,
            max_identical_consecutive_lines: 3,
        };

        let log = "INFO: healthcheck\nINFO: healthcheck\nINFO: healthcheck\nINFO: healthcheck\nINFO: healthcheck\nINFO: done";
        let res = compact_text(log, &config);
        assert!(res.contains("[... 同一行がさらに 2 回繰り返されたため省略 ...]"));
        assert!(res.contains("INFO: done"));
    }
}
