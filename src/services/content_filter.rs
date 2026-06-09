//! Port of `services/content_filter.py` — request content moderation.
//!
//! Two layers, mirroring the Python module:
//!   1. A cheap local **sensitive-word** scan over the raw request text
//!      (no network), sourced from `config.sensitive_words`.
//!   2. An optional upstream **AI moderation** call (OpenAI chat-completion
//!      shaped) gated on `config.ai_review.enabled`.
//!
//! The AI moderation step **fails open** by default: on any network error,
//! non-JSON / malformed response, or ambiguous verdict the request is allowed
//! through. Set `config.ai_review.fail_open = false` for strict deployments,
//! in which case those same conditions surface as an HTTP 503-shaped error.
//!
//! The free functions `request_text` / `request_shape` are direct ports of the
//! Python module-level helpers and do not touch config or the network.

use std::time::Duration;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::config::Config;

/// Default review prompt (verbatim from the Python module).
const DEFAULT_REVIEW_PROMPT: &str = "判断用户请求是否允许。只回答 ALLOW 或 REJECT。";

/// Cap aligned to the upstream review service's max context (character count).
const MAX_REVIEW_TEXT_LEN: usize = 100_000;
const TRUNCATION_MARKER: &str = "\n…[truncated]…\n";

/// Strip base64 image data URIs before review: a text-only review model can't
/// analyze image bytes, and a single inlined image easily blows past the token
/// budget of the upstream review service.
static BASE64_DATA_URI: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"data:[\w/.+;-]+;base64,[A-Za-z0-9+/=]+").unwrap());

/// Error raised by content filtering. `status_code` mirrors the Python
/// `HTTPException` status (400 for a rejected request, 503 when strict
/// `fail_open=false` review is unavailable); `detail` is the human message.
#[derive(Debug, Clone)]
pub struct ContentFilterError {
    pub status_code: u16,
    pub detail: String,
}

impl ContentFilterError {
    fn new(status_code: u16, detail: impl Into<String>) -> Self {
        Self { status_code, detail: detail.into() }
    }

    /// `{"error": detail}` — mirrors the Python `HTTPException` detail body.
    pub fn detail_json(&self) -> Value {
        json!({ "error": self.detail })
    }
}

impl std::fmt::Display for ContentFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {}: {}", self.status_code, self.detail)
    }
}
impl std::error::Error for ContentFilterError {}

// ---- pure text/shape helpers (no config, no network) ----

/// Recursively flatten a JSON value into review text (port of `_text`).
fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr.iter().map(text_of).collect::<Vec<_>>().join("\n"),
        Value::Object(_) => {
            const KEYS: [&str; 7] =
                ["text", "input_text", "content", "input", "instructions", "system", "prompt"];
            KEYS.iter()
                .map(|k| text_of(value.get(*k).unwrap_or(&Value::Null)))
                .collect::<Vec<_>>()
                .join("\n")
        }
        // numbers, bools, null -> "" (matches Python's catch-all)
        _ => String::new(),
    }
}

/// Python-style truthiness for an optional value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map_or(false, |f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Port of `str(x or "")`: empty string for falsy/missing, else the string form.
fn or_empty_str(value: Option<&Value>) -> String {
    match value {
        Some(v) if truthy(Some(v)) => match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        },
        _ => String::new(),
    }
}

/// Structural summary counters for `request_shape`, in declared output order.
#[derive(Default)]
struct ShapeStats {
    response_message_items: i64,
    input_image_parts: i64,
    image_url_parts: i64,
    image_parts: i64,
    data_url_images: i64,
    remote_image_urls: i64,
    literal_image_placeholders: i64,
}

impl ShapeStats {
    fn walk(&mut self, value: &Value, key: &str) {
        match value {
            Value::String(s) => {
                let text = s.trim();
                let lower = text.to_lowercase();
                if lower.contains("<image>") {
                    self.literal_image_placeholders += 1;
                }
                if lower.starts_with("data:image/") {
                    self.data_url_images += 1;
                } else if (key == "image_url" || key == "url")
                    && (lower.starts_with("http://") || lower.starts_with("https://"))
                {
                    self.remote_image_urls += 1;
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    self.walk(item, key);
                }
            }
            Value::Object(map) => {
                let item_type = match map.get("type") {
                    Some(Value::String(s)) => s.trim().to_string(),
                    _ => String::new(),
                };
                match item_type.as_str() {
                    "message" => self.response_message_items += 1,
                    "input_image" => self.input_image_parts += 1,
                    "image_url" => self.image_url_parts += 1,
                    "image" => self.image_parts += 1,
                    _ => {}
                }
                for (child_key, child) in map {
                    self.walk(child, child_key);
                }
            }
            _ => {}
        }
    }

    /// Emit a map with only the non-zero counters (matches Python).
    fn into_value(self) -> Value {
        let mut out = Map::new();
        let pairs = [
            ("response_message_items", self.response_message_items),
            ("input_image_parts", self.input_image_parts),
            ("image_url_parts", self.image_url_parts),
            ("image_parts", self.image_parts),
            ("data_url_images", self.data_url_images),
            ("remote_image_urls", self.remote_image_urls),
            ("literal_image_placeholders", self.literal_image_placeholders),
        ];
        for (k, v) in pairs {
            if v != 0 {
                out.insert(k.to_string(), json!(v));
            }
        }
        Value::Object(out)
    }
}

/// Strip base64 data URIs and truncate to the review-service context limit.
/// Returns `(sanitized_text, base64_blocks_stripped, truncated_chars)`.
fn sanitize_for_review(text: &str) -> (String, i64, i64) {
    let base64_blocks_stripped = BASE64_DATA_URI.find_iter(text).count() as i64;
    let sanitized = BASE64_DATA_URI.replace_all(text, "[image]").into_owned();

    let chars: Vec<char> = sanitized.chars().collect();
    if chars.len() > MAX_REVIEW_TEXT_LEN {
        let marker_len = TRUNCATION_MARKER.chars().count();
        // Reserve marker space so the result stays within the cap.
        let half = (MAX_REVIEW_TEXT_LEN - marker_len) / 2;
        let truncated_chars = (chars.len() - 2 * half) as i64;
        let head: String = chars[..half].iter().collect();
        let tail: String = chars[chars.len() - half..].iter().collect();
        let out = format!("{head}{TRUNCATION_MARKER}{tail}");
        return (out, base64_blocks_stripped, truncated_chars);
    }
    (sanitized, base64_blocks_stripped, 0)
}

/// Defensively pull the verdict text out of the review-service response.
/// Returns `None` when the shape doesn't match the chat-completion contract.
fn extract_review_decision(data: &Value) -> Option<String> {
    let choices = data.get("choices")?.as_array()?;
    let first = choices.first()?.as_object()?;
    let message = first.get("message")?.as_object()?;
    let content = message.get("content")?;
    if content.is_null() {
        return None;
    }
    let s = match content {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    Some(s.trim().to_lowercase())
}

fn is_allow_decision(decision: &str) -> bool {
    ["allow", "pass", "true", "yes", "通过", "允许", "安全"]
        .iter()
        .any(|p| decision.starts_with(p))
}

fn is_reject_decision(decision: &str) -> bool {
    ["reject", "deny", "block", "false", "no", "拒绝", "不允许", "违规", "禁止"]
        .iter()
        .any(|p| decision.starts_with(p))
}

/// Resolve `fail_open` from review config. Defaults to `true`.
fn resolve_fail_open(review: &Value) -> bool {
    match review.get("fail_open") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => {
            matches!(s.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
        }
        other => truthy(other),
    }
}

/// On review failure: log, and if not failing open surface a 503-shaped error.
fn on_failure(payload: Value, fail_open: bool) -> Result<(), ContentFilterError> {
    tracing::warn!(target: "content_filter", "{payload}");
    if fail_open {
        Ok(())
    } else {
        Err(ContentFilterError::new(503, "AI 审核服务暂时不可用，请稍后重试"))
    }
}

/// Content moderation gate (port of `services/content_filter.py`).
pub struct ContentFilter {
    config: Config,
}

impl ContentFilter {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Flatten and join the non-empty text of each value (port of `request_text`).
    pub fn request_text(values: &[&Value]) -> String {
        values
            .iter()
            .map(|v| text_of(v).trim().to_string())
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Safe structural summary without logging prompts or image bytes
    /// (port of `request_shape`). Only non-zero counters are present.
    pub fn request_shape(values: &[&Value]) -> Value {
        let mut stats = ShapeStats::default();
        for value in values {
            stats.walk(value, "");
        }
        stats.into_value()
    }

    /// Build a moderation HTTP client honoring the configured proxy.
    fn build_client(&self) -> Result<wreq::Client, wreq::Error> {
        let mut builder = wreq::Client::builder().emulation(wreq_util::Emulation::Chrome137);
        let proxy = self.config.proxy_setting();
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        builder.build()
    }

    /// Check a request's text. `Ok(())` allows it; `Err` rejects it
    /// (HTTP 400 sensitive-word / AI reject, or HTTP 503 strict-review outage).
    ///
    /// Faithful to the Python `check_request`: empty text passes, sensitive
    /// words reject immediately, and AI moderation fails open by default.
    pub async fn check_request(&self, text: &str) -> Result<(), ContentFilterError> {
        if text.trim().is_empty() {
            return Ok(());
        }

        // Local sensitive-word match runs on the raw text (cheap, no network).
        for word in self.config.sensitive_words() {
            if text.contains(&word) {
                return Err(ContentFilterError::new(400, "检测到敏感词，拒绝本次任务"));
            }
        }

        let review = self.config.ai_review();
        if !truthy(review.get("enabled")) {
            return Ok(());
        }

        let base_url = or_empty_str(review.get("base_url")).trim().trim_end_matches('/').to_string();
        let api_key = or_empty_str(review.get("api_key")).trim().to_string();
        let model = or_empty_str(review.get("model")).trim().to_string();
        if base_url.is_empty() || api_key.is_empty() || model.is_empty() {
            return Err(ContentFilterError::new(400, "ai review config is incomplete"));
        }

        let fail_open = resolve_fail_open(&review);

        let (review_text, base64_blocks_stripped, truncated_chars) = sanitize_for_review(text);
        if base64_blocks_stripped != 0 || truncated_chars != 0 {
            tracing::info!(
                target: "content_filter",
                "{}",
                json!({
                    "event": "ai_review_text_sanitized",
                    "original_text_len": text.chars().count(),
                    "review_text_len": review_text.chars().count(),
                    "base64_blocks_stripped": base64_blocks_stripped,
                    "truncated_chars": truncated_chars,
                })
            );
        }

        let prompt = {
            let p = or_empty_str(review.get("prompt"));
            let p = if p.is_empty() { DEFAULT_REVIEW_PROMPT.to_string() } else { p };
            p.trim().to_string()
        };
        let content = format!("{prompt}\n\n用户请求:\n{review_text}\n\n只回答 ALLOW 或 REJECT。");

        let original_len = text.chars().count();
        let review_len = review_text.chars().count();

        let client = match self.build_client() {
            Ok(c) => c,
            Err(exc) => {
                on_failure(
                    json!({
                        "event": "ai_review_request_failed",
                        "error": exc.to_string(),
                        "error_type": "ClientBuildError",
                        "review_text_len": review_len,
                        "original_text_len": original_len,
                    }),
                    fail_open,
                )?;
                return Ok(());
            }
        };

        let mut headers = wreq::header::HeaderMap::new();
        for (k, v) in [
            ("Authorization", format!("Bearer {api_key}")),
            ("Content-Type", "application/json".to_string()),
        ] {
            if let (Ok(name), Ok(val)) = (
                wreq::header::HeaderName::from_bytes(k.as_bytes()),
                wreq::header::HeaderValue::from_str(&v),
            ) {
                headers.insert(name, val);
            }
        }

        let payload = json!({
            "model": model,
            "messages": [{ "role": "user", "content": content }],
            "temperature": 0,
        });

        let resp = match client
            .post(format!("{base_url}/v1/chat/completions"))
            .headers(headers)
            .json(&payload)
            .timeout(Duration::from_secs(60))
            .send()
            .await
        {
            Ok(r) => r,
            Err(exc) => {
                on_failure(
                    json!({
                        "event": "ai_review_request_failed",
                        "error": exc.to_string(),
                        "error_type": "RequestError",
                        "review_text_len": review_len,
                        "original_text_len": original_len,
                    }),
                    fail_open,
                )?;
                return Ok(());
            }
        };

        let status_code = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        let data: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(exc) => {
                on_failure(
                    json!({
                        "event": "ai_review_response_not_json",
                        "status_code": status_code,
                        "body_preview": body.chars().take(200).collect::<String>(),
                        "error": exc.to_string(),
                    }),
                    fail_open,
                )?;
                return Ok(());
            }
        };

        let decision = match extract_review_decision(&data) {
            Some(d) => d,
            None => {
                on_failure(
                    json!({
                        "event": "ai_review_malformed_response",
                        "status_code": status_code,
                        "body_preview": data.to_string().chars().take(300).collect::<String>(),
                        "review_text_len": review_len,
                        "original_text_len": original_len,
                    }),
                    fail_open,
                )?;
                return Ok(());
            }
        };

        if is_allow_decision(&decision) {
            return Ok(());
        }
        if is_reject_decision(&decision) {
            return Err(ContentFilterError::new(400, "AI 审核未通过，拒绝本次任务"));
        }

        // Ambiguous decisions (e.g. "MAYBE", empty content) fall back to fail-open policy.
        on_failure(
            json!({
                "event": "ai_review_ambiguous_decision",
                "decision": decision.chars().take(100).collect::<String>(),
                "review_text_len": review_len,
            }),
            fail_open,
        )?;
        Ok(())
    }
}
