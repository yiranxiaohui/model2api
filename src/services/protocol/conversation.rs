//! Port of `services/protocol/conversation.py` — the core orchestrator the API
//! routes call. Translates OpenAI/Anthropic-shaped requests into engine calls,
//! drives the SSE conversation state machine, and runs the text + image flows
//! (single-image, N-image, account-pool rotation with retries).
//!
//! Async adaptation notes (see the report for the full list):
//!   * Python's blocking generators become channel-based streams so axum can
//!     stream: `stream_text_deltas` / `stream_image_outputs_with_pool` spawn a
//!     tokio task and feed an `mpsc` receiver.
//!   * The Rust engine splits the Python `conversation_events` into
//!     `stream_conversation` (text) and `stream_picture_conversation` (image);
//!     polling is encapsulated inside `resolve_conversation_image_urls`.
//!   * Engine internals the Python retry loops used directly
//!     (`_poll_image_results`, `find_conversation_by_prompt`,
//!     `_query_backend_tasks`, `check_task_error`) are private in Rust, so the
//!     text-reply / fallback retry paths use `resolve_conversation_image_urls`
//!     (which re-polls) instead. These spots are marked `// TODO(engine-internal)`.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};
use tiktoken_rs::CoreBPE;
use tokio::sync::mpsc;

use crate::config::Config;
use crate::error::AppError;
use crate::services::account_service::AccountService;
use crate::services::image_storage_service::ImageStorageService;
use crate::services::openai_backend_api::{EngineError, OpenAIBackendAPI};
use crate::utils::helper::{is_codex_image_model, is_supported_image_model, split_image_model, IMAGE_MODEL_PLAN_TYPES};
use crate::utils::image_tokens::count_image_content_tokens;

// ---------------------------------------------------------------------------
// regexes (mirror the Python module-level patterns exactly)
// ---------------------------------------------------------------------------

static REFERENCED_IMAGE_IDS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#""referenced_image_ids"\s*:\s*\[([^\]]+)\]"#).unwrap());
static TOOL_PARAMS_JSON_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"\{\s*"size"\s*:\s*"\d+x\d+"\s*,\s*"n"\s*:\s*\d+\s*\}"#).unwrap());

static CONVERSATION_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#""conversation_id"\s*:\s*"([^"]+)""#).unwrap());
static FILE_SERVICE_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"file-service://([A-Za-z0-9_-]+)").unwrap());
static REAL_IMAGE_FILE_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bfile_00000000[a-f0-9]{24}\b").unwrap());
static SEDIMENT_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"sediment://([A-Za-z0-9_-]+)").unwrap());

// sanitize_output_text patterns (private-use unicode markers).
static RE_ANNOT: Lazy<Regex> = Lazy::new(|| Regex::new("\u{e200}([^\u{e201}]*)\u{e201}").unwrap());
static RE_ANNOT_TAIL: Lazy<Regex> = Lazy::new(|| Regex::new("\u{e200}[^\u{e201}]*$").unwrap());
static RE_SPACE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+([.,;:!?])").unwrap());
static RE_INTERNAL1: Lazy<Regex> = Lazy::new(|| Regex::new(r"^turn\d+[a-z]*\d*$").unwrap());
static RE_INTERNAL2: Lazy<Regex> = Lazy::new(|| Regex::new(r"^turn\d+\w*$").unwrap());

// tiktoken: cache the BPE once (o200k_base is the default for current models).
static O200K: Lazy<CoreBPE> = Lazy::new(|| tiktoken_rs::o200k_base().expect("o200k_base"));

fn encode_len(text: &str) -> i64 {
    O200K.encode_with_special_tokens(text).len() as i64
}

fn now_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn model_value(model: &str) -> Value {
    Value::String(model.to_string())
}

// ---------------------------------------------------------------------------
// error classifiers (mirror the Python helpers verbatim)
// ---------------------------------------------------------------------------

pub fn public_image_error_message(message: &str) -> String {
    let text = message.trim();
    let lower = text.to_lowercase();
    const NEEDLES: [&str; 5] = ["backend-api/", "status=", "body=", "chatgpt.com", "upstreamhttperror"];
    if NEEDLES.iter().any(|n| lower.contains(n)) {
        return "The image generation request failed. Please try again later.".to_string();
    }
    if text.is_empty() {
        "The image generation request failed. Please try again later.".to_string()
    } else {
        text.to_string()
    }
}

pub fn is_token_invalid_error(message: &str) -> bool {
    let text = message.to_lowercase();
    text.contains("token_invalidated")
        || text.contains("token_revoked")
        || text.contains("authentication token has been invalidated")
        || text.contains("invalidated oauth token")
}

pub fn is_tls_connection_error(message: &str) -> bool {
    let text = message.to_lowercase();
    text.contains("curl: (35)")
        || text.contains("tls connect error")
        || text.contains("openssl_internal")
        || text.contains("ssl: wrong_version_number")
        || text.contains("ssl: certificate_verify_failed")
        || text.contains("connection aborted")
        || text.contains("remote disconnected")
        || text.contains("connection reset by peer")
}

pub fn is_connection_timeout_error(message: &str) -> bool {
    let text = message.to_lowercase();
    text.contains("curl: (28)")
        || text.contains("operation timed out")
        || text.contains("connection timed out")
        || text.contains("read timed out")
        || text.contains("connect timeout")
}

pub fn image_stream_error_message(message: &str) -> String {
    if is_token_invalid_error(message) {
        return "image generation failed".to_string();
    }
    if is_tls_connection_error(message) {
        return "upstream image connection failed, please retry later".to_string();
    }
    if is_connection_timeout_error(message) {
        return "upstream connection timed out, please retry later".to_string();
    }
    if message.is_empty() {
        "image generation failed".to_string()
    } else {
        message.to_string()
    }
}

pub fn is_model_text_reply_instead_of_image(message: &str) -> bool {
    if message.is_empty() {
        return false;
    }
    REFERENCED_IMAGE_IDS_RE.is_match(message) || TOOL_PARAMS_JSON_RE.is_match(message)
}

// ---------------------------------------------------------------------------
// ImageGenerationError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ImageGenerationError {
    pub message: String,
    pub status_code: u16,
    pub error_type: String,
    pub code: Option<String>,
    pub param: Option<String>,
    pub account_email: String,
    pub conversation_id: String,
}

impl ImageGenerationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status_code: 502,
            error_type: "server_error".to_string(),
            code: Some("upstream_error".to_string()),
            param: None,
            account_email: String::new(),
            conversation_id: String::new(),
        }
    }

    pub fn to_openai_error(&self) -> Value {
        let mut error = Map::new();
        error.insert("message".into(), json!(public_image_error_message(&self.message)));
        error.insert("type".into(), json!(self.error_type));
        error.insert(
            "param".into(),
            self.param.clone().map(Value::String).unwrap_or(Value::Null),
        );
        error.insert(
            "code".into(),
            self.code.clone().map(Value::String).unwrap_or(Value::Null),
        );
        if !self.account_email.is_empty() {
            error.insert("account_email".into(), json!(self.account_email));
        }
        json!({ "error": Value::Object(error) })
    }
}

impl std::fmt::Display for ImageGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ImageGenerationError {}

/// Internal error union used by the image stream functions so the orchestrator
/// can distinguish engine errors (poll-timeout / content-policy / network) from
/// generation errors (the Python `ImageGenerationError` branch).
enum ImgErr {
    Engine(EngineError),
    Gen(ImageGenerationError),
}

impl From<EngineError> for ImgErr {
    fn from(e: EngineError) -> Self {
        ImgErr::Engine(e)
    }
}

impl From<ImageGenerationError> for ImgErr {
    fn from(e: ImageGenerationError) -> Self {
        ImgErr::Gen(e)
    }
}

// ---------------------------------------------------------------------------
// dependency context + request/output types
// ---------------------------------------------------------------------------

/// Explicitly-passed dependencies (no globals). Cheap to clone.
#[derive(Clone)]
pub struct ConvDeps {
    pub config: Config,
    pub accounts: Arc<AccountService>,
    pub image_storage: ImageStorageService,
}

#[derive(Clone, Debug)]
pub struct ConversationRequest {
    pub model: String,
    pub prompt: String,
    pub messages: Option<Vec<Value>>,
    pub images: Option<Vec<String>>,
    pub n: i64,
    pub size: Option<String>,
    pub quality: String,
    pub response_format: String,
    pub base_url: Option<String>,
    pub message_as_error: bool,
}

impl Default for ConversationRequest {
    fn default() -> Self {
        Self {
            model: "auto".to_string(),
            prompt: String::new(),
            messages: None,
            images: None,
            n: 1,
            size: None,
            quality: "auto".to_string(),
            response_format: "b64_json".to_string(),
            base_url: None,
            message_as_error: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImageOutput {
    pub kind: String,
    pub model: String,
    pub index: i64,
    pub total: i64,
    pub created: i64,
    pub text: String,
    pub upstream_event_type: String,
    pub data: Vec<Value>,
    pub account_email: String,
    pub conversation_id: String,
}

impl ImageOutput {
    fn progress(model: &str, index: i64, total: i64, text: String, upstream: &str) -> Self {
        Self {
            kind: "progress".to_string(),
            model: model.to_string(),
            index,
            total,
            created: now_ts(),
            text,
            upstream_event_type: upstream.to_string(),
            data: Vec::new(),
            account_email: String::new(),
            conversation_id: String::new(),
        }
    }

    fn message(model: &str, index: i64, total: i64, text: String, conversation_id: String) -> Self {
        Self {
            kind: "message".to_string(),
            model: model.to_string(),
            index,
            total,
            created: now_ts(),
            text,
            upstream_event_type: String::new(),
            data: Vec::new(),
            account_email: String::new(),
            conversation_id,
        }
    }

    fn result(model: &str, index: i64, total: i64, data: Vec<Value>, conversation_id: String) -> Self {
        Self {
            kind: "result".to_string(),
            model: model.to_string(),
            index,
            total,
            created: now_ts(),
            text: String::new(),
            upstream_event_type: String::new(),
            data,
            account_email: String::new(),
            conversation_id,
        }
    }

    pub fn to_chunk(&self) -> Value {
        let mut chunk = Map::new();
        chunk.insert("object".into(), json!("image.generation.chunk"));
        chunk.insert("created".into(), json!(self.created));
        chunk.insert("model".into(), json!(self.model));
        chunk.insert("index".into(), json!(self.index));
        chunk.insert("total".into(), json!(self.total));
        chunk.insert("progress_text".into(), json!(self.text));
        chunk.insert("upstream_event_type".into(), json!(self.upstream_event_type));
        chunk.insert("data".into(), json!([]));
        if !self.account_email.is_empty() {
            chunk.insert("_account_email".into(), json!(self.account_email));
        }
        if !self.conversation_id.is_empty() {
            chunk.insert("_conversation_id".into(), json!(self.conversation_id));
        }
        if self.kind == "message" {
            chunk.insert("object".into(), json!("image.generation.message"));
            chunk.insert("message".into(), json!(self.text));
            chunk.remove("progress_text");
            chunk.remove("upstream_event_type");
        } else if self.kind == "result" {
            chunk.insert("object".into(), json!("image.generation.result"));
            chunk.insert("data".into(), json!(self.data));
            chunk.remove("progress_text");
            chunk.remove("upstream_event_type");
        }
        Value::Object(chunk)
    }
}

// ---------------------------------------------------------------------------
// pure helpers: text extraction, normalization, prompts
// ---------------------------------------------------------------------------

fn json_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::String(s) => s.is_empty(),
        Value::Number(n) => n.as_f64().map(|f| f == 0.0).unwrap_or(false),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

pub fn message_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut parts = String::new();
            for item in items {
                if let Some(s) = item.as_str() {
                    parts.push_str(s);
                } else if let Some(obj) = item.as_object() {
                    let t = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if matches!(t, "text" | "input_text" | "output_text") {
                        parts.push_str(obj.get("text").and_then(|v| v.as_str()).unwrap_or(""));
                    }
                }
            }
            parts
        }
        _ => String::new(),
    }
}

/// Collect image-bearing parts of a user message so token counting still sees
/// them. NOTE: this diverges from Python's `extract_image_from_message_content`
/// (which returns decoded `(bytes, mime)` pairs) because that helper is not
/// ported in the Rust `utils::helper`; downstream the engine drops non-text
/// parts for plain chat anyway, so only the shapes the token counter
/// understands (`image`/`image_url`/`input_image`/`source`) are preserved.
fn collect_image_parts(content: &Value) -> Vec<Value> {
    let Some(arr) = content.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for part in arr {
        let Some(obj) = part.as_object() else { continue };
        let t = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if matches!(t, "image" | "image_url" | "input_image") || obj.contains_key("source") {
            out.push(part.clone());
        }
    }
    out
}

pub fn normalize_messages(config: &Config, messages: &[Value], system: Option<&Value>) -> Vec<Value> {
    let mut normalized: Vec<Value> = Vec::new();
    let gsp = config.global_system_prompt();
    if !gsp.is_empty() {
        normalized.push(json!({"role": "system", "content": gsp}));
    }
    let system_text = system.map(message_text).unwrap_or_default();
    if !system_text.is_empty() {
        normalized.push(json!({"role": "system", "content": system_text}));
    }
    for message in messages {
        let Some(obj) = message.as_object() else { continue };
        let role = obj.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = obj.get("content").cloned().unwrap_or(Value::Null);
        let text = message_text(&content);
        let image_parts = if role == "user" {
            collect_image_parts(&content)
        } else {
            Vec::new()
        };
        if !image_parts.is_empty() {
            let mut parts: Vec<Value> = Vec::new();
            if !text.is_empty() {
                parts.push(json!({"type": "text", "text": text}));
            }
            parts.extend(image_parts);
            normalized.push(json!({"role": role, "content": parts}));
        } else {
            normalized.push(json!({"role": role, "content": text}));
        }
    }
    normalized
}

fn prompt_with_global_system(config: &Config, prompt: &str) -> String {
    let gsp = config.global_system_prompt();
    if gsp.is_empty() {
        prompt.to_string()
    } else {
        format!("{gsp}\n\n{prompt}")
    }
}

fn assistant_history_text(messages: &[Value]) -> String {
    let mut out = String::new();
    for item in messages {
        if item.get("role").and_then(|v| v.as_str()) == Some("assistant") {
            out.push_str(item.get("content").and_then(|v| v.as_str()).unwrap_or(""));
        }
    }
    out
}

fn assistant_history_messages(messages: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    for item in messages {
        if item.get("role").and_then(|v| v.as_str()) == Some("assistant") {
            let c = item.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if !c.is_empty() {
                out.push(c.to_string());
            }
        }
    }
    out
}

pub fn build_image_prompt(prompt: &str, size: Option<&str>, quality: &str) -> String {
    let mut hints: Vec<String> = Vec::new();
    if let Some(s) = size {
        if !s.is_empty() {
            hints.push(format!("输出图片尺寸为 {s}。"));
        }
    }
    if !quality.is_empty() {
        hints.push(format!("输出图片质量为 {quality}。"));
    }
    if hints.is_empty() {
        prompt.to_string()
    } else {
        format!("{}\n\n{}", prompt.trim(), hints.join(""))
    }
}

// ---------------------------------------------------------------------------
// token counting (tiktoken via o200k_base)
// ---------------------------------------------------------------------------

pub fn count_text_tokens(text: &str, _model: &str) -> i64 {
    encode_len(text)
}

pub fn count_message_tokens(messages: &[Value], model: &str) -> i64 {
    count_message_text_tokens(messages, model) + count_message_image_tokens(messages, model)
}

fn count_message_text_tokens(messages: &[Value], _model: &str) -> i64 {
    let mut total = 0i64;
    for msg in messages {
        total += 3;
        let Some(obj) = msg.as_object() else { continue };
        for (key, value) in obj {
            let counted = if key == "content" && value.is_array() {
                Some(encode_len(&message_text(value)))
            } else if let Some(s) = value.as_str() {
                Some(encode_len(s))
            } else {
                None
            };
            if let Some(c) = counted {
                total += c;
                if key == "name" {
                    total += 1;
                }
            }
        }
    }
    total + 3
}

fn count_message_image_tokens(messages: &[Value], model: &str) -> i64 {
    let mut total = 0i64;
    for msg in messages {
        if let Some(content) = msg.get("content") {
            total += count_image_content_tokens(content, model, "auto");
        }
    }
    total
}

// ---------------------------------------------------------------------------
// sanitize_output_text
// ---------------------------------------------------------------------------

fn is_internal_annotation_part(value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() {
        return true;
    }
    let lower = v.to_lowercase();
    RE_INTERNAL1.is_match(&lower)
        || RE_INTERNAL2.is_match(&lower)
        || lower.starts_with("turn")
        || lower.starts_with("source")
}

fn readable_annotation_part(parts: &[String]) -> String {
    for part in parts {
        let value = part.trim();
        if !value.is_empty() && !is_internal_annotation_part(value) {
            return value.to_string();
        }
    }
    String::new()
}

fn replace_annotation(payload: &str) -> String {
    let parts: Vec<String> = payload.split('\u{e202}').map(|p| p.trim().to_string()).collect();
    let kind = parts.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let data: Vec<String> = if parts.len() > 1 {
        parts[1..].to_vec()
    } else {
        Vec::new()
    };
    if kind == "url" {
        let label = data.first().cloned().unwrap_or_default();
        let url = data.get(1).cloned().unwrap_or_default();
        if !label.is_empty() && (url.starts_with("http://") || url.starts_with("https://")) {
            return format!("{label} ({url})");
        }
        return if !label.is_empty() { label } else { url };
    }
    // "cite" and the default both fall back to the first readable part.
    readable_annotation_part(&data)
}

pub fn sanitize_output_text(text: &str) -> String {
    let step1 = RE_ANNOT.replace_all(text, |caps: &regex::Captures| replace_annotation(&caps[1]));
    let step2 = RE_ANNOT_TAIL.replace_all(&step1, "");
    let step3 = RE_SPACE_PUNCT.replace_all(&step2, "$1");
    step3.into_owned()
}

// ---------------------------------------------------------------------------
// assistant text / patch application
// ---------------------------------------------------------------------------

fn strip_history(text: &str, history_text: &str) -> String {
    let mut text = text.to_string();
    while !history_text.is_empty() && text.starts_with(history_text) {
        text = text[history_text.len()..].to_string();
    }
    text
}

fn assistant_message_text(message: &Value) -> String {
    let content = message.get("content").cloned().unwrap_or(json!({}));
    if let Some(parts) = content.get("parts").and_then(|v| v.as_array()) {
        if !parts.is_empty() {
            let text: String = parts.iter().filter_map(|p| p.as_str()).collect();
            if !text.is_empty() {
                return text;
            }
        }
    }
    content.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string()
}

fn apply_patch_op(operation: &Value, current_text: &str, history_text: &str) -> String {
    let op = operation.get("o").and_then(|v| v.as_str());
    let value = match operation.get("v") {
        Some(Value::String(s)) => s.clone(),
        Some(other) if !json_falsy(other) => other.to_string(),
        _ => String::new(),
    };
    if op == Some("append") {
        return format!("{current_text}{value}");
    }
    if op == Some("replace") {
        return strip_history(&value, history_text);
    }
    current_text.to_string()
}

fn apply_text_patch(event: &Value, current_text: &str, history_text: &str) -> String {
    if event.get("p").and_then(|v| v.as_str()) == Some("/message/content/parts/0") {
        return apply_patch_op(event, current_text, history_text);
    }
    let operations = event.get("v");
    if let Some(s) = operations.and_then(|v| v.as_str()) {
        let p_falsy = event.get("p").map_or(true, json_falsy);
        let o_falsy = event.get("o").map_or(true, json_falsy);
        if !current_text.is_empty() && p_falsy && o_falsy {
            return format!("{current_text}{s}");
        }
    }
    if event.get("o").and_then(|v| v.as_str()) == Some("patch") {
        if let Some(arr) = operations.and_then(|v| v.as_array()) {
            let mut text = current_text.to_string();
            for item in arr {
                if item.is_object() {
                    text = apply_text_patch(item, &text, history_text);
                }
            }
            return text;
        }
    }
    let Some(arr) = operations.and_then(|v| v.as_array()) else {
        return current_text.to_string();
    };
    let mut text = current_text.to_string();
    for item in arr {
        if item.is_object() {
            text = apply_text_patch(item, &text, history_text);
        }
    }
    text
}

fn assistant_raw_text(event: &Value, current_text: &str, history_text: &str) -> String {
    for candidate in [Some(event), event.get("v")] {
        let Some(c) = candidate else { continue };
        if !c.is_object() {
            continue;
        }
        let Some(message) = c.get("message").filter(|m| m.is_object()) else {
            continue;
        };
        let role = message
            .get("author")
            .and_then(|a| a.get("role"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if role != "assistant" {
            continue;
        }
        let text = assistant_message_text(message);
        if !text.is_empty() {
            return strip_history(&text, history_text);
        }
    }
    apply_text_patch(event, current_text, history_text)
}

pub fn assistant_text(event: &Value, current_text: &str, history_text: &str) -> String {
    sanitize_output_text(&assistant_raw_text(event, current_text, history_text))
}

fn event_assistant_text(event: &Value, history_text: &str) -> String {
    for candidate in [Some(event), event.get("v")] {
        let Some(c) = candidate else { continue };
        if !c.is_object() {
            continue;
        }
        let Some(message) = c.get("message").filter(|m| m.is_object()) else {
            continue;
        };
        if message.get("author").and_then(|a| a.get("role")).and_then(|v| v.as_str()) == Some("assistant") {
            return strip_history(&assistant_message_text(message), history_text);
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// conversation id / image-tool-event extraction + state machine
// ---------------------------------------------------------------------------

fn add_unique(values: &mut Vec<String>, candidates: Vec<String>) {
    for candidate in candidates {
        if !candidate.is_empty() && !values.contains(&candidate) {
            values.push(candidate);
        }
    }
}

pub fn extract_conversation_ids(payload: &str) -> (String, Vec<String>, Vec<String>) {
    let conversation_id = CONVERSATION_ID_RE
        .captures(payload)
        .map(|c| c[1].to_string())
        .unwrap_or_default();
    let mut file_ids: Vec<String> = Vec::new();
    add_unique(
        &mut file_ids,
        FILE_SERVICE_ID_RE.captures_iter(payload).map(|c| c[1].to_string()).collect(),
    );
    add_unique(
        &mut file_ids,
        REAL_IMAGE_FILE_ID_RE.find_iter(payload).map(|m| m.as_str().to_string()).collect(),
    );
    let sediment_ids: Vec<String> = SEDIMENT_ID_RE.captures_iter(payload).map(|c| c[1].to_string()).collect();
    (conversation_id, file_ids, sediment_ids)
}

/// `event.get("message") or value.get("message")` with Python truthiness
/// (an empty object counts as falsy).
fn event_message(event: &Value) -> Option<&Value> {
    let non_empty_obj = |v: &Value| v.as_object().map_or(false, |o| !o.is_empty());
    if let Some(m) = event.get("message").filter(|m| non_empty_obj(m)) {
        return Some(m);
    }
    event
        .get("v")
        .filter(|v| v.is_object())
        .and_then(|v| v.get("message"))
        .filter(|m| non_empty_obj(m))
}

pub fn is_image_tool_event(event: &Value) -> bool {
    let Some(message) = event_message(event) else {
        return false;
    };
    let role = message.get("author").and_then(|a| a.get("role")).and_then(|v| v.as_str());
    if role != Some("tool") {
        return false;
    }
    if message
        .get("metadata")
        .and_then(|m| m.get("async_task_type"))
        .and_then(|v| v.as_str())
        == Some("image_gen")
    {
        return true;
    }
    let content = message.get("content");
    if content.and_then(|c| c.get("content_type")).and_then(|v| v.as_str()) != Some("multimodal_text") {
        return false;
    }
    let Some(parts) = content.and_then(|c| c.get("parts")).and_then(|v| v.as_array()) else {
        return false;
    };
    parts.iter().any(|part| {
        part.is_object()
            && (part.get("content_type").and_then(|v| v.as_str()) == Some("image_asset_pointer")
                || part
                    .get("asset_pointer")
                    .and_then(|v| v.as_str())
                    .map_or(false, |ap| ap.starts_with("file-service://") || ap.starts_with("sediment://")))
    })
}

fn is_user_message_event(event: &Value) -> bool {
    let Some(message) = event_message(event) else {
        return false;
    };
    message
        .get("author")
        .and_then(|a| a.get("role"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase()
        == "user"
}

#[derive(Default)]
struct ConversationState {
    text: String,
    raw_text: String,
    conversation_id: String,
    file_ids: Vec<String>,
    sediment_ids: Vec<String>,
    blocked: bool,
    tool_invoked: Option<bool>,
    turn_use_case: String,
}

pub fn update_conversation_state(state: &mut ConversationStateView, payload: &str, event: Option<&Value>) {
    state.update(payload, event);
}

/// Public view wrapper so the function name matches the Python API while the
/// real state stays private to the decoder.
pub struct ConversationStateView {
    inner: ConversationState,
}

impl ConversationStateView {
    pub fn new() -> Self {
        Self {
            inner: ConversationState::default(),
        }
    }

    fn update(&mut self, payload: &str, event: Option<&Value>) {
        update_state(&mut self.inner, payload, event);
    }
}

impl Default for ConversationStateView {
    fn default() -> Self {
        Self::new()
    }
}

fn update_state(state: &mut ConversationState, payload: &str, event: Option<&Value>) {
    let (conversation_id, file_ids, sediment_ids) = extract_conversation_ids(payload);
    if !conversation_id.is_empty() && state.conversation_id.is_empty() {
        state.conversation_id = conversation_id;
    }
    let is_patch_event = event
        .map(|e| e.get("o").and_then(|v| v.as_str()) == Some("patch"))
        .unwrap_or(false);
    let is_user_msg = event.map(is_user_message_event).unwrap_or(false);
    let image_context = event.map(is_image_tool_event).unwrap_or(false)
        || (state.tool_invoked == Some(true) && !is_user_msg)
        || (is_patch_event
            && !is_user_msg
            && (payload.contains("asset_pointer") || payload.contains("file-service://")));
    if image_context {
        add_unique(&mut state.file_ids, file_ids);
        add_unique(&mut state.sediment_ids, sediment_ids);
    }
    let Some(event) = event else {
        return;
    };
    if let Some(cid) = event.get("conversation_id").and_then(|v| v.as_str()) {
        if !cid.is_empty() {
            state.conversation_id = cid.to_string();
        }
    }
    if let Some(v) = event.get("v").filter(|v| v.is_object()) {
        if let Some(cid) = v.get("conversation_id").and_then(|x| x.as_str()) {
            if !cid.is_empty() {
                state.conversation_id = cid.to_string();
            }
        }
    }
    if event.get("type").and_then(|v| v.as_str()) == Some("moderation") {
        if let Some(m) = event.get("moderation_response").filter(|m| m.is_object()) {
            if m.get("blocked").and_then(|v| v.as_bool()) == Some(true) {
                state.blocked = true;
            }
        }
    }
    if event.get("type").and_then(|v| v.as_str()) == Some("server_ste_metadata") {
        if let Some(meta) = event.get("metadata").filter(|m| m.is_object()) {
            if let Some(ti) = meta.get("tool_invoked").and_then(|v| v.as_bool()) {
                state.tool_invoked = Some(ti);
            }
            if let Some(tuc) = meta.get("turn_use_case").and_then(|v| v.as_str()) {
                if !tuc.is_empty() {
                    state.turn_use_case = tuc.to_string();
                }
            }
        }
    }
}

/// Streaming decoder mirroring `iter_conversation_payloads`: feed each SSE
/// payload via [`ConversationDecoder::push`] to get the conversation events it
/// produces (`conversation.delta` / `.event` / `.raw` / `.done`).
struct ConversationDecoder {
    state: ConversationState,
    history_text: String,
    history_messages: Vec<String>,
    history_index: usize,
}

impl ConversationDecoder {
    fn new(history_text: String, history_messages: Vec<String>) -> Self {
        Self {
            state: ConversationState::default(),
            history_text,
            history_messages,
            history_index: 0,
        }
    }

    fn base_event(&self, event_type: &str, extra: &[(&str, Value)]) -> Value {
        let mut obj = Map::new();
        obj.insert("type".into(), json!(event_type));
        obj.insert("text".into(), json!(self.state.text));
        obj.insert("conversation_id".into(), json!(self.state.conversation_id));
        obj.insert("file_ids".into(), json!(self.state.file_ids));
        obj.insert("sediment_ids".into(), json!(self.state.sediment_ids));
        obj.insert("blocked".into(), json!(self.state.blocked));
        obj.insert(
            "tool_invoked".into(),
            self.state.tool_invoked.map(Value::Bool).unwrap_or(Value::Null),
        );
        obj.insert("turn_use_case".into(), json!(self.state.turn_use_case));
        for (k, v) in extra {
            obj.insert((*k).to_string(), v.clone());
        }
        Value::Object(obj)
    }

    fn push(&mut self, payload: &str) -> Vec<Value> {
        if payload.is_empty() {
            return Vec::new();
        }
        if payload == "[DONE]" {
            return vec![self.base_event("conversation.done", &[("done", json!(true))])];
        }
        let event: Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => {
                update_state(&mut self.state, payload, None);
                return vec![self.base_event("conversation.raw", &[("payload", json!(payload))])];
            }
        };
        if !event.is_object() {
            return vec![self.base_event("conversation.event", &[("raw", event.clone())])];
        }
        update_state(&mut self.state, payload, Some(&event));
        if self.history_index < self.history_messages.len()
            && event_assistant_text(&event, &self.history_text) == self.history_messages[self.history_index]
        {
            self.history_index += 1;
            self.state.raw_text = String::new();
            self.state.text = String::new();
            return Vec::new();
        }
        let next_raw_text = assistant_raw_text(&event, &self.state.raw_text, &self.history_text);
        let next_text = sanitize_output_text(&next_raw_text);
        self.state.raw_text = next_raw_text;
        if next_text != self.state.text {
            let delta = if next_text.starts_with(&self.state.text) {
                next_text[self.state.text.len()..].to_string()
            } else {
                next_text.clone()
            };
            self.state.text = next_text;
            return vec![self.base_event(
                "conversation.delta",
                &[("raw", event.clone()), ("delta", json!(delta))],
            )];
        }
        vec![self.base_event("conversation.event", &[("raw", event)])]
    }
}

// ---------------------------------------------------------------------------
// format_image_result (async — persists via image storage)
// ---------------------------------------------------------------------------

pub async fn format_image_result(
    storage: &ImageStorageService,
    items: &[Value],
    prompt: &str,
    response_format: &str,
    base_url: Option<&str>,
    created: Option<i64>,
    message: &str,
) -> Value {
    let mut data: Vec<Value> = Vec::new();
    for item in items {
        let b64 = item.get("b64_json").and_then(|v| v.as_str()).unwrap_or("").trim();
        if b64.is_empty() {
            continue;
        }
        let revised_prompt = {
            let r = item.get("revised_prompt").and_then(|v| v.as_str()).unwrap_or("");
            let r = if r.is_empty() { prompt } else { r };
            let t = r.trim();
            if t.is_empty() {
                prompt.to_string()
            } else {
                t.to_string()
            }
        };
        let Ok(bytes) = B64.decode(b64) else {
            continue;
        };
        let url = storage.save(&bytes, base_url).await.url;
        if response_format == "b64_json" {
            data.push(json!({"b64_json": b64, "url": url, "revised_prompt": revised_prompt}));
        } else {
            data.push(json!({"url": url, "revised_prompt": revised_prompt}));
        }
    }
    let mut result = json!({"created": created.unwrap_or_else(now_ts), "data": data});
    if !message.is_empty() && result["data"].as_array().map_or(true, |a| a.is_empty()) {
        result["message"] = json!(message);
    }
    result
}

// ---------------------------------------------------------------------------
// text flow
// ---------------------------------------------------------------------------

fn messages_for_request(request: &ConversationRequest) -> Vec<Value> {
    match &request.messages {
        Some(m) if !m.is_empty() => m.clone(),
        _ => {
            if !request.prompt.is_empty() {
                vec![json!({"role": "user", "content": request.prompt})]
            } else {
                Vec::new()
            }
        }
    }
}

/// Run one text conversation against `token`, sending text deltas into `tx`.
/// Returns `Err(message)` on failure (string-matched by the retry loop);
/// `emitted` is set true once any non-empty delta has been sent.
async fn run_text_stream(
    deps: &ConvDeps,
    token: &str,
    request: &ConversationRequest,
    tx: &mpsc::Sender<Result<String, AppError>>,
    emitted: &mut bool,
) -> Result<(), String> {
    let account = deps.accounts.get_account(token).unwrap_or(json!({}));
    let mut engine = OpenAIBackendAPI::new(deps.config.clone(), token.to_string(), account)
        .map_err(|e| e.to_string())?;
    let input = messages_for_request(request);
    let normalized = normalize_messages(&deps.config, &input, None);
    let history_text = assistant_history_text(&normalized);
    let history_messages = assistant_history_messages(&normalized);
    let mut stream = engine
        .stream_conversation(Some(normalized), &request.model, &request.prompt)
        .await
        .map_err(|e| e.to_string())?;
    let mut decoder = ConversationDecoder::new(history_text, history_messages);
    loop {
        let payload = match stream.next_payload().await.map_err(|e| e.to_string())? {
            Some(p) => p,
            None => break,
        };
        for ev in decoder.push(&payload) {
            if ev.get("type").and_then(|v| v.as_str()) == Some("conversation.delta") {
                let delta = ev.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if !delta.is_empty() {
                    *emitted = true;
                    if tx.send(Ok(delta.to_string())).await.is_err() {
                        return Ok(());
                    }
                }
            }
        }
        if payload == "[DONE]" {
            break;
        }
    }
    Ok(())
}

/// Stream text deltas for `request`, rotating text accounts on token invalidation
/// (port of `stream_text_deltas`). The initial token is fetched from the pool
/// (Python's caller built the backend with `get_text_access_token()`).
pub fn stream_text_deltas(
    deps: ConvDeps,
    request: ConversationRequest,
) -> mpsc::Receiver<Result<String, AppError>> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut attempted: HashSet<String> = HashSet::new();
        let mut token = deps.accounts.get_text_access_token(&attempted).await;
        let mut emitted = false;
        loop {
            if !token.is_empty() && attempted.contains(&token) {
                let _ = tx.send(Err(AppError::internal("no available text account"))).await;
                return;
            }
            if !token.is_empty() {
                attempted.insert(token.clone());
            }
            match run_text_stream(&deps, &token, &request, &tx, &mut emitted).await {
                Ok(()) => {
                    deps.accounts.mark_text_used(&token);
                    return;
                }
                Err(error_message) => {
                    if !token.is_empty() && !emitted && is_token_invalid_error(&error_message) {
                        let refreshed = deps
                            .accounts
                            .refresh_access_token(&token, true, "text_stream")
                            .await;
                        if !refreshed.is_empty() && refreshed != token && !attempted.contains(&refreshed) {
                            token = refreshed;
                        } else {
                            deps.accounts.remove_invalid_token(&token, "text_stream", false);
                            token = deps.accounts.get_text_access_token(&attempted).await;
                        }
                        if !token.is_empty() {
                            continue;
                        }
                    }
                    let _ = tx.send(Err(AppError::upstream(error_message))).await;
                    return;
                }
            }
        }
    });
    rx
}

/// Collect the full text response (drains [`stream_text_deltas`]).
pub async fn collect_text(deps: ConvDeps, request: ConversationRequest) -> Result<String, AppError> {
    let mut rx = stream_text_deltas(deps, request);
    let mut out = String::new();
    while let Some(item) = rx.recv().await {
        out.push_str(&item?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// image flow — single-account stream functions
// ---------------------------------------------------------------------------

fn codex_response_images(value: &Value) -> Vec<String> {
    match value {
        Value::Object(m) => {
            if m.get("type").and_then(|v| v.as_str()) == Some("image_generation_call") {
                if let Some(result) = m.get("result").and_then(|v| v.as_str()) {
                    let result = result.trim();
                    if !result.is_empty() {
                        let item = if result.starts_with("data:image/") {
                            result.splitn(2, ',').nth(1).unwrap_or(result).to_string()
                        } else {
                            result.to_string()
                        };
                        return vec![item];
                    }
                }
            }
            let mut out = Vec::new();
            for v in m.values() {
                out.extend(codex_response_images(v));
            }
            out
        }
        Value::Array(a) => {
            let mut out = Vec::new();
            for v in a {
                out.extend(codex_response_images(v));
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Download the resolved URLs and format them into result `data` entries.
async fn download_and_format(
    deps: &ConvDeps,
    backend: &OpenAIBackendAPI,
    request: &ConversationRequest,
    urls: &[String],
) -> Result<Vec<Value>, ImgErr> {
    let bytes = backend.download_image_bytes(urls).await?;
    let items: Vec<Value> = bytes
        .iter()
        .map(|b| json!({"b64_json": B64.encode(b)}))
        .collect();
    let result = format_image_result(
        &deps.image_storage,
        &items,
        &request.prompt,
        &request.response_format,
        request.base_url.as_deref(),
        Some(now_ts()),
        "",
    )
    .await;
    Ok(result["data"].as_array().cloned().unwrap_or_default())
}

/// Port of `stream_codex_image_outputs`: one codex/responses image call.
async fn stream_codex_image_outputs(
    deps: &ConvDeps,
    backend: &OpenAIBackendAPI,
    request: &ConversationRequest,
    index: i64,
    total: i64,
    out: &mut Vec<ImageOutput>,
) -> Result<(), ImgErr> {
    let images = request.images.clone().unwrap_or_default();
    let events = backend
        .codex_image_response_events(&request.prompt, &images, request.size.as_deref(), &request.quality)
        .await?;
    let b64_images = codex_response_images(&Value::Array(events));
    if b64_images.is_empty() {
        return Err(ImgErr::Gen(ImageGenerationError::new("No image result found in response")));
    }
    let items: Vec<Value> = b64_images
        .iter()
        .map(|item| json!({"b64_json": item, "revised_prompt": request.prompt}))
        .collect();
    let result = format_image_result(
        &deps.image_storage,
        &items,
        &request.prompt,
        &request.response_format,
        request.base_url.as_deref(),
        Some(now_ts()),
        "",
    )
    .await;
    let data = result["data"].as_array().cloned().unwrap_or_default();
    if !data.is_empty() {
        out.push(ImageOutput::result(&request.model, index, total, data, String::new()));
        return Ok(());
    }
    Err(ImgErr::Gen(ImageGenerationError::new("No image result found in response")))
}

/// Port of `stream_image_outputs`: drive the picture_v2 SSE stream, then resolve
/// + download the generated image(s). Progress + terminal outputs are pushed
/// into `out` (so the caller can compute `emitted_for_token`).
async fn stream_image_outputs(
    deps: &ConvDeps,
    backend: &mut OpenAIBackendAPI,
    request: &ConversationRequest,
    index: i64,
    total: i64,
    out: &mut Vec<ImageOutput>,
) -> Result<(), ImgErr> {
    let images = request.images.clone().unwrap_or_default();
    let final_prompt = prompt_with_global_system(
        &deps.config,
        &build_image_prompt(&request.prompt, request.size.as_deref(), &request.quality),
    );
    let mut stream = backend
        .stream_picture_conversation(&final_prompt, &request.model, &images)
        .await?;
    // Image flow uses no assistant history (matches the Python image branch).
    let mut decoder = ConversationDecoder::new(String::new(), Vec::new());
    let mut last: Value = json!({});
    loop {
        let payload = match stream.next_payload().await? {
            Some(p) => p,
            None => break,
        };
        for ev in decoder.push(&payload) {
            let t = ev.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if t == "conversation.delta" {
                let delta = ev.get("delta").and_then(|v| v.as_str()).unwrap_or("").to_string();
                out.push(ImageOutput::progress(&request.model, index, total, delta, "conversation.delta"));
            } else if t == "conversation.event" {
                let raw_type = ev
                    .get("raw")
                    .and_then(|r| r.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                out.push(ImageOutput::progress(&request.model, index, total, String::new(), &raw_type));
            }
            last = ev;
        }
        if payload == "[DONE]" {
            break;
        }
    }

    let conversation_id = last.get("conversation_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let file_ids: Vec<String> = last
        .get("file_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let sediment_ids: Vec<String> = last
        .get("sediment_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let message = last.get("text").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let blocked = last.get("blocked").and_then(|v| v.as_bool()).unwrap_or(false);
    let turn_use_case = last.get("turn_use_case").and_then(|v| v.as_str()).unwrap_or("");

    let no_ids = file_ids.is_empty() && sediment_ids.is_empty();

    if !message.is_empty() && no_ids && blocked {
        // TODO(engine-internal): Python queried /backend-api/tasks for a
        // structured error here (`_get_detailed_error_from_tasks`); that engine
        // helper is private, so we fall back to the SSE message text.
        let error_text = if message.is_empty() {
            "Image generation was rejected by upstream policy.".to_string()
        } else {
            message.clone()
        };
        out.push(ImageOutput::message(&request.model, index, total, error_text, conversation_id));
        return Ok(());
    }

    let should_poll = !images.is_empty() || turn_use_case == "image gen";
    if !message.is_empty() && no_ids && !should_poll {
        out.push(ImageOutput::message(&request.model, index, total, message, conversation_id));
        return Ok(());
    }

    let is_text_reply = !message.is_empty() && is_model_text_reply_instead_of_image(&message);
    // TODO(engine-internal): Python recovered a lost conversation_id via
    // `find_conversation_by_prompt` and pre-checked tasks for a moderation error
    // before polling. Both rely on private engine helpers; omitted here.

    let mut poll_timeout = deps.config.image_poll_timeout_secs() as f64;
    if is_text_reply && !conversation_id.is_empty() {
        poll_timeout = poll_timeout.max(300.0);
    }

    let resolve_result = backend
        .resolve_conversation_image_urls(
            &conversation_id,
            file_ids.clone(),
            sediment_ids.clone(),
            true,
            Some(poll_timeout),
        )
        .await;
    let image_urls: Vec<String> = match resolve_result {
        Ok(urls) => urls,
        Err(EngineError::ImageContentPolicy(_)) if is_text_reply => Vec::new(),
        Err(e @ EngineError::ImageContentPolicy(_)) => return Err(e.into()),
        Err(e @ EngineError::ImagePollTimeout { .. }) => return Err(e.into()),
        Err(e) => {
            if is_text_reply && !conversation_id.is_empty() {
                Vec::new()
            } else {
                return Err(e.into());
            }
        }
    };

    if !image_urls.is_empty() {
        let data = download_and_format(deps, backend, request, &image_urls).await?;
        if !data.is_empty() {
            out.push(ImageOutput::result(&request.model, index, total, data, conversation_id));
        }
        return Ok(());
    }

    if !message.is_empty() {
        // Python retry-polled here via `_poll_image_results`; we re-poll through
        // resolve_conversation_image_urls (which polls internally) instead.
        if is_text_reply && !conversation_id.is_empty() {
            let retry_timeout = (deps.config.image_poll_timeout_secs() as f64).max(300.0);
            if let Ok(urls) = backend
                .resolve_conversation_image_urls(
                    &conversation_id,
                    file_ids.clone(),
                    sediment_ids.clone(),
                    true,
                    Some(retry_timeout),
                )
                .await
            {
                if !urls.is_empty() {
                    if let Ok(data) = download_and_format(deps, backend, request, &urls).await {
                        if !data.is_empty() {
                            out.push(ImageOutput::result(&request.model, index, total, data, conversation_id));
                            return Ok(());
                        }
                    }
                }
            }
        }
        out.push(ImageOutput::message(&request.model, index, total, message, conversation_id));
        return Ok(());
    }

    // Fallback: empty message and no image. Retry-poll if we expected an image.
    if should_poll && !conversation_id.is_empty() {
        let retry_timeout = (deps.config.image_poll_timeout_secs() as f64).max(300.0);
        let wait = 30.0_f64.min(deps.config.image_poll_initial_wait_secs()).max(0.0);
        if wait > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
        }
        if let Ok(urls) = backend
            .resolve_conversation_image_urls(
                &conversation_id,
                file_ids.clone(),
                sediment_ids.clone(),
                true,
                Some(retry_timeout),
            )
            .await
        {
            if !urls.is_empty() {
                if let Ok(data) = download_and_format(deps, backend, request, &urls).await {
                    if !data.is_empty() {
                        out.push(ImageOutput::result(&request.model, index, total, data, conversation_id));
                        return Ok(());
                    }
                }
            }
        }
        out.push(ImageOutput::message(
            &request.model,
            index,
            total,
            "Image generation completed upstream but the result could not be retrieved. \
             The image may still be processing. Please try again in a moment."
                .to_string(),
            conversation_id,
        ));
    } else {
        out.push(ImageOutput::message(
            &request.model,
            index,
            total,
            "Image generation started upstream but the response was incomplete. Please try again."
                .to_string(),
            conversation_id,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// image flow — single-image orchestration with retries (port of
// `_generate_single_image`)
// ---------------------------------------------------------------------------

async fn generate_single_image(
    deps: &ConvDeps,
    request: &ConversationRequest,
    index: i64,
    total: i64,
) -> Result<Vec<ImageOutput>, ImageGenerationError> {
    const MAX_TEXT_REPLY_RETRIES: i64 = 3;
    const MAX_TLS_RETRIES: i64 = 3;
    const MAX_CONN_TIMEOUT_RETRIES: i64 = 3;
    const MAX_POLL_TIMEOUT_RETRIES: i64 = 4;

    let mut text_reply_retry = 0i64;
    let mut tls_retry = 0i64;
    let mut conn_timeout_retry = 0i64;
    let mut poll_timeout_retry = 0i64;
    let mut account_email = String::new();

    let model_val = model_value(&request.model);
    let codex_model = is_codex_image_model(&model_val);
    let (plan_type, _) = split_image_model(&model_val);

    loop {
        let plan_types: Vec<String> = if codex_model && plan_type.is_none() {
            IMAGE_MODEL_PLAN_TYPES.iter().map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        };
        let source_type = if codex_model { Some("codex") } else { None };

        let token = match deps
            .accounts
            .get_available_access_token(plan_type.as_deref(), source_type, &plan_types)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                let mut err = ImageGenerationError::new(if e.is_empty() {
                    "image generation failed".to_string()
                } else {
                    e
                });
                err.account_email = account_email.clone();
                return Err(err);
            }
        };

        let account = deps.accounts.get_account(&token).unwrap_or(json!({}));
        account_email = account.get("email").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

        let mut outputs: Vec<ImageOutput> = Vec::new();
        let stream_result: Result<(), ImgErr> =
            match OpenAIBackendAPI::new(deps.config.clone(), token.clone(), account) {
                Ok(mut backend) => {
                    if codex_model {
                        stream_codex_image_outputs(deps, &backend, request, index, total, &mut outputs).await
                    } else {
                        stream_image_outputs(deps, &mut backend, request, index, total, &mut outputs).await
                    }
                }
                Err(e) => Err(ImgErr::Engine(e)),
            };

        // Inject account_email into produced outputs (Python does this per output).
        for o in outputs.iter_mut() {
            if !account_email.is_empty() && o.account_email.is_empty() {
                o.account_email = account_email.clone();
            }
        }

        let emitted_for_token = !outputs.is_empty();

        // message_as_error: a "message" output becomes a content-policy error,
        // routed through the same handling as a generation error.
        let outcome: Result<(), ImgErr> = stream_result.and_then(|()| {
            if request.message_as_error {
                if let Some(o) = outputs.iter().find(|o| o.kind == "message") {
                    return Err(ImgErr::Gen(ImageGenerationError {
                        message: if o.text.is_empty() {
                            "Image generation was rejected by upstream policy.".to_string()
                        } else {
                            o.text.clone()
                        },
                        status_code: 400,
                        error_type: "invalid_request_error".to_string(),
                        code: Some("content_policy_violation".to_string()),
                        param: None,
                        account_email: account_email.clone(),
                        conversation_id: o.conversation_id.clone(),
                    }));
                }
            }
            Ok(())
        });

        match outcome {
            Ok(()) => {
                let returned_message = outputs.last().map(|o| o.kind == "message").unwrap_or(false);
                let returned_result = outputs.iter().any(|o| o.kind == "result");
                if returned_message {
                    deps.accounts.mark_image_result(&token, false);
                    return Ok(outputs);
                }
                if !returned_result {
                    deps.accounts.mark_image_result(&token, false);
                    if emitted_for_token {
                        let conv_id = outputs.last().map(|o| o.conversation_id.clone()).unwrap_or_default();
                        return Err(ImageGenerationError {
                            message: "upstream completed without generating images".to_string(),
                            status_code: 400,
                            error_type: "invalid_request_error".to_string(),
                            code: Some("no_image_generated".to_string()),
                            param: None,
                            account_email: account_email.clone(),
                            conversation_id: conv_id,
                        });
                    }
                    return Ok(outputs);
                }
                deps.accounts.mark_image_result(&token, true);
                return Ok(outputs);
            }
            Err(ImgErr::Engine(EngineError::ImagePollTimeout { message, conversation_id, .. })) => {
                deps.accounts.mark_image_result(&token, false);
                if !emitted_for_token {
                    poll_timeout_retry += 1;
                    if poll_timeout_retry <= MAX_POLL_TIMEOUT_RETRIES {
                        continue;
                    }
                }
                // Adaptation: Rust needs a single error type, so the engine's
                // ImagePollTimeout becomes an ImageGenerationError (504).
                return Err(ImageGenerationError {
                    message,
                    status_code: 504,
                    error_type: "server_error".to_string(),
                    code: Some("image_poll_timeout".to_string()),
                    param: None,
                    account_email: account_email.clone(),
                    conversation_id,
                });
            }
            Err(ImgErr::Engine(EngineError::ImageContentPolicy(m))) => {
                deps.accounts.mark_image_result(&token, false);
                return Err(ImageGenerationError {
                    message: if m.is_empty() {
                        "Image generation was rejected by upstream policy.".to_string()
                    } else {
                        m
                    },
                    status_code: 400,
                    error_type: "invalid_request_error".to_string(),
                    code: Some("content_policy_violation".to_string()),
                    param: None,
                    account_email: account_email.clone(),
                    conversation_id: String::new(),
                });
            }
            Err(ImgErr::Gen(mut gen)) => {
                deps.accounts.mark_image_result(&token, false);
                if !account_email.is_empty() && gen.account_email.is_empty() {
                    gen.account_email = account_email.clone();
                }
                let error_text = gen.message.clone();
                if is_model_text_reply_instead_of_image(&error_text) && !emitted_for_token {
                    text_reply_retry += 1;
                    if text_reply_retry <= MAX_TEXT_REPLY_RETRIES {
                        continue;
                    }
                    return Err(ImageGenerationError {
                        message: "Image generation failed: the upstream model returned a text description \
                                  instead of generating an image. Please try again later."
                            .to_string(),
                        status_code: 502,
                        error_type: "server_error".to_string(),
                        code: Some("upstream_text_reply".to_string()),
                        param: None,
                        account_email: account_email.clone(),
                        conversation_id: gen.conversation_id,
                    });
                }
                return Err(gen);
            }
            Err(ImgErr::Engine(other)) => {
                deps.accounts.mark_image_result(&token, false);
                let last_error = other.to_string();
                if !emitted_for_token && is_token_invalid_error(&last_error) {
                    let refreshed = deps
                        .accounts
                        .refresh_access_token(&token, true, "image_stream")
                        .await;
                    if !(!refreshed.is_empty() && refreshed != token) {
                        deps.accounts.remove_invalid_token(&token, "image_stream", false);
                    }
                    continue;
                }
                if !emitted_for_token && is_tls_connection_error(&last_error) {
                    tls_retry += 1;
                    if tls_retry <= MAX_TLS_RETRIES {
                        let secs = (2.0 * tls_retry as f64).min(10.0);
                        tokio::time::sleep(Duration::from_secs_f64(secs)).await;
                        continue;
                    }
                }
                if !emitted_for_token && is_connection_timeout_error(&last_error) {
                    conn_timeout_retry += 1;
                    if conn_timeout_retry <= MAX_CONN_TIMEOUT_RETRIES {
                        let secs = (3.0 * conn_timeout_retry as f64).min(9.0);
                        tokio::time::sleep(Duration::from_secs_f64(secs)).await;
                        continue;
                    }
                }
                return Err(ImageGenerationError {
                    message: image_stream_error_message(&last_error),
                    status_code: 502,
                    error_type: "server_error".to_string(),
                    code: Some("upstream_error".to_string()),
                    param: None,
                    account_email: account_email.clone(),
                    conversation_id: String::new(),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// image flow — pool orchestration (port of `stream_image_outputs_with_pool`)
// ---------------------------------------------------------------------------

fn supported_image_models_listing() -> String {
    let mut models: Vec<String> = vec!["gpt-image-2".to_string(), "codex-gpt-image-2".to_string()];
    for plan in IMAGE_MODEL_PLAN_TYPES {
        models.push(format!("{plan}-codex-gpt-image-2"));
    }
    models.sort();
    models.join(", ")
}

/// Generate N images using the account pool, one task per image (parallel or
/// serial per config), rotating accounts and retrying as in Python.
///
/// Deviation from the requested `Receiver<ImageOutput>`: the receiver yields
/// `Result<ImageOutput, ImageGenerationError>` so the route layer can recover
/// the error status/type/code (Python relied on exception propagation; a bare
/// `ImageOutput` channel would silently drop those).
pub fn stream_image_outputs_with_pool(
    deps: ConvDeps,
    request: ConversationRequest,
) -> mpsc::Receiver<Result<ImageOutput, ImageGenerationError>> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        if !is_supported_image_model(&model_value(&request.model)) {
            let _ = tx
                .send(Err(ImageGenerationError::new(format!(
                    "unsupported image model,supported models: {}",
                    supported_image_models_listing()
                ))))
                .await;
            return;
        }

        // Single image: run directly (no task-spawn overhead).
        if request.n <= 1 {
            match generate_single_image(&deps, &request, 1, 1).await {
                Ok(outputs) => {
                    for o in outputs {
                        if tx.send(Ok(o)).await.is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
            return;
        }

        // Serial multi-image generation.
        if !deps.config.image_parallel_generation() {
            for index in 1..=request.n {
                match generate_single_image(&deps, &request, index, request.n).await {
                    Ok(outputs) => {
                        for o in outputs {
                            if tx.send(Ok(o)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }
            return;
        }

        // Parallel multi-image generation: one task per image, then emit in
        // index order, not letting a low-index failure block a high-index hit.
        let mut handles: Vec<(i64, tokio::task::JoinHandle<Result<Vec<ImageOutput>, ImageGenerationError>>)> =
            Vec::new();
        for index in 1..=request.n {
            let d = deps.clone();
            let r = request.clone();
            let total = request.n;
            handles.push((
                index,
                tokio::spawn(async move { generate_single_image(&d, &r, index, total).await }),
            ));
        }

        let mut results: BTreeMap<i64, Vec<ImageOutput>> = BTreeMap::new();
        let mut errors: BTreeMap<i64, ImageGenerationError> = BTreeMap::new();
        for (index, handle) in handles {
            match handle.await {
                Ok(Ok(outputs)) => {
                    results.insert(index, outputs);
                }
                Ok(Err(e)) => {
                    errors.insert(index, e);
                }
                Err(join_err) => {
                    errors.insert(index, ImageGenerationError::new(join_err.to_string()));
                }
            }
        }

        let mut emitted = false;
        let mut last_error = String::new();
        for index in 1..=request.n {
            if let Some(outputs) = results.remove(&index) {
                for o in outputs {
                    emitted = true;
                    if tx.send(Ok(o)).await.is_err() {
                        return;
                    }
                }
            } else if let Some(e) = errors.get(&index) {
                last_error = e.to_string();
            }
        }

        if !emitted {
            if last_error.is_empty() {
                last_error = "no account in the pool could generate images — check account quota and rate-limit status"
                    .to_string();
            }
            let _ = tx
                .send(Err(ImageGenerationError::new(image_stream_error_message(&last_error))))
                .await;
        }
    });
    rx
}

/// Drain the image-output receiver into the final `{created, data, ...}` result
/// (port of `collect_image_outputs`). Propagates the first error encountered.
pub async fn collect_image_outputs(
    mut rx: mpsc::Receiver<Result<ImageOutput, ImageGenerationError>>,
) -> Result<Value, ImageGenerationError> {
    let mut created: Option<i64> = None;
    let mut data: Vec<Value> = Vec::new();
    let mut message = String::new();
    let mut progress_parts: Vec<String> = Vec::new();
    let mut account_email = String::new();

    while let Some(item) = rx.recv().await {
        let output = item?;
        if created.is_none() {
            created = Some(output.created);
        }
        if !output.account_email.is_empty() && account_email.is_empty() {
            account_email = output.account_email.clone();
        }
        match output.kind.as_str() {
            "progress" => {
                if !output.text.is_empty() {
                    progress_parts.push(output.text);
                }
            }
            "message" => message = output.text,
            "result" => data.extend(output.data),
            _ => {}
        }
    }

    let mut result = json!({ "created": created.unwrap_or_else(now_ts), "data": data.clone() });
    if data.is_empty() {
        let text = if !message.is_empty() {
            message
        } else {
            progress_parts.join("").trim().to_string()
        };
        if !text.is_empty() {
            result["message"] = json!(text);
        }
    }
    if !account_email.is_empty() {
        result["_account_email"] = json!(account_email);
    }
    Ok(result)
}

/// Map image outputs to streaming chunks (port of `stream_image_chunks`).
/// Errors are forwarded so the caller can surface them.
pub fn stream_image_chunks(
    mut rx: mpsc::Receiver<Result<ImageOutput, ImageGenerationError>>,
) -> mpsc::Receiver<Result<Value, ImageGenerationError>> {
    let (tx, out_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        while let Some(item) = rx.recv().await {
            let mapped = item.map(|o| o.to_chunk());
            let is_err = mapped.is_err();
            if tx.send(mapped).await.is_err() {
                return;
            }
            if is_err {
                return;
            }
        }
    });
    out_rx
}

