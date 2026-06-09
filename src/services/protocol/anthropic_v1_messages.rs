//! Port of `services/protocol/anthropic_v1_messages.py` — the Anthropic
//! `/v1/messages` protocol adapter.
//!
//! Translates an Anthropic Messages request (`system` + `messages` + `tools`)
//! into the internal text conversation, then formats the upstream text reply
//! back into the Anthropic response shape — both the non-streaming `message`
//! object and the streaming SSE event sequence:
//!
//! ```text
//! message_start
//! content_block_start / content_block_delta* / content_block_stop   (text)
//! [ content_block_start / content_block_delta / content_block_stop ]* (tool_use)
//! message_delta   (stop_reason + output usage)
//! message_stop
//! ```
//!
//! Tools are emulated via an XML tool-call convention injected into the system
//! prompt (`build_tool_prompt` / `merge_system`); assistant `tool_use` /
//! `tool_result` history blocks are flattened to text so the text engine can
//! consume them, and the upstream model's XML output is parsed back into
//! Anthropic `tool_use` content blocks.
//!
//! Async adaptation notes:
//!   * Python consumed OpenAI chat-completion chunks via
//!     `stream_text_chat_completion`; the Rust engine exposes raw text deltas
//!     through [`stream_text_deltas`] / [`collect_text`], so the streaming state
//!     machine (`stream_events`) is driven directly off the delta channel and
//!     runs its "finish" logic when the channel closes (the implicit
//!     `finish_reason`).
//!   * Errors mid-stream have no equivalent in the Python event vocabulary
//!     (Python let the exception propagate out of the generator). Here a single
//!     Anthropic `error` SSE event is emitted and the stream ends — see
//!     `error_event`. This is the one behavioural deviation from the source.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use crate::config::Config;
use crate::error::{error_message_from_detail, AppError};
use crate::services::protocol::conversation::*;
use crate::utils::helper::{anthropic_sse_event, new_uuid};

// ---------------------------------------------------------------------------
// constants (mirror the Python module-level strings verbatim)
// ---------------------------------------------------------------------------

const XML_TOOL_RULE: &str = "Tool output adapter: when calling tools, output ONLY this XML and no prose/markdown:\n<tool_calls><tool_call><tool_name>TOOL_NAME</tool_name><parameters><PARAM><![CDATA[value]]></PARAM></parameters></tool_call></tool_calls>";

/// The trailing block appended by `build_tool_prompt`. In Python this is a
/// triple-quoted string with `.strip()` applied, so the leading blank lines are
/// removed and there is **no** separator between the joined tool blocks and
/// "Tool use rules:" — reproduced exactly here.
const TOOL_RULES_SUFFIX: &str = "Tool use rules:\n- If the user asks to list/read/search files, inspect project state, run a command, or answer from local code, you MUST call a suitable tool first. Do not say you cannot access files.\n- To call tools, output ONLY XML and no prose/markdown:\n<tool_calls><tool_call><tool_name>TOOL_NAME</tool_name><parameters><PARAM><![CDATA[value]]></PARAM></parameters></tool_call></tool_calls>\n- Put parameters under <parameters> using the exact schema names.";

// ---------------------------------------------------------------------------
// regexes (mirror the Python module patterns)
// ---------------------------------------------------------------------------

static STRIP_TOOL_MARKUP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?is)<tool_calls\b[^>]*>.*?</tool_calls>|<tool_call\b[^>]*>.*?</tool_call>|<function_call\b[^>]*>.*?</function_call>|<invoke\b[^>]*>.*?</invoke>",
    )
    .unwrap()
});

static STREAMABLE_TEXT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<tool_calls\b|<tool_call\b|<function_call\b|<invoke\b").unwrap());

static TOOL_CALL_BLOCKS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?is)<tool_call\b[^>]*>(.*?)</tool_call>|<function_call\b[^>]*>(.*?)</function_call>|<invoke\b[^>]*>(.*?)</invoke>",
    )
    .unwrap()
});

static CODE_FENCE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)```.*?```").unwrap());

static XML_OPEN_TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<([\w.-]+)\b[^>]*>").unwrap());

static CDATA_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)^<!\[CDATA\[(.*?)\]\]>$").unwrap());

// ---------------------------------------------------------------------------
// small JSON / string helpers
// ---------------------------------------------------------------------------

fn now_ts() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Python truthiness for a JSON value (used wherever the source relies on
/// `x or y` / `if x:` short-circuiting).
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

/// `isinstance(tools, list) and bool(tools)`.
fn tools_active(tools: &Value) -> bool {
    tools.as_array().map_or(false, |a| !a.is_empty())
}

/// ASCII case-insensitive substring search returning a byte offset. `needle`
/// must be ASCII; offsets land on char boundaries because ASCII bytes never
/// appear inside a multi-byte UTF-8 sequence.
fn find_ci_ascii(hay: &str, needle: &str) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// Minimal HTML entity decoder (Python uses `html.unescape`). Handles the
/// common named entities plus numeric (`&#NN;` / `&#xHH;`) references; unknown
/// entities are left untouched. NOTE: this is a deliberate subset of the full
/// HTML5 named-entity table.
fn html_unescape(input: &str) -> String {
    if !input.contains('&') {
        return input.to_string();
    }
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '&' {
            let limit = (i + 33).min(chars.len());
            if let Some(semi) = (i + 1..limit).find(|&j| chars[j] == ';') {
                let entity: String = chars[i + 1..semi].iter().collect();
                if let Some(decoded) = decode_entity(&entity) {
                    out.push_str(&decoded);
                    i = semi + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn decode_entity(entity: &str) -> Option<String> {
    if let Some(num) = entity.strip_prefix('#') {
        let code = if let Some(hex) = num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
            u32::from_str_radix(hex, 16).ok()?
        } else {
            num.parse::<u32>().ok()?
        };
        return char::from_u32(code).map(|c| c.to_string());
    }
    let mapped = match entity {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => "\u{a0}",
        _ => return None,
    };
    Some(mapped.to_string())
}

// ---------------------------------------------------------------------------
// tool prompt construction
// ---------------------------------------------------------------------------

/// `tool.get(k) or fn.get(k) or ""` then `str(...).strip()` — returns the first
/// truthy string-ish value.
fn truthy_string(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(other) if !matches!(other, Value::String(_)) && !json_falsy(other) => {
            Some(other.to_string())
        }
        _ => None,
    }
}

fn first_truthy(candidates: &[Option<&Value>]) -> Value {
    for c in candidates {
        if let Some(v) = c {
            if !json_falsy(v) {
                return (*v).clone();
            }
        }
    }
    json!({})
}

fn tool_meta(tool: &Value) -> (String, String, Value) {
    let func = tool.get("function").and_then(|v| v.as_object());
    let func_get = |key: &str| func.and_then(|f| f.get(key));

    let name = truthy_string(tool.get("name"))
        .or_else(|| truthy_string(func_get("name")))
        .unwrap_or_default()
        .trim()
        .to_string();
    let desc = truthy_string(tool.get("description"))
        .or_else(|| truthy_string(func_get("description")))
        .unwrap_or_default()
        .trim()
        .to_string();
    let schema = first_truthy(&[
        tool.get("input_schema"),
        tool.get("parameters"),
        func_get("input_schema"),
        func_get("parameters"),
    ]);
    (name, desc, schema)
}

fn build_tool_prompt(tools: &Value) -> String {
    let Some(arr) = tools.as_array() else {
        return String::new();
    };
    let mut blocks: Vec<String> = Vec::new();
    for tool in arr {
        if !tool.is_object() {
            continue;
        }
        let (name, desc, schema) = tool_meta(tool);
        if !name.is_empty() {
            let schema_str = serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_string());
            blocks.push(format!(
                "Tool: {name}\nDescription: {desc}\nParameters: {schema_str}"
            ));
        }
    }
    if blocks.is_empty() {
        return String::new();
    }
    // No separator between the joined blocks and the suffix — see the
    // `TOOL_RULES_SUFFIX` doc comment (Python `.strip()` quirk).
    format!("Available tools:\n{}{}", blocks.join("\n"), TOOL_RULES_SUFFIX)
}

// ---------------------------------------------------------------------------
// system handling
// ---------------------------------------------------------------------------

fn has_claude_code_system(system: &Value) -> bool {
    match system {
        Value::String(s) => s.contains("You are Claude Code"),
        Value::Array(arr) => arr.iter().any(|item| {
            item.get("text")
                .and_then(|v| v.as_str())
                .map_or(false, |t| t.contains("You are Claude Code"))
        }),
        _ => false,
    }
}

/// `compact_system` — effectively identity (the Python `_compact_system_text`
/// only coalesced `None` text to `""`).
fn compact_system(system: &Value) -> Value {
    match system {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                if let Some(obj) = item.as_object() {
                    if obj.get("type").and_then(|v| v.as_str()).unwrap_or("") == "text" {
                        let mut copied = obj.clone();
                        let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        copied.insert("text".into(), json!(text));
                        out.push(Value::Object(copied));
                        continue;
                    }
                }
                out.push(item.clone());
            }
            Value::Array(out)
        }
        other => other.clone(),
    }
}

fn merge_system(system: &Value, extra: &str) -> Value {
    let system = compact_system(system);
    let extra = if has_claude_code_system(&system) {
        XML_TOOL_RULE.to_string()
    } else {
        extra.to_string()
    };
    if extra.is_empty() {
        return system;
    }
    match &system {
        Value::String(s) if !s.trim().is_empty() => json!(format!("{}\n\n{}", s.trim(), extra)),
        Value::Array(arr) => {
            let mut new_arr = arr.clone();
            new_arr.push(json!({"type": "text", "text": extra}));
            Value::Array(new_arr)
        }
        _ => json!(extra),
    }
}

// ---------------------------------------------------------------------------
// message preprocessing (tool_use / tool_result history -> text)
// ---------------------------------------------------------------------------

/// `str(content or '')` — Python stringification used for tool_result content.
/// NOTE: for list/dict content Python produces its native `repr`; here we fall
/// back to a JSON serialization (deviation called out in the port report).
fn content_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other if json_falsy(other) => String::new(),
        other => other.to_string(),
    }
}

fn preprocess_block(block: &Value) -> Value {
    let Some(obj) = block.as_object() else {
        return block.clone();
    };
    match obj.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "text" => {
            // `_compact_message_text` is identity; keep the block as-is.
            let mut item = obj.clone();
            let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
            item.insert("text".into(), json!(text));
            Value::Object(item)
        }
        "tool_use" => {
            let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let input = obj.get("input").cloned().unwrap_or_else(|| json!({}));
            let input = if json_falsy(&input) { json!({}) } else { input };
            let input_str = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
            json!({
                "type": "text",
                "text": format!(
                    "<tool_calls><tool_call><tool_name>{name}</tool_name><parameters>{input_str}</parameters></tool_call></tool_calls>"
                ),
            })
        }
        "tool_result" => {
            let id = obj.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
            let content = content_to_string(obj.get("content").unwrap_or(&Value::Null));
            json!({"type": "text", "text": format!("Tool result {id}: {content}")})
        }
        _ => block.clone(),
    }
}

fn preprocess_messages(messages: &Value) -> Vec<Value> {
    let Some(arr) = messages.as_array() else {
        return Vec::new();
    };
    let mut result = Vec::with_capacity(arr.len());
    for message in arr {
        let Some(obj) = message.as_object() else {
            continue;
        };
        let mut item = obj.clone();
        match obj.get("content") {
            Some(Value::String(_)) => {} // identity text mapper — leave unchanged
            Some(Value::Array(blocks)) => {
                let new_blocks: Vec<Value> = blocks.iter().map(preprocess_block).collect();
                item.insert("content".into(), Value::Array(new_blocks));
            }
            _ => {}
        }
        result.push(Value::Object(item));
    }
    result
}

// ---------------------------------------------------------------------------
// request assembly
// ---------------------------------------------------------------------------

struct MessageRequest {
    messages: Vec<Value>,
    model: String,
    tools: Value,
}

/// Port of `message_request` (without the backend/token construction — the Rust
/// conversation layer acquires the account internally). Produces the normalized
/// messages, model and the (original) tools list.
fn message_request(config: &Config, body: &Value) -> MessageRequest {
    let messages_pre = preprocess_messages(body.get("messages").unwrap_or(&Value::Null));
    let tools = body.get("tools").cloned().unwrap_or(Value::Null);
    let system_in = body.get("system").cloned().unwrap_or(Value::Null);
    let system = merge_system(&system_in, &build_tool_prompt(&tools));
    let normalized = normalize_messages(config, &messages_pre, Some(&system));
    let model = {
        let m = body.get("model").and_then(|v| v.as_str()).unwrap_or("").trim();
        if m.is_empty() {
            "auto".to_string()
        } else {
            m.to_string()
        }
    };
    MessageRequest {
        messages: normalized,
        model,
        tools,
    }
}

fn conversation_request(req: &MessageRequest, base_url: Option<String>) -> ConversationRequest {
    ConversationRequest {
        model: req.model.clone(),
        messages: Some(req.messages.clone()),
        base_url,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// tool-call markup parsing
// ---------------------------------------------------------------------------

fn strip_tool_markup(text: &str) -> String {
    STRIP_TOOL_MARKUP_RE.replace_all(text, "").trim().to_string()
}

fn streamable_text(text: &str) -> String {
    match STREAMABLE_TEXT_RE.find(text) {
        Some(m) => text[..m.start()].trim_end().to_string(),
        None => text.to_string(),
    }
}

/// Extract the inner value of `<tag ...>...</tag>` (first match), unwrapping a
/// single CDATA section and HTML-unescaping the result. Returns `""` if absent.
fn xml_value(text: &str, tag: &str) -> String {
    let re = Regex::new(&format!(
        r"(?is)<{tag}\b[^>]*>(.*?)</{tag}>",
        tag = regex::escape(tag)
    ))
    .unwrap();
    let Some(caps) = re.captures(text) else {
        return String::new();
    };
    let value = caps
        .get(1)
        .map(|m| m.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let inner = match CDATA_RE.captures(&value) {
        Some(c) => c.get(1).map(|m| m.as_str()).unwrap_or("").to_string(),
        None => value,
    };
    html_unescape(&inner).trim().to_string()
}

/// Manual replacement for the Python `<([\w.-]+)\b[^>]*>(.*?)</\1>` finditer:
/// the `regex` crate has no backreferences, so we scan for each opening tag and
/// pair it with the first matching (case-insensitive) closing tag.
fn parse_xml_pairs(raw: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < raw.len() {
        let rest = &raw[pos..];
        let Some(caps) = XML_OPEN_TAG_RE.captures(rest) else {
            break;
        };
        let m0 = caps.get(0).unwrap();
        let tag = caps.get(1).unwrap().as_str().to_string();
        let open_end = pos + m0.end();
        let close = format!("</{tag}>");
        match find_ci_ascii(&raw[open_end..], &close) {
            Some(rel) => {
                let content_end = open_end + rel;
                let content = raw[open_end..content_end].to_string();
                out.push((tag, content));
                pos = content_end + close.len();
            }
            None => {
                pos = open_end;
            }
        }
    }
    out
}

fn first_nonempty(values: &[String]) -> String {
    values.iter().find(|v| !v.is_empty()).cloned().unwrap_or_default()
}

fn parse_tool_value(raw: &str) -> Value {
    let wrapped = format!("<x>{raw}</x>");
    let value = xml_value(&wrapped, "x");
    serde_json::from_str::<Value>(&value).unwrap_or(Value::String(value))
}

fn parse_tool_params(raw: &str) -> Value {
    let raw = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
        return if parsed.is_object() { parsed } else { json!({}) };
    }
    let mut map = Map::new();
    for (key, content) in parse_xml_pairs(raw) {
        map.insert(key, parse_tool_value(&content));
    }
    Value::Object(map)
}

fn parse_tool_calls(text: &str) -> Vec<(String, Value)> {
    let cleaned = CODE_FENCE_RE.replace_all(text, "");
    let cleaned = cleaned.trim();
    let mut result = Vec::new();
    for caps in TOOL_CALL_BLOCKS_RE.captures_iter(cleaned) {
        // Exactly one alternation branch participates; take whichever matched.
        let block = caps
            .get(1)
            .or_else(|| caps.get(2))
            .or_else(|| caps.get(3))
            .map(|m| m.as_str())
            .unwrap_or("");
        let name = first_nonempty(&[
            xml_value(block, "tool_name"),
            xml_value(block, "name"),
            xml_value(block, "function"),
        ]);
        let params = {
            let p = first_nonempty(&[
                xml_value(block, "parameters"),
                xml_value(block, "input"),
                xml_value(block, "arguments"),
            ]);
            if p.is_empty() {
                "{}".to_string()
            } else {
                p
            }
        };
        if !name.is_empty() {
            result.push((name, parse_tool_params(&params)));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// content blocks / non-streaming response
// ---------------------------------------------------------------------------

fn content_blocks(text: &str, tools: &Value) -> (Vec<Value>, String) {
    let calls = if tools_active(tools) {
        parse_tool_calls(text)
    } else {
        Vec::new()
    };
    let text = strip_tool_markup(text);
    if !calls.is_empty() {
        let mut content: Vec<Value> = Vec::new();
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
        for (name, args) in calls {
            content.push(json!({
                "type": "tool_use",
                "id": format!("toolu_{}", new_uuid()),
                "name": name,
                "input": args,
            }));
        }
        (content, "tool_use".to_string())
    } else {
        (vec![json!({"type": "text", "text": text})], "end_turn".to_string())
    }
}

fn message_response(
    model: &str,
    text: &str,
    input_tokens: i64,
    output_tokens: i64,
    tools: &Value,
) -> Value {
    let (content, stop_reason) = content_blocks(text, tools);
    json!({
        "id": format!("msg_{}", new_uuid()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })
}

/// Public API: handle a non-streaming Anthropic `/v1/messages` request.
pub async fn messages_once(
    deps: ConvDeps,
    body: Value,
    base_url: Option<String>,
) -> Result<Value, AppError> {
    let req = message_request(&deps.config, &body);
    let input_tokens = count_message_tokens(&req.messages, &req.model);
    let conv = conversation_request(&req, base_url);
    let text = collect_text(deps, conv).await?;
    let output_tokens = count_text_tokens(&text, &req.model);
    Ok(message_response(&req.model, &text, input_tokens, output_tokens, &req.tools))
}

// ---------------------------------------------------------------------------
// streaming response (SSE event sequence)
// ---------------------------------------------------------------------------

/// Port of `_stream_buffered_blocks`: emit start/delta/stop for each buffered
/// (already-complete) content block. Text blocks stream their full text in one
/// delta; tool_use blocks stream their input as a single `input_json_delta`.
fn stream_buffered_blocks(content: &[Value], start_index: i64) -> Vec<Value> {
    let mut out = Vec::new();
    for (offset, block) in content.iter().enumerate() {
        let index = start_index + offset as i64;
        let (start, delta) = if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
            let id = block.get("id").cloned().unwrap_or(Value::Null);
            let name = block.get("name").cloned().unwrap_or(Value::Null);
            let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
            let input = if json_falsy(&input) { json!({}) } else { input };
            let partial = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
            (
                json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                json!({"type": "input_json_delta", "partial_json": partial}),
            )
        } else {
            let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
            (
                json!({"type": "text", "text": ""}),
                json!({"type": "text_delta", "text": text}),
            )
        };
        out.push(json!({"type": "content_block_start", "index": index, "content_block": start}));
        out.push(json!({"type": "content_block_delta", "index": index, "delta": delta}));
        out.push(json!({"type": "content_block_stop", "index": index}));
    }
    out
}

fn error_event(err: &AppError) -> Value {
    let message = {
        let m = error_message_from_detail(&err.detail);
        if m.is_empty() {
            "stream failed".to_string()
        } else {
            m
        }
    };
    json!({"type": "error", "error": {"type": "api_error", "message": message}})
}

/// Public API: handle a streaming Anthropic `/v1/messages` request. Each item
/// in the returned channel is a ready-to-send SSE frame (built via
/// [`anthropic_sse_event`]).
pub fn messages_stream(
    deps: ConvDeps,
    body: Value,
    base_url: Option<String>,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>(64);
    tokio::spawn(async move {
        macro_rules! send {
            ($ev:expr) => {
                if tx.send(anthropic_sse_event(&$ev)).await.is_err() {
                    return;
                }
            };
        }

        let req = message_request(&deps.config, &body);
        let input_tokens = count_message_tokens(&req.messages, &req.model);
        let model = req.model.clone();
        let tools = req.tools.clone();
        let conv = conversation_request(&req, base_url);
        let mut deltas = stream_text_deltas(deps, conv);

        let message_id = format!("msg_{}", new_uuid());
        let created = now_ts();
        let mut current_text = String::new();
        let mut streamed_text = String::new();
        let tool_mode = tools_active(&tools);
        let mut tool_started = false;
        let mut text_open = false;

        send!(json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }));

        if !tool_mode {
            text_open = true;
            send!(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}));
        }

        let mut errored = false;
        while let Some(item) = deltas.recv().await {
            let text_delta = match item {
                Ok(d) => d,
                Err(e) => {
                    send!(error_event(&e));
                    errored = true;
                    break;
                }
            };
            if text_delta.is_empty() {
                continue;
            }
            current_text.push_str(&text_delta);
            if !tool_started {
                let visible_text = if !tool_mode {
                    current_text.clone()
                } else {
                    streamable_text(&current_text)
                };
                if visible_text.starts_with(&streamed_text) {
                    let skip = streamed_text.chars().count();
                    let new_delta: String = visible_text.chars().skip(skip).collect();
                    if !new_delta.is_empty() {
                        if !text_open {
                            text_open = true;
                            send!(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}));
                        }
                        streamed_text = visible_text.clone();
                        send!(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": new_delta}}));
                    }
                }
                tool_started = tool_mode && visible_text != current_text;
            }
        }

        if errored {
            return;
        }

        // Finish (the implicit `finish_reason` once the delta channel closes).
        let (content, stop_reason) = content_blocks(&current_text, &tools);
        if text_open {
            send!(json!({"type": "content_block_stop", "index": 0}));
        }
        if stop_reason == "tool_use" {
            let mut start_index = if text_open { 1 } else { 0 };
            let mut content = content;
            if !content.is_empty()
                && content[0].get("type").and_then(|v| v.as_str()) == Some("text")
            {
                let block_text = content[0]
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let skip = streamed_text.chars().count();
                let remaining: String = block_text.chars().skip(skip).collect();
                if !remaining.is_empty() {
                    if !text_open {
                        send!(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}));
                    }
                    send!(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": remaining}}));
                    if !text_open {
                        send!(json!({"type": "content_block_stop", "index": 0}));
                    }
                }
                start_index = 1;
                content = content[1..].to_vec();
            }
            for ev in stream_buffered_blocks(&content, start_index) {
                send!(ev);
            }
        }

        let output_tokens = count_text_tokens(&current_text, &model);
        send!(json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
            "usage": {"output_tokens": output_tokens},
        }));
        send!(json!({"type": "message_stop", "created": created}));
    });
    rx
}
