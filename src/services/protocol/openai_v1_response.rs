//! Port of `services/protocol/openai_v1_response.py` — the `/v1/responses`
//! endpoint (text + image-generation tool, streaming + buffered).
//!
//! Mirrors the OpenAI Responses wire shapes: `resp_…` / `msg_…` ids,
//! `response.created` → `response.output_item.added` →
//! `response.output_text.delta` … → `response.output_text.done` →
//! `response.output_item.done` → `response.completed` event sequence, the
//! `image_generation_call` output items, and the Responses `usage` block.
//!
//! Public entry points:
//!   * [`response_once`] — buffered, returns the final `response` object.
//!   * [`response_stream`] — returns a receiver of ready-to-send SSE frames
//!     (the Responses stream uses the same generic `data: {event}` framing as
//!     chat completions, per `sse_json_stream`).
//!
//! Adaptation notes:
//!   * The Python `chat_completion_cache` layer is omitted (no cache handle in
//!     the entry signature); message normalization is still applied.
//!   * `extract_image_from_message_content` / `encode_images` are not present in
//!     the Rust `utils::helper`, so a local decoder pulls inline (data-URL /
//!     base64) images out of the response `input` (remote URLs are skipped).
#![allow(dead_code)]

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use chrono::Utc;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::config::Config;
use crate::error::{openai_error_payload, AppError};
use crate::services::protocol::chat_completion_cache::normalize_text_messages;
use crate::services::protocol::conversation::{
    count_message_tokens, count_text_tokens, normalize_messages, stream_image_outputs_with_pool, stream_text_deltas,
    ConvDeps, ConversationRequest, ImageGenerationError, ImageOutput,
};
use crate::utils::helper::{
    decode_json_image_string, extract_response_prompt, has_response_image_generation_tool, sse_data, sse_done, sse_open,
};
use crate::utils::image_tokens::{count_image_content_tokens, count_image_output_items_tokens, image_usage, token_usage};

const TOOL_UNAVAILABLE_SYSTEM_MESSAGE: &str = "This compatibility backend cannot execute local tools, shell commands, web searches, or file operations. Do not claim to have run tools or inspected external resources. If a user asks you to use a tool, say that tool execution is unavailable through this backend.";

const RESPONSE_CONTENT_PART_TYPES: [&str; 6] =
    ["text", "input_text", "output_text", "image_url", "input_image", "image"];

type EvTx = mpsc::Sender<Result<Value, AppError>>;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn hex_id() -> String {
    Uuid::new_v4().simple().to_string()
}

fn now_ts() -> i64 {
    Utc::now().timestamp()
}

fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::String(s) => s.is_empty(),
        Value::Number(n) => n.as_f64().map(|f| f == 0.0).unwrap_or(false),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn str_or_trim(body: &Value, key: &str, default: &str) -> String {
    let s = body.get(key).and_then(|v| v.as_str()).unwrap_or(default).trim();
    if s.is_empty() { default.to_string() } else { s.to_string() }
}

fn message_image_tokens(messages: &[Value], model: &str) -> i64 {
    messages
        .iter()
        .filter_map(|m| m.get("content"))
        .map(|c| count_image_content_tokens(c, model, "auto"))
        .sum()
}

// ---------------------------------------------------------------------------
// request classification / tool helpers
// ---------------------------------------------------------------------------

fn is_text_response_request(body: &Value) -> bool {
    !has_response_image_generation_tool(body)
}

fn has_non_image_tools(body: &Value) -> bool {
    let Some(tools) = body.get("tools").and_then(|v| v.as_array()) else {
        return false;
    };
    tools.iter().any(|tool| {
        tool.is_object() && tool.get("type").and_then(|v| v.as_str()).unwrap_or("").trim() != "image_generation"
    })
}

fn response_image_tool(body: &Value) -> Value {
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if tool.get("type").and_then(|v| v.as_str()) == Some("image_generation") {
                return tool.clone();
            }
        }
    }
    json!({})
}

// ---------------------------------------------------------------------------
// inline image extraction (mirrors extract_image_from_message_content)
// ---------------------------------------------------------------------------

fn decode_part_image(part: &Value) -> Option<Vec<u8>> {
    let obj = part.as_object()?;
    let ptype = obj.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
    let dec = |s: &str, mime: Option<&str>| decode_json_image_string(s, 1, None, mime).ok().map(|d| d.data);

    if ptype == "image_url" {
        let iu = obj.get("image_url");
        let (url, mime) = match iu {
            Some(Value::Object(m)) => (
                m.get("url").or_else(|| m.get("image_url")).and_then(|v| v.as_str()),
                m.get("mime_type").or_else(|| m.get("mimeType")).and_then(|v| v.as_str()),
            ),
            Some(Value::String(s)) => (Some(s.as_str()), None),
            _ => (obj.get("url").and_then(|v| v.as_str()), None),
        };
        return url.and_then(|u| dec(u, mime));
    }

    if ptype == "input_image" || ptype == "image" {
        if let Some(s) = obj.get("b64_json").or_else(|| obj.get("base64")).and_then(|v| v.as_str()) {
            let mime = obj
                .get("mime")
                .or_else(|| obj.get("mime_type"))
                .or_else(|| obj.get("mimeType"))
                .and_then(|v| v.as_str());
            if let Some(d) = dec(s, mime) {
                return Some(d);
            }
        }
        for key in ["image_url", "url"] {
            if let Some(v) = obj.get(key) {
                let url = match v {
                    Value::Object(m) => m.get("url").or_else(|| m.get("image_url")).and_then(|x| x.as_str()),
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                };
                if let Some(d) = url.and_then(|u| dec(u, None)) {
                    return Some(d);
                }
            }
        }
        if let Some(src) = obj.get("source").and_then(|v| v.as_object()) {
            if src.get("type").and_then(|v| v.as_str()) == Some("base64") {
                let data = src.get("data").and_then(|v| v.as_str()).unwrap_or("");
                let mime = src.get("media_type").or_else(|| src.get("mime_type")).and_then(|v| v.as_str());
                if let Some(d) = dec(data, mime) {
                    return Some(d);
                }
            }
        }
    }
    None
}

fn extract_image_from_message_content(content: &Value) -> Vec<Vec<u8>> {
    let Some(arr) = content.as_array() else {
        return Vec::new();
    };
    arr.iter().filter_map(decode_part_image).collect()
}

/// Mirror of `extract_response_image`: find the newest inline image in `input`.
fn extract_response_image(input_value: Option<&Value>) -> Option<Vec<u8>> {
    let input_value = input_value?;
    if input_value.is_object() {
        let ptype = input_value.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
        if ptype == "input_image" {
            return extract_image_from_message_content(&json!([input_value.clone()])).into_iter().next();
        }
        return extract_image_from_message_content(input_value.get("content").unwrap_or(&Value::Null))
            .into_iter()
            .next();
    }
    let arr = input_value.as_array()?;
    for item in arr.iter().rev() {
        if item.is_object() {
            let ptype = item.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
            if ptype == "input_image" {
                if let Some(d) = extract_image_from_message_content(&json!([item.clone()])).into_iter().next() {
                    return Some(d);
                }
            }
            if let Some(d) =
                extract_image_from_message_content(item.get("content").unwrap_or(&Value::Null)).into_iter().next()
            {
                return Some(d);
            }
        }
    }
    None
}

/// Mirror of `_input_image_parts`: collect content parts for image-token counting.
fn input_image_parts(input_value: Option<&Value>) -> Vec<Value> {
    let Some(input_value) = input_value else {
        return Vec::new();
    };
    if let Some(obj) = input_value.as_object() {
        let mut parts = Vec::new();
        if let Some(arr) = obj.get("content").and_then(|v| v.as_array()) {
            parts.extend(arr.iter().filter(|i| i.is_object()).cloned());
        }
        return parts;
    }
    let Some(items) = input_value.as_array() else {
        return Vec::new();
    };
    // `all(isinstance(item, dict) and item.get("type") for item in input)`
    let all_typed = items.iter().all(|it| {
        it.as_object().map_or(false, |o| o.get("type").map_or(false, |t| !is_falsy(t)))
    });
    if all_typed {
        return items.iter().filter(|i| i.is_object()).cloned().collect();
    }
    let mut parts = Vec::new();
    for item in items {
        if let Some(arr) = item.get("content").and_then(|v| v.as_array()) {
            parts.extend(arr.iter().filter(|p| p.is_object()).cloned());
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// input -> messages conversion
// ---------------------------------------------------------------------------

fn is_response_content_part(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    let part_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
    RESPONSE_CONTENT_PART_TYPES.contains(&part_type) || (obj.contains_key("image_url") && part_type != "message")
}

fn message_content_from_response_item(item: &Value) -> Value {
    match item.get("content") {
        Some(Value::Array(parts)) => Value::Array(parts.clone()),
        Some(Value::String(s)) => Value::String(s.clone()),
        other => {
            let prompt = extract_response_prompt(&json!([item.clone()]));
            if !prompt.is_empty() {
                Value::String(prompt)
            } else {
                match other {
                    Some(c) if !is_falsy(c) => c.clone(),
                    _ => Value::String(String::new()),
                }
            }
        }
    }
}

fn append_response_message(messages: &mut Vec<Value>, role: Option<&Value>, content: &Value) {
    let role_str = role
        .and_then(|r| r.as_str().map(String::from))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "user".to_string());
    match content {
        Value::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                messages.push(json!({ "role": role_str, "content": t }));
            }
        }
        Value::Array(a) => {
            if !a.is_empty() {
                messages.push(json!({ "role": role_str, "content": a.clone() }));
            }
        }
        _ => {}
    }
}

fn messages_from_input(input_value: Option<&Value>, instructions: Option<&Value>) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    let system_text = instructions.map(value_to_string).unwrap_or_default();
    let system_text = system_text.trim();
    if !system_text.is_empty() {
        messages.push(json!({ "role": "system", "content": system_text }));
    }

    let Some(input_value) = input_value else {
        return messages;
    };

    match input_value {
        Value::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                messages.push(json!({ "role": "user", "content": t }));
            }
        }
        Value::Object(_) => {
            if is_response_content_part(input_value) {
                append_response_message(&mut messages, None, &json!([input_value.clone()]));
            } else {
                let role = input_value.get("role").cloned();
                let content = message_content_from_response_item(input_value);
                append_response_message(&mut messages, role.as_ref(), &content);
            }
        }
        Value::Array(items) => {
            if items.iter().all(is_response_content_part) {
                let parts: Vec<Value> = items.iter().filter(|i| i.is_object()).cloned().collect();
                append_response_message(&mut messages, None, &Value::Array(parts));
                return messages;
            }
            let mut pending: Vec<Value> = Vec::new();
            for item in items {
                if is_response_content_part(item) {
                    pending.push(item.clone());
                    continue;
                }
                if !pending.is_empty() {
                    append_response_message(&mut messages, None, &Value::Array(std::mem::take(&mut pending)));
                }
                if !item.is_object() {
                    continue;
                }
                let role = item.get("role").cloned();
                let content = message_content_from_response_item(item);
                append_response_message(&mut messages, role.as_ref(), &content);
            }
            if !pending.is_empty() {
                append_response_message(&mut messages, None, &Value::Array(pending));
            }
        }
        _ => {}
    }
    messages
}

// ---------------------------------------------------------------------------
// output item / event builders
// ---------------------------------------------------------------------------

fn text_output_item(text: &str, item_id: Option<&str>, status: &str) -> Value {
    let id = item_id.map(String::from).unwrap_or_else(|| format!("msg_{}", hex_id()));
    json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{ "type": "output_text", "text": text, "annotations": [] }],
    })
}

fn image_output_items(prompt: &str, data: &[Value], item_id: Option<&str>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for item in data {
        let b64 = item.get("b64_json").and_then(|v| v.as_str()).unwrap_or("").trim();
        if b64.is_empty() {
            continue;
        }
        let raw = item
            .get("revised_prompt")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(prompt);
        let stripped = raw.trim();
        let revised = if stripped.is_empty() { prompt.to_string() } else { stripped.to_string() };
        let id = item_id.map(String::from).unwrap_or_else(|| format!("ig_{}", out.len() + 1));
        out.push(json!({
            "id": id,
            "type": "image_generation_call",
            "status": "completed",
            "result": b64,
            "revised_prompt": revised,
        }));
    }
    out
}

fn response_created(response_id: &str, model: &str, created: i64) -> Value {
    json!({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "error": Value::Null,
            "incomplete_details": Value::Null,
            "model": model,
            "output": [],
            "parallel_tool_calls": false,
        },
    })
}

fn response_completed(response_id: &str, model: &str, created: i64, output: Vec<Value>, usage: Option<Value>) -> Value {
    let mut response = json!({
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "model": model,
        "output": output,
        "parallel_tool_calls": false,
    });
    if let Some(usage) = usage {
        response["usage"] = usage;
    }
    json!({ "type": "response.completed", "response": response })
}

// ---------------------------------------------------------------------------
// text response flow
// ---------------------------------------------------------------------------

fn text_response_parts(config: &Config, body: &Value) -> (String, Vec<Value>) {
    let model = str_or_trim(body, "model", "auto");
    let base = messages_from_input(body.get("input"), body.get("instructions"));
    let normalized = normalize_messages(config, &base, None);
    let mut messages = normalize_text_messages(config, &normalized);
    if has_non_image_tools(body) {
        messages.insert(0, json!({ "role": "system", "content": TOOL_UNAVAILABLE_SYSTEM_MESSAGE }));
    }
    (model, messages)
}

async fn stream_text_response(deps: ConvDeps, model: String, messages: Vec<Value>, tx: EvTx) {
    let response_id = format!("resp_{}", hex_id());
    let item_id = format!("msg_{}", hex_id());
    let created = now_ts();

    if tx.send(Ok(response_created(&response_id, &model, created))).await.is_err() {
        return;
    }
    let added = json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": text_output_item("", Some(&item_id), "in_progress"),
    });
    if tx.send(Ok(added)).await.is_err() {
        return;
    }

    let request = ConversationRequest {
        model: model.clone(),
        messages: Some(messages.clone()),
        ..Default::default()
    };
    let mut drx = stream_text_deltas(deps, request);
    let mut full_text = String::new();
    while let Some(item) = drx.recv().await {
        match item {
            Ok(delta) => {
                full_text.push_str(&delta);
                let ev = json!({
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                });
                if tx.send(Ok(ev)).await.is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }

    let done = json!({
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_text,
    });
    if tx.send(Ok(done)).await.is_err() {
        return;
    }
    let item = text_output_item(&full_text, Some(&item_id), "completed");
    if tx
        .send(Ok(json!({ "type": "response.output_item.done", "output_index": 0, "item": item.clone() })))
        .await
        .is_err()
    {
        return;
    }
    let image = message_image_tokens(&messages, &model);
    let total = count_message_tokens(&messages, &model);
    let usage = token_usage(total - image, image, count_text_tokens(&full_text, &model), 0);
    let _ = tx
        .send(Ok(response_completed(&response_id, &model, created, vec![item], Some(usage))))
        .await;
}

// ---------------------------------------------------------------------------
// image response flow
// ---------------------------------------------------------------------------

async fn stream_image_response(
    tx: EvTx,
    mut rx: mpsc::Receiver<Result<ImageOutput, ImageGenerationError>>,
    prompt: String,
    model: String,
    input_image_tokens: i64,
    size_val: Value,
    quality: String,
) {
    let response_id = format!("resp_{}", hex_id());
    let created = now_ts();
    if tx.send(Ok(response_created(&response_id, &model, created))).await.is_err() {
        return;
    }

    while let Some(item) = rx.recv().await {
        let output = match item {
            Ok(o) => o,
            Err(e) => {
                let _ = tx.send(Err(image_err_to_app(e))).await;
                return;
            }
        };

        if output.kind == "message" {
            let text = output.text.clone();
            let item = text_output_item(&text, None, "completed");
            let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let usage = token_usage(
                count_text_tokens(&prompt, &model),
                input_image_tokens,
                count_text_tokens(&text, &model),
                0,
            );
            let delta = json!({
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            });
            let done = json!({
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            });
            let item_done = json!({ "type": "response.output_item.done", "output_index": 0, "item": item.clone() });
            for ev in [
                delta,
                done,
                item_done,
                response_completed(&response_id, &model, created, vec![item], Some(usage)),
            ] {
                if tx.send(Ok(ev)).await.is_err() {
                    return;
                }
            }
            return;
        }

        if output.kind != "result" {
            continue;
        }

        let items = image_output_items(&prompt, &output.data, None);
        if !items.is_empty() {
            let usage = image_usage(
                count_text_tokens(&prompt, &model),
                input_image_tokens,
                count_image_output_items_tokens(&Value::Array(output.data.clone()), &size_val, &quality),
            );
            for (idx, item) in items.iter().enumerate() {
                let ev = json!({ "type": "response.output_item.done", "output_index": idx, "item": item });
                if tx.send(Ok(ev)).await.is_err() {
                    return;
                }
            }
            let _ = tx
                .send(Ok(response_completed(&response_id, &model, created, items, Some(usage))))
                .await;
            return;
        }
    }

    let _ = tx.send(Err(AppError::upstream("image generation failed"))).await;
}

fn image_err_to_app(e: ImageGenerationError) -> AppError {
    let status = axum::http::StatusCode::from_u16(e.status_code).unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
    AppError::new(status, e.to_openai_error())
}

// ---------------------------------------------------------------------------
// event orchestration
// ---------------------------------------------------------------------------

async fn run_response_events(deps: ConvDeps, body: Value, base_url: Option<String>, tx: EvTx) {
    if is_text_response_request(&body) {
        let (model, messages) = text_response_parts(&deps.config, &body);
        stream_text_response(deps, model, messages, tx).await;
        return;
    }

    let prompt = extract_response_prompt(body.get("input").unwrap_or(&Value::Null));
    if prompt.is_empty() {
        let _ = tx.send(Err(AppError::bad_request("input text is required"))).await;
        return;
    }
    let model = str_or_trim(&body, "model", "gpt-image-2");
    let images = extract_response_image(body.get("input")).map(|bytes| vec![B64.encode(&bytes)]);
    let input_image_tokens =
        count_image_content_tokens(&Value::Array(input_image_parts(body.get("input"))), &model, "auto");
    let tool = response_image_tool(&body);
    let size = tool.get("size").and_then(|v| v.as_str()).map(String::from);
    let size_val = tool.get("size").cloned().unwrap_or(Value::Null);
    let quality = {
        let q = tool.get("quality").and_then(|v| v.as_str()).unwrap_or("");
        if q.is_empty() { "auto".to_string() } else { q.to_string() }
    };
    let request = ConversationRequest {
        prompt: prompt.clone(),
        model: model.clone(),
        size,
        quality: quality.clone(),
        response_format: "b64_json".to_string(),
        images,
        base_url,
        ..Default::default()
    };
    let rx = stream_image_outputs_with_pool(deps, request);
    stream_image_response(tx, rx, prompt, model, input_image_tokens, size_val, quality).await;
}

fn response_event_channel(deps: ConvDeps, body: Value, base_url: Option<String>) -> mpsc::Receiver<Result<Value, AppError>> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        run_response_events(deps, body, base_url, tx).await;
    });
    rx
}

// ---------------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------------

/// Buffered responses call: drain events and return the final `response` object.
pub async fn response_once(deps: ConvDeps, body: Value, base_url: Option<String>) -> Result<Value, AppError> {
    let mut rx = response_event_channel(deps, body, base_url);
    let mut completed: Option<Value> = None;
    while let Some(item) = rx.recv().await {
        let event = item?;
        if event.get("type").and_then(|v| v.as_str()) == Some("response.completed") {
            if let Some(resp) = event.get("response").filter(|r| r.as_object().map_or(false, |o| !o.is_empty())) {
                completed = Some(resp.clone());
            }
        }
    }
    completed.ok_or_else(|| AppError::upstream("response generation failed"))
}

/// Streaming responses call: emit each event as a ready-to-send SSE frame.
pub fn response_stream(deps: ConvDeps, body: Value, base_url: Option<String>) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let _ = tx.send(sse_open()).await;
        let mut erx = response_event_channel(deps, body, base_url);
        while let Some(item) = erx.recv().await {
            match item {
                Ok(event) => {
                    if tx.send(sse_data(&event)).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(sse_data(&openai_error_payload(&e.detail, e.status.as_u16()))).await;
                    break;
                }
            }
        }
        let _ = tx.send(sse_done()).await;
    });
    rx
}
