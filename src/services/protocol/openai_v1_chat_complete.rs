//! Port of `services/protocol/openai_v1_chat_complete.py` — the
//! `/v1/chat/completions` endpoint (text + image-chat, streaming + buffered).
//!
//! Mirrors the OpenAI chat-completion wire shapes exactly: `chatcmpl-…` ids,
//! `chat.completion` / `chat.completion.chunk` objects, role-first streaming
//! deltas, a trailing `finish_reason:"stop"` chunk, and the detailed `usage`
//! block (text/image prompt-token split + completion-token details).
//!
//! Public entry points:
//!   * [`chat_complete_once`] — buffered, returns the response JSON.
//!   * [`chat_complete_stream`] — returns a receiver of ready-to-send SSE frames
//!     (`: stream-open`, `data: {chunk}` …, `data: [DONE]`).
//!
//! Adaptation notes:
//!   * The Python `chat_completion_cache` TTL/in-flight layer is omitted: the
//!     fixed `(deps, body, base_url)` entry signature carries no cache handle
//!     (the app/route layer owns caching). Message normalization
//!     (`normalize_text_messages`) is still applied.
//!   * `extract_chat_image` + `encode_images` are not present in the Rust
//!     `utils::helper`, so a local image extractor decodes the data-URL / base64
//!     image parts of the latest user message (remote `http(s)` URLs are
//!     skipped — they need the HTTP fetcher, which lives with the engine).
#![allow(dead_code)]

use axum::http::StatusCode;
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
    collect_image_outputs, collect_text, count_message_tokens, count_text_tokens, normalize_messages,
    stream_image_outputs_with_pool, stream_text_deltas, ConvDeps, ConversationRequest, ImageGenerationError,
};
use crate::utils::helper::{
    build_chat_image_markdown_content, decode_json_image_string, extract_chat_prompt, is_image_chat_request,
    parse_image_count, sse_data, sse_done, sse_open,
};
use crate::utils::image_tokens::{
    chat_usage_from_image_usage, count_image_content_tokens, count_image_input_tokens,
    count_image_output_items_tokens, image_size_from_bytes, image_usage,
};

const TOOL_UNAVAILABLE_SYSTEM_MESSAGE: &str = "This compatibility backend cannot execute local tools, shell commands, web searches, or file operations. Do not claim to have run tools or inspected external resources. If a user asks you to use a tool, say that tool execution is unavailable through this backend.";

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn hex_id() -> String {
    Uuid::new_v4().simple().to_string()
}

fn now_ts() -> i64 {
    Utc::now().timestamp()
}

fn image_err_to_app(e: ImageGenerationError) -> AppError {
    let status = StatusCode::from_u16(e.status_code).unwrap_or(StatusCode::BAD_GATEWAY);
    AppError::new(status, e.to_openai_error())
}

fn app_err_frame(e: &AppError) -> String {
    sse_data(&openai_error_payload(&e.detail, e.status.as_u16()))
}

/// Image-token count for the prompt's content parts (mirrors the private
/// `count_message_image_tokens`, which uses detail="auto").
fn message_image_tokens(messages: &[Value], model: &str) -> i64 {
    messages
        .iter()
        .filter_map(|m| m.get("content"))
        .map(|c| count_image_content_tokens(c, model, "auto"))
        .sum()
}

/// Local count of input-image tokens from decoded image bytes (mirrors
/// `count_image_inputs_tokens`).
fn count_input_image_tokens(images: &[Vec<u8>], model: &str) -> i64 {
    let mut total = 0;
    for data in images {
        if let Some((w, h)) = image_size_from_bytes(data) {
            total += count_image_input_tokens(w, h, model, "auto");
        }
    }
    total
}

// ---------------------------------------------------------------------------
// response builders
// ---------------------------------------------------------------------------

fn completion_chunk(model: &str, delta: Value, finish_reason: Option<&str>, completion_id: &str, created: i64) -> Value {
    json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    })
}

fn completion_response(model: &str, content: &str, created: Option<i64>, messages: Option<&[Value]>) -> Value {
    let (prompt_text_tokens, prompt_image_tokens, prompt_tokens, completion_tokens) = match messages {
        Some(msgs) => {
            let image = message_image_tokens(msgs, model);
            let total = count_message_tokens(msgs, model);
            let text = total - image;
            (text, image, total, count_text_tokens(content, model))
        }
        None => (0, 0, 0, 0),
    };
    json!({
        "id": format!("chatcmpl-{}", hex_id()),
        "object": "chat.completion",
        "created": created.unwrap_or_else(now_ts),
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "text_tokens": prompt_text_tokens,
                "image_tokens": prompt_image_tokens,
                "cached_tokens": 0,
            },
            "completion_tokens_details": {
                "text_tokens": completion_tokens,
                "image_tokens": 0,
                "reasoning_tokens": 0,
            },
        },
    })
}

// ---------------------------------------------------------------------------
// request parsing
// ---------------------------------------------------------------------------

fn chat_messages_from_body(body: &Value) -> Result<Vec<Value>, AppError> {
    if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
        if !arr.is_empty() {
            return Ok(arr.iter().filter(|m| m.is_object()).cloned().collect());
        }
    }
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim();
    if !prompt.is_empty() {
        return Ok(vec![json!({ "role": "user", "content": prompt })]);
    }
    Err(AppError::bad_request("messages or prompt is required"))
}

fn text_chat_parts(config: &Config, body: &Value) -> Result<(String, Vec<Value>), AppError> {
    let model = {
        let m = body.get("model").and_then(|v| v.as_str()).unwrap_or("auto").trim();
        if m.is_empty() { "auto".to_string() } else { m.to_string() }
    };
    let base = chat_messages_from_body(body)?;
    let normalized = normalize_messages(config, &base, None);
    let mut messages = normalize_text_messages(config, &normalized);
    if body.get("tools").and_then(|v| v.as_array()).map_or(false, |a| !a.is_empty()) {
        messages.insert(0, json!({ "role": "system", "content": TOOL_UNAVAILABLE_SYSTEM_MESSAGE }));
    }
    Ok((model, messages))
}

fn chat_image_args(body: &Value) -> Result<(String, String, i64, Vec<Vec<u8>>), AppError> {
    let model = {
        let m = body.get("model").and_then(|v| v.as_str()).unwrap_or("gpt-image-2").trim();
        if m.is_empty() { "gpt-image-2".to_string() } else { m.to_string() }
    };
    let prompt = extract_chat_prompt(body);
    if prompt.is_empty() {
        return Err(AppError::bad_request("prompt is required"));
    }
    let images = extract_chat_images(body);
    let n = parse_image_count(body.get("n").unwrap_or(&Value::Null))?;
    Ok((model, prompt, n, images))
}

/// Decode a single content part into raw image bytes (mirrors
/// `_decode_message_image_url` / `_decode_message_image_object`, minus remote
/// fetching). Returns `None` if the part carries no decodable inline image.
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

fn extract_images_from_content(content: Option<&Value>) -> Vec<Vec<u8>> {
    let Some(arr) = content.and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter().filter_map(decode_part_image).collect()
}

/// Mirror of `extract_chat_image` + `encode_images`: scan messages newest-first
/// for a user message carrying images, returning their raw bytes.
fn extract_chat_images(body: &Value) -> Vec<Vec<u8>> {
    let Some(messages) = body.get("messages").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    for message in messages.iter().rev() {
        let Some(obj) = message.as_object() else { continue };
        if obj.get("role").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase() != "user" {
            continue;
        }
        let imgs = extract_images_from_content(obj.get("content"));
        if !imgs.is_empty() {
            return imgs;
        }
    }
    Vec::new()
}

fn image_request(model: &str, prompt: &str, n: i64, images: &[Vec<u8>], base_url: Option<String>) -> ConversationRequest {
    let encoded: Vec<String> = images.iter().map(|b| B64.encode(b)).collect();
    ConversationRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        n,
        response_format: "b64_json".to_string(),
        images: if encoded.is_empty() { None } else { Some(encoded) },
        base_url,
        ..Default::default()
    }
}

fn image_result_content(result: &Value) -> String {
    if result.get("data").and_then(|v| v.as_array()).map_or(false, |a| !a.is_empty()) {
        build_chat_image_markdown_content(result)
    } else {
        let m = result.get("message").and_then(|v| v.as_str()).unwrap_or("");
        if m.is_empty() {
            "Image generation completed.".to_string()
        } else {
            m.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// buffered (non-stream) entry
// ---------------------------------------------------------------------------

pub async fn chat_complete_once(deps: ConvDeps, body: Value, base_url: Option<String>) -> Result<Value, AppError> {
    if is_image_chat_request(&body) {
        image_chat_response(deps, &body, base_url).await
    } else {
        text_chat_response(deps, &body).await
    }
}

async fn text_chat_response(deps: ConvDeps, body: &Value) -> Result<Value, AppError> {
    let (model, messages) = text_chat_parts(&deps.config, body)?;
    let request = ConversationRequest {
        model: model.clone(),
        messages: Some(messages.clone()),
        ..Default::default()
    };
    let content = collect_text(deps, request).await?;
    Ok(completion_response(&model, &content, None, Some(&messages)))
}

async fn image_chat_response(deps: ConvDeps, body: &Value, base_url: Option<String>) -> Result<Value, AppError> {
    let (model, prompt, n, images) = chat_image_args(body)?;
    let request = image_request(&model, &prompt, n, &images, base_url);
    let rx = stream_image_outputs_with_pool(deps, request);
    let result = collect_image_outputs(rx).await.map_err(image_err_to_app)?;
    let created = result.get("created").and_then(|v| v.as_i64()).filter(|c| *c != 0);
    let content = image_result_content(&result);
    let mut response = completion_response(&model, &content, created, None);
    let usage = image_usage(
        count_text_tokens(&prompt, &model),
        count_input_image_tokens(&images, &model),
        count_image_output_items_tokens(result.get("data").unwrap_or(&Value::Null), &Value::Null, "auto"),
    );
    response["usage"] = chat_usage_from_image_usage(&usage);
    Ok(response)
}

// ---------------------------------------------------------------------------
// streaming entry
// ---------------------------------------------------------------------------

pub fn chat_complete_stream(deps: ConvDeps, body: Value, base_url: Option<String>) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let _ = tx.send(sse_open()).await;
        let result = if is_image_chat_request(&body) {
            stream_image_chat(&deps, &body, base_url, &tx).await
        } else {
            stream_text_chat(&deps, &body, &tx).await
        };
        if let Err(frame) = result {
            let _ = tx.send(frame).await;
        }
        let _ = tx.send(sse_done()).await;
    });
    rx
}

/// Returns `Err(sse_frame)` carrying a ready-to-send error frame on failure.
async fn stream_text_chat(deps: &ConvDeps, body: &Value, tx: &mpsc::Sender<String>) -> Result<(), String> {
    let (model, messages) = text_chat_parts(&deps.config, body).map_err(|e| app_err_frame(&e))?;
    let completion_id = format!("chatcmpl-{}", hex_id());
    let created = now_ts();
    let request = ConversationRequest {
        model: model.clone(),
        messages: Some(messages),
        ..Default::default()
    };
    let mut drx = stream_text_deltas(deps.clone(), request);
    let mut sent_role = false;
    while let Some(item) = drx.recv().await {
        match item {
            Ok(delta) => {
                let chunk = if !sent_role {
                    sent_role = true;
                    completion_chunk(&model, json!({ "role": "assistant", "content": delta }), None, &completion_id, created)
                } else {
                    completion_chunk(&model, json!({ "content": delta }), None, &completion_id, created)
                };
                if tx.send(sse_data(&chunk)).await.is_err() {
                    return Ok(());
                }
            }
            Err(e) => return Err(app_err_frame(&e)),
        }
    }
    if !sent_role {
        let chunk = completion_chunk(&model, json!({ "role": "assistant", "content": "" }), None, &completion_id, created);
        let _ = tx.send(sse_data(&chunk)).await;
    }
    let chunk = completion_chunk(&model, json!({}), Some("stop"), &completion_id, created);
    let _ = tx.send(sse_data(&chunk)).await;
    Ok(())
}

async fn stream_image_chat(
    deps: &ConvDeps,
    body: &Value,
    base_url: Option<String>,
    tx: &mpsc::Sender<String>,
) -> Result<(), String> {
    let (model, prompt, n, images) = chat_image_args(body).map_err(|e| app_err_frame(&e))?;
    let request = image_request(&model, &prompt, n, &images, base_url);
    let mut rx = stream_image_outputs_with_pool(deps.clone(), request);
    let completion_id = format!("chatcmpl-{}", hex_id());
    let created = now_ts();
    let mut sent_role = false;
    let mut sent_text = String::new();
    while let Some(item) = rx.recv().await {
        let output = match item {
            Ok(o) => o,
            Err(e) => return Err(sse_data(&e.to_openai_error())),
        };
        let content = match output.kind.as_str() {
            "progress" => {
                sent_text.push_str(&output.text);
                output.text.clone()
            }
            "result" => build_chat_image_markdown_content(&json!({ "data": output.data })),
            "message" => {
                if output.text.starts_with(&sent_text) {
                    output.text[sent_text.len()..].to_string()
                } else {
                    output.text.clone()
                }
            }
            _ => String::new(),
        };
        if content.is_empty() {
            continue;
        }
        let chunk = if !sent_role {
            sent_role = true;
            completion_chunk(&model, json!({ "role": "assistant", "content": content }), None, &completion_id, created)
        } else {
            completion_chunk(&model, json!({ "content": content }), None, &completion_id, created)
        };
        if tx.send(sse_data(&chunk)).await.is_err() {
            return Ok(());
        }
    }
    if !sent_role {
        let chunk = completion_chunk(&model, json!({ "role": "assistant", "content": "" }), None, &completion_id, created);
        let _ = tx.send(sse_data(&chunk)).await;
    }
    let chunk = completion_chunk(&model, json!({}), Some("stop"), &completion_id, created);
    let _ = tx.send(sse_data(&chunk)).await;
    Ok(())
}
