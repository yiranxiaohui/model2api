//! Port of `services/protocol/openai_v1_image_generations.py` — the
//! `/v1/images/generations` endpoint (text-to-image, streaming + buffered).
//!
//! Buffered calls return the `{created, data, usage}` result with the OpenAI
//! image `usage` block; streaming calls relay `image.generation.*` chunks as
//! SSE frames. `message_as_error` is set so a refusal/empty result surfaces as
//! an error (matching the Python handler).
//!
//! Adaptation: the Python `progress_callback` hook is dropped (progress is
//! delivered as streamed chunks). Internal `_account_email` / `_conversation_id`
//! fields are stripped from the wire output (the Python route's log layer did
//! this).
#![allow(dead_code)]

use axum::http::StatusCode;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::error::AppError;
use crate::services::protocol::conversation::{
    collect_image_outputs, count_text_tokens, stream_image_chunks, stream_image_outputs_with_pool, ConvDeps,
    ConversationRequest, ImageGenerationError,
};
use crate::utils::helper::{sse_data, sse_done, sse_open};
use crate::utils::image_tokens::{count_image_output_items_tokens, image_usage};

fn image_err_to_app(e: ImageGenerationError) -> AppError {
    let status = StatusCode::from_u16(e.status_code).unwrap_or(StatusCode::BAD_GATEWAY);
    AppError::new(status, e.to_openai_error())
}

/// `int(body.get("n") or 1)` — falsy (missing/0/empty) coerces to 1.
fn parse_n(body: &Value) -> i64 {
    match body.get("n") {
        Some(Value::Number(num)) => {
            let i = num.as_i64().or_else(|| num.as_f64().map(|f| f as i64)).unwrap_or(1);
            if i == 0 { 1 } else { i }
        }
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() { 1 } else { t.parse::<i64>().unwrap_or(1) }
        }
        _ => 1,
    }
}

/// `str(body.get(key) or default)`.
fn str_or(body: &Value, key: &str, default: &str) -> String {
    match body.get(key).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => default.to_string(),
    }
}

fn resolve_base_url(body: &Value, arg: Option<String>) -> Option<String> {
    match body.get("base_url").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => Some(s.to_string()),
        _ => arg,
    }
}

fn build_request(body: &Value, base_url: Option<String>) -> ConversationRequest {
    ConversationRequest {
        prompt: body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        model: str_or(body, "model", "gpt-image-2"),
        n: parse_n(body),
        size: body.get("size").and_then(|v| v.as_str()).map(String::from),
        quality: str_or(body, "quality", "auto"),
        response_format: str_or(body, "response_format", "b64_json"),
        base_url: resolve_base_url(body, base_url),
        message_as_error: true,
        ..Default::default()
    }
}

fn strip_internal(mut chunk: Value) -> Value {
    if let Some(map) = chunk.as_object_mut() {
        map.remove("_account_email");
        map.remove("_conversation_id");
    }
    chunk
}

/// Buffered generation: collect outputs and attach the image `usage` block.
pub async fn image_generations_once(deps: ConvDeps, body: Value, base_url: Option<String>) -> Result<Value, AppError> {
    let request = build_request(&body, base_url);
    let prompt = request.prompt.clone();
    let model = request.model.clone();
    let quality = request.quality.clone();
    let size_val = body.get("size").cloned().unwrap_or(Value::Null);

    let rx = stream_image_outputs_with_pool(deps, request);
    let mut result = collect_image_outputs(rx).await.map_err(image_err_to_app)?;
    let usage = image_usage(
        count_text_tokens(&prompt, &model),
        0,
        count_image_output_items_tokens(result.get("data").unwrap_or(&Value::Null), &size_val, &quality),
    );
    if let Some(map) = result.as_object_mut() {
        map.insert("usage".to_string(), usage);
        map.remove("_account_email");
    }
    Ok(result)
}

/// Streaming generation: relay `image.generation.*` chunks as SSE frames.
pub fn image_generations_stream(deps: ConvDeps, body: Value, base_url: Option<String>) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let _ = tx.send(sse_open()).await;
        let request = build_request(&body, base_url);
        let pool_rx = stream_image_outputs_with_pool(deps, request);
        let mut chunk_rx = stream_image_chunks(pool_rx);
        while let Some(item) = chunk_rx.recv().await {
            match item {
                Ok(chunk) => {
                    if tx.send(sse_data(&strip_internal(chunk))).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(sse_data(&e.to_openai_error())).await;
                    break;
                }
            }
        }
        let _ = tx.send(sse_done()).await;
    });
    rx
}
