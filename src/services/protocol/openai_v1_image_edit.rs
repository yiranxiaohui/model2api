//! Port of `services/protocol/openai_v1_image_edit.py` — the
//! `/v1/images/edits` endpoint (image-to-image, streaming + buffered).
//!
//! Like image generation, but the request must carry one or more source images
//! (decoded from the `image` / `images` JSON fields). Buffered calls add an
//! image `usage` block that also accounts for the input-image tokens.
//!
//! Adaptation notes:
//!   * The Python route pre-decoded the upload into `(bytes, name, mime)` tuples
//!     then `encode_images`'d them. Here the raw JSON body is decoded via
//!     `normalize_json_edit_images` (data-URL / base64 entries) and re-encoded to
//!     base64 strings for the conversation request.
//!   * Missing images surface as a 400 ("image file is required") from
//!     `normalize_json_edit_images` rather than the Python
//!     `ImageGenerationError("image is required")` (502); the intent is the same.
//!   * The Python `progress_callback` is dropped; internal `_account_email` /
//!     `_conversation_id` fields are stripped from the wire output.
#![allow(dead_code)]

use axum::http::StatusCode;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::error::{openai_error_payload, AppError};
use crate::services::protocol::conversation::{
    collect_image_outputs, count_text_tokens, stream_image_chunks, stream_image_outputs_with_pool, ConvDeps,
    ConversationRequest, ImageGenerationError,
};
use crate::utils::helper::{normalize_json_edit_images, sse_data, sse_done, sse_open, DecodedImage};
use crate::utils::image_tokens::{count_image_input_tokens, count_image_output_items_tokens, image_size_from_bytes, image_usage};

fn image_err_to_app(e: ImageGenerationError) -> AppError {
    let status = StatusCode::from_u16(e.status_code).unwrap_or(StatusCode::BAD_GATEWAY);
    AppError::new(status, e.to_openai_error())
}

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

fn strip_internal(mut chunk: Value) -> Value {
    if let Some(map) = chunk.as_object_mut() {
        map.remove("_account_email");
        map.remove("_conversation_id");
    }
    chunk
}

fn count_input_image_tokens(images: &[DecodedImage], model: &str) -> i64 {
    let mut total = 0;
    for img in images {
        if let Some((w, h)) = image_size_from_bytes(&img.data) {
            total += count_image_input_tokens(w, h, model, "auto");
        }
    }
    total
}

/// Decode source images and build the conversation request. Returns the request
/// plus the pieces needed to compute usage.
fn build_request(
    body: &Value,
    base_url: Option<String>,
) -> Result<(ConversationRequest, String, String, Value, String, Vec<DecodedImage>), AppError> {
    let images = normalize_json_edit_images(body.get("image"), body.get("images"))?;
    let encoded: Vec<String> = images.iter().map(|i| B64.encode(&i.data)).collect();
    let request = ConversationRequest {
        prompt: body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        model: str_or(body, "model", "gpt-image-2"),
        n: parse_n(body),
        size: body.get("size").and_then(|v| v.as_str()).map(String::from),
        quality: str_or(body, "quality", "auto"),
        response_format: str_or(body, "response_format", "b64_json"),
        base_url: resolve_base_url(body, base_url),
        images: Some(encoded),
        message_as_error: true,
        ..Default::default()
    };
    let prompt = request.prompt.clone();
    let model = request.model.clone();
    let quality = request.quality.clone();
    let size_val = body.get("size").cloned().unwrap_or(Value::Null);
    Ok((request, prompt, model, size_val, quality, images))
}

/// Buffered edit: collect outputs and attach the image `usage` block.
pub async fn image_edit_once(deps: ConvDeps, body: Value, base_url: Option<String>) -> Result<Value, AppError> {
    let (request, prompt, model, size_val, quality, images) = build_request(&body, base_url)?;
    let rx = stream_image_outputs_with_pool(deps, request);
    let mut result = collect_image_outputs(rx).await.map_err(image_err_to_app)?;
    let usage = image_usage(
        count_text_tokens(&prompt, &model),
        count_input_image_tokens(&images, &model),
        count_image_output_items_tokens(result.get("data").unwrap_or(&Value::Null), &size_val, &quality),
    );
    if let Some(map) = result.as_object_mut() {
        map.insert("usage".to_string(), usage);
        map.remove("_account_email");
    }
    Ok(result)
}

/// Streaming edit: relay `image.generation.*` chunks as SSE frames.
pub fn image_edit_stream(deps: ConvDeps, body: Value, base_url: Option<String>) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let _ = tx.send(sse_open()).await;
        match build_request(&body, base_url) {
            Ok((request, _, _, _, _, _)) => {
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
            }
            Err(e) => {
                let _ = tx.send(sse_data(&openai_error_payload(&e.detail, e.status.as_u16()))).await;
            }
        }
        let _ = tx.send(sse_done()).await;
    });
    rx
}
