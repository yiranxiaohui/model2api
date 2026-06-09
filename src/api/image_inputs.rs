//! Port of `api/image_inputs.py` — multipart / JSON parsing helpers for the
//! image-edit (`/v1/images/edits`, `/api/image-tasks/edits`) endpoints.
//!
//! This module is a *helper* layer (it defines no router). It is consumed by
//! `ai.rs` and `image_tasks.rs` to turn an inbound request — either a
//! `multipart/form-data` upload or an `application/json` body — into a normalized
//! payload plus the list of source images decoded to **raw base64 strings**
//! (`base64::STANDARD` of the image bytes, no `data:` prefix), which is exactly
//! the form [`ConversationRequest::images`](crate::services::protocol::conversation)
//! expects (see `openai_v1_image_edit::build_request`).
//!
//! Public entry points:
//!   * [`parse_image_edit_request`] — content-type dispatcher (mirrors the Python
//!     `parse_image_edit_request(request)`). Takes the whole
//!     [`axum::extract::Request`], branches on the `Content-Type`, and returns
//!     `(payload, base64_images)`.
//!   * [`parse_image_edit_multipart`] — the `multipart/form-data` branch.
//!   * [`parse_image_edit_json`] — the `application/json` branch (uses
//!     [`normalize_json_edit_images`]).
//!
//! Adaptation notes vs. the Python source:
//!   * Remote `http(s)://` image URLs are **not** fetched (the Python original
//!     downloaded them with `curl_cffi`). The Rust engine rejects remote URLs in
//!     `decode_json_image_string`, so a remote `image_url` yields a 400 here too.
//!     Only uploaded files, `data:` URLs, plain base64, and `b64_json`/`base64`
//!     object references are supported.
//!   * `read_image_sources` is folded into the parse functions: sources are
//!     resolved to [`DecodedImage`] and base64-encoded inline, rather than
//!     returning intermediate `(bytes, name, mime)` tuples.
#![allow(dead_code)]

use axum::extract::{FromRequest, Multipart, Request};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use serde_json::{Map, Value};

use crate::error::AppError;
use crate::utils::helper::{
    decode_json_image_string, normalize_json_edit_images, parse_image_count, DecodedImage,
};

/// Generous body cap for the JSON branch (image edits can carry inline base64).
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Form field names that hold image references (port of `IMAGE_REFERENCE_FIELDS`).
const IMAGE_REFERENCE_FIELDS: [&str; 6] =
    ["image", "image[]", "images", "images[]", "image_url", "image_url[]"];

/// Scalar form fields lifted into the normalized payload.
const PAYLOAD_FIELDS: [&str; 8] = [
    "client_task_id",
    "prompt",
    "model",
    "n",
    "size",
    "quality",
    "response_format",
    "stream",
];

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// `str(value or "").strip()` for JSON values.
fn clean(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string().trim().to_string(),
    }
}

fn clean_or(value: Option<&Value>, default: &str) -> String {
    let v = clean(value);
    if v.is_empty() {
        default.to_string()
    } else {
        v
    }
}

/// Port of `_parse_bool` — returns `null` for missing/empty, `bool` otherwise.
fn parse_bool(value: Option<&Value>) -> Result<Value, AppError> {
    match value {
        None | Some(Value::Null) => Ok(Value::Null),
        Some(Value::Bool(b)) => Ok(Value::Bool(*b)),
        Some(other) => {
            let text = clean(Some(other)).to_ascii_lowercase();
            if text.is_empty() {
                return Ok(Value::Null);
            }
            match text.as_str() {
                "true" | "1" | "yes" | "y" | "on" => Ok(Value::Bool(true)),
                "false" | "0" | "no" | "n" | "off" => Ok(Value::Bool(false)),
                _ => Err(AppError::bad_request("stream must be a boolean")),
            }
        }
    }
}

/// Port of `_payload_from_fields` — assemble the normalized edit payload.
fn payload_from_fields(fields: &Map<String, Value>) -> Result<Value, AppError> {
    let prompt = clean(fields.get("prompt"));
    if prompt.is_empty() {
        return Err(AppError::bad_request("prompt is required"));
    }
    let size = clean(fields.get("size"));
    let mut payload = Map::new();
    payload.insert("prompt".into(), Value::String(prompt));
    payload.insert("model".into(), Value::String(clean_or(fields.get("model"), "gpt-image-2")));
    payload.insert(
        "n".into(),
        Value::from(parse_image_count(fields.get("n").unwrap_or(&Value::Null))?),
    );
    payload.insert(
        "size".into(),
        if size.is_empty() { Value::Null } else { Value::String(size) },
    );
    payload.insert("quality".into(), Value::String(clean_or(fields.get("quality"), "auto")));
    payload.insert(
        "response_format".into(),
        Value::String(clean_or(fields.get("response_format"), "b64_json")),
    );
    payload.insert("stream".into(), parse_bool(fields.get("stream"))?);
    if fields.contains_key("client_task_id") {
        payload.insert("client_task_id".into(), Value::String(clean(fields.get("client_task_id"))));
    }
    Ok(Value::Object(payload))
}

/// Decode a string image source (`data:` URL or plain base64) to [`DecodedImage`].
/// Remote `http(s)://` URLs are rejected (consistent with the Rust engine).
fn decode_string_source(text: &str, index: usize) -> Result<DecodedImage, AppError> {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return Err(AppError::bad_request(
            "remote image URLs are not supported; use a data URL or base64",
        ));
    }
    decode_json_image_string(trimmed, index, None, None)
}

/// Port of `_sources_from_value` — flatten a JSON-ish image reference into
/// decoded images. Strings that are themselves JSON arrays/objects are parsed
/// and recursed (port of `_json_reference_value`).
fn collect_sources_from_value(value: &Value, out: &mut Vec<DecodedImage>) -> Result<(), AppError> {
    match value {
        Value::Null => Ok(()),
        Value::String(s) => {
            let text = s.trim();
            if text.is_empty() {
                return Ok(());
            }
            if matches!(text.chars().next(), Some('[') | Some('{')) {
                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                    return collect_sources_from_value(&parsed, out);
                }
            }
            let idx = out.len() + 1;
            out.push(decode_string_source(text, idx)?);
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                collect_sources_from_value(item, out)?;
            }
            Ok(())
        }
        Value::Object(obj) => collect_source_from_object(obj, out),
        _ => Err(AppError::bad_request("invalid image reference")),
    }
}

/// Port of `_source_from_object` — `image_url`/`url`/`b64_json`/`base64`, reject
/// `file_id`.
fn collect_source_from_object(
    obj: &Map<String, Value>,
    out: &mut Vec<DecodedImage>,
) -> Result<(), AppError> {
    if obj.get("file_id").map(|v| !v.is_null()).unwrap_or(false) {
        return Err(AppError::bad_request(
            "file_id image references are not supported; use image_url instead",
        ));
    }
    let inline = obj.get("b64_json").or_else(|| obj.get("base64")).and_then(|v| v.as_str());
    if let Some(inline) = inline {
        if !inline.trim().is_empty() {
            let filename = obj
                .get("filename")
                .or_else(|| obj.get("file_name"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let mime = obj
                .get("mime_type")
                .or_else(|| obj.get("mimeType"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let idx = out.len() + 1;
            out.push(decode_json_image_string(inline.trim(), idx, filename, mime)?);
            return Ok(());
        }
    }
    if !(obj.contains_key("image_url") || obj.contains_key("url")) {
        return Err(AppError::bad_request("image reference must include image_url"));
    }
    let mut image_url = obj.get("image_url").or_else(|| obj.get("url")).cloned().unwrap_or(Value::Null);
    if let Value::Object(inner) = &image_url {
        image_url = inner.get("url").cloned().unwrap_or(Value::Null);
    }
    collect_sources_from_value(&image_url, out)
}

fn encode_images(images: &[DecodedImage]) -> Vec<String> {
    images.iter().map(|img| B64.encode(&img.data)).collect()
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

/// Parse an image-edit request, dispatching on `Content-Type`.
///
/// `application/json` → [`parse_image_edit_json`]; anything else is treated as
/// `multipart/form-data` → [`parse_image_edit_multipart`].
///
/// Returns `(payload, base64_images)` where `payload` is the normalized field map
/// (`prompt`/`model`/`n`/`size`/`quality`/`response_format`/`stream`, plus
/// `client_task_id` when present) and `base64_images` are raw base64 strings.
pub async fn parse_image_edit_request(req: Request) -> Result<(Value, Vec<String>), AppError> {
    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    if content_type == "application/json" {
        let bytes = axum::body::to_bytes(req.into_body(), MAX_BODY_BYTES)
            .await
            .map_err(|_| AppError::bad_request("failed to read request body"))?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| AppError::bad_request("invalid JSON body"))?;
        return parse_image_edit_json(&body);
    }

    let multipart = Multipart::from_request(req, &())
        .await
        .map_err(|_| AppError::bad_request("invalid multipart form"))?;
    parse_image_edit_multipart(multipart).await
}

/// Parse a `multipart/form-data` image-edit request.
///
/// Scalar fields populate the payload; image-reference fields (uploads, `data:`
/// URLs, base64, or JSON-encoded references) are decoded and base64-encoded.
pub async fn parse_image_edit_multipart(
    mut multipart: Multipart,
) -> Result<(Value, Vec<String>), AppError> {
    let mut fields: Map<String, Value> = Map::new();
    let mut images: Vec<DecodedImage> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| AppError::bad_request("invalid multipart form"))?
    {
        let name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());
        let is_image_field = IMAGE_REFERENCE_FIELDS.contains(&name.as_str());

        if file_name.is_some() {
            // An uploaded file part.
            if !is_image_field {
                continue;
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|_| AppError::bad_request("failed to read uploaded image"))?;
            if bytes.is_empty() {
                return Err(AppError::bad_request("image file is empty"));
            }
            images.push(DecodedImage {
                data: bytes.to_vec(),
                filename: file_name.unwrap_or_else(|| "image.png".to_string()),
                mime_type: content_type.unwrap_or_else(|| "image/png".to_string()),
            });
            continue;
        }

        // A scalar text part.
        let text = field
            .text()
            .await
            .map_err(|_| AppError::bad_request("invalid form field"))?;
        if is_image_field {
            collect_sources_from_value(&Value::String(text), &mut images)?;
        } else if PAYLOAD_FIELDS.contains(&name.as_str()) {
            fields.insert(name, Value::String(text));
        }
    }

    let payload = payload_from_fields(&fields)?;
    if images.is_empty() {
        return Err(AppError::bad_request("image file or image_url is required"));
    }
    Ok((payload, encode_images(&images)))
}

/// Parse an `application/json` image-edit body.
///
/// Image references come from the `image` / `image_url` / `images` fields via
/// [`normalize_json_edit_images`].
pub fn parse_image_edit_json(body: &Value) -> Result<(Value, Vec<String>), AppError> {
    let obj = body
        .as_object()
        .ok_or_else(|| AppError::bad_request("JSON body must be an object"))?;
    let payload = payload_from_fields(obj)?;
    let decoded = normalize_json_edit_images(
        body.get("image").or_else(|| body.get("image_url")),
        body.get("images"),
    )?;
    Ok((payload, encode_images(&decoded)))
}
