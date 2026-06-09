//! Port of the pure/algorithmic parts of `utils/helper.py`: image-model name
//! handling, JSON image decoding, prompt extraction, SSE framing helpers and the
//! `UpstreamHTTPError` type. Remote `image_url` fetching (which needs the HTTP
//! client) is added alongside the backend client.

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use serde_json::{json, Value};

use crate::error::AppError;

pub const BASE_IMAGE_MODELS: [&str; 2] = ["gpt-image-2", "codex-gpt-image-2"];
pub const IMAGE_MODEL_PLAN_TYPES: [&str; 3] = ["plus", "team", "pro"];
pub const CODEX_IMAGE_MODEL: &str = "codex-gpt-image-2";

const SUPPORTED_JSON_IMAGE_MIME_TYPES: [&str; 5] = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
];
const MAX_JSON_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_JSON_EDIT_IMAGES: usize = 10;

pub fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Split a possibly plan-prefixed image model into `(plan_type, base_model)`.
pub fn split_image_model(model: &Value) -> (Option<String>, Option<String>) {
    let normalized = value_to_string(model).trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return (None, None);
    }
    if BASE_IMAGE_MODELS.contains(&normalized.as_str()) {
        return (None, Some(normalized));
    }
    for plan in IMAGE_MODEL_PLAN_TYPES {
        let prefix = format!("{plan}-");
        if let Some(base) = normalized.strip_prefix(&prefix) {
            if base == CODEX_IMAGE_MODEL {
                return (Some(plan.to_string()), Some(base.to_string()));
            }
        }
    }
    (None, None)
}

pub fn is_supported_image_model(model: &Value) -> bool {
    split_image_model(model).1.is_some()
}

pub fn is_codex_image_model(model: &Value) -> bool {
    split_image_model(model).1.as_deref() == Some(CODEX_IMAGE_MODEL)
}

pub fn is_image_chat_request(body: &Value) -> bool {
    let model = body.get("model").cloned().unwrap_or(Value::Null);
    if is_supported_image_model(&model) {
        return true;
    }
    if let Some(modalities) = body.get("modalities").and_then(|v| v.as_array()) {
        return modalities
            .iter()
            .any(|m| value_to_string(m).trim().to_ascii_lowercase() == "image");
    }
    false
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// A decoded image: raw bytes, filename and mime type.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub data: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
}

fn image_extension(mime_type: &str) -> String {
    let image_type = mime_type
        .split_once('/')
        .map(|(_, rest)| rest.split(';').next().unwrap_or("png"))
        .unwrap_or("png")
        .to_ascii_lowercase();
    if image_type == "jpeg" {
        "jpg".to_string()
    } else if image_type.is_empty() {
        "png".to_string()
    } else {
        image_type
    }
}

/// Port of `_decode_json_image_string`.
pub fn decode_json_image_string(
    value: &str,
    index: usize,
    filename: Option<&str>,
    mime_type: Option<&str>,
) -> Result<DecodedImage, AppError> {
    let text = value.trim();
    if text.is_empty() {
        return Err(AppError::bad_request("image file is empty"));
    }
    let re = regex::Regex::new(r"(?s)^data:(?P<mime>[-+./\w]+);base64,(?P<data>.*)$").unwrap();
    let (mut resolved_mime, encoded) = if let Some(cap) = re.captures(text) {
        (
            cap.name("mime").map(|m| m.as_str()).unwrap_or("image/png").to_ascii_lowercase(),
            cap.name("data").map(|m| m.as_str()).unwrap_or("").to_string(),
        )
    } else {
        if text.starts_with("http://") || text.starts_with("https://") {
            return Err(AppError::bad_request("remote image URLs are not supported"));
        }
        (
            mime_type.unwrap_or("image/png").to_ascii_lowercase(),
            text.to_string(),
        )
    };
    if resolved_mime == "image/jpg" {
        resolved_mime = "image/jpeg".to_string();
    }
    if !SUPPORTED_JSON_IMAGE_MIME_TYPES.contains(&resolved_mime.as_str()) {
        return Err(AppError::bad_request("unsupported image mime type"));
    }
    let image_data = BASE64_STANDARD
        .decode(encoded.trim())
        .map_err(|_| AppError::bad_request("invalid base64 image data"))?;
    if image_data.is_empty() {
        return Err(AppError::bad_request("image file is empty"));
    }
    if image_data.len() > MAX_JSON_IMAGE_BYTES {
        return Err(AppError::bad_request("image file is too large"));
    }
    let filename = filename
        .map(String::from)
        .unwrap_or_else(|| format!("image_{index}.{}", image_extension(&resolved_mime)));
    Ok(DecodedImage {
        data: image_data,
        filename,
        mime_type: resolved_mime,
    })
}

fn extract_json_image_value(item: &Value) -> Result<(String, Option<String>, Option<String>), AppError> {
    if let Value::String(s) = item {
        return Ok((s.clone(), None, None));
    }
    let Some(obj) = item.as_object() else {
        return Err(AppError::bad_request(
            "image entry must be a base64 string or object",
        ));
    };
    let get_str = |keys: &[&str]| -> Option<String> {
        for k in keys {
            if let Some(s) = obj.get(*k).and_then(|v| v.as_str()) {
                let s = s.trim();
                if !s.is_empty() {
                    return Some(s.to_string());
                }
            }
        }
        None
    };
    let mut filename = get_str(&["filename", "file_name"]);
    let mut mime_type = get_str(&["mime_type", "mimeType"]);
    let mut value = obj
        .get("b64_json")
        .or_else(|| obj.get("base64"))
        .and_then(|v| v.as_str())
        .map(String::from);

    if value.is_none() {
        let image_url = obj.get("image_url").or_else(|| obj.get("url"));
        match image_url {
            Some(Value::Object(m)) => {
                if filename.is_none() {
                    filename = m
                        .get("filename")
                        .or_else(|| m.get("file_name"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                }
                if mime_type.is_none() {
                    mime_type = m
                        .get("mime_type")
                        .or_else(|| m.get("mimeType"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                }
                value = m
                    .get("url")
                    .or_else(|| m.get("image_url"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
            Some(Value::String(s)) => value = Some(s.clone()),
            _ => {}
        }
    }

    match value {
        Some(v) if !v.trim().is_empty() => Ok((v, filename, mime_type)),
        _ => Err(AppError::bad_request("image entry must include image data")),
    }
}

/// Port of `normalize_json_edit_images`.
pub fn normalize_json_edit_images(
    image: Option<&Value>,
    images: Option<&Value>,
) -> Result<Vec<DecodedImage>, AppError> {
    let raw = images.or(image);
    let Some(raw) = raw else {
        return Err(AppError::bad_request("image file is required"));
    };
    let entries: Vec<Value> = match raw {
        Value::Array(a) => a.clone(),
        Value::Null => return Err(AppError::bad_request("image file is required")),
        other => vec![other.clone()],
    };
    if entries.is_empty() {
        return Err(AppError::bad_request("image file is required"));
    }
    if entries.len() > MAX_JSON_EDIT_IMAGES {
        return Err(AppError::bad_request(format!(
            "images supports up to {MAX_JSON_EDIT_IMAGES} items"
        )));
    }
    let mut out = Vec::with_capacity(entries.len());
    for (i, item) in entries.iter().enumerate() {
        let (value, filename, mime) = extract_json_image_value(item)?;
        out.push(decode_json_image_string(
            &value,
            i + 1,
            filename.as_deref(),
            mime.as_deref(),
        )?);
    }
    Ok(out)
}

/// Port of `parse_image_count` (1..=4).
pub fn parse_image_count(raw: &Value) -> Result<i64, AppError> {
    let value = match raw {
        Value::Null => 1,
        Value::Number(n) => n.as_i64().ok_or_else(|| AppError::bad_request("n must be an integer"))?,
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| AppError::bad_request("n must be an integer"))?,
        _ => return Err(AppError::bad_request("n must be an integer")),
    };
    if !(1..=4).contains(&value) {
        return Err(AppError::bad_request("n must be between 1 and 4"));
    }
    Ok(value)
}

/// Upstream HTTP error carrying status / body / retry-after.
#[derive(Debug, Clone)]
pub struct UpstreamHttpError {
    pub context: String,
    pub status_code: u16,
    pub body: Value,
    pub retry_after: Option<u64>,
}

impl UpstreamHttpError {
    pub fn new(context: impl Into<String>, status_code: u16, body: Value, retry_after: Option<u64>) -> Self {
        Self {
            context: context.into(),
            status_code,
            body,
            retry_after,
        }
    }
}

impl std::fmt::Display for UpstreamHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut body_str = match &self.body {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        const LIMIT: usize = 500;
        if body_str.len() > LIMIT {
            body_str.truncate(LIMIT);
            body_str.push_str("…[truncated]");
        }
        write!(
            f,
            "{} failed: status={}, body={}",
            self.context, self.status_code, body_str
        )
    }
}

impl std::error::Error for UpstreamHttpError {}

// ---- Prompt extraction (pure) ----

pub fn extract_prompt_from_message_content(content: &Value) -> String {
    if let Some(s) = content.as_str() {
        return s.trim().to_string();
    }
    let Some(items) = content.as_array() else {
        return String::new();
    };
    let mut parts = Vec::new();
    for item in items {
        let Some(obj) = item.as_object() else { continue };
        let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
        match item_type {
            "text" => {
                let t = obj.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
            "input_text" => {
                let t = obj
                    .get("text")
                    .or_else(|| obj.get("input_text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
            _ => {}
        }
    }
    parts.join("\n").trim().to_string()
}

pub fn extract_chat_prompt(body: &Value) -> String {
    if let Some(p) = body.get("prompt").and_then(|v| v.as_str()) {
        let p = p.trim();
        if !p.is_empty() {
            return p.to_string();
        }
    }
    let Some(messages) = body.get("messages").and_then(|v| v.as_array()) else {
        return String::new();
    };
    let mut parts = Vec::new();
    for message in messages {
        let Some(obj) = message.as_object() else { continue };
        if obj.get("role").and_then(|v| v.as_str()).unwrap_or("").trim().to_ascii_lowercase() != "user" {
            continue;
        }
        let prompt = extract_prompt_from_message_content(obj.get("content").unwrap_or(&Value::Null));
        if !prompt.is_empty() {
            parts.push(prompt);
        }
    }
    parts.join("\n").trim().to_string()
}

pub fn extract_response_prompt(input_value: &Value) -> String {
    match input_value {
        Value::String(s) => s.trim().to_string(),
        Value::Object(obj) => {
            let role = obj.get("role").and_then(|v| v.as_str()).unwrap_or("").trim().to_ascii_lowercase();
            if !role.is_empty() && role != "user" {
                return String::new();
            }
            extract_prompt_from_message_content(obj.get("content").unwrap_or(&Value::Null))
        }
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                let Some(obj) = item.as_object() else { continue };
                let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
                if item_type == "input_text" {
                    let t = obj.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
                    if !t.is_empty() {
                        parts.push(t.to_string());
                    }
                    continue;
                }
                let role = obj.get("role").and_then(|v| v.as_str()).unwrap_or("").trim().to_ascii_lowercase();
                if !role.is_empty() && role != "user" {
                    continue;
                }
                let prompt = extract_prompt_from_message_content(obj.get("content").unwrap_or(&Value::Null));
                if !prompt.is_empty() {
                    parts.push(prompt);
                }
            }
            parts.join("\n").trim().to_string()
        }
        _ => String::new(),
    }
}

pub fn has_response_image_generation_tool(body: &Value) -> bool {
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if tool.get("type").and_then(|v| v.as_str()).unwrap_or("").trim() == "image_generation" {
                return true;
            }
        }
    }
    body.get("tool_choice")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim() == "image_generation")
        .unwrap_or(false)
}

pub fn build_chat_image_markdown_content(image_result: &Value) -> String {
    let items = image_result.get("data").and_then(|v| v.as_array());
    let mut markdown = Vec::new();
    if let Some(items) = items {
        for (i, item) in items.iter().enumerate() {
            if let Some(b64) = item.get("b64_json").and_then(|v| v.as_str()) {
                let b64 = b64.trim();
                if !b64.is_empty() {
                    markdown.push(format!("![image_{}](data:image/png;base64,{})", i + 1, b64));
                }
            }
        }
    }
    if markdown.is_empty() {
        "Image generation completed.".to_string()
    } else {
        markdown.join("\n\n")
    }
}

pub fn anonymize_token(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let value = token.trim();
    if value.is_empty() {
        return "token:empty".to_string();
    }
    let digest = hex::encode(Sha256::digest(value.as_bytes()));
    format!("token:{}", &digest[..10])
}

// ---- SSE framing helpers ----

pub fn sse_open() -> String {
    ": stream-open\n\n".to_string()
}

pub fn sse_data(value: &Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(value).unwrap_or_default())
}

pub fn sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}

pub fn anthropic_sse_event(value: &Value) -> String {
    let event = value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message_delta");
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(value).unwrap_or_default()
    )
}

/// Build an OpenAI-style error SSE data frame.
pub fn sse_error(message: &str, error_type: &str) -> String {
    sse_data(&json!({ "error": { "message": message, "type": error_type } }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_models() {
        assert_eq!(split_image_model(&json!("gpt-image-2")), (None, Some("gpt-image-2".into())));
        assert_eq!(
            split_image_model(&json!("plus-codex-gpt-image-2")),
            (Some("plus".into()), Some("codex-gpt-image-2".into()))
        );
        assert_eq!(split_image_model(&json!("gpt-4o")), (None, None));
        assert!(is_codex_image_model(&json!("team-codex-gpt-image-2")));
    }

    #[test]
    fn image_count_bounds() {
        assert_eq!(parse_image_count(&json!(2)).unwrap(), 2);
        assert!(parse_image_count(&json!(0)).is_err());
        assert!(parse_image_count(&json!(5)).is_err());
        assert_eq!(parse_image_count(&Value::Null).unwrap(), 1);
    }

    #[test]
    fn decode_data_url_image() {
        // 1x1 transparent PNG
        let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
        let url = format!("data:image/png;base64,{png_b64}");
        let img = decode_json_image_string(&url, 1, None, None).unwrap();
        assert_eq!(img.mime_type, "image/png");
        assert!(img.filename.ends_with(".png"));
        assert!(!img.data.is_empty());
    }

    #[test]
    fn prompt_extraction() {
        let body = json!({"messages": [{"role": "user", "content": "hello"}]});
        assert_eq!(extract_chat_prompt(&body), "hello");
        let body2 = json!({"prompt": "direct"});
        assert_eq!(extract_chat_prompt(&body2), "direct");
    }
}
