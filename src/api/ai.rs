//! Port of `api/ai.py` — the OpenAI-/Anthropic-compatible inference routes plus
//! the editable-file (PPT/PSD) task endpoints.
//!
//! Covers: `/v1/models`, `/v1/images/{generations,edits}`,
//! `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/search`,
//! `/v1/editable-file-tasks`, `/files/{*path}`, and `/v1/{ppt,psd}/generations`.
//!
//! Each handler authenticates via [`require_identity`], runs the request text
//! through the content filter (`filter_or_log`), then dispatches to the matching
//! protocol translator — streaming (`sse_response`) or buffered (`json_ok`)
//! depending on the request's `stream` flag.
//!
//! Adaptation notes:
//!   * The multipart/JSON image-edit body parsing (`api/image_inputs.py`'s
//!     `parse_image_edit_request` / `read_image_sources`) is inlined here. Remote
//!     `http(s)` image URLs are NOT downloaded — the Rust image helper only
//!     decodes inline data-URL / base64 sources (consistent with the rest of the
//!     port); such references surface as a 400 from the translator.
//!   * `LoggedCall` here is the lightweight `routes_common` variant; it records a
//!     `call` log entry on success/failure (the Python `request_text` /
//!     `request_shape` log fields are dropped — that struct does not carry them).
//!   * Editable-file (PPT/PSD) submission and `/files` downloads are backed by
//!     the stub `EditableFileTaskService` (engine export pending); submissions
//!     report "unsupported" and downloads 404 until that lands.
#![allow(dead_code)]

use std::collections::HashMap;

use axum::body::{to_bytes, Body};
use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use serde_json::{json, Value};

use crate::api::routes_common::{
    authorization, header_str, json_ok, scheme, sse_response, LoggedCall,
};
use crate::api::support::{require_identity, resolve_image_base_url};
use crate::error::AppError;
use crate::services::content_filter::ContentFilter;
use crate::services::protocol::{
    anthropic_v1_messages, openai_search, openai_v1_chat_complete, openai_v1_image_edit,
    openai_v1_image_generations, openai_v1_models, openai_v1_response,
};
use crate::state::SharedState;

const MAX_EDIT_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Form field names that carry image references (mirrors `IMAGE_REFERENCE_FIELDS`).
const IMAGE_REFERENCE_FIELDS: [&str; 6] =
    ["image", "image[]", "images", "images[]", "image_url", "image_url[]"];

/// Scalar form fields copied into the image-edit payload.
const EDIT_SCALAR_FIELDS: [&str; 8] = [
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
// router
// ---------------------------------------------------------------------------

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/images/generations", post(generate_images))
        .route("/v1/images/edits", post(edit_images))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/responses", post(create_response))
        .route("/v1/messages", post(create_message))
        .route("/v1/search", post(search))
        .route("/v1/editable-file-tasks", get(list_editable_file_tasks))
        .route("/files/{*file_path}", get(download_editable_file))
        .route("/v1/ppt/generations", post(create_ppt_task))
        .route("/v1/psd/generations", post(create_psd_task))
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn is_stream(body: &Value) -> bool {
    body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false)
}

fn resolve_base_url(state: &SharedState, headers: &HeaderMap) -> String {
    resolve_image_base_url(state, header_str(headers, "host").as_deref(), &scheme(headers))
}

/// `body.get(key)` as a non-empty trimmed string, else `default`.
fn str_field(body: &Value, key: &str, default: &str) -> String {
    body.get(key)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(default)
        .to_string()
}

/// Port of `filter_or_log`: run the content filter, logging + mapping a failure.
async fn filter_or_log(
    state: &SharedState,
    call: &LoggedCall,
    text: &str,
    endpoint: &str,
) -> Result<(), AppError> {
    if let Err(e) = state.content_filter.check_request(text).await {
        call.finish("调用失败", "failed", Some(json!({ "error": e.detail.clone() })));
        let status = StatusCode::from_u16(e.status_code).unwrap_or(StatusCode::BAD_REQUEST);
        return Err(AppError::message(status, e.detail).with_path(endpoint));
    }
    Ok(())
}

/// Log a failed buffered call and return the error tagged with the endpoint path.
fn finish_err(call: &LoggedCall, e: AppError, endpoint: &str) -> AppError {
    call.finish("调用失败", "failed", Some(json!({ "error": e.detail.clone() })));
    e.with_path(endpoint)
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

async fn list_models(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_identity(&state, authorization(&headers).as_deref())?;
    let result = openai_v1_models::list_models(state.conv.clone())
        .await
        .map_err(|e| e.with_path("/v1/models"))?;
    Ok(json_ok(result))
}

// ---------------------------------------------------------------------------
// /v1/images/generations
// ---------------------------------------------------------------------------

async fn generate_images(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/images/generations";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();

    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err(AppError::bad_request("prompt is required").with_path(ENDPOINT));
    }
    let model = str_field(&body, "model", "gpt-image-2");

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, &model, "文生图");
    filter_or_log(&state, &call, &prompt, ENDPOINT).await?;

    let base_url = Some(resolve_base_url(&state, &headers));
    let deps = state.conv.clone();
    if is_stream(&body) {
        let rx = openai_v1_image_generations::image_generations_stream(deps, body, base_url);
        call.finish("", "success", None);
        Ok(sse_response(rx))
    } else {
        match openai_v1_image_generations::image_generations_once(deps, body, base_url).await {
            Ok(v) => {
                call.finish("", "success", None);
                Ok(json_ok(v))
            }
            Err(e) => Err(finish_err(&call, e, ENDPOINT)),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/images/edits  (multipart form OR JSON body)
// ---------------------------------------------------------------------------

async fn edit_images(
    State(state): State<SharedState>,
    headers: HeaderMap,
    request: Request,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/images/edits";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();

    let (body, prompt, model) = parse_image_edit_request(request).await?;

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, &model, "图生图");
    filter_or_log(&state, &call, &prompt, ENDPOINT).await?;

    let base_url = Some(resolve_base_url(&state, &headers));
    let deps = state.conv.clone();
    if is_stream(&body) {
        let rx = openai_v1_image_edit::image_edit_stream(deps, body, base_url);
        call.finish("", "success", None);
        Ok(sse_response(rx))
    } else {
        match openai_v1_image_edit::image_edit_once(deps, body, base_url).await {
            Ok(v) => {
                call.finish("", "success", None);
                Ok(json_ok(v))
            }
            Err(e) => Err(finish_err(&call, e, ENDPOINT)),
        }
    }
}

/// Boolean form field (port of `_parse_bool`).
fn parse_bool_field(value: Option<&String>) -> Result<Option<bool>, AppError> {
    match value.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(text) => match text.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" | "on" => Ok(Some(true)),
            "false" | "0" | "no" | "n" | "off" => Ok(Some(false)),
            _ => Err(AppError::bad_request("stream must be a boolean")),
        },
    }
}

/// Count form field, 1..=4 (port of `_parse_count`).
fn parse_count_field(value: Option<&String>) -> Result<i64, AppError> {
    let n = match value.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => 1,
        Some(text) => text.parse::<i64>().map_err(|_| AppError::bad_request("n must be an integer"))?,
    };
    if !(1..=4).contains(&n) {
        return Err(AppError::bad_request("n must be between 1 and 4"));
    }
    Ok(n)
}

/// Parse the image-edit request, accepting multipart form data OR a JSON body.
/// Returns `(translator_body, prompt, model)`.
async fn parse_image_edit_request(request: Request) -> Result<(Value, String, String), AppError> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    if content_type == "application/json" {
        let bytes = to_bytes(request.into_body(), MAX_EDIT_BODY_BYTES)
            .await
            .map_err(|_| AppError::bad_request("invalid request body"))?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| AppError::bad_request("invalid JSON body"))?;
        if !body.is_object() {
            return Err(AppError::bad_request("JSON body must be an object"));
        }
        let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            return Err(AppError::bad_request("prompt is required"));
        }
        let model = str_field(&body, "model", "gpt-image-2");
        // The translator reads `image` / `images` directly off the body.
        return Ok((body, prompt, model));
    }

    // multipart/form-data
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|e| AppError::bad_request(format!("invalid multipart form: {e}")))?;

    let mut fields: HashMap<String, String> = HashMap::new();
    let mut image_entries: Vec<Value> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("invalid multipart form: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        if IMAGE_REFERENCE_FIELDS.contains(&name.as_str()) {
            let filename = field.file_name().map(str::to_string);
            let mime = field.content_type().map(str::to_string);
            let is_file = filename.is_some() || mime.is_some();
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::bad_request(format!("invalid image upload: {e}")))?;
            if is_file {
                image_entries.push(json!({
                    "b64_json": B64.encode(&data),
                    "mime_type": mime.unwrap_or_else(|| "image/png".to_string()),
                    "filename": filename.unwrap_or_else(|| "image.png".to_string()),
                }));
            } else {
                // Text value: a data-URL / base64 string (remote URLs unsupported).
                let text = String::from_utf8_lossy(&data).trim().to_string();
                if !text.is_empty() {
                    image_entries.push(Value::String(text));
                }
            }
        } else if EDIT_SCALAR_FIELDS.contains(&name.as_str()) {
            let text = field
                .text()
                .await
                .map_err(|e| AppError::bad_request(format!("invalid form field: {e}")))?;
            fields.insert(name, text);
        }
    }

    let prompt = fields.get("prompt").map(|s| s.trim()).unwrap_or("").to_string();
    if prompt.is_empty() {
        return Err(AppError::bad_request("prompt is required"));
    }
    let model = fields
        .get("model")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("gpt-image-2")
        .to_string();
    let n = parse_count_field(fields.get("n"))?;
    let stream = parse_bool_field(fields.get("stream"))?;

    let mut body = json!({
        "prompt": prompt,
        "model": model,
        "n": n,
        "quality": fields.get("quality").map(|s| s.trim()).filter(|s| !s.is_empty()).unwrap_or("auto"),
        "response_format": fields.get("response_format").map(|s| s.trim()).filter(|s| !s.is_empty()).unwrap_or("b64_json"),
        "images": image_entries,
    });
    if let Some(size) = fields.get("size").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        body["size"] = json!(size);
    }
    if let Some(stream) = stream {
        body["stream"] = json!(stream);
    }
    if let Some(client_task_id) = fields.get("client_task_id") {
        body["client_task_id"] = json!(client_task_id);
    }

    Ok((body, prompt, model))
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

async fn create_chat_completion(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/chat/completions";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();
    let model = str_field(&body, "model", "auto");

    let null = Value::Null;
    let preview = ContentFilter::request_text(&[
        body.get("prompt").unwrap_or(&null),
        body.get("messages").unwrap_or(&null),
    ]);

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, &model, "文本生成");
    filter_or_log(&state, &call, &preview, ENDPOINT).await?;

    let base_url = Some(resolve_base_url(&state, &headers));
    let deps = state.conv.clone();
    if is_stream(&body) {
        let rx = openai_v1_chat_complete::chat_complete_stream(deps, body, base_url);
        call.finish("", "success", None);
        Ok(sse_response(rx))
    } else {
        match openai_v1_chat_complete::chat_complete_once(deps, body, base_url).await {
            Ok(v) => {
                call.finish("", "success", None);
                Ok(json_ok(v))
            }
            Err(e) => Err(finish_err(&call, e, ENDPOINT)),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/responses
// ---------------------------------------------------------------------------

async fn create_response(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/responses";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();
    let model = str_field(&body, "model", "auto");

    let null = Value::Null;
    let preview = ContentFilter::request_text(&[
        body.get("input").unwrap_or(&null),
        body.get("instructions").unwrap_or(&null),
    ]);

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, &model, "Responses");
    filter_or_log(&state, &call, &preview, ENDPOINT).await?;

    let base_url = Some(resolve_base_url(&state, &headers));
    let deps = state.conv.clone();
    if is_stream(&body) {
        let rx = openai_v1_response::response_stream(deps, body, base_url);
        call.finish("", "success", None);
        Ok(sse_response(rx))
    } else {
        match openai_v1_response::response_once(deps, body, base_url).await {
            Ok(v) => {
                call.finish("", "success", None);
                Ok(json_ok(v))
            }
            Err(e) => Err(finish_err(&call, e, ENDPOINT)),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/messages  (Anthropic; also accepts x-api-key)
// ---------------------------------------------------------------------------

async fn create_message(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/messages";
    let auth = authorization(&headers)
        .or_else(|| header_str(&headers, "x-api-key").map(|key| format!("Bearer {key}")));
    let identity = require_identity(&state, auth.as_deref())?;
    let identity_value = identity.to_json();
    let model = str_field(&body, "model", "auto");

    let null = Value::Null;
    let preview = ContentFilter::request_text(&[
        body.get("system").unwrap_or(&null),
        body.get("messages").unwrap_or(&null),
        body.get("tools").unwrap_or(&null),
    ]);

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, &model, "Messages");
    filter_or_log(&state, &call, &preview, ENDPOINT).await?;

    let base_url = Some(resolve_base_url(&state, &headers));
    let deps = state.conv.clone();
    if is_stream(&body) {
        let rx = anthropic_v1_messages::messages_stream(deps, body, base_url);
        call.finish("", "success", None);
        Ok(sse_response(rx))
    } else {
        match anthropic_v1_messages::messages_once(deps, body, base_url).await {
            Ok(v) => {
                call.finish("", "success", None);
                Ok(json_ok(v))
            }
            Err(e) => Err(finish_err(&call, e, ENDPOINT)),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/search
// ---------------------------------------------------------------------------

async fn search(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/search";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();

    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if prompt.trim().is_empty() {
        return Err(AppError::bad_request("prompt is required").with_path(ENDPOINT));
    }

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, openai_search::MODEL, "搜索");
    filter_or_log(&state, &call, &prompt, ENDPOINT).await?;

    let base_url = Some(resolve_base_url(&state, &headers));
    match openai_search::search(state.conv.clone(), body, base_url).await {
        Ok(v) => {
            call.finish("", "success", None);
            Ok(json_ok(v))
        }
        Err(e) => Err(finish_err(&call, e, ENDPOINT)),
    }
}

// ---------------------------------------------------------------------------
// /v1/editable-file-tasks  +  /files/{*path}
// ---------------------------------------------------------------------------

async fn list_editable_file_tasks(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let task_ids: Vec<String> = query
        .get("ids")
        .map(|s| s.as_str())
        .unwrap_or("")
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    let result = state.editable_tasks.list_tasks(&identity.to_json(), &task_ids);
    Ok(json_ok(result))
}

async fn download_editable_file(
    State(state): State<SharedState>,
    Path(file_path): Path<String>,
) -> Result<Response, AppError> {
    let path = state
        .editable_tasks
        .public_file_path(&file_path)
        .ok_or_else(|| AppError::not_found("file not found"))?;
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|_| AppError::not_found("file not found"))?;
    let mime = mime_guess::from_path(&path).first_or_octet_stream().to_string();
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download")
        .to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{filename}\""))
        .body(Body::from(bytes))
        .map_err(|e| AppError::internal(e.to_string()))
}

// ---------------------------------------------------------------------------
// /v1/ppt/generations  +  /v1/psd/generations
// ---------------------------------------------------------------------------

fn editable_task_args(body: &Value) -> (String, Vec<String>, String) {
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let base64_images: Vec<String> = body
        .get("base64_images")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let client_task_id = body.get("client_task_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    (prompt, base64_images, client_task_id)
}

async fn create_ppt_task(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/ppt/generations";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();
    let (prompt, base64_images, client_task_id) = editable_task_args(&body);

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, "gpt-5-5-thinking", "PPT生成任务");
    filter_or_log(&state, &call, &prompt, ENDPOINT).await?;

    let base_url = resolve_base_url(&state, &headers);
    let result = state
        .editable_tasks
        .submit_ppt(&identity_value, &client_task_id, &prompt, &base64_images, &base_url);
    call.finish("", "success", None);
    Ok(json_ok(result))
}

async fn create_psd_task(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    const ENDPOINT: &str = "/v1/psd/generations";
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let identity_value = identity.to_json();
    let (prompt, base64_images, client_task_id) = editable_task_args(&body);

    let call = LoggedCall::new(state.log.clone(), &identity_value, ENDPOINT, "gpt-5-5-thinking", "PSD生成任务");
    filter_or_log(&state, &call, &prompt, ENDPOINT).await?;

    let base_url = resolve_base_url(&state, &headers);
    let result = state
        .editable_tasks
        .submit_psd(&identity_value, &client_task_id, &prompt, &base64_images, &base_url);
    call.finish("", "success", None);
    Ok(json_ok(result))
}
