//! Port of `api/image_tasks.py` — the async image-task routes mounted under
//! `/api/image-tasks`. Clients submit text-to-image (`generations`) and
//! image-to-image (`edits`) tasks keyed by a caller-supplied `client_task_id`,
//! then poll task status via the list endpoint. All routes require a valid
//! identity ([`require_identity`]); the heavy lifting lives in
//! [`ImageTaskService`](crate::services::image_task_service).
//!
//! Routes (matching the Python original exactly):
//!   * `GET  /api/image-tasks`              — list tasks (`?ids=a,b,c`)
//!   * `POST /api/image-tasks/generations`  — submit a text-to-image task
//!   * `POST /api/image-tasks/edits`        — submit an image-to-image task
//!   * `POST /api/image-tasks/{task_id}/resume-poll` — see note below
//!
//! Adaptation notes:
//!   * `resume-poll` is **not supported**: the underlying
//!     `ImageTaskService::resume_poll` was not ported (it relied on private engine
//!     poll/download helpers — see `image_task_service.rs`). The route is kept for
//!     URL compatibility but returns `501 Not Implemented`.
//!   * Python's `filter_or_log` (run the content filter, log on rejection) is
//!     reproduced via [`filter_or_log`]; the success/failure call log is emitted
//!     inside the service when the task completes.
#![allow(dead_code)]

use std::collections::HashMap;

use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::api::image_inputs::parse_image_edit_request;
use crate::api::routes_common::{authorization, header_str, json_ok, scheme, LoggedCall};
use crate::api::support::{require_identity, resolve_image_base_url, Identity};
use crate::error::AppError;
use crate::state::SharedState;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/api/image-tasks", get(list_image_tasks))
        .route("/api/image-tasks/generations", post(create_generation_task))
        .route("/api/image-tasks/edits", post(create_edit_task))
        .route("/api/image-tasks/{task_id}/resume-poll", post(resume_image_poll))
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Port of `_parse_task_ids` — comma-split, trim, drop empties.
fn parse_task_ids(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

fn clean_field(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string().trim().to_string(),
    }
}

fn clean_or(value: Option<&Value>, default: &str) -> String {
    let v = clean_field(value);
    if v.is_empty() {
        default.to_string()
    } else {
        v
    }
}

fn base_url_for(state: &SharedState, headers: &HeaderMap) -> String {
    let host = header_str(headers, "host");
    resolve_image_base_url(state, host.as_deref(), &scheme(headers))
}

/// Port of `filter_or_log` — run the content filter; on rejection, write a
/// failed-call log entry and surface the error.
async fn filter_or_log(
    state: &SharedState,
    identity: &Identity,
    endpoint: &str,
    model: &str,
    summary: &str,
    text: &str,
) -> Result<(), AppError> {
    if let Err(err) = state.content_filter.check_request(text).await {
        let id = identity.to_json();
        let call = LoggedCall::new(state.log.clone(), &id, endpoint, model, summary);
        call.finish("调用失败", "failed", Some(json!({ "error": err.detail })));
        let status = StatusCode::from_u16(err.status_code).unwrap_or(StatusCode::BAD_REQUEST);
        return Err(AppError::new(status, err.detail_json()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// handlers
// ---------------------------------------------------------------------------

async fn list_image_tasks(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let ids = parse_task_ids(params.get("ids").map(String::as_str).unwrap_or(""));
    let result = state.image_tasks.list_tasks(&identity.to_json(), &ids);
    Ok(json_ok(result))
}

async fn create_generation_task(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    let identity = require_identity(&state, authorization(&headers).as_deref())?;

    let client_task_id = clean_field(body.get("client_task_id"));
    if client_task_id.is_empty() {
        return Err(AppError::bad_request("client_task_id is required"));
    }
    let prompt = clean_field(body.get("prompt"));
    if prompt.is_empty() {
        return Err(AppError::bad_request("prompt is required"));
    }
    let model = clean_or(body.get("model"), "gpt-image-2");
    let size = clean_field(body.get("size"));
    let quality = clean_or(body.get("quality"), "auto");
    let base_url = base_url_for(&state, &headers);

    filter_or_log(&state, &identity, "/api/image-tasks/generations", &model, "文生图任务", &prompt)
        .await?;

    let result = state
        .image_tasks
        .submit_generation(
            &identity.to_json(),
            &client_task_id,
            &prompt,
            &model,
            if size.is_empty() { None } else { Some(size.as_str()) },
            &quality,
            &base_url,
        )
        .await;
    Ok(json_ok(result))
}

async fn create_edit_task(
    State(state): State<SharedState>,
    request: Request,
) -> Result<Response, AppError> {
    let headers = request.headers().clone();
    let identity = require_identity(&state, authorization(&headers).as_deref())?;
    let base_url = base_url_for(&state, &headers);

    let (payload, images) = parse_image_edit_request(request).await?;

    let client_task_id = clean_field(payload.get("client_task_id"));
    if client_task_id.is_empty() {
        return Err(AppError::bad_request("client_task_id is required"));
    }
    let prompt = clean_field(payload.get("prompt"));
    let model = clean_or(payload.get("model"), "gpt-image-2");
    let size = clean_field(payload.get("size"));
    let quality = clean_or(payload.get("quality"), "auto");

    filter_or_log(&state, &identity, "/api/image-tasks/edits", &model, "图生图任务", &prompt).await?;

    let result = state
        .image_tasks
        .submit_edit(
            &identity.to_json(),
            &client_task_id,
            &prompt,
            &model,
            if size.is_empty() { None } else { Some(size.as_str()) },
            &quality,
            &base_url,
            images,
        )
        .await;
    Ok(json_ok(result))
}

/// `resume-poll` is not supported in this build (see module docs).
async fn resume_image_poll(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(task_id): Path<String>,
    _body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    // Still authenticate so unauthorized callers get a 401 rather than a 501.
    require_identity(&state, authorization(&headers).as_deref())?;
    let _ = task_id;
    Err(AppError::message(
        StatusCode::NOT_IMPLEMENTED,
        "resume-poll is not supported in this build",
    ))
}
