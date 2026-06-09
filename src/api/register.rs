//! Port of `api/register.py` — the registration-orchestrator routes mounted under
//! `/api/register`. Every route is **admin-only** ([`require_admin`]); the work is
//! delegated to [`RegisterService`](crate::services::register_service).
//!
//! Routes (matching the Python original exactly):
//!   * `GET  /api/register`        — current config + recent logs
//!   * `POST /api/register`        — patch the config
//!   * `POST /api/register/start`  — start (or re-enable) the run
//!   * `POST /api/register/stop`   — request a cooperative stop
//!   * `POST /api/register/reset`  — clear logs/stats (when idle)
//!   * `GET  /api/register/events` — SSE stream of the config snapshot (auth via
//!     `?token=...`, since `EventSource` cannot send an `Authorization` header)
//!
//! Each JSON response is wrapped as `{ "register": <snapshot> }`, matching Python.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Map, Value};

use crate::api::routes_common::{authorization, json_ok, sse_response};
use crate::api::support::require_admin;
use crate::error::AppError;
use crate::state::SharedState;

/// Fields accepted by the config-patch endpoint (port of `RegisterConfigRequest`,
/// `model_dump(exclude_none=True)`): only these keys, and only when non-null.
const REGISTER_FIELDS: [&str; 8] = [
    "mail",
    "proxy",
    "total",
    "threads",
    "mode",
    "target_quota",
    "target_available",
    "check_interval",
];

pub fn router() -> Router<SharedState> {
    Router::new()
        .route(
            "/api/register",
            get(get_register_config).post(update_register_config),
        )
        .route("/api/register/start", post(start_register))
        .route("/api/register/stop", post(stop_register))
        .route("/api/register/reset", post(reset_register))
        .route("/api/register/events", get(register_events))
}

/// Keep only the whitelisted, non-null fields from the request body.
fn sanitize_updates(body: Value) -> Value {
    let mut out = Map::new();
    if let Some(obj) = body.as_object() {
        for key in REGISTER_FIELDS {
            if let Some(value) = obj.get(key) {
                if !value.is_null() {
                    out.insert(key.to_string(), value.clone());
                }
            }
        }
    }
    Value::Object(out)
}

async fn get_register_config(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({ "register": state.register.get() })))
}

async fn update_register_config(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let updates = sanitize_updates(body.map(|Json(v)| v).unwrap_or_else(|| json!({})));
    Ok(json_ok(json!({ "register": state.register.update(updates) })))
}

async fn start_register(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({ "register": state.register.start().await })))
}

async fn stop_register(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({ "register": state.register.stop() })))
}

async fn reset_register(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({ "register": state.register.reset() })))
}

/// SSE stream of the register snapshot. Auth comes from the `token` query param
/// (treated as a Bearer token), mirroring the Python `require_admin(f"Bearer {token}")`.
async fn register_events(
    State(state): State<SharedState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    let token = params.get("token").cloned().unwrap_or_default();
    require_admin(&state, Some(&format!("Bearer {token}")))?;

    let (tx, rx) = tokio::sync::mpsc::channel::<String>(8);
    let register = state.register.clone();
    tokio::spawn(async move {
        let mut last = String::new();
        loop {
            let payload = serde_json::to_string(&register.get()).unwrap_or_default();
            if payload != last {
                last = payload.clone();
                if tx.send(format!("data: {payload}\n\n")).await.is_err() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });
    Ok(sse_response(rx))
}
