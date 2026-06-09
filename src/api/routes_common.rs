//! Shared helpers for the axum route layer: SSE streaming responses, header
//! extraction, and a lightweight logged-call wrapper (port of the `LoggedCall`
//! behaviour in `log_service.py`).

use std::convert::Infallible;

use axum::body::Body;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::services::log_service::{LogService, LOG_TYPE_CALL};

/// Read the `Authorization` header as `&str`.
pub fn authorization(headers: &HeaderMap) -> Option<String> {
    headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).map(String::from)
}

/// Read a header by name.
pub fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name).and_then(|v| v.to_str().ok()).map(String::from)
}

/// Derive the request scheme (best-effort; honours `x-forwarded-proto`).
pub fn scheme(headers: &HeaderMap) -> String {
    header_str(headers, "x-forwarded-proto").unwrap_or_else(|| "http".to_string())
}

/// Build a `text/event-stream` response from a channel of ready SSE frames.
pub fn sse_response(rx: tokio::sync::mpsc::Receiver<String>) -> Response {
    let stream = ReceiverStream::new(rx).map(|frame| Ok::<_, Infallible>(frame.into_bytes()));
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap()
}

/// A JSON 200 response.
pub fn json_ok(value: Value) -> Response {
    Json(value).into_response()
}

/// Lightweight equivalent of the Python `LoggedCall`: records a `call` log entry
/// with key/endpoint/model/status/duration. Construct, then call `finish`.
pub struct LoggedCall {
    log: LogService,
    identity: Value,
    endpoint: String,
    model: String,
    summary: String,
    started: std::time::Instant,
}

impl LoggedCall {
    pub fn new(
        log: LogService,
        identity: &Value,
        endpoint: &str,
        model: &str,
        summary: &str,
    ) -> Self {
        Self {
            log,
            identity: identity.clone(),
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            summary: summary.to_string(),
            started: std::time::Instant::now(),
        }
    }

    fn base_detail(&self, status: &str) -> Value {
        serde_json::json!({
            "key_id": self.identity.get("id"),
            "key_name": self.identity.get("name"),
            "role": self.identity.get("role"),
            "endpoint": self.endpoint,
            "model": self.model,
            "status": status,
            "duration_ms": self.started.elapsed().as_millis() as i64,
        })
    }

    /// Log a completion/failure with the given status and optional extra detail.
    pub fn finish(&self, summary: &str, status: &str, extra: Option<Value>) {
        let mut detail = self.base_detail(status);
        if let (Some(obj), Some(Value::Object(extra))) = (detail.as_object_mut(), extra) {
            for (k, v) in extra {
                obj.insert(k, v);
            }
        }
        let summary = if summary.is_empty() { &self.summary } else { summary };
        self.log.add(LOG_TYPE_CALL, summary, detail);
    }
}
