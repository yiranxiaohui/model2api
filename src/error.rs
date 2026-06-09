//! Port of `api/errors.py` + `services/protocol/error_response.py`.
//!
//! `AppError` is the unified error type returned by handlers. It carries an
//! HTTP status and a `detail` payload (string or JSON). When rendered, the
//! request path decides whether to emit an OpenAI-compatible, Anthropic-
//! compatible, or plain `{"detail": ...}` body — matching FastAPI's exception
//! handlers.

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct AppError {
    pub status: StatusCode,
    /// Either a plain message or a structured detail (e.g. `{"error": "..."}`).
    pub detail: Value,
    pub headers: Vec<(String, String)>,
    /// Optional request path override used to pick the error envelope.
    pub path: Option<String>,
}

impl AppError {
    pub fn new(status: StatusCode, detail: impl Into<Value>) -> Self {
        Self {
            status,
            detail: detail.into(),
            headers: Vec::new(),
            path: None,
        }
    }

    pub fn message(status: StatusCode, msg: impl Into<String>) -> Self {
        Self::new(status, json!({ "error": msg.into() }))
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::message(StatusCode::BAD_REQUEST, msg)
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self::message(StatusCode::UNAUTHORIZED, msg)
    }

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self::message(StatusCode::FORBIDDEN, msg)
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::message(StatusCode::NOT_FOUND, msg)
    }

    pub fn upstream(msg: impl Into<String>) -> Self {
        Self::message(StatusCode::BAD_GATEWAY, msg)
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::message(StatusCode::INTERNAL_SERVER_ERROR, msg)
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status, error_message_from_detail(&self.detail))
    }
}

impl std::error::Error for AppError {}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::internal(e.to_string())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status;
        let path = self.path.clone().unwrap_or_default();
        let body = if is_anthropic_messages_path(&path) {
            anthropic_error_payload(&self.detail, status.as_u16())
        } else if is_openai_compatible_path(&path) {
            openai_error_payload(&self.detail, status.as_u16())
        } else {
            json!({ "detail": self.detail })
        };

        let mut headers = HeaderMap::new();
        for (k, v) in &self.headers {
            if let (Ok(name), Ok(val)) = (
                axum::http::HeaderName::try_from(k.as_str()),
                axum::http::HeaderValue::try_from(v.as_str()),
            ) {
                headers.insert(name, val);
            }
        }
        (status, headers, axum::Json(body)).into_response()
    }
}

pub fn is_openai_compatible_path(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

pub fn is_anthropic_messages_path(path: &str) -> bool {
    path == "/v1/messages"
}

fn message_from_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(map) => {
            if let Some(Value::String(m)) = map.get("message") {
                if !m.is_empty() {
                    return m.clone();
                }
            }
            map.get("error")
                .map(message_from_value)
                .unwrap_or_default()
        }
        _ => String::new(),
    }
}

pub fn error_message_from_detail(detail: &Value) -> String {
    match detail {
        Value::Array(items) => {
            let mut messages = Vec::new();
            for item in items {
                let Some(obj) = item.as_object() else { continue };
                let location = obj
                    .get("loc")
                    .and_then(|v| v.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| {
                                let s = match p {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };
                                if s == "body" {
                                    None
                                } else {
                                    Some(s)
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(".")
                    })
                    .unwrap_or_default();
                let message = obj
                    .get("msg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !location.is_empty() && !message.is_empty() {
                    messages.push(format!("{location}: {message}"));
                } else if !message.is_empty() {
                    messages.push(message);
                }
            }
            messages.join("; ")
        }
        Value::Object(map) => {
            let from_error = map
                .get("error")
                .map(message_from_value)
                .unwrap_or_default();
            if !from_error.is_empty() {
                from_error
            } else {
                message_from_value(detail)
            }
        }
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn default_error_type(status_code: u16) -> &'static str {
    match status_code {
        401 => "authentication_error",
        403 => "permission_error",
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        _ => "server_error",
    }
}

fn default_error_code(status_code: u16) -> &'static str {
    match status_code {
        401 => "invalid_api_key",
        403 => "permission_denied",
        429 => "rate_limit_exceeded",
        400..=499 => "bad_request",
        _ => "upstream_error",
    }
}

pub fn openai_error_payload(detail: &Value, status_code: u16) -> Value {
    if let Some(Value::Object(error_detail)) = detail.as_object().and_then(|m| m.get("error")) {
        let msg = {
            let m = error_message_from_detail(&Value::Object(error_detail.clone()));
            if m.is_empty() {
                "request failed".to_string()
            } else {
                m
            }
        };
        return json!({
            "error": {
                "message": msg,
                "type": error_detail.get("type").and_then(|v| v.as_str())
                    .unwrap_or(default_error_type(status_code)),
                "param": error_detail.get("param").cloned().unwrap_or(Value::Null),
                "code": error_detail.get("code").cloned()
                    .unwrap_or_else(|| json!(default_error_code(status_code))),
            }
        });
    }
    let msg = {
        let m = error_message_from_detail(detail);
        if m.is_empty() {
            "request failed".to_string()
        } else {
            m
        }
    };
    json!({
        "error": {
            "message": msg,
            "type": default_error_type(status_code),
            "param": Value::Null,
            "code": default_error_code(status_code),
        }
    })
}

pub fn anthropic_error_payload(detail: &Value, status_code: u16) -> Value {
    let error_type = if status_code >= 500 {
        "api_error"
    } else {
        default_error_type(status_code)
    };
    let msg = {
        let m = error_message_from_detail(detail);
        if m.is_empty() {
            "request failed".to_string()
        } else {
            m
        }
    };
    json!({
        "type": "error",
        "error": { "type": error_type, "message": msg },
    })
}

pub type AppResult<T> = Result<T, AppError>;
