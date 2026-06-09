//! Port of `utils/log.py` — thin wrapper mapping structured log calls to
//! `tracing`. The Python `logger` accepts dict payloads; here we accept any
//! `serde_json::Value` / `Display` and forward to tracing.

use serde_json::Value;

pub fn debug(payload: impl Into<Value>) {
    tracing::debug!("{}", render(payload.into()));
}

pub fn info(payload: impl Into<Value>) {
    tracing::info!("{}", render(payload.into()));
}

pub fn warning(payload: impl Into<Value>) {
    tracing::warn!("{}", render(payload.into()));
}

pub fn error(payload: impl Into<Value>) {
    tracing::error!("{}", render(payload.into()));
}

fn render(value: Value) -> String {
    match value {
        Value::String(s) => s,
        other => other.to_string(),
    }
}
