//! Global outbound-proxy helpers (port of `services/proxy_service.py`).
//! `build_session_kwargs` is inlined where clients are constructed; this module
//! ports `test_proxy`.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

fn is_valid_proxy_url(url: &str) -> bool {
    match wreq::Url::parse(url) {
        Ok(parsed) => {
            matches!(parsed.scheme(), "http" | "https" | "socks5" | "socks5h") && parsed.host().is_some()
        }
        Err(_) => false,
    }
}

/// Probe a proxy by hitting ChatGPT's CSRF endpoint; returns
/// `{ok, status, latency_ms, error}`.
pub async fn test_proxy(url: &str, timeout_secs: f64) -> Value {
    let candidate = url.trim();
    if candidate.is_empty() {
        return json!({"ok": false, "status": 0, "latency_ms": 0, "error": "proxy url is required"});
    }
    if !is_valid_proxy_url(candidate) {
        return json!({"ok": false, "status": 0, "latency_ms": 0, "error": "invalid proxy url"});
    }
    let proxy = match wreq::Proxy::all(candidate) {
        Ok(p) => p,
        Err(e) => return json!({"ok": false, "status": 0, "latency_ms": 0, "error": e.to_string()}),
    };
    let client = match wreq::Client::builder()
        .emulation(wreq_util::Emulation::Edge101)
        .proxy(proxy)
        .build()
    {
        Ok(c) => c,
        Err(e) => return json!({"ok": false, "status": 0, "latency_ms": 0, "error": e.to_string()}),
    };
    let started = Instant::now();
    let result = client
        .get("https://chatgpt.com/api/auth/csrf")
        .header("user-agent", "Mozilla/5.0 (chatgpt2api proxy test)")
        .timeout(Duration::from_secs_f64(timeout_secs))
        .send()
        .await;
    let latency_ms = started.elapsed().as_millis() as i64;
    match result {
        Ok(resp) => {
            let status = resp.status().as_u16();
            json!({
                "ok": status < 500,
                "status": status,
                "latency_ms": latency_ms,
                "error": if status < 500 { Value::Null } else { Value::String(format!("HTTP {status}")) },
            })
        }
        Err(e) => json!({"ok": false, "status": 0, "latency_ms": latency_ms, "error": e.to_string()}),
    }
}
