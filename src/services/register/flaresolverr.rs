//! Minimal FlareSolverr v1 client. FlareSolverr drives a real headless browser
//! that solves the Cloudflare JS/Turnstile challenge for a URL and returns the
//! resulting cookies plus the exact User-Agent its browser used. Both are needed
//! to *reuse* the `cf_clearance` cookie on our own HTTP client: the cookie is
//! bound to the (egress IP, User-Agent) pair, so the register flow must send the
//! same UA and egress from the same IP (i.e. the same proxy) as the solve call.

use std::time::Duration;

use serde_json::{json, Value};

/// A cookie returned by FlareSolverr's browser.
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: String,
}

/// The useful parts of a FlareSolverr `request.get` solution.
pub struct Solution {
    pub cookies: Vec<Cookie>,
    pub user_agent: String,
    pub status: i64,
}

/// POST `{cmd: request.get}` to a FlareSolverr `/v1` endpoint and return the
/// solved cookies + User-Agent. `proxy` (when non-empty) is forwarded so the
/// browser egresses from the same IP the register flow will use.
pub async fn solve(
    endpoint: &str,
    target_url: &str,
    proxy: &str,
    max_timeout_ms: u64,
) -> Result<Solution, String> {
    let mut payload = json!({
        "cmd": "request.get",
        "url": target_url,
        "maxTimeout": max_timeout_ms,
    });
    if !proxy.trim().is_empty() {
        payload["proxy"] = json!({ "url": proxy.trim() });
    }

    let client = wreq::Client::builder()
        .build()
        .map_err(|e| format!("flaresolverr client build: {e}"))?;

    // Give the HTTP call headroom over FlareSolverr's own browser timeout.
    let resp = client
        .post(endpoint)
        .json(&payload)
        .timeout(Duration::from_millis(max_timeout_ms + 15_000))
        .send()
        .await
        .map_err(|e| format!("flaresolverr request: {e}"))?;

    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    let data: Value = serde_json::from_str(&text).map_err(|e| {
        let snippet: String = text.chars().take(300).collect();
        format!("flaresolverr bad json (http {status}): {e}; body={snippet}")
    })?;

    if data.get("status").and_then(|v| v.as_str()) != Some("ok") {
        let msg = data
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(format!("flaresolverr status!=ok: {msg}"));
    }

    let sol = data
        .get("solution")
        .ok_or_else(|| "flaresolverr: response has no solution".to_string())?;

    let user_agent = sol
        .get("userAgent")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let sol_status = sol.get("status").and_then(|v| v.as_i64()).unwrap_or(0);
    let cookies = sol
        .get("cookies")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    let name = c.get("name")?.as_str()?.to_string();
                    let value = c.get("value")?.as_str()?.to_string();
                    let domain = c
                        .get("domain")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    Some(Cookie {
                        name,
                        value,
                        domain,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(Solution {
        cookies,
        user_agent,
        status: sol_status,
    })
}
