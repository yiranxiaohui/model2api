//! Port of `services/register/openai_register.py` — the OpenAI *platform*
//! account self-registration flow (authorization-code + PKCE).
//!
//! The flow, mirroring the Python `PlatformRegistrar`:
//!   1. `authorize` — hit `/api/accounts/authorize` with a freshly minted PKCE
//!      challenge + `login_hint` so the backend opens the signup screen.
//!   2. `register user` — POST the email/password to `/api/accounts/user/register`.
//!   3. `send OTP` — GET `/api/accounts/email-otp/send`.
//!   4. poll the temp-mailbox (via the [`MailProvider`] abstraction) for the
//!      6-digit code, then POST it to `/api/accounts/email-otp/validate`.
//!   5. `create_account` — POST name/birthdate; the response carries a
//!      `continue_url` whose `code` query param is the OAuth authorization code.
//!   6. exchange that code at `/api/accounts/oauth/token` for the
//!      `{access_token, refresh_token, id_token}` triple.
//!
//! Deviations from the Python original:
//! * The Sentinel proof-of-work header (`openai-sentinel-token`, produced by
//!   `build_sentinel_token`) is **deferred** — the full HTTP requirements
//!   handshake is not yet ported. Every site where Python attaches the header is
//!   marked with a `// TODO: sentinel token (deferred)` and the request proceeds
//!   without it (best-effort). See [`PlatformRegistrar::validate_otp`] etc.
//! * Threading / global stats / `account_service` persistence from the Python
//!   `worker` are out of scope here; [`register_one`] returns the result object
//!   and the caller persists it.
//! * The `oai-did` cookie Python pins on the `curl_cffi` session is installed in
//!   a [`wreq::cookie::Jar`] at client-build time; server `Set-Cookie`s persist
//!   in the same jar across the flow.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{SecondsFormat, Utc};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use serde_json::{json, Value};

use crate::config::Config;
use crate::services::register::constants::{
    common_headers, platform_oauth_redirect_uri, AUTH_BASE, PLATFORM_AUTH0_CLIENT, PLATFORM_BASE,
    PLATFORM_OAUTH_AUDIENCE, PLATFORM_OAUTH_CLIENT_ID, SEC_CH_UA, SEC_CH_UA_FULL_VERSION_LIST,
    USER_AGENT,
};
use crate::services::register::mail_provider;
use crate::utils::pkce::generate_pkce;

/// Default per-request timeout, in seconds (port of `default_timeout = 30`).
const DEFAULT_TIMEOUT: u64 = 30;

// ---------------------------------------------------------------------------
// Error type (mirrors Python's RuntimeError surface)
// ---------------------------------------------------------------------------

/// Error raised by the registration flow (port of Python `RuntimeError`).
#[derive(Debug, Clone)]
pub struct RegisterError(pub String);

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for RegisterError {}

fn rerr(msg: impl Into<String>) -> RegisterError {
    RegisterError(msg.into())
}

// ---------------------------------------------------------------------------
// Local header set (port of `navigate_headers`; `common_headers` lives in
// constants.rs and is imported above).
// ---------------------------------------------------------------------------

/// The `navigate_headers` dict (top-level document navigations).
fn navigate_headers() -> Vec<(&'static str, String)> {
    vec![
        (
            "accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
                .to_string(),
        ),
        ("accept-encoding", "gzip, deflate, br".to_string()),
        ("accept-language", "en-US,en;q=0.9".to_string()),
        ("cache-control", "max-age=0".to_string()),
        ("connection", "keep-alive".to_string()),
        ("dnt", "1".to_string()),
        ("sec-gpc", "1".to_string()),
        ("sec-ch-ua", SEC_CH_UA.to_string()),
        ("sec-ch-ua-arch", "\"x86_64\"".to_string()),
        ("sec-ch-ua-bitness", "\"64\"".to_string()),
        (
            "sec-ch-ua-full-version-list",
            SEC_CH_UA_FULL_VERSION_LIST.to_string(),
        ),
        ("sec-ch-ua-mobile", "?0".to_string()),
        ("sec-ch-ua-model", "\"\"".to_string()),
        ("sec-ch-ua-platform", "\"Windows\"".to_string()),
        ("sec-ch-ua-platform-version", "\"10.0.0\"".to_string()),
        ("sec-fetch-dest", "document".to_string()),
        ("sec-fetch-mode", "navigate".to_string()),
        ("sec-fetch-site", "same-origin".to_string()),
        ("sec-fetch-user", "?1".to_string()),
        ("upgrade-insecure-requests", "1".to_string()),
        ("user-agent", USER_AGENT.to_string()),
    ]
}

// ---------------------------------------------------------------------------
// Logging helpers (port of `log` / `step`)
// ---------------------------------------------------------------------------

fn step(index: i64, text: &str) {
    tracing::info!("[任务{index}] {text}");
}

fn step_warn(index: i64, text: &str) {
    tracing::warn!("[任务{index}] {text}");
}

// ---------------------------------------------------------------------------
// Random identity helpers (port of `_random_password` / `_random_name` /
// `_random_birthdate` / trace + token helpers)
// ---------------------------------------------------------------------------

const ASCII_UPPER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ASCII_LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const DIGITS: &[u8] = b"0123456789";
const SPECIALS: &[u8] = b"!@#$%";

fn rand_byte(chars: &[u8]) -> char {
    let mut rng = rand::thread_rng();
    chars[rng.gen_range(0..chars.len())] as char
}

/// Port of `_random_password`. Guarantees one upper, one lower, one digit and
/// one special character, then fills + shuffles.
fn random_password(length: usize) -> String {
    // chars = ascii_letters + digits + "!@#$%"
    let mut pool: Vec<u8> = Vec::new();
    pool.extend_from_slice(ASCII_UPPER);
    pool.extend_from_slice(ASCII_LOWER);
    pool.extend_from_slice(DIGITS);
    pool.extend_from_slice(SPECIALS);

    let mut value: Vec<char> = vec![
        rand_byte(ASCII_UPPER),
        rand_byte(ASCII_LOWER),
        rand_byte(DIGITS),
        rand_byte(SPECIALS),
    ];
    let remaining = length.saturating_sub(4);
    for _ in 0..remaining {
        value.push(rand_byte(&pool));
    }
    value.shuffle(&mut rand::thread_rng());
    value.into_iter().collect()
}

/// Port of `_random_name`.
fn random_name() -> (String, String) {
    let first = [
        "James", "Robert", "John", "Michael", "David", "Mary", "Emma", "Olivia",
    ];
    let last = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    ];
    let mut rng = rand::thread_rng();
    (
        first.choose(&mut rng).unwrap().to_string(),
        last.choose(&mut rng).unwrap().to_string(),
    )
}

/// Port of `_random_birthdate`.
fn random_birthdate() -> String {
    let mut rng = rand::thread_rng();
    format!(
        "{:04}-{:02}-{:02}",
        rng.gen_range(1996..=2006),
        rng.gen_range(1..=12),
        rng.gen_range(1..=28)
    )
}

/// `secrets.token_urlsafe(n)` — `n` random bytes, base64url-encoded, unpadded.
fn token_urlsafe(n: usize) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    let mut bytes = vec![0u8; n];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Port of `_make_trace_headers` (Datadog RUM tracing headers).
fn make_trace_headers() -> Vec<(String, String)> {
    let mut rng = rand::thread_rng();
    let trace_id: u64 = rng.gen();
    let parent_id: u64 = rng.gen();
    vec![
        (
            "traceparent".to_string(),
            format!("00-{}-{:016x}-01", uuid::Uuid::new_v4().simple(), parent_id),
        ),
        ("tracestate".to_string(), "dd=s:1;o:rum".to_string()),
        ("x-datadog-origin".to_string(), "rum".to_string()),
        ("x-datadog-parent-id".to_string(), parent_id.to_string()),
        ("x-datadog-sampling-priority".to_string(), "1".to_string()),
        ("x-datadog-trace-id".to_string(), trace_id.to_string()),
    ]
}

// ---------------------------------------------------------------------------
// HTTP response wrapper (port of the `_response_*` helpers)
// ---------------------------------------------------------------------------

/// Captured HTTP response (status + lowercased headers + body text).
struct Resp {
    status: u16,
    url: String,
    headers: HashMap<String, String>,
    text: String,
}

impl Resp {
    /// Port of `_response_json` — returns a JSON object, or `{}`.
    fn json(&self) -> Value {
        match serde_json::from_str::<Value>(&self.text) {
            Ok(v) if v.is_object() => v,
            _ => json!({}),
        }
    }

    fn header(&self, key: &str) -> &str {
        self.headers.get(key).map(|s| s.as_str()).unwrap_or("")
    }
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Port of `_response_debug_detail`.
fn response_debug_detail(resp: Option<&Resp>, limit: usize) -> String {
    let Some(resp) = resp else {
        return String::new();
    };
    let data = resp.json();
    let mut parts = vec![
        format!("url={}", truncate(&resp.url, 300)),
        format!("content_type={}", resp.header("content-type")),
    ];
    for key in ["cf-ray", "x-request-id", "openai-processing-ms"] {
        let value = resp.header(key).trim();
        if !value.is_empty() {
            parts.push(format!("{key}={value}"));
        }
    }
    if data.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
        parts.push(format!(
            "json={}",
            truncate(&serde_json::to_string(&data).unwrap_or_default(), limit)
        ));
    } else {
        parts.push(format!("body={}", truncate(&resp.text, limit)));
    }
    parts.join(", ")
}

/// Port of `_is_cloudflare_challenge`.
fn is_cloudflare_challenge(resp: Option<&Resp>) -> bool {
    let Some(resp) = resp else {
        return false;
    };
    let text = resp.text.to_lowercase();
    let server = resp.header("server").to_lowercase();
    server.contains("cloudflare")
        || text.contains("challenges.cloudflare.com")
        || text.contains("<title>just a moment")
}

/// Port of `extract_oauth_callback_params_from_url` (the `code` is all we need).
fn extract_callback_code(url: &str) -> Option<String> {
    if url.is_empty() {
        return None;
    }
    let parsed = wreq::Url::parse(url).ok()?;
    let mut code = String::new();
    for (k, v) in parsed.query_pairs() {
        if k.as_ref() == "code" {
            code = v.trim().to_string();
            break;
        }
    }
    if code.is_empty() {
        None
    } else {
        Some(code)
    }
}

fn header_map(headers: &[(String, String)]) -> wreq::header::HeaderMap {
    let mut hm = wreq::header::HeaderMap::new();
    for (k, v) in headers {
        if let (Ok(name), Ok(val)) = (
            wreq::header::HeaderName::from_bytes(k.as_bytes()),
            wreq::header::HeaderValue::from_str(v),
        ) {
            hm.insert(name, val);
        }
    }
    hm
}

// ---------------------------------------------------------------------------
// PlatformRegistrar (port of the Python class)
// ---------------------------------------------------------------------------

/// Drives a single account registration (port of `PlatformRegistrar`).
struct PlatformRegistrar {
    client: wreq::Client,
    device_id: String,
    code_verifier: String,
    platform_auth_code: String,
}

impl PlatformRegistrar {
    /// Port of `create_session` + `__init__`: build a Chrome-emulated `wreq`
    /// client, pin the `oai-did` cookie, honor the configured proxy.
    fn new(config: &Config, proxy_override: &str) -> Result<Self, RegisterError> {
        let device_id = uuid::Uuid::new_v4().to_string();

        let jar = Arc::new(wreq::cookie::Jar::default());
        // Python: session.cookies.set("oai-did", device_id, domain=".auth.openai.com")
        if let Ok(url) = wreq::Url::parse(AUTH_BASE) {
            jar.add_cookie_str(
                &format!("oai-did={device_id}; Domain=.auth.openai.com; Path=/"),
                &url,
            );
        }

        let mut builder = wreq::Client::builder()
            .emulation(wreq_util::Emulation::Chrome137)
            .cert_verification(false)
            .cookie_provider(jar);

        // Python `worker` builds the session from the register config's own
        // `proxy` (config["proxy"]); fall back to the global config proxy only
        // when the register proxy is empty.
        let proxy = if proxy_override.trim().is_empty() {
            config.proxy_setting()
        } else {
            proxy_override.trim().to_string()
        };
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        let client = builder
            .build()
            .map_err(|e| rerr(format!("register client build: {e}")))?;

        Ok(Self {
            client,
            device_id,
            code_verifier: String::new(),
            platform_auth_code: String::new(),
        })
    }

    fn navigate_headers(&self, referer: &str) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = navigate_headers()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        if !referer.is_empty() {
            headers.push(("referer".to_string(), referer.to_string()));
        }
        headers
    }

    fn json_headers(&self, referer: &str) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = common_headers()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        headers.push(("referer".to_string(), referer.to_string()));
        headers.push(("oai-device-id".to_string(), self.device_id.clone()));
        headers.extend(make_trace_headers());
        headers
    }

    /// One HTTP round-trip. Returns `Ok(Resp)` for any HTTP response (regardless
    /// of status) and `Err(detail)` only on transport failure.
    async fn send_once(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        params: &[(&str, String)],
        body: Option<&Value>,
        timeout_secs: u64,
    ) -> Result<Resp, String> {
        let m = wreq::Method::from_bytes(method.to_ascii_uppercase().as_bytes())
            .map_err(|e| format!("invalid HTTP method {method}: {e}"))?;
        let mut req = self.client.request(m, url).headers(header_map(headers));
        if !params.is_empty() {
            req = req.query(params);
        }
        if let Some(b) = body {
            req = req.json(b);
        }
        // Raw request trace (enable with RUST_LOG=model2api=debug). Captures what
        // we actually send so a failure can be diagnosed independently of the
        // `is_cloudflare_challenge` heuristic.
        let ua = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("user-agent"))
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        tracing::debug!(
            target: "model2api::register",
            method = %method.to_ascii_uppercase(),
            %url,
            params = params.len(),
            has_body = body.is_some(),
            timeout_secs,
            ua,
            "register → request"
        );
        let resp = req
            .timeout(Duration::from_secs(timeout_secs))
            .send()
            .await
            .map_err(|e| {
                tracing::debug!(
                    target: "model2api::register",
                    %url,
                    error = %e,
                    "register ✗ transport error"
                );
                format!("网络请求异常: {e}")
            })?;
        let status = resp.status().as_u16();
        let url = resp.url().to_string();
        let mut hm = HashMap::new();
        for (k, v) in resp.headers().iter() {
            hm.insert(k.as_str().to_lowercase(), v.to_str().unwrap_or("").to_string());
        }
        let text = resp.text().await.unwrap_or_default();
        // Raw response trace: status, final URL, every response header, and a
        // body snippet — the ground truth behind whatever message we surface.
        if tracing::enabled!(target: "model2api::register", tracing::Level::DEBUG) {
            let mut header_keys: Vec<&String> = hm.keys().collect();
            header_keys.sort();
            let header_dump = header_keys
                .iter()
                .map(|k| format!("{k}: {}", hm[*k]))
                .collect::<Vec<_>>()
                .join(" | ");
            tracing::debug!(
                target: "model2api::register",
                status,
                final_url = %url,
                body_len = text.len(),
                headers = %header_dump,
                body = %truncate(&text, 1500),
                "register ← response"
            );
        }
        Ok(Resp {
            status,
            url,
            headers: hm,
            text,
        })
    }

    /// Port of `request_with_local_retry`: retry transport errors up to
    /// `retry_attempts` times with a 1s backoff. HTTP error statuses are *not*
    /// retried (returned as `Ok`).
    async fn request_with_retry(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        params: &[(&str, String)],
        body: Option<&Value>,
        timeout_secs: u64,
        retry_attempts: u32,
    ) -> (Option<Resp>, String) {
        let mut last_error = String::new();
        let attempts = retry_attempts.max(1);
        for attempt in 1..=attempts {
            match self
                .send_once(method, url, headers, params, body, timeout_secs)
                .await
            {
                Ok(resp) => return (Some(resp), String::new()),
                Err(e) => {
                    tracing::debug!(
                        target: "model2api::register",
                        %url,
                        attempt,
                        attempts,
                        error = %e,
                        "register ✗ attempt failed, retrying"
                    );
                    last_error = e;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
        (None, last_error)
    }

    /// Port of `_platform_authorize`.
    async fn platform_authorize(&mut self, email: &str, index: i64) -> Result<(), RegisterError> {
        step(index, "开始 platform authorize");
        let (code_verifier, code_challenge) = generate_pkce();
        self.code_verifier = code_verifier;

        let params: Vec<(&str, String)> = vec![
            ("issuer", AUTH_BASE.to_string()),
            ("client_id", PLATFORM_OAUTH_CLIENT_ID.to_string()),
            ("audience", PLATFORM_OAUTH_AUDIENCE.to_string()),
            ("redirect_uri", platform_oauth_redirect_uri()),
            ("device_id", self.device_id.clone()),
            ("screen_hint", "login_or_signup".to_string()),
            ("max_age", "0".to_string()),
            ("login_hint", email.to_string()),
            ("scope", "openid profile email offline_access".to_string()),
            ("response_type", "code".to_string()),
            ("response_mode", "query".to_string()),
            ("state", token_urlsafe(32)),
            ("nonce", token_urlsafe(32)),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256".to_string()),
            ("auth0Client", PLATFORM_AUTH0_CLIENT.to_string()),
        ];
        let headers = self.navigate_headers(&format!("{PLATFORM_BASE}/"));
        let (resp, error) = self
            .request_with_retry(
                "get",
                &format!("{AUTH_BASE}/api/accounts/authorize"),
                &headers,
                &params,
                None,
                DEFAULT_TIMEOUT,
                3,
            )
            .await;

        let ok = resp.as_ref().map(|r| r.status == 200).unwrap_or(false);
        if !ok {
            if is_cloudflare_challenge(resp.as_ref()) {
                // Surface the diagnostics instead of swallowing them: status code,
                // the `server` header and `cf-ray` (Cloudflare's request id, useful
                // for correlating the block) plus a body snippet.
                let status = resp
                    .as_ref()
                    .map(|r| r.status.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let server = resp.as_ref().map(|r| r.header("server")).unwrap_or("");
                let debug = response_debug_detail(resp.as_ref(), 800);
                return Err(rerr(format!(
                    "被 Cloudflare 拦截，请更换 IP 重试 (cf_block http_{status}, server={server}, {debug})"
                )));
            }
            let err_obj = resp
                .as_ref()
                .map(|r| r.json())
                .and_then(|d| d.get("error").cloned())
                .unwrap_or(Value::Null);
            let detail = if err_obj.is_object() {
                let code = err_obj.get("code").and_then(|v| v.as_str()).unwrap_or("");
                let message = err_obj.get("message").and_then(|v| v.as_str()).unwrap_or("");
                format!(": {code} - {message}")
                    .trim_matches(|c| c == ' ' || c == '-')
                    .to_string()
            } else {
                String::new()
            };
            let status = resp
                .as_ref()
                .map(|r| r.status.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let debug = response_debug_detail(resp.as_ref(), 800);
            let msg = if error.is_empty() {
                format!("platform_authorize_http_{status}{detail}, {debug}")
            } else {
                error
            };
            return Err(rerr(msg));
        }
        step(index, "platform authorize 完成");
        Ok(())
    }

    /// Port of `_register_user`.
    async fn register_user(
        &self,
        email: &str,
        password: &str,
        index: i64,
    ) -> Result<(), RegisterError> {
        step(index, "开始提交注册密码");
        let headers = self.json_headers(&format!("{AUTH_BASE}/create-account/password"));
        // TODO: sentinel token (deferred) — Python sets
        // headers["openai-sentinel-token"] = build_sentinel_token(.., "username_password_create").
        let payload = json!({"username": email, "password": password});
        let (resp, error) = self
            .request_with_retry(
                "post",
                &format!("{AUTH_BASE}/api/accounts/user/register"),
                &headers,
                &[],
                Some(&payload),
                DEFAULT_TIMEOUT,
                3,
            )
            .await;

        let ok = resp.as_ref().map(|r| r.status == 200).unwrap_or(false);
        if !ok {
            let data = resp.as_ref().map(|r| r.json()).unwrap_or(json!({}));
            if data.get("message").and_then(|v| v.as_str())
                == Some("Failed to create account. Please try again.")
            {
                step_warn(index, "注册失败提示: 邮箱域名很可能因滥用被封禁，请更换邮箱域名");
            }
            let detail = if data.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
                format!(", detail={}", serde_json::to_string(&data).unwrap_or_default())
            } else {
                String::new()
            };
            let status = resp
                .as_ref()
                .map(|r| r.status.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let msg = if error.is_empty() {
                format!("user_register_http_{status}{detail}")
            } else {
                error
            };
            return Err(rerr(msg));
        }
        step(index, "提交注册密码完成");
        Ok(())
    }

    /// Port of `_send_otp`.
    async fn send_otp(&self, index: i64) -> Result<(), RegisterError> {
        step(index, "开始发送验证码");
        let headers = self.navigate_headers(&format!("{AUTH_BASE}/create-account/password"));
        let (resp, error) = self
            .request_with_retry(
                "get",
                &format!("{AUTH_BASE}/api/accounts/email-otp/send"),
                &headers,
                &[],
                None,
                DEFAULT_TIMEOUT,
                3,
            )
            .await;
        let ok = resp
            .as_ref()
            .map(|r| r.status == 200 || r.status == 302)
            .unwrap_or(false);
        if !ok {
            let status = resp
                .as_ref()
                .map(|r| r.status.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let msg = if error.is_empty() {
                format!("send_otp_http_{status}")
            } else {
                error
            };
            return Err(rerr(msg));
        }
        step(index, "发送验证码完成");
        Ok(())
    }

    /// Port of the module-level `validate_otp`: first attempt without the
    /// Sentinel header; on non-200 the Python retries *with* a freshly minted
    /// sentinel token. That token is deferred, so the retry is a plain re-send.
    async fn validate_otp(&self, code: &str) -> (Option<Resp>, String) {
        let mut headers = common_headers()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<Vec<_>>();
        headers.push(("referer".to_string(), format!("{AUTH_BASE}/email-verification")));
        headers.push(("oai-device-id".to_string(), self.device_id.clone()));
        headers.extend(make_trace_headers());

        let payload = json!({"code": code});
        let url = format!("{AUTH_BASE}/api/accounts/email-otp/validate");
        let (resp, error) = self
            .request_with_retry("post", &url, &headers, &[], Some(&payload), DEFAULT_TIMEOUT, 3)
            .await;
        if resp.as_ref().map(|r| r.status == 200).unwrap_or(false) {
            return (resp, error);
        }
        // TODO: sentinel token (deferred) — Python sets
        // headers["openai-sentinel-token"] = build_sentinel_token(.., "authorize_continue")
        // before this second attempt.
        self.request_with_retry("post", &url, &headers, &[], Some(&payload), DEFAULT_TIMEOUT, 3)
            .await
    }

    /// Port of `_validate_otp`.
    async fn validate_otp_step(&self, code: &str, index: i64) -> Result<(), RegisterError> {
        step(index, &format!("开始校验验证码 {code}"));
        let (resp, error) = self.validate_otp(code).await;
        let ok = resp.as_ref().map(|r| r.status == 200).unwrap_or(false);
        if !ok {
            let body = resp
                .as_ref()
                .map(|r| truncate(&r.text, 500))
                .unwrap_or_default();
            let status = resp
                .as_ref()
                .map(|r| r.status.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let msg = if error.is_empty() {
                format!("validate_otp_http_{status}_body={body}")
            } else {
                error
            };
            return Err(rerr(msg));
        }
        step(index, "验证码校验完成");
        Ok(())
    }

    /// Port of `_create_account`.
    async fn create_account(
        &mut self,
        name: &str,
        birthdate: &str,
        index: i64,
    ) -> Result<(), RegisterError> {
        step(index, "开始创建账号资料");
        let headers = self.json_headers(&format!("{AUTH_BASE}/about-you"));
        // TODO: sentinel token (deferred) — Python sets
        // headers["openai-sentinel-token"] = build_sentinel_token(.., "oauth_create_account").
        let payload = json!({"name": name, "birthdate": birthdate});
        let (resp, error) = self
            .request_with_retry(
                "post",
                &format!("{AUTH_BASE}/api/accounts/create_account"),
                &headers,
                &[],
                Some(&payload),
                DEFAULT_TIMEOUT,
                3,
            )
            .await;

        let ok = resp
            .as_ref()
            .map(|r| r.status == 200 || r.status == 302)
            .unwrap_or(false);
        if !ok {
            let data = resp.as_ref().map(|r| r.json()).unwrap_or(json!({}));
            if data.get("message").and_then(|v| v.as_str())
                == Some("Failed to create account. Please try again.")
            {
                step_warn(index, "创建账号失败提示: 邮箱域名很可能因滥用被封禁，请更换邮箱域名");
            }
            let detail = if data.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
                format!(", detail={}", serde_json::to_string(&data).unwrap_or_default())
            } else {
                String::new()
            };
            let status = resp
                .as_ref()
                .map(|r| r.status.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let msg = if error.is_empty() {
                format!("create_account_http_{status}{detail}")
            } else {
                error
            };
            return Err(rerr(msg));
        }
        let data = resp.as_ref().map(|r| r.json()).unwrap_or(json!({}));
        let continue_url = data
            .get("continue_url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        self.platform_auth_code = extract_callback_code(&continue_url).unwrap_or_default();
        step(index, "创建账号资料完成");
        Ok(())
    }

    /// Port of `request_platform_oauth_token` + `_exchange_registered_tokens`.
    async fn exchange_registered_tokens(&self, index: i64) -> Result<Value, RegisterError> {
        step(index, "开始换 token");
        let headers: Vec<(String, String)> = vec![
            ("accept".to_string(), "*/*".to_string()),
            ("accept-language".to_string(), "zh-CN,zh;q=0.9".to_string()),
            ("auth0-client".to_string(), PLATFORM_AUTH0_CLIENT.to_string()),
            ("cache-control".to_string(), "no-cache".to_string()),
            ("content-type".to_string(), "application/json".to_string()),
            ("origin".to_string(), PLATFORM_BASE.to_string()),
            ("pragma".to_string(), "no-cache".to_string()),
            ("priority".to_string(), "u=1, i".to_string()),
            ("referer".to_string(), format!("{PLATFORM_BASE}/")),
            ("sec-ch-ua".to_string(), SEC_CH_UA.to_string()),
            ("sec-ch-ua-mobile".to_string(), "?0".to_string()),
            ("sec-ch-ua-platform".to_string(), "\"Windows\"".to_string()),
            ("sec-fetch-dest".to_string(), "empty".to_string()),
            ("sec-fetch-mode".to_string(), "cors".to_string()),
            ("sec-fetch-site".to_string(), "same-site".to_string()),
            ("user-agent".to_string(), USER_AGENT.to_string()),
        ];
        let payload = json!({
            "client_id": PLATFORM_OAUTH_CLIENT_ID,
            "code_verifier": self.code_verifier,
            "grant_type": "authorization_code",
            "code": self.platform_auth_code,
            "redirect_uri": platform_oauth_redirect_uri(),
        });
        let resp = self
            .send_once(
                "post",
                &format!("{AUTH_BASE}/api/accounts/oauth/token"),
                &headers,
                &[],
                Some(&payload),
                60,
            )
            .await
            .map_err(rerr)?;
        if resp.status != 200 {
            tracing::warn!("[register] oauth/token rejected: status={} body={}", resp.status, truncate(&resp.text, 300));
            return Err(rerr("token换取失败"));
        }
        step(index, "token 换取完成");
        Ok(resp.json())
    }
}

// ---------------------------------------------------------------------------
// Public API (port of `register` + the orchestration in `worker`)
// ---------------------------------------------------------------------------

/// Run a full registration against a freshly created temp-mailbox. Returns the
/// account result object (port of `PlatformRegistrar.register`).
///
/// `mail_config` is the `mail` section of the register config (the same shape
/// consumed by [`mail_provider::create_mailbox`] / [`mail_provider::wait_for_code`]).
async fn register(
    registrar: &mut PlatformRegistrar,
    config: &Config,
    mail_config: &Value,
    index: i64,
) -> Result<Value, RegisterError> {
    step(index, "开始创建邮箱");
    let mut mailbox = mail_provider::create_mailbox(mail_config, config, None)
        .await
        .map_err(|e| rerr(e.to_string()))?;
    let email = mailbox
        .get("address")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if email.is_empty() {
        return Err(rerr("邮箱服务未返回 address"));
    }
    let label = mailbox.get("label").and_then(|v| v.as_str()).unwrap_or("");
    step(index, &format!("邮箱创建完成[{label}]: {email}"));

    let password = random_password(16);
    let (first_name, last_name) = random_name();

    registrar.platform_authorize(&email, index).await?;
    registrar.register_user(&email, &password, index).await?;
    registrar.send_otp(index).await?;

    step(index, "开始等待注册验证码");
    let code = mail_provider::wait_for_code(mail_config, config, &mut mailbox)
        .await
        .map_err(|e| rerr(e.to_string()))?;
    let code = match code {
        Some(c) if !c.is_empty() => c,
        _ => return Err(rerr("等待注册验证码超时")),
    };
    step(index, &format!("收到注册验证码: {code}"));

    registrar.validate_otp_step(&code, index).await?;
    registrar
        .create_account(&format!("{first_name} {last_name}"), &random_birthdate(), index)
        .await?;
    let tokens = registrar.exchange_registered_tokens(index).await?;

    let access_token = tokens
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = tokens
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let id_token = tokens
        .get("id_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(json!({
        "email": email,
        "password": password,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "id_token": id_token,
        "source_type": "web",
        "created_at": Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false),
    }))
}

/// Register a single account end-to-end (port of the success/error envelope in
/// `worker`). On success returns
/// `{ok: true, index, email, password, access_token, refresh_token, id_token,
/// source_type, created_at}`; on failure `{ok: false, index, error}`.
///
/// Persisting the result (Python's `account_service.add_account_items` /
/// `refresh_accounts`) is left to the caller.
pub async fn register_one(config: &Config, mail_config: &Value, index: i64, proxy: &str) -> Value {
    let mut registrar = match PlatformRegistrar::new(config, proxy) {
        Ok(r) => r,
        Err(e) => return json!({"ok": false, "index": index, "error": e.to_string()}),
    };
    step(index, "任务启动");
    match register(&mut registrar, config, mail_config, index).await {
        Ok(result) => {
            let mut obj = result.as_object().cloned().unwrap_or_default();
            obj.insert("ok".to_string(), json!(true));
            obj.insert("index".to_string(), json!(index));
            tracing::info!(
                "[任务{index}] {} 注册成功",
                obj.get("email").and_then(|v| v.as_str()).unwrap_or("")
            );
            Value::Object(obj)
        }
        Err(e) => {
            tracing::error!("[任务{index}] 注册失败，原因: {e}");
            json!({"ok": false, "index": index, "error": e.to_string()})
        }
    }
}
