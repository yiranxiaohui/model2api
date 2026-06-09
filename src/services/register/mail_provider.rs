//! Port of `services/register/mail_provider.py` — temp-email provider
//! abstraction used by the account-registration flow.
//!
//! The Python module defines a `BaseMailProvider` base class plus a family of
//! concrete temp-mail backends (Cloudflare temp mail, DDG/DuckDuckGo forwarding,
//! CloudMailGen, TempMail.lol, DuckMail, GptMail, MoEmail, Inbucket, YYDS mail),
//! a factory that selects a provider from the `mail` config section, and the
//! high-level `create_mailbox` / `wait_for_code` / `get_existing_mailbox`
//! helpers. This is a faithful Rust port using `wreq` (async) for HTTP.
//!
//! Notable deviations from the Python original (see crate report):
//! * `curl_cffi`'s full MIME (`email.message_from_string`) parsing of a raw
//!   message is simplified — when only a raw RFC822 blob is present it is
//!   surfaced as the text body rather than walked for multipart parts.
//! * The Rust `regex` crate has no look-behind, so the bare 6-digit branch of
//!   `_extract_code` is emulated by a manual preceding-character check.
//! * The proxy is sourced from [`Config::proxy_setting`] (falling back to the
//!   `mail.proxy` config key) instead of only the mail config block.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::Rng;
use regex::Regex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;

// ---------------------------------------------------------------------------
// Error type (mirrors Python's RuntimeError surface)
// ---------------------------------------------------------------------------

/// Error raised by mail-provider operations (port of Python `RuntimeError`).
#[derive(Debug, Clone)]
pub struct MailError(pub String);

impl std::fmt::Display for MailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for MailError {}

fn merr(msg: impl Into<String>) -> MailError {
    MailError(msg.into())
}

/// Sentinel substring used to detect the DDG daily-limit soft error so the
/// orchestration loop can fall through to the next provider.
const DDG_LIMIT_SENTINEL: &str = "DDG日上限已达";

// ---------------------------------------------------------------------------
// Mail runtime configuration (port of `_config`)
// ---------------------------------------------------------------------------

/// Per-request runtime knobs derived from the `mail` config section.
#[derive(Clone, Debug)]
pub struct MailConf {
    pub request_timeout: f64,
    pub wait_timeout: f64,
    pub wait_interval: f64,
    pub user_agent: String,
    pub proxy: String,
}

fn as_f64_default(v: Option<&Value>, default: f64) -> f64 {
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(default),
        _ => default,
    }
}

/// Build [`MailConf`] from the `mail` config object (port of `_config`). The
/// proxy is taken from [`Config::proxy_setting`], falling back to the mail
/// config's own `proxy` field.
pub fn mail_conf(mail_config: &Value, config: &Config) -> MailConf {
    let rt = as_f64_default(mail_config.get("request_timeout"), 0.0);
    let wt = as_f64_default(mail_config.get("wait_timeout"), 0.0);
    let wi = as_f64_default(mail_config.get("wait_interval"), 0.0);
    let ua = {
        let s = estr(mail_config, "user_agent");
        if s.is_empty() { "Mozilla/5.0".to_string() } else { s }
    };
    let proxy = {
        let from_config = config.proxy_setting();
        if !from_config.trim().is_empty() {
            from_config.trim().to_string()
        } else {
            estr(mail_config, "proxy")
        }
    };
    MailConf {
        request_timeout: if rt > 0.0 { rt } else { 30.0 },
        wait_timeout: if wt > 0.0 { wt } else { 30.0 },
        wait_interval: if wi > 0.0 { wi } else { 2.0 },
        user_agent: ua,
        proxy,
    }
}

// ---------------------------------------------------------------------------
// Global rotation state + caches (port of module-level globals)
// ---------------------------------------------------------------------------

static DOMAIN_INDEX: AtomicUsize = AtomicUsize::new(0);
static PROVIDER_INDEX: AtomicUsize = AtomicUsize::new(0);

/// CloudMailGen admin-token cache: key -> (token, expiry_epoch_secs).
static CLOUDMAIL_TOKEN_CACHE: Lazy<Mutex<HashMap<String, (String, f64)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Serializes reads/writes of the DDG alias file.
static DDG_ALIASES_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

// ---------------------------------------------------------------------------
// Generic JSON / string helpers
// ---------------------------------------------------------------------------

/// Python-ish `str(value)` for a JSON value (`None`/`null` -> empty string).
fn vstring(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        _ => v.to_string(),
    }
}

/// Python truthiness for a JSON value.
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `str(entry.get(key) or "").strip()` for truthy values.
fn estr(entry: &Value, key: &str) -> String {
    match entry.get(key) {
        Some(v) if is_truthy(v) => vstring(v).trim().to_string(),
        _ => String::new(),
    }
}

/// [`estr`] with a default when the value is missing/empty.
fn estr_or(entry: &Value, key: &str, default: &str) -> String {
    let v = estr(entry, key);
    if v.is_empty() {
        default.to_string()
    } else {
        v
    }
}

/// `[str(x).strip() for x in (list) if str(x).strip()]` (list-typed only).
fn str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a
            .iter()
            .map(vstring)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// Port of `_normalize_string_list` (accepts list or scalar).
fn normalize_string_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a
            .iter()
            .map(vstring)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        other => {
            let t = other.map(vstring).unwrap_or_default().trim().to_string();
            if t.is_empty() {
                vec![]
            } else {
                vec![t]
            }
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn now_secs() -> f64 {
    Utc::now().timestamp_millis() as f64 / 1000.0
}

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.trim().parse::<f64>().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// First truthy value among `keys`.
fn first_truthy_value<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    for k in keys {
        if let Some(val) = v.get(*k) {
            if is_truthy(val) {
                return Some(val);
            }
        }
    }
    None
}

/// `str(first truthy among keys or "")`.
fn first_truthy_str(v: &Value, keys: &[&str]) -> String {
    first_truthy_value(v, keys).map(vstring).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Random identifiers (port of `_random_mailbox_name` / `_random_subdomain_label`)
// ---------------------------------------------------------------------------

const LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const DIGITS: &[u8] = b"0123456789";
const ALNUM_LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
const ALNUM: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

fn rand_from(chars: &[u8], n: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| chars[rng.gen_range(0..chars.len())] as char).collect()
}

fn random_mailbox_name() -> String {
    let d: usize = rand::thread_rng().gen_range(1..=3);
    let t: usize = rand::thread_rng().gen_range(1..=3);
    format!("{}{}{}", rand_from(LOWER, 5), rand_from(DIGITS, d), rand_from(LOWER, t))
}

fn random_subdomain_label() -> String {
    let n: usize = rand::thread_rng().gen_range(4..=10);
    rand_from(ALNUM_LOWER, n)
}

/// Round-robin pick across configured domains (port of `_next_domain`).
fn next_domain(domains: &[String]) -> Result<String, MailError> {
    let domains: Vec<String> = domains
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if domains.is_empty() {
        return Err(merr("mail.domain 不能为空"));
    }
    if domains.len() == 1 {
        return Ok(domains[0].clone());
    }
    let i = DOMAIN_INDEX.fetch_add(1, Ordering::Relaxed) % domains.len();
    Ok(domains[i].clone())
}

// ---------------------------------------------------------------------------
// Date parsing + message content/code extraction
// ---------------------------------------------------------------------------

/// Port of `_parse_received_at`. Returns a UTC datetime, or `None`.
fn parse_received_at(value: &Value) -> Option<DateTime<Utc>> {
    match value {
        Value::Number(n) => {
            let f = n.as_f64()?;
            let secs = f.trunc() as i64;
            let nanos = ((f - f.trunc()) * 1e9).round() as u32;
            Utc.timestamp_opt(secs, nanos).single()
        }
        Value::String(s) => {
            let text = s.trim();
            if text.is_empty() {
                return None;
            }
            let normalized = if text.ends_with('Z') {
                format!("{}+00:00", &text[..text.len() - 1])
            } else {
                text.to_string()
            };
            if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
                return Some(dt.with_timezone(&Utc));
            }
            for fmt in [
                "%Y-%m-%dT%H:%M:%S%.f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S%.f",
                "%Y-%m-%d %H:%M:%S",
            ] {
                if let Ok(ndt) = NaiveDateTime::parse_from_str(text, fmt) {
                    return Some(Utc.from_utc_datetime(&ndt));
                }
            }
            if let Ok(dt) = DateTime::parse_from_rfc2822(text) {
                return Some(dt.with_timezone(&Utc));
            }
            None
        }
        _ => None,
    }
}

/// `received_at` -> normalized JSON value (RFC3339 string or null).
fn received_at_value(item: &Value, keys: &[&str]) -> Value {
    match first_truthy_value(item, keys).and_then(parse_received_at) {
        Some(dt) => json!(dt.to_rfc3339()),
        None => Value::Null,
    }
}

/// Sortable timestamp (seconds) for the first truthy key, else 0.
fn received_ts(item: &Value, keys: &[&str]) -> f64 {
    first_truthy_value(item, keys)
        .and_then(parse_received_at)
        .map(|d| d.timestamp() as f64)
        .unwrap_or(0.0)
}

/// Pick the latest message, tie-broken by an id-like string (port of the
/// `max(..., key=lambda v: (ts, str(id)))` idiom).
fn pick_latest(messages: &[Value], date_keys: &[&str], id_keys: &[&str]) -> Option<Value> {
    messages.iter().cloned().max_by(|a, b| {
        let ta = received_ts(a, date_keys);
        let tb = received_ts(b, date_keys);
        ta.partial_cmp(&tb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| first_truthy_str(a, id_keys).cmp(&first_truthy_str(b, id_keys)))
    })
}

/// Port of `_extract_content`.
fn extract_content(data: &Value) -> (String, String) {
    let text = first_str(data, &["text_content", "text", "body", "content"]);
    let html = first_str(data, &["html_content", "html", "html_body", "body_html"]);
    if !text.is_empty() || !html.is_empty() {
        return (text, html);
    }
    // Fallback: raw RFC822 blob. Full MIME walking is simplified to surfacing
    // the raw text as the plain body (see module docs).
    match data.get("raw") {
        Some(Value::String(raw)) if !raw.trim().is_empty() => (raw.clone(), String::new()),
        _ => (String::new(), String::new()),
    }
}

/// First non-empty stringified value among `keys`.
fn first_str(v: &Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(val) = v.get(*k) {
            let t = vstring(val);
            if !t.is_empty() {
                return t;
            }
        }
    }
    String::new()
}

/// Resolve a sender field that may be a string or `{address,email,name}` object
/// (port of the repeated `if isinstance(sender, dict)` idiom).
fn extract_sender(item: &Value, keys: &[&str]) -> String {
    match first_truthy_value(item, keys) {
        Some(Value::Object(o)) => {
            let obj = Value::Object(o.clone());
            first_str(&obj, &["address", "email", "name"])
        }
        Some(val) => vstring(val),
        None => String::new(),
    }
}

/// Port of `_extract_text_candidates`.
fn extract_text_candidates(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Object(o) => {
            let mut out = Vec::new();
            for k in ["address", "email", "name", "value"] {
                if let Some(val) = o.get(k) {
                    if is_truthy(val) {
                        out.extend(extract_text_candidates(val));
                    }
                }
            }
            out
        }
        Value::Array(a) => a.iter().flat_map(extract_text_candidates).collect(),
        _ => Vec::new(),
    }
}

/// Port of `_message_matches_email`.
fn message_matches_email(data: &Value, email: &str) -> bool {
    let target = email.trim().to_lowercase();
    let mut candidates: Vec<String> = Vec::new();
    for k in ["to", "mailTo", "receiver", "receivers", "address", "email", "envelope_to"] {
        if let Some(v) = data.get(k) {
            candidates.extend(extract_text_candidates(v));
        }
    }
    if target.is_empty() || candidates.is_empty() {
        return true;
    }
    candidates.iter().any(|c| {
        let t = c.trim().to_lowercase();
        !t.is_empty() && t.contains(&target)
    })
}

static RE_BG: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)background-color:\s*#F3F3F3[^>]*>[\s\S]*?(\d{6})[\s\S]*?</p>").unwrap()
});
static RE_KEYWORD: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:Verification code|code is|代码为|验证码)[:\s]*(\d{6})").unwrap()
});
static RE_SIX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d{6}\b").unwrap());

/// Port of `_extract_code`.
fn extract_code(message: &Value) -> Option<String> {
    let content = format!(
        "{}\n{}\n{}",
        message.get("subject").map(vstring).unwrap_or_default(),
        message.get("text_content").map(vstring).unwrap_or_default(),
        message.get("html_content").map(vstring).unwrap_or_default()
    )
    .trim()
    .to_string();
    if content.is_empty() {
        return None;
    }
    if let Some(c) = RE_BG.captures(&content) {
        return Some(c[1].to_string());
    }
    if let Some(c) = RE_KEYWORD.captures(&content) {
        let v = c[1].to_string();
        if v != "177010" {
            return Some(v);
        }
    }
    // Bare 6-digit branch: `(?<![#&])\b(\d{6})\b` emulated via manual check.
    for m in RE_SIX.find_iter(&content) {
        let prev = content[..m.start()].chars().last();
        if matches!(prev, Some('#') | Some('&')) {
            continue;
        }
        let v = m.as_str().to_string();
        if v != "177010" {
            return Some(v);
        }
    }
    None
}

/// Port of `_message_tracking_ref`.
fn message_tracking_ref(message: &Value) -> String {
    let provider = estr(message, "provider");
    let mailbox = estr(message, "mailbox");
    let message_id = estr(message, "message_id");
    if !message_id.is_empty() {
        return format!("id:{provider}:{mailbox}:{message_id}");
    }
    let received_value = match message.get("received_at") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => vstring(other),
    };
    let content = ["subject", "sender", "text_content", "html_content"]
        .iter()
        .map(|k| message.get(*k).map(vstring).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hex::encode(hasher.finalize());
    format!("content:{provider}:{mailbox}:{received_value}:{digest}")
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

/// Build an emulated `wreq` client, honoring the configured proxy.
fn build_client(conf: &MailConf) -> Result<wreq::Client, MailError> {
    let mut builder = wreq::Client::builder()
        .emulation(wreq_util::Emulation::Chrome131)
        .cert_verification(false);
    if !conf.proxy.trim().is_empty() {
        if let Ok(p) = wreq::Proxy::all(conf.proxy.trim()) {
            builder = builder.proxy(p);
        }
    }
    builder.build().map_err(|e| merr(format!("mail client build: {e}")))
}

async fn http_request(
    client: &wreq::Client,
    method: &str,
    url: &str,
    headers: &[(String, String)],
    params: &[(&str, String)],
    json_body: Option<&Value>,
    timeout: f64,
) -> Result<(u16, String), MailError> {
    let m = wreq::Method::from_bytes(method.to_ascii_uppercase().as_bytes())
        .map_err(|e| merr(format!("invalid HTTP method {method}: {e}")))?;
    let mut req = client.request(m, url);
    if !headers.is_empty() {
        let mut hm = wreq::header::HeaderMap::new();
        for (k, v) in headers {
            if let (Ok(name), Ok(val)) = (
                wreq::header::HeaderName::from_bytes(k.as_bytes()),
                wreq::header::HeaderValue::from_str(v),
            ) {
                hm.insert(name, val);
            }
        }
        req = req.headers(hm);
    }
    if !params.is_empty() {
        req = req.query(params);
    }
    if let Some(b) = json_body {
        req = req.json(b);
    }
    let resp = req
        .timeout(Duration::from_secs_f64(timeout.max(0.0)))
        .send()
        .await
        .map_err(|e| merr(format!("网络请求异常: {e}")))?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    Ok((status, text))
}

fn check_status(
    status: u16,
    expected: &[u16],
    text: &str,
    prefix: &str,
    method: &str,
    path: &str,
) -> Result<(), MailError> {
    if !expected.contains(&status) {
        return Err(merr(format!(
            "{prefix} 请求失败: {method} {path}, HTTP {status}, body={}",
            truncate(text, 300)
        )));
    }
    Ok(())
}

fn body_json(status: u16, text: &str) -> Value {
    if status == 204 {
        return json!({});
    }
    serde_json::from_str(text).unwrap_or(Value::Null)
}

#[allow(clippy::too_many_arguments)]
async fn do_json(
    client: &wreq::Client,
    timeout: f64,
    method: &str,
    base: &str,
    path: &str,
    headers: &[(String, String)],
    params: &[(&str, String)],
    payload: Option<&Value>,
    expected: &[u16],
    prefix: &str,
) -> Result<Value, MailError> {
    let url = format!("{base}{path}");
    let (status, text) = http_request(client, method, &url, headers, params, payload, timeout).await?;
    check_status(status, expected, &text, prefix, method, path)?;
    Ok(body_json(status, &text))
}

// ---------------------------------------------------------------------------
// MailProvider trait
// ---------------------------------------------------------------------------

/// Common interface exposed by every temp-mail backend (port of
/// `BaseMailProvider` + the concrete provider methods).
#[async_trait]
pub trait MailProvider: Send + Sync {
    /// Provider type identifier (e.g. `"duckmail"`).
    fn name(&self) -> &str;

    /// The unique `type#idx` reference assigned by the entry builder.
    fn provider_ref(&self) -> &str;

    /// Runtime configuration knobs.
    fn conf(&self) -> &MailConf;

    /// Create a fresh mailbox; returns the mailbox descriptor object.
    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError>;

    /// Fetch the latest (normalized) message for `mailbox`, if any.
    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError>;

    /// Look up the JWT/token for a pre-existing address. Only supported by some
    /// providers; the default mirrors Python's "unsupported" branch.
    async fn get_existing_mailbox(&self, _email: &str) -> Result<Value, MailError> {
        Err(merr(format!("邮箱提供商 {} 不支持查询已有邮箱", self.name())))
    }

    /// Poll the inbox until a verification code arrives or the wait times out
    /// (port of `wait_for_code` + `wait_for`). Records seen message refs into
    /// `mailbox["_seen_code_message_refs"]` to avoid re-reporting codes.
    async fn wait_for_code(&self, mailbox: &mut Value) -> Option<String> {
        let conf = self.conf();
        let wait_interval = conf.wait_interval.max(0.2);
        let deadline = Instant::now() + Duration::from_secs_f64(conf.wait_timeout.max(0.0));

        let mut seen: HashSet<String> = mailbox
            .get("_seen_code_message_refs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(vstring).collect())
            .unwrap_or_default();

        loop {
            if Instant::now() >= deadline {
                break;
            }
            if let Ok(Some(message)) = self.fetch_latest_message(mailbox).await {
                let r = message_tracking_ref(&message);
                if !seen.contains(&r) {
                    if let Some(code) = extract_code(&message) {
                        seen.insert(r.clone());
                        if let Some(obj) = mailbox.as_object_mut() {
                            let arr = obj
                                .entry("_seen_code_message_refs")
                                .or_insert_with(|| json!([]));
                            if let Some(a) = arr.as_array_mut() {
                                a.push(json!(r));
                            }
                        }
                        return Some(code);
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs_f64(wait_interval)).await;
        }
        None
    }

    /// Release any held resources (no-op for pooled `wreq` clients).
    async fn close(&self) {}
}

// ---------------------------------------------------------------------------
// CloudflareTempMailProvider
// ---------------------------------------------------------------------------

/// Port of `CloudflareTempMailProvider`.
pub struct CloudflareTempMailProvider {
    provider_ref: String,
    api_base: String,
    admin_password: String,
    domain: Vec<String>,
    conf: MailConf,
    client: wreq::Client,
}

impl CloudflareTempMailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_base: estr(entry, "api_base").trim_end_matches('/').to_string(),
            admin_password: estr(entry, "admin_password"),
            domain: str_list(entry.get("domain")),
            conf,
            client,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        extra: Vec<(String, String)>,
        params: &[(&str, String)],
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let mut headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
        ];
        headers.extend(extra);
        do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            &self.api_base,
            path,
            &headers,
            params,
            payload,
            expected,
            "CloudflareTempMail",
        )
        .await
    }
}

#[async_trait]
impl MailProvider for CloudflareTempMailProvider {
    fn name(&self) -> &str {
        "cloudflare_temp_email"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let name = username
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(random_mailbox_name);
        let domain = next_domain(&self.domain)?;
        let payload = json!({"enablePrefix": true, "name": name, "domain": domain});
        let data = self
            .request(
                "POST",
                "/admin/new_address",
                vec![("x-admin-auth".into(), self.admin_password.clone())],
                &[],
                Some(&payload),
                &[200],
            )
            .await?;
        let address = estr(&data, "address");
        let token = estr(&data, "jwt");
        if address.is_empty() || token.is_empty() {
            return Err(merr("CloudflareTempMail 缺少 address 或 jwt"));
        }
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "token": token,
        }))
    }

    async fn get_existing_mailbox(&self, email: &str) -> Result<Value, MailError> {
        let payload = json!({"address": email});
        let data = self
            .request(
                "POST",
                "/admin/get_address",
                vec![("x-admin-auth".into(), self.admin_password.clone())],
                &[],
                Some(&payload),
                &[200],
            )
            .await?;
        let address = estr(&data, "address");
        let token = estr(&data, "jwt");
        if address.is_empty() || token.is_empty() {
            return Err(merr(format!(
                "CloudflareTempMail 无法获取已有邮箱 {email} 的 JWT"
            )));
        }
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "token": token,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let token = estr(mailbox, "token");
        let address = estr(mailbox, "address");
        let data = self
            .request(
                "GET",
                "/api/mails",
                vec![("Authorization".into(), format!("Bearer {token}"))],
                &[("limit", "10".into()), ("offset", "0".into())],
                None,
                &[200],
            )
            .await?;
        let raw_list: Vec<Value> = if let Some(o) = data.as_object() {
            o.get("results")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        } else if let Some(a) = data.as_array() {
            a.clone()
        } else {
            vec![]
        };
        let messages: Vec<Value> = raw_list
            .into_iter()
            .filter(|i| i.is_object() && message_matches_email(i, &address))
            .collect();
        let Some(item) = messages.into_iter().next() else {
            return Ok(None);
        };
        let (text_content, html_content) = extract_content(&item);
        let sender = extract_sender(&item, &["from", "sender"]);
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": first_truthy_str(&item, &["id", "_id"]),
            "subject": estr(&item, "subject"),
            "sender": sender,
            "text_content": text_content,
            "html_content": html_content,
            "received_at": received_at_value(&item, &["createdAt", "created_at", "receivedAt", "date", "timestamp"]),
            "raw": item,
        })))
    }
}

// ---------------------------------------------------------------------------
// DDG (DuckDuckGo) alias bookkeeping (port of the `_*_ddg_alias*` helpers)
// ---------------------------------------------------------------------------

fn ddg_aliases_file(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("ddg_aliases.json")
}

fn load_ddg_aliases(data_dir: &PathBuf) -> HashSet<String> {
    let path = ddg_aliases_file(data_dir);
    if let Ok(text) = std::fs::read_to_string(&path) {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&text) {
            return arr
                .iter()
                .map(vstring)
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    HashSet::new()
}

fn save_ddg_aliases(data_dir: &PathBuf, aliases: &HashSet<String>) {
    let path = ddg_aliases_file(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut sorted: Vec<&String> = aliases.iter().collect();
    sorted.sort();
    if let Ok(body) = serde_json::to_string_pretty(&sorted) {
        std::fs::write(&path, body + "\n").ok();
    }
}

fn is_ddg_alias_duplicate(data_dir: &PathBuf, address: &str) -> bool {
    let target = address.trim().to_lowercase();
    if target.is_empty() {
        return false;
    }
    let _guard = DDG_ALIASES_LOCK.lock();
    load_ddg_aliases(data_dir).contains(&target)
}

fn record_ddg_alias(data_dir: &PathBuf, address: &str) {
    let target = address.trim().to_lowercase();
    if target.is_empty() {
        return;
    }
    let _guard = DDG_ALIASES_LOCK.lock();
    let mut used = load_ddg_aliases(data_dir);
    used.insert(target);
    save_ddg_aliases(data_dir, &used);
}

static RE_TO_HEADER: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?im)^To:\s*(.+?)$").unwrap());
static RE_ANGLE_ADDR: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*<[^>]*>").unwrap());

// ---------------------------------------------------------------------------
// DDGMailProvider
// ---------------------------------------------------------------------------

/// Port of `DDGMailProvider` (DuckDuckGo email alias forwarded to a Cloudflare
/// temp-mail inbox).
pub struct DDGMailProvider {
    provider_ref: String,
    label: String,
    ddg_token: String,
    cf_api_base: String,
    cf_inbox_jwt: String,
    cf_admin_password: String,
    cf_api_key: String,
    cf_auth_mode: String,
    #[allow(dead_code)]
    cf_domain: Vec<String>,
    #[allow(dead_code)]
    cf_create_path: String,
    cf_messages_path: String,
    conf: MailConf,
    client: wreq::Client,
    data_dir: PathBuf,
}

impl DDGMailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client, data_dir: PathBuf) -> Self {
        let provider_ref = estr(entry, "provider_ref");
        let label = estr_or(entry, "label", &provider_ref);
        let cf_api_base = {
            let a = estr(entry, "api_base");
            let v = if !a.is_empty() { a } else { estr(entry, "cf_api_base") };
            v.trim_end_matches('/').to_string()
        };
        Self {
            provider_ref,
            label,
            ddg_token: estr(entry, "ddg_token"),
            cf_api_base,
            cf_inbox_jwt: estr(entry, "cf_inbox_jwt"),
            cf_admin_password: estr(entry, "admin_password"),
            cf_api_key: estr(entry, "cf_api_key"),
            cf_auth_mode: estr_or(entry, "cf_auth_mode", "none").to_lowercase(),
            cf_domain: str_list(entry.get("cf_domain")),
            cf_create_path: estr_or(entry, "cf_create_path", "/api/new_address"),
            cf_messages_path: estr_or(entry, "cf_messages_path", "/api/mails"),
            conf,
            client,
            data_dir,
        }
    }

    fn cf_build_headers(&self, content_type: bool) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if content_type {
            headers.push(("Content-Type".to_string(), "application/json".to_string()));
        }
        if !self.cf_api_key.is_empty() {
            if self.cf_auth_mode == "x-api-key" {
                headers.push(("X-API-Key".to_string(), self.cf_api_key.clone()));
            } else if self.cf_auth_mode != "none" {
                headers.push(("Authorization".to_string(), format!("Bearer {}", self.cf_api_key)));
            }
        }
        headers
    }

    async fn cf_request(
        &self,
        method: &str,
        path: &str,
        extra: Vec<(String, String)>,
        params: &[(&str, String)],
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let mut headers = self.cf_build_headers(true);
        headers.extend(extra);
        headers.push(("User-Agent".to_string(), self.conf.user_agent.clone()));
        if !self.cf_admin_password.is_empty() && method.eq_ignore_ascii_case("POST") {
            headers.push(("x-admin-auth".to_string(), self.cf_admin_password.clone()));
        }
        let mut all_params: Vec<(&str, String)> = params.to_vec();
        if !self.cf_api_key.is_empty() && self.cf_auth_mode == "query-key" {
            all_params.push(("key", self.cf_api_key.clone()));
        }
        do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            &self.cf_api_base,
            path,
            &headers,
            &all_params,
            payload,
            expected,
            "DDGMail CF",
        )
        .await
    }

    async fn ddg_request(
        &self,
        method: &str,
        path: &str,
        payload: Option<&Value>,
    ) -> Result<Value, MailError> {
        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", self.ddg_token)),
            ("Content-Type".to_string(), "application/json".to_string()),
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
        ];
        do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            "https://quack.duckduckgo.com",
            path,
            &headers,
            &[],
            payload,
            &[200, 201],
            "DDG API",
        )
        .await
    }

    fn cf_list_payload(data: &Value) -> Vec<Value> {
        if let Some(a) = data.as_array() {
            return a.clone();
        }
        if let Some(o) = data.as_object() {
            for key in ["results", "hydra:member", "data", "messages"] {
                match o.get(key) {
                    Some(Value::Array(a)) => return a.clone(),
                    Some(Value::Object(inner)) => {
                        if let Some(Value::Array(a)) = inner.get("messages") {
                            return a.clone();
                        }
                    }
                    _ => {}
                }
            }
        }
        Vec::new()
    }

    fn parse_raw_recipient(raw_text: &str) -> String {
        if raw_text.is_empty() {
            return String::new();
        }
        if let Some(c) = RE_TO_HEADER.captures(raw_text) {
            let addr = c[1].trim().to_string();
            let stripped = RE_ANGLE_ADDR.replace_all(&addr, "");
            return stripped.trim().to_lowercase();
        }
        String::new()
    }
}

#[async_trait]
impl MailProvider for DDGMailProvider {
    fn name(&self) -> &str {
        "ddg_mail"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, _username: Option<&str>) -> Result<Value, MailError> {
        let ddg_data = self
            .ddg_request("POST", "/api/email/addresses", Some(&json!({})))
            .await?;
        let ddg_address_part = estr(&ddg_data, "address");
        if ddg_address_part.is_empty() {
            return Err(merr("DDG API 返回无 address 字段"));
        }
        let ddg_address = format!("{ddg_address_part}@duck.com");

        if is_ddg_alias_duplicate(&self.data_dir, &ddg_address) {
            return Err(merr(format!(
                "[{}] {DDG_LIMIT_SENTINEL}，别名 {ddg_address} 已存在，自动切换邮箱提供商",
                self.label
            )));
        }
        record_ddg_alias(&self.data_dir, &ddg_address);

        if self.cf_inbox_jwt.is_empty() {
            return Err(merr(
                "DDGMail 需要 cf_inbox_jwt（DDG 转发目标的固定收件箱 JWT），请在邮箱配置中填写 CF Inbox JWT",
            ));
        }

        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": ddg_address,
            "token": self.cf_inbox_jwt,
            "label": self.label,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let target_address = estr(mailbox, "address").to_lowercase();
        let token = estr(mailbox, "token");
        let address = estr(mailbox, "address");
        let path = self.cf_messages_path.clone();
        let data = self
            .cf_request(
                "GET",
                &path,
                vec![("Authorization".into(), format!("Bearer {token}"))],
                &[("limit", "30".into()), ("offset", "0".into())],
                None,
                &[200],
            )
            .await?;
        let messages: Vec<Value> = Self::cf_list_payload(&data)
            .into_iter()
            .filter(|i| i.is_object())
            .collect();
        if messages.is_empty() {
            return Ok(None);
        }
        for item in messages {
            let message_id = first_truthy_str(&item, &["id", "msgid", "_id"]);
            let raw_text = estr(&item, "raw");
            let raw_recipient = Self::parse_raw_recipient(&raw_text);
            if !target_address.is_empty()
                && !raw_recipient.is_empty()
                && !raw_recipient.contains(&target_address)
            {
                continue;
            }
            let (text_content, html_content) = extract_content(&item);
            let subject = estr(&item, "subject");
            let sender = extract_sender(&item, &["from", "sender", "source"]);
            return Ok(Some(json!({
                "provider": self.name(),
                "mailbox": address,
                "message_id": message_id,
                "subject": subject,
                "sender": sender,
                "text_content": text_content,
                "html_content": html_content,
                "received_at": received_at_value(&item, &["createdAt", "created_at", "receivedAt", "date", "timestamp"]),
                "raw": item,
            })));
        }
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// CloudMailGenProvider
// ---------------------------------------------------------------------------

/// Port of `CloudMailGenProvider`.
pub struct CloudMailGenProvider {
    provider_ref: String,
    api_base: String,
    admin_email: String,
    admin_password: String,
    domain: Vec<String>,
    subdomain: Vec<String>,
    email_prefix: String,
    conf: MailConf,
    client: wreq::Client,
}

impl CloudMailGenProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_base: estr(entry, "api_base").trim_end_matches('/').to_string(),
            admin_email: estr(entry, "admin_email"),
            admin_password: estr(entry, "admin_password"),
            domain: normalize_string_list(entry.get("domain")),
            subdomain: normalize_string_list(entry.get("subdomain")),
            email_prefix: estr(entry, "email_prefix"),
            conf,
            client,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        extra: Vec<(String, String)>,
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let mut headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
        ];
        headers.extend(extra);
        do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            &self.api_base,
            path,
            &headers,
            &[],
            payload,
            expected,
            "CloudMailGen",
        )
        .await
    }

    fn cache_key(&self) -> String {
        format!("{}|{}", self.api_base, self.admin_email)
    }

    async fn get_token(&self) -> Result<String, MailError> {
        if self.admin_email.is_empty() || self.admin_password.is_empty() {
            return Err(merr("CloudMailGen 缺少 admin_email 或 admin_password"));
        }
        let cache_key = self.cache_key();
        let now = now_secs();
        {
            let cache = CLOUDMAIL_TOKEN_CACHE.lock();
            if let Some((tok, exp)) = cache.get(&cache_key) {
                if now < exp - 300.0 {
                    return Ok(tok.clone());
                }
            }
        }
        let data = self
            .request(
                "POST",
                "/api/public/genToken",
                vec![],
                Some(&json!({"email": self.admin_email, "password": self.admin_password})),
                &[200],
            )
            .await?;
        let token = if data.get("code").and_then(|v| v.as_i64()) == Some(200) {
            data.get("data").map(|d| estr(d, "token")).unwrap_or_default()
        } else {
            String::new()
        };
        if token.is_empty() {
            return Err(merr(format!("CloudMailGen genToken 返回异常: {data}")));
        }
        CLOUDMAIL_TOKEN_CACHE
            .lock()
            .insert(cache_key, (token.clone(), now + 24.0 * 3600.0));
        Ok(token)
    }

    fn resolve_address(&self, username: Option<&str>) -> Result<String, MailError> {
        let mut domain = next_domain(&self.domain)?;
        if !self.subdomain.is_empty() {
            let sub = self
                .subdomain
                .choose(&mut rand::thread_rng())
                .cloned()
                .unwrap_or_default();
            domain = format!("{sub}.{domain}");
        }
        let local_part = match username {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => {
                if !self.email_prefix.is_empty() {
                    format!("{}_{}", self.email_prefix, rand_from(ALNUM_LOWER, 6))
                } else {
                    random_mailbox_name()
                }
            }
        };
        Ok(format!("{local_part}@{domain}"))
    }
}

#[async_trait]
impl MailProvider for CloudMailGenProvider {
    fn name(&self) -> &str {
        "cloudmail_gen"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        if self.domain.is_empty() {
            return Err(merr("CloudMailGen 需要至少配置一个 domain"));
        }
        let address = self.resolve_address(username)?;
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let address = estr(mailbox, "address");
        if address.is_empty() {
            return Err(merr("CloudMailGen 缺少 address"));
        }
        let token = self.get_token().await?;
        let data = self
            .request(
                "POST",
                "/api/public/emailList",
                vec![("Authorization".into(), token)],
                Some(&json!({"toEmail": address, "size": 20, "timeSort": "desc"})),
                &[200],
            )
            .await?;
        let items: Vec<Value> = if data.get("code").and_then(|v| v.as_i64()) == Some(200) {
            data.get("data").and_then(|v| v.as_array()).cloned().unwrap_or_default()
        } else {
            vec![]
        };
        let messages: Vec<Value> = items
            .into_iter()
            .filter(|i| i.is_object() && message_matches_email(i, &address))
            .collect();
        let Some(item) = messages.into_iter().next() else {
            return Ok(None);
        };
        let (text_content, html_content) = extract_content(&item);
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": first_truthy_str(&item, &["id", "_id", "messageId"]),
            "subject": estr(&item, "subject"),
            "sender": first_truthy_str(&item, &["from", "sender"]),
            "text_content": text_content,
            "html_content": html_content,
            "received_at": received_at_value(&item, &["createdAt", "created_at", "receivedAt", "date", "timestamp"]),
            "to": item.get("to").or_else(|| item.get("toEmail")).or_else(|| item.get("mailTo")).cloned().unwrap_or(Value::Null),
            "raw": item,
        })))
    }
}

// ---------------------------------------------------------------------------
// TempMailLolProvider
// ---------------------------------------------------------------------------

/// Port of `TempMailLolProvider`.
pub struct TempMailLolProvider {
    provider_ref: String,
    api_key: String,
    domain: Vec<String>,
    conf: MailConf,
    client: wreq::Client,
}

impl TempMailLolProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_key: estr(entry, "api_key"),
            domain: str_list(entry.get("domain")),
            conf,
            client,
        }
    }

    fn resolve_domain(domain: &str) -> (String, bool) {
        let text = domain.trim().to_lowercase();
        if text.starts_with("*.") && text.len() > 2 {
            return (format!("{}.{}", random_subdomain_label(), &text[2..]), true);
        }
        (text, false)
    }

    fn base_headers(&self) -> Vec<(String, String)> {
        let mut headers = vec![
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
            ("Accept".to_string(), "application/json".to_string()),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if !self.api_key.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {}", self.api_key)));
        }
        headers
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        params: &[(&str, String)],
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let data = do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            "https://api.tempmail.lol/v2",
            path,
            &self.base_headers(),
            params,
            payload,
            expected,
            "TempMail.lol",
        )
        .await?;
        if !data.is_object() {
            return Err(merr(format!("TempMail.lol {method} {path} 返回结构不是对象")));
        }
        Ok(data)
    }
}

#[async_trait]
impl MailProvider for TempMailLolProvider {
    fn name(&self) -> &str {
        "tempmail_lol"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let mut payload = Map::new();
        if !self.domain.is_empty() {
            let chosen = self.domain.choose(&mut rand::thread_rng()).cloned().unwrap_or_default();
            let (domain, force_random_prefix) = Self::resolve_domain(&chosen);
            payload.insert("domain".into(), json!(domain));
            if force_random_prefix {
                payload.insert("prefix".into(), json!(random_mailbox_name()));
            }
        }
        if let Some(u) = username {
            if !u.is_empty() && !payload.contains_key("prefix") {
                payload.insert("prefix".into(), json!(u));
            }
        }
        let data = self
            .request("POST", "/inbox/create", &[], Some(&Value::Object(payload)), &[200, 201])
            .await?;
        let address = estr(&data, "address");
        let token = estr(&data, "token");
        if address.is_empty() || token.is_empty() {
            return Err(merr("TempMail.lol 缺少 address 或 token"));
        }
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "token": token,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let token = estr(mailbox, "token");
        let address = estr(mailbox, "address");
        let data = self.request("GET", "/inbox", &[("token", token)], None, &[200]).await?;
        let items = data
            .get("emails")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| data.get("messages").and_then(|v| v.as_array()).cloned())
            .unwrap_or_default();
        let messages: Vec<Value> = items.into_iter().filter(|i| i.is_object()).collect();
        if messages.is_empty() {
            return Ok(None);
        }
        let date_keys = ["created_at", "createdAt", "date", "received_at", "timestamp"];
        let item = pick_latest(&messages, &date_keys, &["id", "token"]).unwrap();
        let (text_content, html_content) = extract_content(&item);
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": first_truthy_str(&item, &["id", "token"]),
            "subject": estr(&item, "subject"),
            "sender": first_truthy_str(&item, &["from", "from_address"]),
            "text_content": text_content,
            "html_content": html_content,
            "received_at": received_at_value(&item, &date_keys),
            "raw": item,
        })))
    }
}

// ---------------------------------------------------------------------------
// DuckMailProvider
// ---------------------------------------------------------------------------

/// Port of `DuckMailProvider`.
pub struct DuckMailProvider {
    provider_ref: String,
    api_key: String,
    default_domain: String,
    conf: MailConf,
    client: wreq::Client,
}

impl DuckMailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_key: estr(entry, "api_key"),
            default_domain: estr_or(entry, "default_domain", "duckmail.sbs"),
            conf,
            client,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        token: &str,
        use_api_key: bool,
        params: &[(&str, String)],
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let mut headers = vec![
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
            ("Accept".to_string(), "application/json".to_string()),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if use_api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {}", self.api_key)));
        } else if !token.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {token}")));
        }
        do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            "https://api.duckmail.sbs",
            path,
            &headers,
            params,
            payload,
            expected,
            "DuckMail",
        )
        .await
    }

    fn items(data: &Value) -> Vec<Value> {
        if let Some(a) = data.as_array() {
            return a.clone();
        }
        for k in ["hydra:member", "member", "data"] {
            if let Some(a) = data.get(k).and_then(|v| v.as_array()) {
                return a.clone();
            }
        }
        Vec::new()
    }
}

#[async_trait]
impl MailProvider for DuckMailProvider {
    fn name(&self) -> &str {
        "duckmail"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let password = rand_from(ALNUM, 12);
        let local = match username {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => random_mailbox_name(),
        };
        let address = format!("{local}@{}", self.default_domain);
        let payload = json!({"address": address, "password": password});
        let account = self
            .request("POST", "/accounts", "", true, &[], Some(&payload), &[200, 201, 204])
            .await?;
        let token_data = self
            .request("POST", "/token", "", true, &[], Some(&payload), &[200, 201, 204])
            .await?;
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "token": estr(&token_data, "token"),
            "password": password,
            "account_id": first_truthy_str(&account, &["id"]),
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let token = estr(mailbox, "token");
        let address = estr(mailbox, "address");
        let data = self
            .request("GET", "/messages", &token, false, &[("page", "1".into())], None, &[200, 201, 204])
            .await?;
        let items = Self::items(&data);
        let Some(mut item) = items.into_iter().next() else {
            return Ok(None);
        };
        let mut message_id = first_truthy_str(&item, &["id", "@id"]).replace("/messages/", "");
        if !message_id.is_empty() {
            item = self
                .request("GET", &format!("/messages/{message_id}"), &token, false, &[], None, &[200, 201, 204])
                .await?;
            message_id = first_truthy_str(&item, &["id", "@id"]).replace("/messages/", "");
        }
        let sender = extract_sender(&item, &["from"]);
        let html_content = match item.get("html") {
            Some(Value::Array(a)) => a.iter().map(vstring).collect::<Vec<_>>().join(""),
            Some(v) => vstring(v),
            None => String::new(),
        };
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": message_id,
            "subject": estr(&item, "subject"),
            "sender": sender,
            "text_content": first_str(&item, &["text", "text_content"]),
            "html_content": html_content,
            "received_at": received_at_value(&item, &["createdAt", "created_at", "receivedAt", "date"]),
            "raw": item,
        })))
    }
}

// ---------------------------------------------------------------------------
// GptMailProvider
// ---------------------------------------------------------------------------

/// Port of `GptMailProvider`.
pub struct GptMailProvider {
    provider_ref: String,
    api_key: String,
    default_domain: String,
    conf: MailConf,
    client: wreq::Client,
}

impl GptMailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_key: estr(entry, "api_key"),
            default_domain: estr(entry, "default_domain"),
            conf,
            client,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        params: &[(&str, String)],
        payload: Option<&Value>,
    ) -> Result<Value, MailError> {
        let headers = vec![
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
            ("Accept".to_string(), "application/json".to_string()),
            ("Content-Type".to_string(), "application/json".to_string()),
            ("X-API-Key".to_string(), self.api_key.clone()),
        ];
        let data = do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            "https://mail.chatgpt.org.uk",
            path,
            &headers,
            params,
            payload,
            &[200],
            "GPTMail",
        )
        .await?;
        if let Some(inner) = data.as_object().and_then(|o| o.get("data")) {
            return Ok(inner.clone());
        }
        Ok(data)
    }
}

#[async_trait]
impl MailProvider for GptMailProvider {
    fn name(&self) -> &str {
        "gptmail"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let mut payload = Map::new();
        if let Some(u) = username {
            if !u.is_empty() {
                payload.insert("prefix".into(), json!(u));
            }
        }
        if !self.default_domain.is_empty() {
            payload.insert("domain".into(), json!(self.default_domain));
        }
        let (method, body) = if payload.is_empty() {
            ("GET", None)
        } else {
            ("POST", Some(Value::Object(payload)))
        };
        let data = self
            .request(method, "/api/generate-email", &[], body.as_ref())
            .await?;
        let email = estr(&data, "email");
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": email,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let address = estr(mailbox, "address");
        let data = self
            .request("GET", "/api/emails", &[("email", address.clone())], None)
            .await?;
        let emails: Vec<Value> = if let Some(a) = data.as_array() {
            a.clone()
        } else {
            data.get("emails").and_then(|v| v.as_array()).cloned().unwrap_or_default()
        };
        if emails.is_empty() {
            return Ok(None);
        }
        let mut item = emails
            .iter()
            .cloned()
            .max_by(|a, b| {
                let ta = a.get("timestamp").map(to_f64).unwrap_or(0.0);
                let tb = b.get("timestamp").map(to_f64).unwrap_or(0.0);
                ta.partial_cmp(&tb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| first_truthy_str(a, &["id"]).cmp(&first_truthy_str(b, &["id"])))
            })
            .unwrap();
        let id = first_truthy_str(&item, &["id"]);
        if !id.is_empty() {
            item = self.request("GET", &format!("/api/email/{id}"), &[], None).await?;
        }
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": first_truthy_str(&item, &["id"]),
            "subject": estr(&item, "subject"),
            "sender": estr(&item, "from_address"),
            "text_content": estr(&item, "content"),
            "html_content": estr(&item, "html_content"),
            "received_at": received_at_value(&item, &["timestamp", "created_at"]),
            "raw": item,
        })))
    }
}

// ---------------------------------------------------------------------------
// MoEmailProvider
// ---------------------------------------------------------------------------

/// Port of `MoEmailProvider`.
pub struct MoEmailProvider {
    provider_ref: String,
    api_base: String,
    api_key: String,
    domain: Vec<String>,
    expiry_time: i64,
    conf: MailConf,
    client: wreq::Client,
}

impl MoEmailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        let expiry_time = match entry.get("expiry_time") {
            Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
            Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(0),
            _ => 0,
        };
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_base: estr(entry, "api_base").trim_end_matches('/').to_string(),
            api_key: estr(entry, "api_key"),
            domain: normalize_string_list(entry.get("domain")),
            expiry_time,
            conf,
            client,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        params: &[(&str, String)],
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let headers = vec![
            ("X-API-Key".to_string(), self.api_key.clone()),
            ("Content-Type".to_string(), "application/json".to_string()),
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
        ];
        let data = do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            &self.api_base,
            path,
            &headers,
            params,
            payload,
            expected,
            "MoEmail",
        )
        .await?;
        if !data.is_object() {
            return Err(merr(format!("MoEmail {method} {path} 返回结构不是对象")));
        }
        Ok(data)
    }
}

#[async_trait]
impl MailProvider for MoEmailProvider {
    fn name(&self) -> &str {
        "moemail"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let name = match username {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => random_mailbox_name(),
        };
        let domain = next_domain(&self.domain)?;
        let payload = json!({"name": name, "expiryTime": self.expiry_time, "domain": domain});
        let data = self
            .request("POST", "/api/emails/generate", &[], Some(&payload), &[200, 201])
            .await?;
        let address = estr(&data, "email");
        let email_id = first_truthy_str(&data, &["id", "email_id"]);
        if address.is_empty() || email_id.is_empty() {
            return Err(merr("MoEmail 缺少 email 或 id"));
        }
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "email_id": email_id,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let email_id = estr(mailbox, "email_id");
        let address = estr(mailbox, "address");
        if email_id.is_empty() {
            return Err(merr("MoEmail 缺少 email_id"));
        }
        let data = self
            .request("GET", &format!("/api/emails/{email_id}"), &[], None, &[200])
            .await?;
        let messages: Vec<Value> = data
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|i| i.is_object())
            .collect();
        if messages.is_empty() {
            return Ok(None);
        }
        let date_keys = ["createdAt", "created_at", "receivedAt", "date", "timestamp"];
        // tie-break by original index (matches Python's enumerate key).
        let item = messages
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                let ta = received_ts(a, &date_keys);
                let tb = received_ts(b, &date_keys);
                ta.partial_cmp(&tb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(ia.cmp(ib))
            })
            .map(|(_, v)| v)
            .unwrap();
        let message_id = first_truthy_str(&item, &["id", "message_id", "_id"]);
        let detail = if !message_id.is_empty() {
            self.request("GET", &format!("/api/emails/{email_id}/{message_id}"), &[], None, &[200])
                .await?
        } else {
            json!({"message": item})
        };
        let message = match detail.get("message") {
            Some(m) if m.is_object() => m.clone(),
            _ => detail.clone(),
        };
        let (text_content, html_content) = extract_content(&message);
        let sender = extract_sender(&message, &["from", "sender"]);
        let subject = {
            let s = estr(&message, "subject");
            if s.is_empty() { estr(&item, "subject") } else { s }
        };
        let received_at = {
            let v = received_at_value(&message, &date_keys);
            if v.is_null() { received_at_value(&item, &date_keys) } else { v }
        };
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": message_id,
            "subject": subject,
            "sender": sender,
            "text_content": text_content,
            "html_content": html_content,
            "received_at": received_at,
            "raw": detail,
        })))
    }
}

// ---------------------------------------------------------------------------
// InbucketMailProvider
// ---------------------------------------------------------------------------

/// Port of `InbucketMailProvider`.
pub struct InbucketMailProvider {
    provider_ref: String,
    api_base: String,
    domain: Vec<String>,
    random_subdomain: bool,
    conf: MailConf,
    client: wreq::Client,
}

impl InbucketMailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        let random_subdomain = match entry.get("random_subdomain") {
            Some(v) => is_truthy(v),
            None => true,
        };
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_base: estr(entry, "api_base").trim_end_matches('/').to_string(),
            domain: normalize_string_list(entry.get("domain")),
            random_subdomain,
            conf,
            client,
        }
    }

    async fn request(&self, method: &str, path: &str, expected: &[u16]) -> Result<Value, MailError> {
        let url = format!("{}{}", self.api_base, path);
        let headers = vec![
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
            ("Accept".to_string(), "application/json".to_string()),
        ];
        let (status, text) =
            http_request(&self.client, method, &url, &headers, &[], None, self.conf.request_timeout)
                .await?;
        check_status(status, expected, &text, "Inbucket", method, path)?;
        if status == 204 {
            return Ok(json!({}));
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    fn resolve_domain(&self) -> Result<String, MailError> {
        if self.domain.is_empty() {
            return Err(merr("Inbucket 需要至少配置一个 domain"));
        }
        next_domain(&self.domain)
    }

    fn mailbox_name(address: &str) -> String {
        address.split('@').next().unwrap_or("").trim().to_string()
    }
}

#[async_trait]
impl MailProvider for InbucketMailProvider {
    fn name(&self) -> &str {
        "inbucket"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let local_part = match username {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => random_mailbox_name(),
        };
        let base_domain = self.resolve_domain()?;
        let domain = if self.random_subdomain {
            format!("{}.{}", random_subdomain_label(), base_domain)
        } else {
            base_domain.clone()
        };
        let address = format!("{local_part}@{domain}");
        let mailbox_name = Self::mailbox_name(&address);
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "base_domain": base_domain,
            "mailbox_name": mailbox_name,
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let address = estr(mailbox, "address");
        let mailbox_name = {
            let m = estr(mailbox, "mailbox_name");
            if m.is_empty() { Self::mailbox_name(&address) } else { m }
        };
        if mailbox_name.is_empty() {
            return Err(merr("Inbucket 缺少 mailbox_name"));
        }
        let data = self
            .request("GET", &format!("/api/v1/mailbox/{mailbox_name}"), &[200])
            .await?;
        let mut items: Vec<Value> = match data {
            Value::Array(a) => a.into_iter().filter(|i| i.is_object()).collect(),
            _ => vec![],
        };
        if items.is_empty() {
            return Ok(None);
        }
        items.sort_by(|a, b| {
            let ta = received_ts(a, &["date"]);
            let tb = received_ts(b, &["date"]);
            tb.partial_cmp(&ta)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| first_truthy_str(b, &["id"]).cmp(&first_truthy_str(a, &["id"])))
        });
        for item in &items {
            let message_id = estr(item, "id");
            if message_id.is_empty() {
                continue;
            }
            let detail = self
                .request("GET", &format!("/api/v1/mailbox/{mailbox_name}/{message_id}"), &[200])
                .await?;
            if !detail.is_object() {
                continue;
            }
            let header = detail.get("header").filter(|v| v.is_object());
            let body = detail.get("body").filter(|v| v.is_object());
            let subject = {
                let s = estr(&detail, "subject");
                if s.is_empty() { estr(item, "subject") } else { s }
            };
            let sender = {
                let s = estr(&detail, "from");
                if s.is_empty() { estr(item, "from") } else { s }
            };
            let normalized = json!({
                "provider": self.name(),
                "mailbox": mailbox_name,
                "message_id": message_id,
                "subject": subject,
                "sender": sender,
                "text_content": body.map(|b| estr(b, "text")).unwrap_or_default(),
                "html_content": body.map(|b| estr(b, "html")).unwrap_or_default(),
                "received_at": received_at_value(&detail, &["date"]),
                "to": header.and_then(|h| h.get("To")).cloned().unwrap_or(Value::Null),
                "raw": detail,
            });
            if message_matches_email(&normalized, &address) {
                return Ok(Some(normalized));
            }
        }
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// YydsMailProvider
// ---------------------------------------------------------------------------

/// Port of `YydsMailProvider`.
pub struct YydsMailProvider {
    provider_ref: String,
    api_base: String,
    api_key: String,
    domain: Vec<String>,
    subdomain: String,
    wildcard: bool,
    conf: MailConf,
    client: wreq::Client,
}

impl YydsMailProvider {
    fn new(entry: &Value, conf: MailConf, client: wreq::Client) -> Self {
        Self {
            provider_ref: estr(entry, "provider_ref"),
            api_base: estr_or(entry, "api_base", "https://maliapi.215.im/v1")
                .trim_end_matches('/')
                .to_string(),
            api_key: estr(entry, "api_key"),
            domain: str_list(entry.get("domain")),
            subdomain: estr(entry, "subdomain"),
            wildcard: entry.get("wildcard").map(is_truthy).unwrap_or(false),
            conf,
            client,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        token: &str,
        params: &[(&str, String)],
        payload: Option<&Value>,
        expected: &[u16],
    ) -> Result<Value, MailError> {
        let mut headers = vec![
            ("User-Agent".to_string(), self.conf.user_agent.clone()),
            ("Accept".to_string(), "application/json".to_string()),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if !token.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {token}")));
        } else {
            headers.push(("X-API-Key".to_string(), self.api_key.clone()));
        }
        let data = do_json(
            &self.client,
            self.conf.request_timeout,
            method,
            &self.api_base,
            path,
            &headers,
            params,
            payload,
            expected,
            "YYDSMail",
        )
        .await?;
        if let Some(o) = data.as_object() {
            if o.get("success") == Some(&Value::Bool(false)) {
                let detail = first_truthy_str(&data, &["errorCode", "error"]);
                return Err(merr(format!("YYDSMail 请求失败: {detail}")));
            }
            if let Some(inner) = o.get("data") {
                if inner.is_object() || inner.is_array() {
                    return Ok(inner.clone());
                }
            }
        }
        Ok(data)
    }

    fn items(data: &Value) -> Vec<Value> {
        if let Some(a) = data.as_array() {
            return a.clone();
        }
        for k in ["items", "messages", "data"] {
            if let Some(a) = data.get(k).and_then(|v| v.as_array()) {
                return a.clone();
            }
        }
        Vec::new()
    }
}

#[async_trait]
impl MailProvider for YydsMailProvider {
    fn name(&self) -> &str {
        "yyds_mail"
    }
    fn provider_ref(&self) -> &str {
        &self.provider_ref
    }
    fn conf(&self) -> &MailConf {
        &self.conf
    }

    async fn create_mailbox(&self, username: Option<&str>) -> Result<Value, MailError> {
        let local = match username {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => random_mailbox_name(),
        };
        let mut payload = Map::new();
        payload.insert("localPart".into(), json!(local));
        if !self.domain.is_empty() {
            payload.insert("domain".into(), json!(next_domain(&self.domain)?));
        }
        if !self.subdomain.is_empty() {
            payload.insert("subdomain".into(), json!(self.subdomain));
        }
        let path = if self.wildcard { "/accounts/wildcard" } else { "/accounts" };
        let data = self
            .request("POST", path, "", &[], Some(&Value::Object(payload)), &[200, 201, 204])
            .await?;
        let address = first_truthy_str(&data, &["address", "email"]);
        let token = first_truthy_str(&data, &["token", "temp_token", "tempToken", "access_token"]);
        if address.is_empty() || token.is_empty() {
            return Err(merr("YYDSMail 缺少 address 或 token"));
        }
        Ok(json!({
            "provider": self.name(),
            "provider_ref": self.provider_ref,
            "address": address,
            "token": token,
            "account_id": first_truthy_str(&data, &["id"]),
        }))
    }

    async fn fetch_latest_message(&self, mailbox: &Value) -> Result<Option<Value>, MailError> {
        let token = estr(mailbox, "token");
        let address = estr(mailbox, "address");
        let data = self
            .request("GET", "/messages", &token, &[("address", address.clone())], None, &[200, 201, 204])
            .await?;
        let messages: Vec<Value> =
            Self::items(&data).into_iter().filter(|i| i.is_object()).collect();
        if messages.is_empty() {
            return Ok(None);
        }
        let date_keys = ["createdAt", "created_at", "receivedAt", "date", "timestamp"];
        let mut item = pick_latest(&messages, &date_keys, &["id"]).unwrap();
        let message_id = first_truthy_str(&item, &["id", "message_id"]);
        if !message_id.is_empty() {
            item = self
                .request(
                    "GET",
                    &format!("/messages/{message_id}"),
                    &token,
                    &[("address", address.clone())],
                    None,
                    &[200, 201, 204],
                )
                .await?;
        }
        let (text_content, html_content) = extract_content(&item);
        let sender = extract_sender(&item, &["from", "sender"]);
        Ok(Some(json!({
            "provider": self.name(),
            "mailbox": address,
            "message_id": message_id,
            "subject": estr(&item, "subject"),
            "sender": sender,
            "text_content": text_content,
            "html_content": html_content,
            "received_at": received_at_value(&item, &date_keys),
            "raw": item,
        })))
    }
}

// ---------------------------------------------------------------------------
// Factory + entry selection (port of `_create_provider` / `_entries` / etc.)
// ---------------------------------------------------------------------------

/// Construct a concrete [`MailProvider`] for `name` from an explicit `entry`
/// (the provider's config object, expected to carry a `provider_ref`) and the
/// resolved [`MailConf`]. Returns `None` for unknown provider types or if the
/// HTTP client cannot be built (port of `_create_provider`'s type dispatch).
pub fn create_mail_provider(
    name: &str,
    config: &Config,
    entry: &Value,
    conf: &MailConf,
) -> Option<Box<dyn MailProvider>> {
    let client = build_client(conf).ok()?;
    let c = conf.clone();
    match name {
        "cloudmail_gen" => Some(Box::new(CloudMailGenProvider::new(entry, c, client))),
        "cloudflare_temp_email" => Some(Box::new(CloudflareTempMailProvider::new(entry, c, client))),
        "ddg_mail" => Some(Box::new(DDGMailProvider::new(
            entry,
            c,
            client,
            config.data_dir().to_path_buf(),
        ))),
        "tempmail_lol" => Some(Box::new(TempMailLolProvider::new(entry, c, client))),
        "duckmail" => Some(Box::new(DuckMailProvider::new(entry, c, client))),
        "gptmail" => Some(Box::new(GptMailProvider::new(entry, c, client))),
        "moemail" => Some(Box::new(MoEmailProvider::new(entry, c, client))),
        "inbucket" => Some(Box::new(InbucketMailProvider::new(entry, c, client))),
        "yyds_mail" => Some(Box::new(YydsMailProvider::new(entry, c, client))),
        _ => None,
    }
}

/// Port of `_entries`: clone each configured provider and stamp it with a
/// `provider_ref` (`type#idx`) and human label.
pub fn mail_entries(mail_config: &Value) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();
    let mut counters: HashMap<String, i64> = HashMap::new();
    if let Some(arr) = mail_config.get("providers").and_then(|v| v.as_array()) {
        for item in arr {
            let idx = result.len() + 1;
            let t = item.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let cnt = {
                let c = counters.entry(t.clone()).or_insert(0);
                *c += 1;
                *c
            };
            let label = if t == "ddg_mail" {
                format!("DDG-{cnt}")
            } else {
                format!("{t}#{idx}")
            };
            let mut obj = item.clone();
            if let Some(o) = obj.as_object_mut() {
                o.insert("provider_ref".into(), json!(format!("{t}#{idx}")));
                o.insert("label".into(), json!(label));
            }
            result.push(obj);
        }
    }
    result
}

/// Port of `_enabled_entries`.
pub fn enabled_entries(mail_config: &Value) -> Result<Vec<Value>, MailError> {
    let items: Vec<Value> = mail_entries(mail_config)
        .into_iter()
        .filter(|e| e.get("enable").map(is_truthy).unwrap_or(false))
        .collect();
    if items.is_empty() {
        return Err(merr("mail.providers 没有启用的 provider"));
    }
    Ok(items)
}

/// Port of `_next_entry` (round-robin across enabled providers).
fn next_entry(mail_config: &Value) -> Result<Value, MailError> {
    let items = enabled_entries(mail_config)?;
    if items.len() == 1 {
        return Ok(items[0].clone());
    }
    let i = PROVIDER_INDEX.fetch_add(1, Ordering::Relaxed) % items.len();
    Ok(items[i].clone())
}

/// Port of `_create_provider`: select an entry (by `provider_ref`, else by
/// enabled type, else round-robin) and instantiate it.
pub fn create_provider_from_config(
    mail_config: &Value,
    config: &Config,
    provider: &str,
    provider_ref: &str,
) -> Result<Box<dyn MailProvider>, MailError> {
    let entries = mail_entries(mail_config);
    let by_ref = entries.iter().find(|e| {
        !provider_ref.is_empty()
            && e.get("provider_ref").and_then(|v| v.as_str()) == Some(provider_ref)
    });
    let entry = if let Some(e) = by_ref {
        e.clone()
    } else {
        let enabled = enabled_entries(mail_config)?;
        match enabled.into_iter().find(|e| {
            !provider.is_empty() && e.get("type").and_then(|v| v.as_str()) == Some(provider)
        }) {
            Some(e) => e,
            None => next_entry(mail_config)?,
        }
    };
    let conf = mail_conf(mail_config, config);
    let t = entry.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
    create_mail_provider(&t, config, &entry, &conf)
        .ok_or_else(|| merr(format!("不支持的 mail.provider: {t}")))
}

// ---------------------------------------------------------------------------
// High-level orchestration (port of the module-level public functions)
// ---------------------------------------------------------------------------

/// Port of module-level `create_mailbox`: try enabled providers in rotation
/// until one creates a mailbox; the DDG daily-limit soft error falls through.
pub async fn create_mailbox(
    mail_config: &Value,
    config: &Config,
    username: Option<&str>,
) -> Result<Value, MailError> {
    let enabled = enabled_entries(mail_config)?;
    let mut tried: HashSet<String> = HashSet::new();
    let mut last_error = String::new();
    for _ in 0..enabled.len() {
        let provider = create_provider_from_config(mail_config, config, "", "")?;
        let key = format!("{}#{}", provider.name(), provider.provider_ref());
        if tried.contains(&key) {
            provider.close().await;
            continue;
        }
        tried.insert(key);
        let result = provider.create_mailbox(username).await;
        provider.close().await;
        match result {
            Ok(mb) => return Ok(mb),
            Err(e) => {
                last_error = e.0.clone();
                if !last_error.contains(DDG_LIMIT_SENTINEL) {
                    return Err(e);
                }
            }
        }
    }
    Err(merr(if last_error.is_empty() {
        "所有启用的邮箱提供商均无法创建邮箱".to_string()
    } else {
        last_error
    }))
}

/// Port of module-level `wait_for_code`.
pub async fn wait_for_code(
    mail_config: &Value,
    config: &Config,
    mailbox: &mut Value,
) -> Result<Option<String>, MailError> {
    let provider = create_provider_from_config(
        mail_config,
        config,
        &estr(mailbox, "provider"),
        &estr(mailbox, "provider_ref"),
    )?;
    let code = provider.wait_for_code(mailbox).await;
    provider.close().await;
    Ok(code)
}

/// Port of module-level `get_existing_mailbox`.
pub async fn get_existing_mailbox(
    mail_config: &Value,
    config: &Config,
    email: &str,
) -> Result<Value, MailError> {
    let enabled = enabled_entries(mail_config)?;
    let mut tried: HashSet<String> = HashSet::new();
    let mut last_error = String::new();
    for _ in 0..enabled.len() {
        let provider = create_provider_from_config(mail_config, config, "", "")?;
        let key = format!("{}#{}", provider.name(), provider.provider_ref());
        if tried.contains(&key) {
            provider.close().await;
            continue;
        }
        tried.insert(key);
        let result = provider.get_existing_mailbox(email).await;
        provider.close().await;
        match result {
            Ok(mb) => return Ok(mb),
            Err(e) => {
                last_error = e.0.clone();
                if !last_error.contains(DDG_LIMIT_SENTINEL) {
                    return Err(e);
                }
            }
        }
    }
    Err(merr(if last_error.is_empty() {
        "所有启用的邮箱提供商均无法查询已有邮箱".to_string()
    } else {
        last_error
    }))
}
