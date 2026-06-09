//! Port of `services/sub2api_service.py` — browse and import ChatGPT OAuth
//! accounts from a sub2api admin server.
//!
//! Mirrors the Python module's three concerns, merged into one cheaply
//! cloneable `Sub2apiService` (an `Arc<Inner>`):
//!   * a persistent server store (`sub2api_config.json` in `data_dir`) with
//!     CRUD that invalidates the per-server JWT cache on mutation;
//!   * remote fetch helpers (`list_remote_accounts` / `list_remote_groups`)
//!     that log in, cache the bearer JWT per server, and page through the
//!     admin API 200 rows at a time;
//!   * a background import (`start_import`) that fetches each account's
//!     access token concurrently, then hands the tokens to `AccountService`.
//!
//! Async notes: HTTP goes through `wreq` (proxy from `config.proxy_setting()`).
//! Mutable state lives behind a `parking_lot::Mutex`; guards are always dropped
//! before any `.await`, so the futures stay `Send` and the background import can
//! run under `tokio::spawn`. The JWT bearer is cached per server until ~5 min
//! before expiry; expiry is taken from the login response's `expires_in` and,
//! when present, refined by the token's own `exp` claim (decoded the same way
//! `account_service` does: split on '.', base64url-decode the payload).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::{STANDARD as B64, URL_SAFE_NO_PAD};
use base64::Engine;
use chrono::Utc;
use futures::stream::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::config::Config;
use crate::services::account_service::AccountService;

/// Token lifetime on sub2api defaults to 24h; refresh 5 min before expiry.
const TOKEN_REFRESH_SKEW: f64 = 5.0 * 60.0;
/// Page size for the paginated admin endpoints.
const PAGE_SIZE: i64 = 200;
/// Bound on concurrent per-account token fetches during an import.
const IMPORT_MAX_WORKERS: usize = 8;

// ---- small value helpers ----

/// Mirror of Python `str(value or "").strip()`.
fn clean(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                String::new()
            }
        }
        Value::String(s) => s.trim().to_string(),
        Value::Number(n) => {
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        other => other.to_string().trim().to_string(),
    }
}

/// `clean()` of a field by key (missing/non-object -> empty string).
fn cstr(v: &Value, key: &str) -> String {
    clean(v.get(key).unwrap_or(&Value::Null))
}

/// Mirror of Python `int(value or 0)` for JSON values.
fn as_int(v: Option<&Value>) -> i64 {
    match v {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(0),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => 0,
    }
}

/// `str(value)` (no falsy collapse) — used where Python applies plain `str()`.
fn to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn head(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn now_epoch() -> f64 {
    Utc::now().timestamp_millis() as f64 / 1000.0
}

fn now_iso() -> String {
    Utc::now().to_rfc3339()
}

fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string().chars().take(12).collect()
}

fn uuid_hex() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

// ---- JWT decode (mirrors account_service) ----

fn decode_jwt_payload(token: &str) -> Value {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() < 2 {
        return json!({});
    }
    let mut payload = parts[1].to_string();
    let pad = (4 - payload.len() % 4) % 4;
    payload.push_str(&"=".repeat(pad));
    match URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .or_else(|_| B64.decode(&payload))
    {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
            .ok()
            .filter(|v| v.is_object())
            .unwrap_or(json!({})),
        Err(_) => json!({}),
    }
}

fn jwt_exp(token: &str) -> i64 {
    decode_jwt_payload(token).get("exp").and_then(|v| v.as_i64()).unwrap_or(0)
}

// ---- envelope / pagination helpers ----

/// Peel sub2api's `{code, message, data}` envelope, returning the inner `data`
/// field when present. Unwrapped responses pass through unchanged.
fn unwrap_envelope(payload: &Value) -> Value {
    if let Some(o) = payload.as_object() {
        if o.contains_key("data") && o.contains_key("code") {
            return o.get("data").cloned().unwrap_or(Value::Null);
        }
    }
    payload.clone()
}

/// Return `(items, total)` from a paginated sub2api response, handling the
/// wrapped shape and a few looser variants.
fn extract_paged_items(payload: &Value) -> (Vec<Value>, i64) {
    let inner = unwrap_envelope(payload);
    if let Some(arr) = inner.as_array() {
        return (arr.clone(), arr.len() as i64);
    }
    if let Some(o) = inner.as_object() {
        for key in ["items", "data", "list"] {
            if let Some(Value::Array(arr)) = o.get(key) {
                let total = {
                    let t = as_int(o.get("total"));
                    if t != 0 {
                        t
                    } else {
                        arr.len() as i64
                    }
                };
                return (arr.clone(), total);
            }
        }
    }
    (Vec::new(), 0)
}

fn extract_access_token(credentials: &Value) -> String {
    if !credentials.is_object() {
        return String::new();
    }
    for key in ["access_token", "accessToken", "token"] {
        let value = cstr(credentials, key);
        if !value.is_empty() {
            return value;
        }
    }
    String::new()
}

// ---- normalization ----

fn normalize_import_job(raw: &Value, fail_unfinished: bool) -> Value {
    if !raw.is_object() {
        return Value::Null;
    }
    let mut status = {
        let s = cstr(raw, "status");
        if s.is_empty() {
            "failed".to_string()
        } else {
            s
        }
    };
    if fail_unfinished && (status == "pending" || status == "running") {
        status = "failed".to_string();
    }
    let created = {
        let c = cstr(raw, "created_at");
        if c.is_empty() {
            now_iso()
        } else {
            c
        }
    };
    let updated = {
        let u = cstr(raw, "updated_at");
        if !u.is_empty() {
            u
        } else {
            let c = cstr(raw, "created_at");
            if !c.is_empty() {
                c
            } else {
                now_iso()
            }
        }
    };
    let errors = match raw.get("errors") {
        Some(Value::Array(a)) => Value::Array(a.clone()),
        _ => json!([]),
    };
    let job_id = { let j = cstr(raw, "job_id"); if j.is_empty() { uuid_hex() } else { j } };
    json!({
        "job_id": job_id,
        "status": status,
        "created_at": created,
        "updated_at": updated,
        "total": as_int(raw.get("total")),
        "completed": as_int(raw.get("completed")),
        "added": as_int(raw.get("added")),
        "skipped": as_int(raw.get("skipped")),
        "refreshed": as_int(raw.get("refreshed")),
        "failed": as_int(raw.get("failed")),
        "errors": errors,
    })
}

fn normalize_server(raw: &Value) -> Value {
    let id = { let i = cstr(raw, "id"); if i.is_empty() { new_id() } else { i } };
    json!({
        "id": id,
        "name": cstr(raw, "name"),
        "base_url": cstr(raw, "base_url"),
        "email": cstr(raw, "email"),
        "password": cstr(raw, "password"),
        "api_key": cstr(raw, "api_key"),
        "group_id": cstr(raw, "group_id"),
        "import_job": normalize_import_job(raw.get("import_job").unwrap_or(&Value::Null), true),
    })
}

// ---- service ----

struct State {
    servers: Vec<Value>,
    /// Per-server cached access token: server_id -> (jwt, expires_at_epoch).
    token_cache: HashMap<String, (String, f64)>,
}

struct Inner {
    config: Config,
    accounts: Arc<AccountService>,
    state: Mutex<State>,
}

#[derive(Clone)]
pub struct Sub2apiService {
    inner: Arc<Inner>,
}

impl Sub2apiService {
    pub fn new(config: Config, accounts: Arc<AccountService>) -> Self {
        let servers = Self::load_servers(&config);
        Self {
            inner: Arc::new(Inner {
                config,
                accounts,
                state: Mutex::new(State {
                    servers,
                    token_cache: HashMap::new(),
                }),
            }),
        }
    }

    // ---- persistence ----

    fn store_file(config: &Config) -> PathBuf {
        config.data_dir().join("sub2api_config.json")
    }

    fn load_servers(config: &Config) -> Vec<Value> {
        let path = Self::store_file(config);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Array(arr)) => arr.iter().filter(|i| i.is_object()).map(normalize_server).collect(),
            _ => Vec::new(),
        }
    }

    fn persist(config: &Config, servers: &[Value]) {
        let path = Self::store_file(config);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let text = serde_json::to_string_pretty(servers).unwrap_or_else(|_| "[]".to_string()) + "\n";
        if let Err(e) = std::fs::write(&path, text) {
            tracing::warn!("save sub2api_config failed: {e}");
        }
    }

    // ---- server CRUD ----

    pub fn list_servers(&self) -> Vec<Value> {
        self.inner.state.lock().servers.clone()
    }

    pub fn get_server(&self, server_id: &str) -> Option<Value> {
        self.inner
            .state
            .lock()
            .servers
            .iter()
            .find(|s| cstr(s, "id") == server_id)
            .cloned()
    }

    pub fn add_server(
        &self,
        name: &str,
        base_url: &str,
        email: &str,
        password: &str,
        api_key: &str,
        group_id: &str,
    ) -> Value {
        let server = normalize_server(&json!({
            "id": new_id(),
            "name": name,
            "base_url": base_url,
            "email": email,
            "password": password,
            "api_key": api_key,
            "group_id": group_id,
        }));
        let mut state = self.inner.state.lock();
        state.servers.push(server.clone());
        Self::persist(&self.inner.config, &state.servers);
        let id = cstr(&server, "id");
        state.token_cache.remove(&id);
        server
    }

    pub fn update_server(&self, server_id: &str, updates: &Value) -> Option<Value> {
        let mut state = self.inner.state.lock();
        let idx = state.servers.iter().position(|s| cstr(s, "id") == server_id)?;
        let mut merged = state.servers[idx].clone();
        if let (Some(m), Some(u)) = (merged.as_object_mut(), updates.as_object()) {
            for (k, v) in u {
                if !v.is_null() {
                    m.insert(k.clone(), v.clone());
                }
            }
        }
        merged["id"] = Value::String(server_id.to_string());
        let normalized = normalize_server(&merged);
        state.servers[idx] = normalized.clone();
        Self::persist(&self.inner.config, &state.servers);
        state.token_cache.remove(server_id);
        Some(normalized)
    }

    pub fn delete_server(&self, server_id: &str) -> bool {
        let mut state = self.inner.state.lock();
        let before = state.servers.len();
        state.servers.retain(|s| cstr(s, "id") != server_id);
        let removed = state.servers.len() < before;
        if removed {
            Self::persist(&self.inner.config, &state.servers);
            state.token_cache.remove(server_id);
        }
        removed
    }

    fn set_import_job(&self, server_id: &str, import_job: &Value) -> Option<Value> {
        let mut state = self.inner.state.lock();
        let idx = state.servers.iter().position(|s| cstr(s, "id") == server_id)?;
        let mut next = state.servers[idx].clone();
        next["import_job"] = normalize_import_job(import_job, false);
        state.servers[idx] = next.clone();
        Self::persist(&self.inner.config, &state.servers);
        Some(next)
    }

    pub fn get_import_job(&self, server_id: &str) -> Option<Value> {
        let state = self.inner.state.lock();
        let server = state.servers.iter().find(|s| cstr(s, "id") == server_id)?;
        match server.get("import_job") {
            Some(v) if v.is_object() => Some(v.clone()),
            _ => None,
        }
    }

    // ---- HTTP plumbing ----

    fn build_client(&self) -> Result<wreq::Client, String> {
        let mut builder = wreq::Client::builder().emulation(wreq_util::Emulation::Chrome137);
        let proxy = self.inner.config.proxy_setting();
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        builder.build().map_err(|e| format!("client build: {e}"))
    }

    async fn send(
        &self,
        client: &wreq::Client,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        params: &[(&str, String)],
        json_body: Option<&Value>,
    ) -> Result<(u16, String), String> {
        let m = wreq::Method::from_bytes(method.as_bytes())
            .map_err(|e| format!("invalid HTTP method {method}: {e}"))?;
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
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("sub2api network error: {e}"))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        Ok((status, text))
    }

    /// Log in with email/password, returning `(jwt, expires_at_epoch)`.
    async fn login(&self, base_url: &str, email: &str, password: &str) -> Result<(String, f64), String> {
        let url = format!("{}/api/v1/auth/login", base_url.trim_end_matches('/'));
        let client = self.build_client()?;
        let body = json!({ "email": email, "password": password });
        let headers = vec![
            ("Accept".to_string(), "application/json".to_string()),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        let (status, text) = self.send(&client, "POST", &url, &headers, &[], Some(&body)).await?;
        if status >= 400 {
            return Err(format!("sub2api login failed: HTTP {status} {}", head(&text, 200)));
        }
        let payload: Value =
            serde_json::from_str(&text).map_err(|e| format!("sub2api login payload is invalid: {e}"))?;
        let body = unwrap_envelope(&payload);
        if !body.is_object() {
            return Err("sub2api login payload is invalid".to_string());
        }
        let token = cstr(&body, "access_token");
        if token.is_empty() {
            return Err("sub2api login did not return access_token".to_string());
        }
        let expires_in = {
            let v = as_int(body.get("expires_in"));
            if v != 0 {
                v
            } else {
                3600
            }
        };
        let mut expires_at = now_epoch() + (expires_in.max(60) as f64) - TOKEN_REFRESH_SKEW;
        // Prefer the JWT's own `exp` claim when present (more accurate).
        let exp = jwt_exp(&token);
        if exp > 0 {
            expires_at = exp as f64 - TOKEN_REFRESH_SKEW;
        }
        Ok((token, expires_at))
    }

    /// Build auth headers, reusing/refreshing the cached bearer JWT as needed.
    async fn auth_headers(&self, server: &Value) -> Result<Vec<(String, String)>, String> {
        let api_key = cstr(server, "api_key");
        if !api_key.is_empty() {
            return Ok(vec![
                ("x-api-key".to_string(), api_key),
                ("Accept".to_string(), "application/json".to_string()),
            ]);
        }

        let email = cstr(server, "email");
        let password = cstr(server, "password");
        if email.is_empty() || password.is_empty() {
            return Err("sub2api server requires email+password or api_key".to_string());
        }
        let server_id = cstr(server, "id");
        let base_url = cstr(server, "base_url");

        {
            let state = self.inner.state.lock();
            if let Some((tok, exp)) = state.token_cache.get(&server_id) {
                if *exp > now_epoch() {
                    return Ok(vec![
                        ("Authorization".to_string(), format!("Bearer {tok}")),
                        ("Accept".to_string(), "application/json".to_string()),
                    ]);
                }
            }
        }

        let (token, expires_at) = self.login(&base_url, &email, &password).await?;
        {
            let mut state = self.inner.state.lock();
            state.token_cache.insert(server_id, (token.clone(), expires_at));
        }
        Ok(vec![
            ("Authorization".to_string(), format!("Bearer {token}")),
            ("Accept".to_string(), "application/json".to_string()),
        ])
    }

    // ---- remote fetch ----

    /// Return a flat list of OpenAI OAuth accounts from a sub2api server.
    pub async fn list_remote_accounts(&self, server: &Value) -> Result<Vec<Value>, String> {
        let base_url = cstr(server, "base_url");
        if base_url.is_empty() {
            return Ok(Vec::new());
        }
        let headers = self.auth_headers(server).await?;
        let group_id = cstr(server, "group_id");
        let client = self.build_client()?;
        let url = format!("{}/api/v1/admin/accounts", base_url.trim_end_matches('/'));

        let mut items: Vec<Value> = Vec::new();
        let mut page: i64 = 1;
        loop {
            let mut params: Vec<(&str, String)> = vec![
                ("platform", "openai".to_string()),
                ("type", "oauth".to_string()),
                ("page", page.to_string()),
                ("page_size", PAGE_SIZE.to_string()),
            ];
            if !group_id.is_empty() {
                params.push(("group", group_id.clone()));
            }
            let (status, text) = self.send(&client, "GET", &url, &headers, &params, None).await?;
            if status >= 400 {
                return Err(format!("sub2api list failed: HTTP {status} {}", head(&text, 200)));
            }
            let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            let (data, total) = extract_paged_items(&payload);
            if data.is_empty() {
                break;
            }

            for account in &data {
                if !account.is_object() {
                    continue;
                }
                let credentials = account
                    .get("credentials")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let access_token = extract_access_token(&credentials);
                if access_token.is_empty() {
                    continue;
                }
                let id = match account.get("id") {
                    Some(Value::Null) | None => cstr(&credentials, "chatgpt_account_id"),
                    Some(v) => to_str(v),
                };
                let email = {
                    let e = cstr(&credentials, "email");
                    if !e.is_empty() {
                        e
                    } else {
                        cstr(account, "name")
                    }
                };
                items.push(json!({
                    "id": id,
                    "name": cstr(account, "name"),
                    "email": email,
                    "plan_type": cstr(&credentials, "plan_type"),
                    "status": cstr(account, "status"),
                    "expires_at": cstr(&credentials, "expires_at"),
                    "has_refresh_token": !cstr(&credentials, "refresh_token").is_empty(),
                }));
            }

            let data_len = data.len() as i64;
            if page * PAGE_SIZE >= total || data_len < PAGE_SIZE {
                break;
            }
            page += 1;
        }
        Ok(items)
    }

    /// Return OpenAI account groups from a sub2api server.
    pub async fn list_remote_groups(&self, server: &Value) -> Result<Vec<Value>, String> {
        let base_url = cstr(server, "base_url");
        if base_url.is_empty() {
            return Ok(Vec::new());
        }
        let headers = self.auth_headers(server).await?;
        let client = self.build_client()?;
        let url = format!("{}/api/v1/admin/groups", base_url.trim_end_matches('/'));

        let mut items: Vec<Value> = Vec::new();
        let mut page: i64 = 1;
        loop {
            let params: Vec<(&str, String)> =
                vec![("page", page.to_string()), ("page_size", PAGE_SIZE.to_string())];
            let (status, text) = self.send(&client, "GET", &url, &headers, &params, None).await?;
            if status >= 400 {
                return Err(format!("sub2api groups failed: HTTP {status} {}", head(&text, 200)));
            }
            let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            let (data, total) = extract_paged_items(&payload);
            if data.is_empty() {
                break;
            }

            for group in &data {
                if !group.is_object() {
                    continue;
                }
                let group_id = match group.get("id") {
                    Some(Value::Null) | None => continue,
                    Some(v) => to_str(v),
                };
                items.push(json!({
                    "id": group_id,
                    "name": cstr(group, "name"),
                    "description": cstr(group, "description"),
                    "platform": cstr(group, "platform"),
                    "status": cstr(group, "status"),
                    "account_count": as_int(group.get("account_count")),
                    "active_account_count": as_int(group.get("active_account_count")),
                }));
            }

            let data_len = data.len() as i64;
            if page * PAGE_SIZE >= total || data_len < PAGE_SIZE {
                break;
            }
            page += 1;
        }
        Ok(items)
    }

    /// Return `(access_token, meta)` for a single sub2api account id.
    async fn fetch_access_token_for_account(
        &self,
        server: &Value,
        account_id: &str,
    ) -> Result<(String, Value), String> {
        let base_url = cstr(server, "base_url");
        let headers = self.auth_headers(server).await?;
        let client = self.build_client()?;
        let url = format!("{}/api/v1/admin/accounts/{}", base_url.trim_end_matches('/'), account_id);
        let (status, text) = self.send(&client, "GET", &url, &headers, &[], None).await?;
        if status >= 400 {
            return Err(format!("HTTP {status}"));
        }
        let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let unwrapped = unwrap_envelope(&payload);
        let account = if unwrapped.is_object() {
            unwrapped
        } else if payload.is_object() {
            payload
        } else {
            json!({})
        };
        let credentials = account
            .get("credentials")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| json!({}));
        let access_token = extract_access_token(&credentials);
        if access_token.is_empty() {
            return Err("missing access_token".to_string());
        }
        let meta = json!({
            "email": cstr(&credentials, "email"),
            "plan_type": cstr(&credentials, "plan_type"),
        });
        Ok((access_token, meta))
    }

    // ---- import ----

    fn update_job(&self, server_id: &str, updates: &Value) {
        let Some(current) = self.get_import_job(server_id) else {
            return;
        };
        let mut next = current;
        if let (Some(n), Some(u)) = (next.as_object_mut(), updates.as_object()) {
            for (k, v) in u {
                n.insert(k.clone(), v.clone());
            }
        }
        next["updated_at"] = Value::String(now_iso());
        self.set_import_job(server_id, &next);
    }

    fn append_error(&self, server_id: &str, account_id: &str, message: &str) {
        let Some(current) = self.get_import_job(server_id) else {
            return;
        };
        let mut errors = current
            .get("errors")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        errors.push(json!({ "name": account_id, "error": message }));
        let failed = errors.len();
        self.update_job(server_id, &json!({ "errors": errors, "failed": failed }));
    }

    /// Kick off a background import of the given account ids; returns the
    /// freshly-created (pending) import job.
    pub fn start_import(&self, server: &Value, account_ids: &[String]) -> Result<Value, String> {
        let ids: Vec<String> = account_ids
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if ids.is_empty() {
            return Err("account ids is required".to_string());
        }
        let server_id = cstr(server, "id");
        let job = json!({
            "job_id": uuid_hex(),
            "status": "pending",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "total": ids.len(),
            "completed": 0,
            "added": 0,
            "skipped": 0,
            "refreshed": 0,
            "failed": 0,
            "errors": [],
        });
        let saved = self
            .set_import_job(&server_id, &job)
            .ok_or_else(|| "server not found".to_string())?;

        let this = self.clone();
        let server = server.clone();
        let sid = server_id.clone();
        tokio::spawn(async move {
            this.run_import(sid, server, ids).await;
        });

        Ok(saved.get("import_job").cloned().unwrap_or(job))
    }

    async fn run_import(&self, server_id: String, server: Value, account_ids: Vec<String>) {
        self.update_job(&server_id, &json!({ "status": "running" }));

        let max_workers = std::cmp::min(IMPORT_MAX_WORKERS, std::cmp::max(1, account_ids.len()));
        let mut tokens: Vec<String> = Vec::new();
        {
            let mut stream = futures::stream::iter(account_ids.iter().cloned().map(|account_id| {
                let this = self.clone();
                let server = server.clone();
                async move {
                    let r = this.fetch_access_token_for_account(&server, &account_id).await;
                    (account_id, r)
                }
            }))
            .buffer_unordered(max_workers);

            while let Some((account_id, res)) = stream.next().await {
                match res {
                    Ok((token, _meta)) => tokens.push(token),
                    Err(e) => {
                        let msg = if e.is_empty() { "unknown error".to_string() } else { e };
                        self.append_error(&server_id, &account_id, &msg);
                    }
                }
                let current = self.get_import_job(&server_id).unwrap_or_else(|| json!({}));
                let failed = current.get("errors").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                let completed = as_int(current.get("completed")) + 1;
                self.update_job(&server_id, &json!({ "completed": completed, "failed": failed }));
            }
        }

        if tokens.is_empty() {
            let current = self.get_import_job(&server_id).unwrap_or_else(|| json!({}));
            let failed = current.get("errors").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            let total = as_int(current.get("total"));
            self.update_job(
                &server_id,
                &json!({ "status": "failed", "completed": total, "failed": failed }),
            );
            return;
        }

        let add_result = self.inner.accounts.add_accounts(&tokens, "codex");
        let refresh_result = self.inner.accounts.refresh_accounts(&tokens, None, false).await;
        let current = self.get_import_job(&server_id).unwrap_or_else(|| json!({}));
        let failed = current.get("errors").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
        self.update_job(
            &server_id,
            &json!({
                "status": "completed",
                "completed": account_ids.len(),
                "added": as_int(add_result.get("added")),
                "skipped": as_int(add_result.get("skipped")),
                "refreshed": as_int(refresh_result.get("refreshed")),
                "failed": failed,
            }),
        );
    }
}
