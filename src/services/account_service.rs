//! Port of `services/account_service.py` — the account pool. Holds a
//! `token -> account` map, round-robins available accounts for image/text
//! requests with per-token concurrency slots, refreshes OAuth access tokens,
//! and tracks per-account statistics.
//!
//! Async notes: the in-memory pool state lives behind a `parking_lot::Mutex`
//! for fast synchronous mutation; engine-calling methods (`fetch_remote_info`,
//! `refresh_access_token`, `get_available_access_token`) are `async`. The image
//! slot `Condition` is modeled with a `tokio::sync::Notify`.
//!
//! Deferred to Phase 9 (needs the sentinel-HTTP password-login machinery shared
//! with the registration flow): `login_with_password` currently returns an
//! error result, so password re-login gracefully marks accounts abnormal.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::{STANDARD as B64, URL_SAFE_NO_PAD};
use base64::Engine;
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use crate::config::Config;
use crate::services::log_service::{LogService, LOG_TYPE_ACCOUNT};
use crate::services::openai_backend_api::{EngineError, OpenAIBackendAPI};
use crate::services::storage::StorageBackend;
use crate::utils::helper::anonymize_token;

const NEW_ACCOUNT_INVALID_GRACE_SECONDS: f64 = 10.0 * 60.0;
const INVALID_CONFIRM_SECONDS: f64 = 30.0;
const ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 24 * 60 * 60;
const REFRESH_TOKEN_KEEPALIVE_SECONDS: i64 = 3 * 24 * 60 * 60;
const REFRESH_TOKEN_KEEPALIVE_ERROR_BACKOFF_SECONDS: f64 = 6.0 * 60.0 * 60.0;
const REFRESH_TOKEN_KEEPALIVE_BATCH_SIZE: usize = 3;
const TOKEN_REFRESH_ERROR_BACKOFF_SECONDS: f64 = 5.0 * 60.0;
const OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OAUTH_CLIENT_ID: &str = "app_2SKx67EdpoN0G6j64rFvigXD";
const OAUTH_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36";

// ---- small helpers on Value objects ----

fn s(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn s_trim(v: &Value, key: &str) -> String {
    s(v, key).trim().to_string()
}

fn now_str() -> String {
    Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn now_local_str() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn decode_jwt_payload(token: &str) -> Value {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() < 2 {
        return json!({});
    }
    let mut payload = parts[1].to_string();
    let pad = (4 - payload.len() % 4) % 4;
    payload.push_str(&"=".repeat(pad));
    match URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).or_else(|_| B64.decode(&payload)) {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
            .ok()
            .filter(|v| v.is_object())
            .unwrap_or(json!({})),
        Err(_) => json!({}),
    }
}

fn jwt_exp(access_token: &str) -> i64 {
    decode_jwt_payload(access_token).get("exp").and_then(|v| v.as_i64()).unwrap_or(0)
}

fn token_expires_in(access_token: &str) -> Option<i64> {
    let exp = jwt_exp(access_token);
    if exp <= 0 {
        None
    } else {
        Some(exp - Utc::now().timestamp())
    }
}

fn token_needs_refresh(access_token: &str, force: bool) -> bool {
    if force {
        return true;
    }
    token_expires_in(access_token).map_or(false, |r| r <= ACCESS_TOKEN_REFRESH_SKEW_SECONDS)
}

fn token_issued_at(access_token: &str) -> Option<DateTime<Utc>> {
    let iat = decode_jwt_payload(access_token).get("iat").and_then(|v| v.as_i64()).unwrap_or(0);
    if iat <= 0 {
        None
    } else {
        DateTime::from_timestamp(iat, 0)
    }
}

fn parse_time(value: &Value) -> Option<DateTime<Utc>> {
    let raw = match value {
        Value::String(s) => s.trim().to_string(),
        Value::Null | _ if value.is_null() => String::new(),
        other => other.to_string(),
    };
    if raw.is_empty() {
        return None;
    }
    let normalized = raw.replace('Z', "+00:00");
    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(&raw, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::from_naive_utc_and_offset(ndt, Utc));
    }
    None
}

fn timestamp_to_iso(value: &Value) -> String {
    let ts = match value {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    };
    let Some(ts) = ts else { return String::new() };
    let tz = chrono::FixedOffset::east_opt(8 * 3600).unwrap();
    match DateTime::from_timestamp(ts, 0) {
        Some(dt) => dt.with_timezone(&tz).to_rfc3339(),
        None => String::new(),
    }
}

fn normalize_source_type(value: &Value) -> String {
    let raw = match value {
        Value::String(s) => s.trim().to_lowercase(),
        Value::Null => String::new(),
        other => other.to_string().trim().to_lowercase(),
    };
    if raw.is_empty() {
        "web".to_string()
    } else {
        raw
    }
}

fn normalize_account_type(value: &Value) -> Option<String> {
    let raw = match value {
        Value::String(s) => s.trim().to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    if raw.is_empty() {
        return None;
    }
    let key = raw.to_lowercase().replace('-', "_").replace(' ', "_");
    let compact = key.replace('_', "");
    let aliases: &[(&str, &str)] = &[
        ("free", "free"),
        ("plus", "Plus"),
        ("pro", "Pro"),
        ("prolite", "ProLite"),
        ("team", "Team"),
        ("business", "Team"),
        ("enterprise", "Enterprise"),
    ];
    for (k, v) in aliases {
        if *k == compact {
            return Some(v.to_string());
        }
    }
    for (k, v) in aliases {
        if *k == key {
            return Some(v.to_string());
        }
    }
    Some(raw)
}

// ---- pool ----

struct Pool {
    accounts: IndexMap<String, Value>,
    index: usize,
    image_inflight: HashMap<String, i64>,
    token_aliases: HashMap<String, String>,
    cumulative_total: i64,
}

pub struct AccountService {
    storage: Arc<dyn StorageBackend>,
    config: Config,
    log: LogService,
    inner: Mutex<Pool>,
    slot_notify: tokio::sync::Notify,
    token_refresh_lock: tokio::sync::Mutex<()>,
    refresh_progress: Mutex<HashMap<String, Value>>,
    relogin_progress: Mutex<HashMap<String, Value>>,
}

impl AccountService {
    pub fn new(storage: Arc<dyn StorageBackend>, config: Config, log: LogService) -> Self {
        let accounts = Self::load_accounts(&storage);
        let cumulative_total = Self::load_cumulative_total(&config, accounts.len() as i64);
        Self {
            storage,
            config,
            log,
            inner: Mutex::new(Pool {
                accounts,
                index: 0,
                image_inflight: HashMap::new(),
                token_aliases: HashMap::new(),
                cumulative_total,
            }),
            slot_notify: tokio::sync::Notify::new(),
            token_refresh_lock: tokio::sync::Mutex::new(()),
            refresh_progress: Mutex::new(HashMap::new()),
            relogin_progress: Mutex::new(HashMap::new()),
        }
    }

    // ---- cumulative total file ----

    fn cumulative_file(config: &Config) -> std::path::PathBuf {
        config.data_dir().join(".cumulative_total")
    }

    fn load_cumulative_total(config: &Config, fallback: i64) -> i64 {
        std::fs::read_to_string(Self::cumulative_file(config))
            .ok()
            .and_then(|s| s.trim().parse::<i64>().ok())
            .unwrap_or(fallback)
    }

    fn save_cumulative_total(&self, total: i64) {
        let _ = std::fs::write(Self::cumulative_file(&self.config), total.to_string());
    }

    // ---- load / save / normalize ----

    fn load_accounts(storage: &Arc<dyn StorageBackend>) -> IndexMap<String, Value> {
        let mut map = IndexMap::new();
        for item in storage.load_accounts() {
            if let Some(norm) = Self::normalize_account(&item) {
                let token = s(&norm, "access_token");
                map.insert(token, norm);
            }
        }
        map
    }

    fn save_accounts_locked(&self, pool: &Pool) {
        let values: Vec<Value> = pool.accounts.values().cloned().collect();
        if let Err(e) = self.storage.save_accounts(&values) {
            tracing::warn!("save_accounts failed: {e}");
        }
    }

    fn normalize_account(item: &Value) -> Option<Value> {
        let obj = item.as_object()?;
        let access_token = {
            let a = obj.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
            let b = obj.get("accessToken").and_then(|v| v.as_str()).unwrap_or("");
            let t = if !a.is_empty() { a } else { b };
            t.to_string()
        };
        if access_token.is_empty() {
            return None;
        }
        let mut n: Map<String, Value> = obj.clone();
        n.remove("accessToken");
        n.insert("access_token".into(), Value::String(access_token));

        if s(item, "type").trim().to_lowercase() == "codex" {
            n.insert("export_type".into(), Value::String("codex".into()));
            n.remove("type");
        }
        let type_val = {
            let t = n.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if t.is_empty() { "free".to_string() } else { t.to_string() }
        };
        n.insert("type".into(), Value::String(type_val));
        let status = {
            let t = n.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if t.is_empty() { "正常".to_string() } else { t.to_string() }
        };
        n.insert("status".into(), Value::String(status));
        let quota = n.get("quota").and_then(|v| v.as_i64()).unwrap_or(0).max(0);
        n.insert("quota".into(), json!(quota));
        let iqu = n.get("image_quota_unknown").and_then(|v| v.as_bool()).unwrap_or(false);
        n.insert("image_quota_unknown".into(), json!(iqu));
        Self::set_or_null(&mut n, "email");
        Self::set_or_null(&mut n, "user_id");
        let proxy = n.get("proxy").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        n.insert("proxy".into(), Value::String(proxy));

        let source_type = {
            let st = n.get("source_type").cloned().unwrap_or(Value::Null);
            let st_empty = st.as_str().map_or(true, |v| v.trim().is_empty());
            if st_empty && n.get("export_type").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase() == "codex" {
                Value::String("codex".into())
            } else {
                st
            }
        };
        n.insert("source_type".into(), Value::String(normalize_source_type(&source_type)));

        let limits = n.get("limits_progress").cloned().unwrap_or(Value::Null);
        n.insert("limits_progress".into(), if limits.is_array() { limits } else { json!([]) });
        Self::set_or_null(&mut n, "default_model_slug");
        Self::set_or_null(&mut n, "restore_at");
        for k in ["success", "fail", "invalid_count"] {
            let v = n.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
            n.insert(k.into(), json!(v));
        }
        for k in [
            "last_used_at",
            "last_invalid_at",
            "last_refresh_error",
            "last_refresh_error_at",
            "last_token_refresh_at",
            "last_token_refresh_error",
            "last_token_refresh_error_at",
        ] {
            Self::set_or_null(&mut n, k);
        }
        let created_at = {
            let t = n.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            if t.is_empty() { now_str() } else { t.to_string() }
        };
        n.insert("created_at".into(), Value::String(created_at));
        Some(Value::Object(n))
    }

    /// Set a key to its string value, or null if empty/missing (Python `x or None`).
    fn set_or_null(n: &mut Map<String, Value>, key: &str) {
        let v = n.get(key).cloned().unwrap_or(Value::Null);
        let keep = match &v {
            Value::String(s) => !s.is_empty(),
            Value::Null => false,
            _ => true,
        };
        n.insert(key.into(), if keep { v } else { Value::Null });
    }

    // ---- token resolution ----

    fn resolve_locked(pool: &Pool, access_token: &str) -> String {
        let mut token = access_token.trim().to_string();
        let mut seen: HashSet<String> = HashSet::new();
        while !token.is_empty()
            && !pool.accounts.contains_key(&token)
            && pool.token_aliases.contains_key(&token)
            && !seen.contains(&token)
        {
            seen.insert(token.clone());
            token = pool.token_aliases.get(&token).cloned().unwrap_or(token);
        }
        token
    }

    pub fn resolve_access_token(&self, access_token: &str) -> String {
        if access_token.is_empty() {
            return String::new();
        }
        let pool = self.inner.lock();
        Self::resolve_locked(&pool, access_token)
    }

    // ---- account availability / matching ----

    fn is_image_account_available(account: &Value) -> bool {
        if !account.is_object() {
            return false;
        }
        let status = account.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if matches!(status, "禁用" | "限流" | "异常") {
            return false;
        }
        if account.get("image_quota_unknown").and_then(|v| v.as_bool()).unwrap_or(false) {
            return true;
        }
        account.get("quota").and_then(|v| v.as_i64()).unwrap_or(0) > 0
    }

    fn account_matches_plan_type(account: &Value, plan_type: Option<&str>) -> bool {
        let Some(plan_type) = plan_type.filter(|s| !s.is_empty()) else {
            return true;
        };
        let np = normalize_account_type(&Value::String(plan_type.to_string()));
        let na = normalize_account_type(account.get("type").unwrap_or(&Value::Null));
        match (np, na) {
            (Some(np), Some(na)) => np.to_lowercase() == na.to_lowercase(),
            _ => false,
        }
    }

    fn account_matches_source_type(account: &Value, source_type: Option<&str>) -> bool {
        let Some(source_type) = source_type.filter(|s| !s.is_empty()) else {
            return true;
        };
        normalize_source_type(account.get("source_type").unwrap_or(&Value::Null))
            == normalize_source_type(&Value::String(source_type.to_string()))
    }

    fn account_matches_any_plan_type(account: &Value, plan_types: &[String]) -> bool {
        if plan_types.is_empty() {
            return true;
        }
        let na = normalize_account_type(account.get("type").unwrap_or(&Value::Null));
        let Some(na) = na else { return false };
        let normalized: HashSet<String> = plan_types
            .iter()
            .filter_map(|p| normalize_account_type(&Value::String(p.clone())))
            .collect();
        normalized.contains(&na)
    }

    // ---- basic getters ----

    pub fn get_account(&self, access_token: &str) -> Option<Value> {
        if access_token.is_empty() {
            return None;
        }
        let pool = self.inner.lock();
        let resolved = Self::resolve_locked(&pool, access_token);
        pool.accounts.get(&resolved).cloned()
    }

    pub fn list_accounts(&self) -> Vec<Value> {
        let pool = self.inner.lock();
        pool.accounts.values().cloned().collect()
    }

    pub fn list_tokens(&self) -> Vec<String> {
        let pool = self.inner.lock();
        pool.accounts.keys().cloned().collect()
    }

    pub fn list_limited_tokens(&self) -> Vec<String> {
        let pool = self.inner.lock();
        pool.accounts
            .values()
            .filter(|a| a.get("status").and_then(|v| v.as_str()) == Some("限流"))
            .map(|a| s(a, "access_token"))
            .filter(|t| !t.is_empty())
            .collect()
    }

    // ---- candidate selection ----

    fn list_ready_candidate_tokens(
        pool: &Pool,
        excluded: &HashSet<String>,
        plan_type: Option<&str>,
        source_type: Option<&str>,
        plan_types: &[String],
    ) -> Vec<String> {
        pool.accounts
            .values()
            .filter(|item| {
                Self::is_image_account_available(item)
                    && Self::account_matches_plan_type(item, plan_type)
                    && Self::account_matches_any_plan_type(item, plan_types)
                    && Self::account_matches_source_type(item, source_type)
            })
            .map(|item| s(item, "access_token"))
            .filter(|t| !t.is_empty() && !excluded.contains(t))
            .collect()
    }

    fn list_available_candidate_tokens(
        &self,
        pool: &Pool,
        excluded: &HashSet<String>,
        plan_type: Option<&str>,
        source_type: Option<&str>,
        plan_types: &[String],
    ) -> Vec<String> {
        let max_concurrency = self.config.image_account_concurrency().max(1);
        Self::list_ready_candidate_tokens(pool, excluded, plan_type, source_type, plan_types)
            .into_iter()
            .filter(|t| pool.image_inflight.get(t).copied().unwrap_or(0) < max_concurrency)
            .collect()
    }

    async fn acquire_next_candidate_token(
        &self,
        excluded: &HashSet<String>,
        plan_type: Option<&str>,
        source_type: Option<&str>,
        plan_types: &[String],
    ) -> Result<String, String> {
        loop {
            {
                let mut pool = self.inner.lock();
                if Self::list_ready_candidate_tokens(&pool, excluded, plan_type, source_type, plan_types)
                    .is_empty()
                {
                    let label = plan_type.or(source_type).unwrap_or("");
                    return Err(if label.is_empty() {
                        "no available image quota".to_string()
                    } else {
                        format!("no available {label} image quota")
                    });
                }
                let tokens =
                    self.list_available_candidate_tokens(&pool, excluded, plan_type, source_type, plan_types);
                if !tokens.is_empty() {
                    let idx = pool.index % tokens.len();
                    let token = tokens[idx].clone();
                    pool.index += 1;
                    *pool.image_inflight.entry(token.clone()).or_insert(0) += 1;
                    return Ok(token);
                }
            }
            let _ = tokio::time::timeout(Duration::from_secs(1), self.slot_notify.notified()).await;
        }
    }

    pub fn release_image_slot(&self, access_token: &str) {
        if access_token.is_empty() {
            return;
        }
        {
            let mut pool = self.inner.lock();
            let token = Self::resolve_locked(&pool, access_token);
            let cur = pool.image_inflight.get(&token).copied().unwrap_or(0);
            if cur <= 1 {
                pool.image_inflight.remove(&token);
            } else {
                pool.image_inflight.insert(token, cur - 1);
            }
        }
        self.slot_notify.notify_waiters();
    }

    /// Acquire a usable image-generation token (remote-validated).
    pub async fn get_available_access_token(
        &self,
        plan_type: Option<&str>,
        source_type: Option<&str>,
        plan_types: &[String],
    ) -> Result<String, String> {
        let max_attempts = 20;
        let mut attempted: HashSet<String> = HashSet::new();
        for _ in 0..max_attempts {
            let access_token = self
                .acquire_next_candidate_token(&attempted, plan_type, source_type, plan_types)
                .await?;
            attempted.insert(access_token.clone());
            let account = match self.fetch_remote_info(&access_token, "get_available_access_token", true).await {
                Ok(a) => a,
                Err(_) => {
                    self.release_image_slot(&access_token);
                    continue;
                }
            };
            let account = account.unwrap_or(json!({}));
            let resolved = s(&account, "access_token");
            if !resolved.is_empty() && resolved != access_token {
                attempted.insert(resolved.clone());
            }
            if Self::is_image_account_available(&account)
                && Self::account_matches_plan_type(&account, plan_type)
                && Self::account_matches_any_plan_type(&account, plan_types)
                && Self::account_matches_source_type(&account, source_type)
            {
                let t = s(&account, "access_token");
                return Ok(if t.is_empty() { access_token } else { t });
            }
            self.release_image_slot(&access_token);
        }
        let label = plan_type.or(source_type).unwrap_or("");
        Err(if label.is_empty() {
            format!("no available image quota (tried {} tokens)", attempted.len())
        } else {
            format!("no available {label} image quota (tried {} tokens)", attempted.len())
        })
    }

    /// Acquire a text-conversation token (round-robin, skipping disabled/abnormal).
    pub async fn get_text_access_token(&self, excluded: &HashSet<String>) -> String {
        let access_token = {
            let mut pool = self.inner.lock();
            let candidates: Vec<String> = pool
                .accounts
                .values()
                .filter(|a| {
                    !matches!(a.get("status").and_then(|v| v.as_str()), Some("禁用") | Some("异常"))
                })
                .map(|a| s(a, "access_token"))
                .filter(|t| !t.is_empty() && !excluded.contains(t))
                .collect();
            if candidates.is_empty() {
                return String::new();
            }
            let idx = pool.index % candidates.len();
            pool.index += 1;
            candidates[idx].clone()
        };
        let refreshed = self.refresh_access_token(&access_token, false, "get_text_access_token").await;
        if refreshed.is_empty() {
            access_token
        } else {
            refreshed
        }
    }

    pub fn mark_text_used(&self, access_token: &str) {
        if access_token.is_empty() {
            return;
        }
        let mut pool = self.inner.lock();
        let token = Self::resolve_locked(&pool, access_token);
        let Some(current) = pool.accounts.get(&token).cloned() else {
            return;
        };
        let mut next = current;
        next["last_used_at"] = Value::String(now_local_str());
        if let Some(account) = Self::normalize_account(&next) {
            pool.accounts.insert(token, account);
            self.save_accounts_locked(&pool);
        }
    }

    // ---- add / delete / update ----

    fn account_payload_token(item: &Value) -> String {
        let a = item.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
        let b = item.get("accessToken").and_then(|v| v.as_str()).unwrap_or("");
        (if !a.is_empty() { a } else { b }).trim().to_string()
    }

    fn prepare_account_payload(item: &Value) -> Option<Value> {
        let obj = item.as_object()?;
        let access_token = Self::account_payload_token(item);
        if access_token.is_empty() {
            return None;
        }
        let mut payload: Map<String, Value> = obj.clone();
        payload.remove("accessToken");
        payload.insert("access_token".into(), Value::String(access_token));
        if payload.get("type").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase() == "codex" {
            payload.insert("export_type".into(), Value::String("codex".into()));
            payload.insert("source_type".into(), Value::String("codex".into()));
            payload.remove("type");
        }
        if payload.get("export_type").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase() == "codex" {
            payload.insert("source_type".into(), Value::String("codex".into()));
        }
        let plan_type = payload.get("plan_type").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let has_type = payload.get("type").and_then(|v| v.as_str()).map_or(false, |s| !s.is_empty());
        if !plan_type.is_empty() && !has_type {
            payload.insert("type".into(), Value::String(plan_type));
        }
        Some(Value::Object(payload))
    }

    pub fn add_account_items(&self, items: &[Value]) -> Value {
        let payloads: Vec<Value> = items.iter().filter_map(Self::prepare_account_payload).collect();
        self.add_account_payloads(payloads)
    }

    pub fn add_accounts(&self, tokens: &[String], source_type: &str) -> Value {
        let mut seen = HashSet::new();
        let unique: Vec<String> = tokens
            .iter()
            .filter(|t| !t.is_empty() && seen.insert((*t).clone()))
            .cloned()
            .collect();
        if unique.is_empty() {
            return json!({"added": 0, "skipped": 0, "items": self.list_accounts()});
        }
        let st = normalize_source_type(&Value::String(source_type.to_string()));
        let payloads: Vec<Value> = unique
            .into_iter()
            .map(|token| json!({"access_token": token, "source_type": st}))
            .collect();
        self.add_account_payloads(payloads)
    }

    fn add_account_payloads(&self, payloads: Vec<Value>) -> Value {
        let mut deduped: IndexMap<String, Value> = IndexMap::new();
        for payload in &payloads {
            if !payload.is_object() {
                continue;
            }
            let access_token = Self::account_payload_token(payload);
            if access_token.is_empty() {
                continue;
            }
            let mut merged = deduped.get(&access_token).cloned().unwrap_or(json!({}));
            merge_objects(&mut merged, payload);
            merged["access_token"] = Value::String(access_token.clone());
            deduped.insert(access_token, merged);
        }
        if deduped.is_empty() {
            return json!({"added": 0, "skipped": 0, "items": self.list_accounts()});
        }
        let mut added = 0;
        let mut skipped = 0;
        {
            let mut pool = self.inner.lock();
            for (access_token, payload) in &deduped {
                let current = pool.accounts.get(access_token).cloned();
                let base = match current {
                    None => {
                        added += 1;
                        pool.cumulative_total += 1;
                        let ct = pool.cumulative_total;
                        self.save_cumulative_total(ct);
                        json!({"created_at": now_str()})
                    }
                    Some(c) => {
                        skipped += 1;
                        c
                    }
                };
                let mut incoming = payload.clone();
                if incoming.get("created_at").and_then(|v| v.as_str()).map_or(true, |s| s.is_empty()) {
                    if let Some(o) = incoming.as_object_mut() {
                        o.remove("created_at");
                    }
                }
                let mut combined = base.clone();
                merge_objects(&mut combined, &incoming);
                combined["access_token"] = Value::String(access_token.clone());
                let type_val = {
                    let inc = incoming.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    let cur = base.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if !inc.is_empty() {
                        inc
                    } else if !cur.is_empty() {
                        cur
                    } else {
                        "free"
                    }
                    .to_string()
                };
                combined["type"] = Value::String(type_val);
                if let Some(account) = Self::normalize_account(&combined) {
                    pool.accounts.insert(access_token.clone(), account);
                }
            }
            self.save_accounts_locked(&pool);
        }
        self.log.add(
            LOG_TYPE_ACCOUNT,
            &format!("新增 {added} 个账号，跳过 {skipped} 个"),
            json!({"added": added, "skipped": skipped}),
        );
        json!({"added": added, "skipped": skipped, "items": self.list_accounts()})
    }

    pub fn delete_accounts(&self, tokens: &[String]) -> Value {
        let target: HashSet<String> = tokens.iter().filter(|t| !t.is_empty()).cloned().collect();
        if target.is_empty() {
            return json!({"removed": 0, "items": self.list_accounts()});
        }
        let mut removed = 0;
        {
            let mut pool = self.inner.lock();
            let resolved: HashSet<String> =
                target.iter().map(|t| Self::resolve_locked(&pool, t)).collect();
            for token in &resolved {
                if pool.accounts.shift_remove(token).is_some() {
                    removed += 1;
                }
                pool.image_inflight.remove(token);
            }
            pool.token_aliases
                .retain(|old, new| !resolved.contains(old) && !resolved.contains(new));
            if removed > 0 {
                if pool.accounts.is_empty() {
                    pool.index = 0;
                } else {
                    pool.index %= pool.accounts.len();
                }
                self.save_accounts_locked(&pool);
                self.log.add(LOG_TYPE_ACCOUNT, &format!("删除 {removed} 个账号"), json!({"removed": removed}));
            }
        }
        json!({"removed": removed, "items": self.list_accounts()})
    }

    pub fn update_account(&self, access_token: &str, updates: &Value, quiet: bool) -> Option<Value> {
        if access_token.is_empty() {
            return None;
        }
        let mut pool = self.inner.lock();
        let token = Self::resolve_locked(&pool, access_token);
        let current = pool.accounts.get(&token).cloned()?;
        let mut combined = current;
        merge_objects(&mut combined, updates);
        combined["access_token"] = Value::String(token.clone());
        let account = Self::normalize_account(&combined)?;
        if account.get("status").and_then(|v| v.as_str()) == Some("限流")
            && self.config.auto_remove_rate_limited_accounts()
        {
            pool.accounts.shift_remove(&token);
            self.save_accounts_locked(&pool);
            self.log.add(LOG_TYPE_ACCOUNT, "自动移除限流账号", json!({"token": anonymize_token(&token)}));
            return None;
        }
        pool.accounts.insert(token.clone(), account.clone());
        self.save_accounts_locked(&pool);
        if !quiet {
            self.log.add(
                LOG_TYPE_ACCOUNT,
                "更新账号",
                json!({"token": anonymize_token(&token), "status": account.get("status")}),
            );
        }
        Some(account)
    }

    pub fn remove_invalid_token(&self, access_token: &str, event: &str, quiet: bool) -> bool {
        if !self.config.auto_remove_invalid_accounts() {
            self.update_account(access_token, &json!({"status": "异常", "quota": 0}), quiet);
            return false;
        }
        let result = self.delete_accounts(&[access_token.to_string()]);
        let removed = result.get("removed").and_then(|v| v.as_i64()).unwrap_or(0) > 0;
        if removed {
            self.log.add(
                LOG_TYPE_ACCOUNT,
                "自动移除异常账号",
                json!({"source": event, "token": anonymize_token(access_token)}),
            );
        } else if !access_token.is_empty() {
            self.update_account(access_token, &json!({"status": "异常", "quota": 0}), quiet);
        }
        removed
    }

    pub fn mark_image_result(&self, access_token: &str, success: bool) -> Option<Value> {
        if access_token.is_empty() {
            return None;
        }
        self.release_image_slot(access_token);
        let mut pool = self.inner.lock();
        let token = Self::resolve_locked(&pool, access_token);
        let current = pool.accounts.get(&token).cloned()?;
        let mut next = current;
        next["last_used_at"] = Value::String(now_local_str());
        let image_quota_unknown = next.get("image_quota_unknown").and_then(|v| v.as_bool()).unwrap_or(false);
        if success {
            let succ = next.get("success").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
            next["success"] = json!(succ);
            if !image_quota_unknown {
                let quota = (next.get("quota").and_then(|v| v.as_i64()).unwrap_or(0) - 1).max(0);
                next["quota"] = json!(quota);
                if quota == 0 {
                    next["status"] = Value::String("限流".into());
                } else if next.get("status").and_then(|v| v.as_str()) == Some("限流") {
                    next["status"] = Value::String("正常".into());
                }
            } else if next.get("status").and_then(|v| v.as_str()) == Some("限流") {
                next["status"] = Value::String("正常".into());
            }
        } else {
            let fail = next.get("fail").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
            next["fail"] = json!(fail);
        }
        let account = Self::normalize_account(&next)?;
        if account.get("status").and_then(|v| v.as_str()) == Some("限流")
            && self.config.auto_remove_rate_limited_accounts()
        {
            pool.accounts.shift_remove(&token);
            self.save_accounts_locked(&pool);
            self.log.add(LOG_TYPE_ACCOUNT, "自动移除限流账号", json!({"token": anonymize_token(&token)}));
            return None;
        }
        pool.accounts.insert(token, account.clone());
        self.save_accounts_locked(&pool);
        Some(account)
    }

    // ---- token refresh ----

    fn record_token_refresh_error(&self, access_token: &str, event: &str, error: &str) {
        let now = Utc::now().to_rfc3339();
        {
            let mut pool = self.inner.lock();
            let token = Self::resolve_locked(&pool, access_token);
            if let Some(current) = pool.accounts.get(&token).cloned() {
                let mut next = current;
                next["last_token_refresh_error"] =
                    Value::String(if error.is_empty() { "refresh token failed".into() } else { error.to_string() });
                next["last_token_refresh_error_at"] = Value::String(now);
                if let Some(account) = Self::normalize_account(&next) {
                    pool.accounts.insert(token, account);
                    self.save_accounts_locked(&pool);
                }
            }
        }
        self.log.add(
            LOG_TYPE_ACCOUNT,
            "refresh_token 刷新 access_token 失败",
            json!({"source": event, "token": anonymize_token(access_token), "error": error}),
        );
    }

    fn recent_token_refresh_error(account: &Value) -> bool {
        match parse_time(account.get("last_token_refresh_error_at").unwrap_or(&Value::Null)) {
            None => false,
            Some(at) => (Utc::now() - at).num_milliseconds() as f64 / 1000.0 < TOKEN_REFRESH_ERROR_BACKOFF_SECONDS,
        }
    }

    async fn request_access_token_refresh(
        &self,
        refresh_token: &str,
        account: &Value,
    ) -> Result<Value, String> {
        let mut builder = wreq::Client::builder().emulation(wreq_util::Emulation::Chrome131);
        let proxy = {
            let ap = s_trim(account, "proxy");
            if !ap.is_empty() { ap } else { self.config.proxy_setting() }
        };
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        let client = builder.build().map_err(|e| format!("client build: {e}"))?;
        let mut headers = wreq::header::HeaderMap::new();
        for (k, v) in [
            ("Accept", "application/json"),
            ("Content-Type", "application/x-www-form-urlencoded"),
            ("User-Agent", OAUTH_USER_AGENT),
        ] {
            if let (Ok(n), Ok(val)) = (
                wreq::header::HeaderName::from_bytes(k.as_bytes()),
                wreq::header::HeaderValue::from_str(v),
            ) {
                headers.insert(n, val);
            }
        }
        let form = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", OAUTH_CLIENT_ID),
        ];
        let resp = client
            .post(OAUTH_TOKEN_URL)
            .headers(headers)
            .form(&form)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("oauth_refresh_network: {e}"))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let data: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let access_token = data.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if status != 200 || !data.is_object() || access_token.is_empty() {
            let mut detail = String::new();
            if let Some(o) = data.as_object() {
                detail = o
                    .get("error_description")
                    .or_else(|| o.get("error"))
                    .or_else(|| o.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
            }
            if detail.is_empty() {
                detail = text.chars().take(300).collect();
            }
            return Err(format!(
                "oauth_refresh_http_{status}{}",
                if detail.is_empty() { String::new() } else { format!(": {detail}") }
            ));
        }
        let refresh_out = data.get("refresh_token").and_then(|v| v.as_str()).unwrap_or(refresh_token).trim().to_string();
        let id_token = data.get("id_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        Ok(json!({
            "access_token": access_token,
            "refresh_token": refresh_out,
            "id_token": id_token,
        }))
    }

    fn apply_refreshed_tokens(&self, old_access_token: &str, token_data: &Value, event: &str) -> String {
        let now = Utc::now().to_rfc3339();
        let (new_token, rotated) = {
            let mut pool = self.inner.lock();
            let old_token = Self::resolve_locked(&pool, old_access_token);
            let Some(current) = pool.accounts.get(&old_token).cloned() else {
                return old_token;
            };
            let new_token = token_data.get("access_token").and_then(|v| v.as_str()).unwrap_or(&old_token).trim().to_string();
            if new_token.is_empty() {
                return old_token;
            }
            let mut next = current;
            next["access_token"] = Value::String(new_token.clone());
            if let Some(rt) = token_data.get("refresh_token").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                next["refresh_token"] = Value::String(rt.trim().to_string());
            }
            if let Some(it) = token_data.get("id_token").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                next["id_token"] = Value::String(it.trim().to_string());
            }
            next["last_token_refresh_at"] = Value::String(now);
            next["last_token_refresh_error"] = Value::Null;
            next["last_token_refresh_error_at"] = Value::Null;
            next["invalid_count"] = json!(0);
            next["last_invalid_at"] = Value::Null;
            next["last_refresh_error"] = Value::Null;
            next["last_refresh_error_at"] = Value::Null;
            let Some(account) = Self::normalize_account(&next) else {
                return old_token;
            };
            let rotated = new_token != old_token;
            if rotated {
                pool.accounts.shift_remove(&old_token);
                pool.token_aliases.insert(old_token.clone(), new_token.clone());
                if let Some(old_inflight) = pool.image_inflight.remove(&old_token) {
                    if old_inflight != 0 {
                        *pool.image_inflight.entry(new_token.clone()).or_insert(0) += old_inflight;
                    }
                }
            }
            pool.accounts.insert(new_token.clone(), account);
            self.save_accounts_locked(&pool);
            (new_token, rotated)
        };
        self.slot_notify.notify_waiters();
        self.log.add(
            LOG_TYPE_ACCOUNT,
            "refresh_token 已刷新 access_token",
            json!({"source": event, "token": anonymize_token(&new_token), "rotated": rotated}),
        );
        new_token
    }

    /// Refresh an access token via its refresh token (if near expiry or forced).
    pub async fn refresh_access_token(&self, access_token: &str, force: bool, event: &str) -> String {
        if access_token.is_empty() {
            return String::new();
        }
        let _guard = self.token_refresh_lock.lock().await;
        let account = match self.get_account(access_token) {
            Some(a) => a,
            None => return access_token.to_string(),
        };
        let resolved = self.resolve_access_token(access_token);
        let active_token = {
            let t = s(&account, "access_token");
            if !t.is_empty() {
                t
            } else if !resolved.is_empty() {
                resolved
            } else {
                access_token.to_string()
            }
        };
        if !token_needs_refresh(&active_token, force) {
            return active_token;
        }
        let refresh_token = s_trim(&account, "refresh_token");
        if refresh_token.is_empty() {
            return active_token;
        }
        if !force && Self::recent_token_refresh_error(&account) {
            return active_token;
        }
        match self.request_access_token_refresh(&refresh_token, &account).await {
            Ok(token_data) => self.apply_refreshed_tokens(&active_token, &token_data, event),
            Err(error_str) => {
                self.record_token_refresh_error(&active_token, event, &error_str);
                // app_session_terminated → password re-login is deferred to Phase 9.
                active_token
            }
        }
    }

    // ---- remote info ----

    /// Refresh account info from the upstream (`get_user_info`), handling token
    /// invalidation/rotation and recording success/failure.
    pub async fn fetch_remote_info(
        &self,
        access_token: &str,
        event: &str,
        defer_invalid_removal: bool,
    ) -> Result<Option<Value>, String> {
        if access_token.is_empty() {
            return Err("access_token is required".to_string());
        }
        let preflight = self.refresh_access_token(access_token, false, &format!("{event}:preflight")).await;
        let mut active_token = if preflight.is_empty() { access_token.to_string() } else { preflight };

        let result = match self.engine_user_info(&active_token).await {
            Ok(r) => r,
            Err(EngineError::InvalidAccessToken(msg)) => {
                let refreshed = self
                    .refresh_access_token(&active_token, true, &format!("{event}:invalid_access_token"))
                    .await;
                if !refreshed.is_empty() && refreshed != active_token {
                    match self.engine_user_info(&refreshed).await {
                        Ok(r) => {
                            active_token = refreshed;
                            r
                        }
                        Err(EngineError::InvalidAccessToken(retry_msg)) => {
                            if self.record_invalid_token_seen(&refreshed, event, &retry_msg, defer_invalid_removal) {
                                self.remove_invalid_token(&refreshed, event, false);
                            }
                            return Err(retry_msg);
                        }
                        Err(e) => return Err(e.to_string()),
                    }
                } else {
                    if self.record_invalid_token_seen(&active_token, event, &msg, defer_invalid_removal) {
                        self.remove_invalid_token(&active_token, event, false);
                    }
                    return Err(msg);
                }
            }
            Err(e) => return Err(e.to_string()),
        };
        self.record_refresh_success(&active_token);
        Ok(self.update_account(&active_token, &result, false))
    }

    async fn engine_user_info(&self, access_token: &str) -> Result<Value, EngineError> {
        let account = self.get_account(access_token).unwrap_or(json!({}));
        let engine = OpenAIBackendAPI::new(self.config.clone(), access_token.to_string(), account)?;
        engine.get_user_info().await
    }

    fn record_refresh_success(&self, access_token: &str) {
        let mut pool = self.inner.lock();
        let token = Self::resolve_locked(&pool, access_token);
        if let Some(current) = pool.accounts.get(&token).cloned() {
            let mut next = current;
            next["invalid_count"] = json!(0);
            next["last_invalid_at"] = Value::Null;
            next["last_refresh_error"] = Value::Null;
            next["last_refresh_error_at"] = Value::Null;
            if let Some(account) = Self::normalize_account(&next) {
                pool.accounts.insert(token, account);
            }
        }
    }

    fn should_defer_invalid_token(account: &Value, now: DateTime<Utc>) -> bool {
        if !account.is_object() {
            return false;
        }
        if let Some(created) = parse_time(account.get("created_at").unwrap_or(&Value::Null)) {
            if (now - created).num_milliseconds() as f64 / 1000.0 < NEW_ACCOUNT_INVALID_GRACE_SECONDS {
                return true;
            }
        }
        let invalid_count = account.get("invalid_count").and_then(|v| v.as_i64()).unwrap_or(0);
        if invalid_count <= 1 {
            return true;
        }
        if let Some(last) = parse_time(account.get("last_invalid_at").unwrap_or(&Value::Null)) {
            if (now - last).num_milliseconds() as f64 / 1000.0 < INVALID_CONFIRM_SECONDS {
                return true;
            }
        }
        false
    }

    fn record_invalid_token_seen(
        &self,
        access_token: &str,
        event: &str,
        error: &str,
        defer_invalid_removal: bool,
    ) -> bool {
        let now = Utc::now();
        let mut pool = self.inner.lock();
        let token = Self::resolve_locked(&pool, access_token);
        let Some(current) = pool.accounts.get(&token).cloned() else {
            return true;
        };
        let should_defer = defer_invalid_removal && Self::should_defer_invalid_token(&current, now);
        let mut next = current;
        let ic = next.get("invalid_count").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
        next["invalid_count"] = json!(ic);
        next["last_invalid_at"] = Value::String(now.to_rfc3339());
        next["last_refresh_error"] =
            Value::String(if error.is_empty() { "invalid access token".into() } else { error.to_string() });
        next["last_refresh_error_at"] = Value::String(now.to_rfc3339());
        if let Some(account) = Self::normalize_account(&next) {
            pool.accounts.insert(token.clone(), account);
            self.save_accounts_locked(&pool);
        }
        if should_defer {
            drop(pool);
            self.log.add(
                LOG_TYPE_ACCOUNT,
                "暂缓标记异常账号",
                json!({"source": event, "token": anonymize_token(&token), "error": error}),
            );
            return false;
        }
        true
    }

    // ---- bulk refresh ----

    pub fn list_expiring_access_tokens(&self) -> Vec<String> {
        let pool = self.inner.lock();
        pool.accounts
            .values()
            .filter(|a| !s_trim(a, "refresh_token").is_empty())
            .map(|a| s_trim(a, "access_token"))
            .filter(|t| !t.is_empty() && token_needs_refresh(t, false))
            .collect()
    }

    fn refresh_token_keepalive_due_at(&self, account: &Value, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        if s_trim(account, "refresh_token").is_empty() {
            return None;
        }
        if account.get("status").and_then(|v| v.as_str()) == Some("禁用") {
            return None;
        }
        if let Some(last) = parse_time(account.get("last_token_refresh_error_at").unwrap_or(&Value::Null)) {
            if (now - last).num_milliseconds() as f64 / 1000.0 < REFRESH_TOKEN_KEEPALIVE_ERROR_BACKOFF_SECONDS {
                return None;
            }
        }
        let anchor = parse_time(account.get("last_token_refresh_at").unwrap_or(&Value::Null))
            .or_else(|| token_issued_at(&s(account, "access_token")))
            .or_else(|| parse_time(account.get("created_at").unwrap_or(&Value::Null)));
        let Some(anchor) = anchor else {
            return Some(now);
        };
        let due_at = anchor + chrono::Duration::seconds(REFRESH_TOKEN_KEEPALIVE_SECONDS);
        if due_at <= now {
            Some(due_at)
        } else {
            None
        }
    }

    pub fn list_refresh_token_keepalive_tokens(&self) -> Vec<String> {
        let now = Utc::now();
        let pool = self.inner.lock();
        let mut due: Vec<(DateTime<Utc>, String)> = Vec::new();
        for account in pool.accounts.values() {
            if let Some(due_at) = self.refresh_token_keepalive_due_at(account, now) {
                let token = s_trim(account, "access_token");
                if !token.is_empty() {
                    due.push((due_at, token));
                }
            }
        }
        due.sort_by_key(|(at, _)| *at);
        due.into_iter().take(REFRESH_TOKEN_KEEPALIVE_BATCH_SIZE).map(|(_, t)| t).collect()
    }

    pub async fn keepalive_refresh_tokens(&self, access_tokens: &[String]) -> Value {
        let mut seen = HashSet::new();
        let tokens: Vec<String> =
            access_tokens.iter().filter(|t| !t.is_empty() && seen.insert((*t).clone())).cloned().collect();
        if tokens.is_empty() {
            return json!({"refreshed": 0, "errors": [], "items": self.list_accounts()});
        }
        let mut refreshed = 0;
        let mut errors: Vec<Value> = Vec::new();
        for token in &tokens {
            let before = self.resolve_access_token(token);
            let after = self.refresh_access_token(&before, true, "refresh_token_keepalive").await;
            let account = self.get_account(&after);
            if let Some(acc) = &account {
                let err = s_trim(acc, "last_token_refresh_error");
                if !err.is_empty() {
                    errors.push(json!({"token": anonymize_token(&before), "error": err}));
                    continue;
                }
                refreshed += 1;
            }
        }
        json!({"refreshed": refreshed, "errors": errors, "items": self.list_accounts(), "relogined": 0})
    }

    /// Refresh many accounts concurrently (bounded), with optional progress.
    pub async fn refresh_accounts(
        &self,
        access_tokens: &[String],
        progress_id: Option<&str>,
        defer_invalid_removal: bool,
    ) -> Value {
        let mut seen = HashSet::new();
        let tokens: Vec<String> =
            access_tokens.iter().filter(|t| !t.is_empty() && seen.insert((*t).clone())).cloned().collect();
        if tokens.is_empty() {
            let result = json!({"refreshed": 0, "errors": [], "items": self.list_accounts(), "relogined": 0});
            if let Some(pid) = progress_id {
                self.finish_refresh_progress(pid, Some(result.clone()), None);
            }
            return result;
        }
        if let Some(pid) = progress_id {
            self.init_refresh_progress(pid, tokens.len());
        }
        use futures::stream::StreamExt;
        let results: Vec<(String, Result<Option<Value>, String>)> = futures::stream::iter(tokens.iter().cloned())
            .map(|token| async move {
                let r = self.fetch_remote_info(&token, "refresh_accounts", defer_invalid_removal).await;
                (token, r)
            })
            .buffer_unordered(10)
            .collect()
            .await;

        let mut refreshed = 0;
        let mut errors: Vec<Value> = Vec::new();
        for (token, r) in results {
            match r {
                Ok(account) => {
                    if account.is_some() {
                        refreshed += 1;
                    }
                }
                Err(error_str) => {
                    if !is_tls_connection_error(&error_str) {
                        errors.push(json!({"token": anonymize_token(&token), "error": error_str}));
                    }
                }
            }
            if let Some(pid) = progress_id {
                self.update_refresh_progress(pid, &token);
            }
        }

        // Auto re-login of abnormal accounts is deferred to Phase 9.
        let relogined = 0;

        let result = json!({
            "refreshed": refreshed,
            "errors": errors,
            "items": self.list_accounts(),
            "relogined": relogined,
        });
        if let Some(pid) = progress_id {
            self.finish_refresh_progress(pid, Some(result.clone()), None);
        }
        result
    }

    /// Password re-login (deferred to Phase 9; currently marks targets abnormal).
    pub fn re_login_accounts(&self, access_tokens: &[String], progress_id: Option<&str>) -> Value {
        let mut seen = HashSet::new();
        let tokens: Vec<String> =
            access_tokens.iter().filter(|t| !t.is_empty() && seen.insert((*t).clone())).cloned().collect();
        if let Some(pid) = progress_id {
            self.init_relogin_progress(pid, tokens.len());
        }
        let mut skipped = 0;
        let mut errors: Vec<Value> = Vec::new();
        for token in &tokens {
            let account = self.get_account(token);
            let Some(account) = account else {
                errors.push(json!({"token": anonymize_token(token), "error": "账号不存在"}));
                if let Some(pid) = progress_id {
                    self.update_relogin_progress(pid, token, "跳过", Some("账号不存在"));
                }
                continue;
            };
            let email = s_trim(&account, "email");
            let password = s_trim(&account, "password");
            if email.is_empty() || password.is_empty() {
                skipped += 1;
                if let Some(pid) = progress_id {
                    self.update_relogin_progress(pid, token, "跳过", Some("无邮箱密码"));
                }
                continue;
            }
            // Login machinery not yet available — mark abnormal.
            if let Some(pid) = progress_id {
                self.update_relogin_progress(pid, token, "异常", Some("password login not implemented"));
            }
        }
        let result = json!({
            "relogined": 0,
            "skipped": skipped,
            "errors": errors,
            "items": self.list_accounts(),
        });
        if let Some(pid) = progress_id {
            self.finish_relogin_progress(pid, Some(result.clone()), None);
        }
        result
    }

    // ---- progress tracking ----

    pub fn init_refresh_progress(&self, progress_id: &str, total: usize) {
        self.refresh_progress.lock().insert(
            progress_id.to_string(),
            json!({
                "total": total,
                "processed": 0,
                "done": false,
                "error": Value::Null,
                "status_counts": {"正常": 0, "限流": 0, "异常": 0, "禁用": 0},
                "total_quota": 0,
            }),
        );
    }

    pub fn update_refresh_progress(&self, progress_id: &str, token: &str) {
        let account = self.get_account(token);
        let status = account.as_ref().map(|a| {
            let st = s_trim(a, "status");
            if st.is_empty() { "正常".to_string() } else { st }
        }).unwrap_or_else(|| "正常".to_string());
        let quota = account.as_ref().and_then(|a| a.get("quota").and_then(|v| v.as_i64())).unwrap_or(0).max(0);
        let mut guard = self.refresh_progress.lock();
        if let Some(progress) = guard.get_mut(progress_id) {
            progress["processed"] = json!(progress["processed"].as_i64().unwrap_or(0) + 1);
            let cur = progress["status_counts"].get(&status).and_then(|v| v.as_i64()).unwrap_or(0);
            progress["status_counts"][&status] = json!(cur + 1);
            progress["total_quota"] = json!(progress["total_quota"].as_i64().unwrap_or(0) + quota);
        }
    }

    pub fn finish_refresh_progress(&self, progress_id: &str, result: Option<Value>, error: Option<&str>) {
        let mut guard = self.refresh_progress.lock();
        if let Some(progress) = guard.get_mut(progress_id) {
            progress["done"] = json!(true);
            if let Some(r) = result {
                progress["result"] = r;
            }
            if let Some(e) = error {
                progress["error"] = Value::String(e.to_string());
            }
        }
    }

    pub fn get_refresh_progress(&self, progress_id: &str) -> Option<Value> {
        self.refresh_progress.lock().get(progress_id).cloned()
    }

    pub fn clean_refresh_progress(&self, progress_id: &str) {
        self.refresh_progress.lock().remove(progress_id);
    }

    pub fn init_relogin_progress(&self, progress_id: &str, total: usize) {
        self.relogin_progress.lock().insert(
            progress_id.to_string(),
            json!({"total": total, "processed": 0, "done": false, "error": Value::Null, "results": []}),
        );
    }

    pub fn update_relogin_progress(&self, progress_id: &str, token: &str, status: &str, error: Option<&str>) {
        let mut guard = self.relogin_progress.lock();
        if let Some(progress) = guard.get_mut(progress_id) {
            let processed = progress["processed"].as_i64().unwrap_or(0) + 1;
            progress["processed"] = json!(processed);
            if let Some(arr) = progress["results"].as_array_mut() {
                arr.push(json!({"token": anonymize_token(token), "status": status, "error": error}));
            }
            let total = progress["total"].as_i64().unwrap_or(0);
            if processed >= total {
                progress["done"] = json!(true);
            }
        }
    }

    pub fn finish_relogin_progress(&self, progress_id: &str, result: Option<Value>, error: Option<&str>) {
        let mut guard = self.relogin_progress.lock();
        if let Some(progress) = guard.get_mut(progress_id) {
            progress["done"] = json!(true);
            if let Some(r) = result {
                progress["result"] = r;
            }
            if let Some(e) = error {
                progress["error"] = Value::String(e.to_string());
            }
        }
    }

    pub fn get_relogin_progress(&self, progress_id: &str) -> Option<Value> {
        self.relogin_progress.lock().get(progress_id).cloned()
    }

    pub fn clean_relogin_progress(&self, progress_id: &str) {
        self.relogin_progress.lock().remove(progress_id);
    }

    // ---- export / stats ----

    pub fn build_export_items(&self, access_tokens: Option<&[String]>) -> Vec<Value> {
        let target: HashSet<String> =
            access_tokens.map(|t| t.iter().filter(|x| !x.is_empty()).cloned().collect()).unwrap_or_default();
        let accounts: Vec<Value> = {
            let pool = self.inner.lock();
            pool.accounts
                .values()
                .filter(|item| target.is_empty() || target.contains(&s(item, "access_token")))
                .cloned()
                .collect()
        };
        let mut items = Vec::new();
        for account in accounts {
            let access_token = s_trim(&account, "access_token");
            let refresh_token = s_trim(&account, "refresh_token");
            let id_token = s_trim(&account, "id_token");
            if access_token.is_empty() || refresh_token.is_empty() || id_token.is_empty() {
                continue;
            }
            let access_payload = decode_jwt_payload(&access_token);
            let id_payload = decode_jwt_payload(&id_token);
            let auth_claim = access_payload.get("https://api.openai.com/auth").cloned().unwrap_or(json!({}));
            let profile_claim = access_payload.get("https://api.openai.com/profile").cloned().unwrap_or(json!({}));

            let email = {
                let a = s_trim(&account, "email");
                let b = profile_claim.get("email").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                let c = id_payload.get("email").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                if !a.is_empty() { a } else if !b.is_empty() { b } else { c }
            };
            let account_id = {
                let a = s_trim(&account, "account_id");
                let b = auth_claim.get("chatgpt_account_id").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                let c = s_trim(&account, "user_id");
                if !a.is_empty() { a } else if !b.is_empty() { b } else { c }
            };
            let export_type = {
                let t = s_trim(&account, "export_type");
                if t.is_empty() { "codex".to_string() } else { t }
            };
            let mut item = json!({
                "type": export_type,
                "email": email,
                "account_id": account_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "id_token": id_token,
                "expired": timestamp_to_iso(access_payload.get("exp").unwrap_or(&Value::Null)),
                "last_refresh": timestamp_to_iso(access_payload.get("iat").unwrap_or(&Value::Null)),
            });
            let password = s_trim(&account, "password");
            if !password.is_empty() {
                item["password"] = Value::String(password);
            }
            items.push(item);
        }
        items
    }

    pub fn get_stats(&self) -> Value {
        let pool = self.inner.lock();
        let items: Vec<&Value> = pool.accounts.values().collect();
        let count_status = |st: &str| items.iter().filter(|a| a.get("status").and_then(|v| v.as_str()) == Some(st)).count();
        let total = items.len();
        let active = count_status("正常");
        let limited = count_status("限流");
        let abnormal = count_status("异常");
        let disabled = count_status("禁用");
        let total_quota: i64 = items
            .iter()
            .filter(|a| a.get("status").and_then(|v| v.as_str()) == Some("正常"))
            .map(|a| a.get("quota").and_then(|v| v.as_i64()).unwrap_or(0).max(0))
            .sum();
        let unlimited = items
            .iter()
            .filter(|a| {
                a.get("status").and_then(|v| v.as_str()) == Some("正常")
                    && a.get("image_quota_unknown").and_then(|v| v.as_bool()).unwrap_or(false)
            })
            .count();
        let total_success: i64 = items.iter().map(|a| a.get("success").and_then(|v| v.as_i64()).unwrap_or(0)).sum();
        let total_fail: i64 = items.iter().map(|a| a.get("fail").and_then(|v| v.as_i64()).unwrap_or(0)).sum();
        let mut by_type: Map<String, Value> = Map::new();
        for a in &items {
            let t = a.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            let cur = by_type.get(&t).and_then(|v| v.as_i64()).unwrap_or(0);
            by_type.insert(t, json!(cur + 1));
        }
        json!({
            "total": total,
            "cumulative_total": pool.cumulative_total,
            "active": active,
            "limited": limited,
            "abnormal": abnormal,
            "disabled": disabled,
            "total_quota": total_quota,
            "unlimited_quota_count": unlimited,
            "total_success": total_success,
            "total_fail": total_fail,
            "by_type": by_type,
        })
    }

    pub fn account_health(&self) -> Value {
        let stats = self.get_stats();
        let active = stats.get("active").and_then(|v| v.as_i64()).unwrap_or(0);
        let unlimited = stats.get("unlimited_quota_count").and_then(|v| v.as_i64()).unwrap_or(0);
        let mut out = stats.clone();
        if let Some(obj) = out.as_object_mut() {
            obj.insert("healthy".into(), json!(active > 0 || unlimited > 0));
            obj.insert("status".into(), Value::String(if active > 0 { "ok".into() } else { "degraded".into() }));
        }
        out
    }
}

/// Merge `src` object fields into `dst` (shallow, like Python `{**a, **b}`).
fn merge_objects(dst: &mut Value, src: &Value) {
    if !dst.is_object() {
        *dst = json!({});
    }
    if let (Some(d), Some(s)) = (dst.as_object_mut(), src.as_object()) {
        for (k, v) in s {
            d.insert(k.clone(), v.clone());
        }
    }
}

/// Heuristic: is this a TLS/proxy connection error (vs. an account failure)?
fn is_tls_connection_error(error: &str) -> bool {
    let e = error.to_lowercase();
    [
        "tls", "ssl", "handshake", "connect", "connection", "proxy", "timed out", "timeout",
        "dns", "unreachable", "reset", "eof",
    ]
    .iter()
    .any(|m| e.contains(m))
}
