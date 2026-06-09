//! Port of `services/cpa_service.py` — CLIProxyAPI (CPA) integration for
//! browsing remote auth files on a remote CLIProxyAPI management endpoint and
//! importing selected tokens into the account pool.
//!
//! A `CpaService` owns a persisted list of "pools" (each is a remote
//! `{base_url, secret_key}` plus its last `import_job`), stored in
//! `data/cpa_config.json`. It can list remote auth files, download their
//! `access_token`s, and bulk-import them.
//!
//! Async notes: the Python original used a `ThreadPoolExecutor` to download
//! tokens concurrently and a background `threading.Thread` to run the import.
//! Here `start_import` is `async`: it stages the job synchronously, then
//! `tokio::spawn`s the download+import work. The service is cheaply cloneable
//! (`Config` + `Arc<AccountService>` + `Arc<Mutex<Inner>>`) so the spawned
//! `'static` task can drive the pool and the account service. Mutable pool /
//! import-job state lives behind a `parking_lot::Mutex`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::config::Config;
use crate::services::account_service::AccountService;

// ---- small helpers on Value objects ----

fn vstr(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn vstr_trim(v: &Value, key: &str) -> String {
    vstr(v, key).trim().to_string()
}

/// Read an integer field, accepting JSON numbers or numeric strings (Python `int(x or 0)`).
fn as_int(v: &Value, key: &str) -> i64 {
    match v.get(key) {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(0),
        _ => 0,
    }
}

fn error_count(job: &Value) -> usize {
    job.get("errors").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)
}

fn new_id() -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    hex[..12].to_string()
}

fn full_uuid() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn management_headers(secret_key: &str) -> wreq::header::HeaderMap {
    let mut headers = wreq::header::HeaderMap::new();
    if let Ok(val) = wreq::header::HeaderValue::from_str(&format!("Bearer {secret_key}")) {
        headers.insert(wreq::header::AUTHORIZATION, val);
    }
    headers.insert(
        wreq::header::ACCEPT,
        wreq::header::HeaderValue::from_static("application/json"),
    );
    headers
}

fn normalize_import_job(raw: Option<&Value>, fail_unfinished: bool) -> Option<Value> {
    let raw = raw?;
    if !raw.is_object() {
        return None;
    }
    let mut status = {
        let s = vstr_trim(raw, "status");
        if s.is_empty() { "failed".to_string() } else { s }
    };
    if fail_unfinished && (status == "pending" || status == "running") {
        status = "failed".to_string();
    }
    let job_id = {
        let s = vstr_trim(raw, "job_id");
        if s.is_empty() { full_uuid() } else { s }
    };
    let created_at = {
        let s = vstr_trim(raw, "created_at");
        if s.is_empty() { now_iso() } else { s }
    };
    let updated_at = {
        let u = vstr_trim(raw, "updated_at");
        if !u.is_empty() {
            u
        } else {
            let c = vstr_trim(raw, "created_at");
            if !c.is_empty() { c } else { now_iso() }
        }
    };
    let errors = raw.get("errors").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    Some(json!({
        "job_id": job_id,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "total": as_int(raw, "total"),
        "completed": as_int(raw, "completed"),
        "added": as_int(raw, "added"),
        "skipped": as_int(raw, "skipped"),
        "refreshed": as_int(raw, "refreshed"),
        "failed": as_int(raw, "failed"),
        "errors": errors,
    }))
}

fn normalize_pool(raw: &Value) -> Value {
    let id = {
        let v = vstr_trim(raw, "id");
        if v.is_empty() { new_id() } else { v }
    };
    json!({
        "id": id,
        "name": vstr_trim(raw, "name"),
        "base_url": vstr_trim(raw, "base_url"),
        "secret_key": vstr_trim(raw, "secret_key"),
        "import_job": normalize_import_job(raw.get("import_job"), true).unwrap_or(Value::Null),
    })
}

// ---- service ----

struct Inner {
    store_file: std::path::PathBuf,
    pools: Vec<Value>,
}

/// CLIProxyAPI integration service: persisted remote pools + token import.
#[derive(Clone)]
pub struct CpaService {
    config: Config,
    accounts: Arc<AccountService>,
    inner: Arc<Mutex<Inner>>,
}

impl CpaService {
    pub fn new(config: Config, accounts: Arc<AccountService>) -> Self {
        let store_file = config.data_dir().join("cpa_config.json");
        let pools = Self::load(&store_file);
        Self {
            config,
            accounts,
            inner: Arc::new(Mutex::new(Inner { store_file, pools })),
        }
    }

    // ---- persistence ----

    fn load(store_file: &Path) -> Vec<Value> {
        if !store_file.exists() {
            return Vec::new();
        }
        let text = match std::fs::read_to_string(store_file) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let raw: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        // Legacy single-pool object form.
        if raw.is_object() && raw.get("base_url").is_some() {
            let pool = normalize_pool(&raw);
            return if !vstr(&pool, "base_url").is_empty() { vec![pool] } else { Vec::new() };
        }
        if let Some(arr) = raw.as_array() {
            return arr.iter().filter(|x| x.is_object()).map(normalize_pool).collect();
        }
        Vec::new()
    }

    fn save_locked(inner: &Inner) {
        if let Some(parent) = inner.store_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(&inner.pools) {
            let _ = std::fs::write(&inner.store_file, text + "\n");
        }
    }

    // ---- pool CRUD ----

    pub fn list_pools(&self) -> Vec<Value> {
        self.inner.lock().pools.clone()
    }

    pub fn get_pool(&self, pool_id: &str) -> Option<Value> {
        self.inner.lock().pools.iter().find(|p| vstr(p, "id") == pool_id).cloned()
    }

    pub fn add_pool(&self, name: &str, base_url: &str, secret_key: &str) -> Value {
        let pool = normalize_pool(&json!({
            "id": new_id(),
            "name": name,
            "base_url": base_url,
            "secret_key": secret_key,
        }));
        let mut inner = self.inner.lock();
        inner.pools.push(pool.clone());
        Self::save_locked(&inner);
        pool
    }

    /// Merge non-null `updates` into a pool and re-normalize (`id` is preserved).
    pub fn update_pool(&self, pool_id: &str, updates: &Value) -> Option<Value> {
        let mut inner = self.inner.lock();
        for i in 0..inner.pools.len() {
            if vstr(&inner.pools[i], "id") != pool_id {
                continue;
            }
            let mut merged = inner.pools[i].clone();
            if let (Some(m), Some(u)) = (merged.as_object_mut(), updates.as_object()) {
                for (k, v) in u {
                    if !v.is_null() {
                        m.insert(k.clone(), v.clone());
                    }
                }
            }
            merged["id"] = Value::String(pool_id.to_string());
            let normalized = normalize_pool(&merged);
            inner.pools[i] = normalized.clone();
            Self::save_locked(&inner);
            return Some(normalized);
        }
        None
    }

    pub fn delete_pool(&self, pool_id: &str) -> bool {
        let mut inner = self.inner.lock();
        let before = inner.pools.len();
        inner.pools.retain(|p| vstr(p, "id") != pool_id);
        if inner.pools.len() < before {
            Self::save_locked(&inner);
            true
        } else {
            false
        }
    }

    // ---- import-job state ----

    fn set_import_job(&self, pool_id: &str, import_job: Option<Value>) -> Option<Value> {
        let mut inner = self.inner.lock();
        for i in 0..inner.pools.len() {
            if vstr(&inner.pools[i], "id") != pool_id {
                continue;
            }
            let mut next = inner.pools[i].clone();
            next["import_job"] = normalize_import_job(import_job.as_ref(), false).unwrap_or(Value::Null);
            inner.pools[i] = next.clone();
            Self::save_locked(&inner);
            return Some(next);
        }
        None
    }

    pub fn get_import_job(&self, pool_id: &str) -> Option<Value> {
        let inner = self.inner.lock();
        for pool in &inner.pools {
            if vstr(pool, "id") == pool_id {
                return pool.get("import_job").filter(|j| j.is_object()).cloned();
            }
        }
        None
    }

    fn update_job(&self, pool_id: &str, updates: Value) -> Option<Value> {
        let mut next = self.get_import_job(pool_id)?;
        if let (Some(n), Some(u)) = (next.as_object_mut(), updates.as_object()) {
            for (k, v) in u {
                n.insert(k.clone(), v.clone());
            }
        }
        next["updated_at"] = Value::String(now_iso());
        let pool = self.set_import_job(pool_id, Some(next))?;
        pool.get("import_job").filter(|j| j.is_object()).cloned()
    }

    fn append_error(&self, pool_id: &str, file_name: &str, message: &str) {
        let Some(current) = self.get_import_job(pool_id) else {
            return;
        };
        let mut errors = current.get("errors").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        errors.push(json!({"name": file_name, "error": message}));
        let failed = errors.len();
        self.update_job(pool_id, json!({"errors": errors, "failed": failed}));
    }

    // ---- remote HTTP ----

    fn build_client(&self) -> Result<wreq::Client, String> {
        // Cert verification stays on (Python used `verify=True`).
        let mut builder = wreq::Client::builder().emulation(wreq_util::Emulation::Chrome137);
        let proxy = self.config.proxy_setting();
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        builder.build().map_err(|e| format!("client build: {e}"))
    }

    /// List remote auth files for a pool: `[{name, email}, ...]`.
    pub async fn list_remote_files(&self, pool: &Value) -> Result<Vec<Value>, String> {
        let base_url = vstr_trim(pool, "base_url");
        let secret_key = vstr_trim(pool, "secret_key");
        if base_url.is_empty() || secret_key.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/v0/management/auth-files", base_url.trim_end_matches('/'));
        let client = self.build_client()?;
        let resp = client
            .get(&url)
            .headers(management_headers(&secret_key))
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("remote list failed: {e}"))?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("remote list failed: HTTP {}", resp.status().as_u16()));
        }
        let text = resp.text().await.unwrap_or_default();
        let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let files = payload
            .get("files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "remote list payload is invalid".to_string())?;

        let mut items = Vec::new();
        for item in files {
            if !item.is_object() {
                continue;
            }
            let name = vstr_trim(item, "name");
            if name.is_empty() {
                continue;
            }
            let email = {
                let a = vstr_trim(item, "email");
                if !a.is_empty() { a } else { vstr_trim(item, "account") }
            };
            items.push(json!({"name": name, "email": email}));
        }
        Ok(items)
    }

    /// Download a single remote auth file and return its `access_token`.
    async fn fetch_remote_access_token(&self, pool: &Value, file_name: &str) -> Result<String, String> {
        let base_url = vstr_trim(pool, "base_url");
        let secret_key = vstr_trim(pool, "secret_key");
        let file_name = file_name.trim();
        if base_url.is_empty() || secret_key.is_empty() || file_name.is_empty() {
            return Err("invalid request".to_string());
        }
        let url = format!("{}/v0/management/auth-files/download", base_url.trim_end_matches('/'));
        let client = self.build_client()?;
        let resp = client
            .get(&url)
            .headers(management_headers(&secret_key))
            .query(&[("name", file_name)])
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if resp.status().as_u16() >= 400 {
            return Err(format!("HTTP {}", resp.status().as_u16()));
        }
        let text = resp.text().await.unwrap_or_default();
        let payload: Value = serde_json::from_str(&text).map_err(|_| "invalid payload".to_string())?;
        if !payload.is_object() {
            return Err("invalid payload".to_string());
        }
        let access_token = vstr_trim(&payload, "access_token");
        if access_token.is_empty() {
            return Err("missing access_token".to_string());
        }
        Ok(access_token)
    }

    // ---- import ----

    /// Stage an import job for `selected_files` and run it in the background.
    /// Returns the freshly-created `import_job` snapshot.
    pub async fn start_import(&self, pool: &Value, selected_files: &[String]) -> Result<Value, String> {
        let names: Vec<String> = selected_files
            .iter()
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .collect();
        if names.is_empty() {
            return Err("selected files is required".to_string());
        }

        let pool_id = vstr_trim(pool, "id");
        let job = json!({
            "job_id": full_uuid(),
            "status": "pending",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "total": names.len(),
            "completed": 0,
            "added": 0,
            "skipped": 0,
            "refreshed": 0,
            "failed": 0,
            "errors": [],
        });
        let saved_pool = self
            .set_import_job(&pool_id, Some(job.clone()))
            .ok_or_else(|| "pool not found".to_string())?;
        let result_job = saved_pool
            .get("import_job")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or(job);

        let this = self.clone();
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            this.run_import(pool_id, pool_clone, names).await;
        });

        Ok(result_job)
    }

    async fn run_import(&self, pool_id: String, pool: Value, names: Vec<String>) {
        self.update_job(&pool_id, json!({"status": "running"}));

        let max_workers = names.len().clamp(1, 16);
        let mut tokens: Vec<String> = Vec::new();
        {
            use futures::stream::StreamExt;
            let pool_ref = &pool;
            let mut stream = futures::stream::iter(names.iter().cloned())
                .map(|name| async move {
                    let r = self.fetch_remote_access_token(pool_ref, &name).await;
                    (name, r)
                })
                .buffer_unordered(max_workers);

            while let Some((name, result)) = stream.next().await {
                match result {
                    Ok(token) => tokens.push(token),
                    Err(error) => self.append_error(&pool_id, &name, &error),
                }
                let current = self.get_import_job(&pool_id).unwrap_or_else(|| json!({}));
                let completed = as_int(&current, "completed") + 1;
                let failed = error_count(&current);
                self.update_job(&pool_id, json!({"completed": completed, "failed": failed}));
            }
        }

        if tokens.is_empty() {
            let current = self.get_import_job(&pool_id).unwrap_or_else(|| json!({}));
            self.update_job(
                &pool_id,
                json!({
                    "status": "failed",
                    "completed": as_int(&current, "total"),
                    "failed": error_count(&current),
                }),
            );
            return;
        }

        let add_result = self.accounts.add_accounts(&tokens, "codex");
        let refresh_result = self.accounts.refresh_accounts(&tokens, None, false).await;
        let current = self.get_import_job(&pool_id).unwrap_or_else(|| json!({}));
        self.update_job(
            &pool_id,
            json!({
                "status": "completed",
                "completed": names.len(),
                "added": as_int(&add_result, "added"),
                "skipped": as_int(&add_result, "skipped"),
                "refreshed": as_int(&refresh_result, "refreshed"),
                "failed": error_count(&current),
            }),
        );
    }
}
