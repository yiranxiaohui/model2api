//! Port of `services/config.py` — configuration store backed by `config.json`
//! plus environment-variable overrides. Values are read lazily via typed
//! getters that mirror the Python `ConfigStore` properties.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::{json, Map, Value};

/// Default `backup.include` map.
fn default_backup_include() -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in [
        ("config", true),
        ("register", true),
        ("cpa", true),
        ("sub2api", true),
        ("logs", true),
        ("image_tasks", true),
        ("accounts_snapshot", true),
        ("auth_keys_snapshot", true),
        ("images", false),
    ] {
        m.insert(k.to_string(), Value::Bool(v));
    }
    m
}

fn normalize_bool(value: Option<&Value>, default: bool) -> bool {
    match value {
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(default),
        Some(Value::Null) | None => default,
        Some(_) => default,
    }
}

fn normalize_positive_int(value: Option<&Value>, default: i64, minimum: i64) -> i64 {
    let parsed = match value {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Some(Value::String(s)) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
    .unwrap_or(default);
    parsed.max(minimum)
}

fn str_field(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Shared, cloneable handle to the configuration store.
#[derive(Clone)]
pub struct Config {
    inner: Arc<ConfigInner>,
}

struct ConfigInner {
    path: PathBuf,
    data_dir: PathBuf,
    version_file: PathBuf,
    data: RwLock<Map<String, Value>>,
}

impl Config {
    /// Load configuration from `config.json` in `base_dir`. Validates that an
    /// auth-key is configured (env `CHATGPT2API_AUTH_KEY` or `auth-key` field).
    pub fn load(base_dir: &Path) -> anyhow::Result<Self> {
        let path = base_dir.join("config.json");
        let data_dir = base_dir.join("data");
        let version_file = base_dir.join("VERSION");
        std::fs::create_dir_all(&data_dir).ok();

        let data = read_json_object(&path);
        let cfg = Config {
            inner: Arc::new(ConfigInner {
                path,
                data_dir,
                version_file,
                data: RwLock::new(data),
            }),
        };

        if cfg.auth_key().is_empty() {
            anyhow::bail!(
                "❌ auth-key 未设置！\n请在环境变量 CHATGPT2API_AUTH_KEY 中设置，或在 config.json 中填写 auth-key。"
            );
        }
        Ok(cfg)
    }

    pub fn data_dir(&self) -> &Path {
        &self.inner.data_dir
    }

    pub fn accounts_file(&self) -> PathBuf {
        self.inner.data_dir.join("accounts.json")
    }

    pub fn images_dir(&self) -> PathBuf {
        let p = self.inner.data_dir.join("images");
        std::fs::create_dir_all(&p).ok();
        p
    }

    pub fn image_thumbnails_dir(&self) -> PathBuf {
        let p = self.inner.data_dir.join("image_thumbnails");
        std::fs::create_dir_all(&p).ok();
        p
    }

    /// Read a top-level raw value (clone).
    fn get_raw(&self, key: &str) -> Option<Value> {
        self.inner.data.read().get(key).cloned()
    }

    pub fn auth_key(&self) -> String {
        if let Ok(env) = std::env::var("CHATGPT2API_AUTH_KEY") {
            let trimmed = env.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
        str_field(self.get_raw("auth-key").as_ref()).trim().to_string()
    }

    pub fn base_url(&self) -> String {
        let env = std::env::var("CHATGPT2API_BASE_URL").unwrap_or_default();
        let val = if !env.trim().is_empty() {
            env
        } else {
            str_field(self.get_raw("base_url").as_ref())
        };
        val.trim().trim_end_matches('/').to_string()
    }

    pub fn refresh_account_interval_minute(&self) -> i64 {
        normalize_positive_int(self.get_raw("refresh_account_interval_minute").as_ref(), 5, 0)
            .max(1)
    }

    pub fn image_retention_days(&self) -> i64 {
        normalize_positive_int(self.get_raw("image_retention_days").as_ref(), 30, 1)
    }

    pub fn image_poll_timeout_secs(&self) -> i64 {
        normalize_positive_int(self.get_raw("image_poll_timeout_secs").as_ref(), 120, 1)
    }

    pub fn image_poll_interval_secs(&self) -> f64 {
        as_f64(self.get_raw("image_poll_interval_secs").as_ref(), 10.0).max(0.5)
    }

    pub fn image_poll_initial_wait_secs(&self) -> f64 {
        as_f64(self.get_raw("image_poll_initial_wait_secs").as_ref(), 10.0).max(0.0)
    }

    pub fn image_account_concurrency(&self) -> i64 {
        normalize_positive_int(self.get_raw("image_account_concurrency").as_ref(), 3, 1)
    }

    pub fn image_parallel_generation(&self) -> bool {
        normalize_bool(self.get_raw("image_parallel_generation").as_ref(), true)
    }

    pub fn image_settle_enabled(&self) -> bool {
        normalize_bool(self.get_raw("image_settle_enabled").as_ref(), true)
    }

    pub fn image_check_before_hit_enabled(&self) -> bool {
        normalize_bool(self.get_raw("image_check_before_hit_enabled").as_ref(), true)
    }

    pub fn image_settle_secs(&self) -> f64 {
        as_f64(self.get_raw("image_settle_secs").as_ref(), 2.0).max(0.5)
    }

    pub fn image_timeout_retry_secs(&self) -> f64 {
        as_f64(self.get_raw("image_timeout_retry_secs").as_ref(), 30.0).max(0.0)
    }

    pub fn image_min_free_mb(&self) -> i64 {
        normalize_positive_int(self.get_raw("image_min_free_mb").as_ref(), 500, 0)
    }

    pub fn auto_remove_invalid_accounts(&self) -> bool {
        normalize_bool(self.get_raw("auto_remove_invalid_accounts").as_ref(), false)
    }

    pub fn auto_remove_rate_limited_accounts(&self) -> bool {
        normalize_bool(self.get_raw("auto_remove_rate_limited_accounts").as_ref(), false)
    }

    pub fn auto_relogin_after_refresh(&self) -> bool {
        normalize_bool(self.get_raw("auto_relogin_after_refresh").as_ref(), false)
    }

    pub fn log_levels(&self) -> Vec<String> {
        let allowed = ["debug", "info", "warning", "error"];
        match self.get_raw("log_levels") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => {
                        let lvl = s.trim().to_ascii_lowercase();
                        allowed.contains(&lvl.as_str()).then_some(lvl)
                    }
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn sensitive_words(&self) -> Vec<String> {
        match self.get_raw("sensitive_words") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn ai_review(&self) -> Value {
        match self.get_raw("ai_review") {
            Some(v @ Value::Object(_)) => v,
            _ => json!({}),
        }
    }

    pub fn global_system_prompt(&self) -> String {
        str_field(self.get_raw("global_system_prompt").as_ref())
            .trim()
            .to_string()
    }

    pub fn proxy_setting(&self) -> String {
        str_field(self.get_raw("proxy").as_ref()).trim().to_string()
    }

    pub fn app_version(&self) -> String {
        match std::fs::read_to_string(&self.inner.version_file) {
            Ok(s) => {
                let v = s.trim().to_string();
                if v.is_empty() {
                    "0.0.0".to_string()
                } else {
                    v
                }
            }
            Err(_) => "0.0.0".to_string(),
        }
    }

    /// Public config snapshot (mirrors `ConfigStore.get`) with `auth-key` stripped.
    pub fn get_public(&self) -> Value {
        let mut data = self.inner.data.read().clone();
        data.insert(
            "refresh_account_interval_minute".into(),
            json!(self.refresh_account_interval_minute()),
        );
        data.insert("image_retention_days".into(), json!(self.image_retention_days()));
        data.insert("image_poll_timeout_secs".into(), json!(self.image_poll_timeout_secs()));
        data.insert("image_poll_interval_secs".into(), json!(self.image_poll_interval_secs()));
        data.insert(
            "image_poll_initial_wait_secs".into(),
            json!(self.image_poll_initial_wait_secs()),
        );
        data.insert(
            "image_account_concurrency".into(),
            json!(self.image_account_concurrency()),
        );
        data.insert(
            "image_parallel_generation".into(),
            json!(self.image_parallel_generation()),
        );
        data.insert(
            "auto_remove_invalid_accounts".into(),
            json!(self.auto_remove_invalid_accounts()),
        );
        data.insert(
            "auto_remove_rate_limited_accounts".into(),
            json!(self.auto_remove_rate_limited_accounts()),
        );
        data.insert(
            "auto_relogin_after_refresh".into(),
            json!(self.auto_relogin_after_refresh()),
        );
        data.insert("log_levels".into(), json!(self.log_levels()));
        data.insert("sensitive_words".into(), json!(self.sensitive_words()));
        data.insert("ai_review".into(), self.ai_review());
        data.insert("global_system_prompt".into(), json!(self.global_system_prompt()));
        data.insert("backup".into(), self.get_backup_settings());
        data.insert("image_storage".into(), self.get_image_storage_settings());
        data.insert(
            "chat_completion_cache".into(),
            self.get_chat_completion_cache_settings(),
        );
        data.remove("auth-key");
        Value::Object(data)
    }

    /// Merge `patch` into config and persist. Returns the public snapshot.
    pub fn update(&self, patch: Value) -> anyhow::Result<Value> {
        if let Value::Object(patch_map) = patch {
            let mut guard = self.inner.data.write();
            for (k, v) in patch_map {
                guard.insert(k, v);
            }
            // normalize nested sections
            if guard.contains_key("backup") {
                let normalized = normalize_backup_settings(guard.get("backup"));
                guard.insert("backup".into(), normalized);
            }
            if guard.contains_key("image_storage") {
                let normalized = normalize_image_storage_settings(guard.get("image_storage"));
                validate_image_storage_settings(&normalized)?;
                guard.insert("image_storage".into(), normalized);
            }
            if guard.contains_key("chat_completion_cache") {
                let normalized =
                    normalize_chat_completion_cache_settings(guard.get("chat_completion_cache"));
                guard.insert("chat_completion_cache".into(), normalized);
            }
            guard.remove("backup_state");
            let to_save = Value::Object(guard.clone());
            drop(guard);
            self.save(&to_save)?;
        }
        Ok(self.get_public())
    }

    fn save(&self, data: &Value) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(data)? + "\n";
        std::fs::write(&self.inner.path, text)?;
        Ok(())
    }

    pub fn get_backup_settings(&self) -> Value {
        normalize_backup_settings(self.get_raw("backup").as_ref())
    }

    pub fn get_image_storage_settings(&self) -> Value {
        normalize_image_storage_settings(self.get_raw("image_storage").as_ref())
    }

    pub fn get_chat_completion_cache_settings(&self) -> Value {
        normalize_chat_completion_cache_settings(self.get_raw("chat_completion_cache").as_ref())
    }

    /// Delete images older than `image_retention_days`. Returns count removed.
    pub fn cleanup_old_images(&self) -> usize {
        let cutoff = std::time::SystemTime::now()
            - std::time::Duration::from_secs((self.image_retention_days() * 86400) as u64);
        let dir = self.images_dir();
        let mut removed = 0;
        for entry in walk_files(&dir) {
            if let Ok(meta) = entry.metadata() {
                if let Ok(modified) = meta.modified() {
                    if modified < cutoff {
                        if std::fs::remove_file(entry.path()).is_ok() {
                            removed += 1;
                        }
                    }
                }
            }
        }
        removed
    }
}

fn as_f64(value: Option<&Value>, default: f64) -> f64 {
    match value {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(default),
        _ => default,
    }
}

fn read_json_object(path: &Path) -> Map<String, Value> {
    if !path.exists() || path.is_dir() {
        return Map::new();
    }
    match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        },
        Err(_) => Map::new(),
    }
}

fn normalize_backup_settings(value: Option<&Value>) -> Value {
    let src = value.and_then(|v| v.as_object());
    let get = |k: &str| src.and_then(|m| m.get(k));
    let prefix = {
        let p = str_field(get("prefix")).trim().trim_matches('/').to_string();
        if p.is_empty() {
            "backups".to_string()
        } else {
            p
        }
    };
    let include = match get("include") {
        Some(Value::Object(m)) => {
            let mut out = default_backup_include();
            for (k, default_v) in out.clone() {
                let dv = default_v.as_bool().unwrap_or(false);
                out.insert(k.clone(), Value::Bool(normalize_bool(m.get(&k), dv)));
            }
            out
        }
        _ => default_backup_include(),
    };
    json!({
        "enabled": normalize_bool(get("enabled"), false),
        "provider": "cloudflare_r2",
        "account_id": str_field(get("account_id")).trim(),
        "access_key_id": str_field(get("access_key_id")).trim(),
        "secret_access_key": str_field(get("secret_access_key")).trim(),
        "bucket": str_field(get("bucket")).trim(),
        "prefix": prefix,
        "interval_minutes": normalize_positive_int(get("interval_minutes"), 360, 1),
        "rotation_keep": normalize_positive_int(get("rotation_keep"), 10, 0),
        "encrypt": normalize_bool(get("encrypt"), false),
        "passphrase": str_field(get("passphrase")).trim(),
        "include": include,
    })
}

fn normalize_image_storage_settings(value: Option<&Value>) -> Value {
    let src = value.and_then(|v| v.as_object());
    let get = |k: &str| src.and_then(|m| m.get(k));
    let enabled = normalize_bool(get("enabled"), false);
    let mut mode = str_field(get("mode")).trim().to_ascii_lowercase();
    if !["local", "webdav", "both"].contains(&mode.as_str()) {
        mode = "local".into();
    }
    if !enabled {
        mode = "local".into();
    }
    let default_root = "chatgpt2api/images";
    let root = {
        let r = str_field(get("webdav_root_path"));
        let r = r.trim().trim_matches('/').to_string();
        if r.is_empty() {
            default_root.to_string()
        } else {
            r
        }
    };
    json!({
        "enabled": enabled,
        "mode": mode,
        "webdav_url": str_field(get("webdav_url")).trim().trim_end_matches('/'),
        "webdav_username": str_field(get("webdav_username")).trim(),
        "webdav_password": str_field(get("webdav_password")).trim(),
        "webdav_root_path": root,
        "public_base_url": str_field(get("public_base_url")).trim().trim_end_matches('/'),
    })
}

fn validate_image_storage_settings(settings: &Value) -> anyhow::Result<()> {
    if !normalize_bool(settings.get("enabled"), false) {
        return Ok(());
    }
    if str_field(settings.get("webdav_url")).trim().is_empty() {
        anyhow::bail!("启用 WebDAV 图片存储后必须填写 WebDAV URL");
    }
    if str_field(settings.get("webdav_password")).trim().is_empty() {
        anyhow::bail!("启用 WebDAV 图片存储后必须填写 WebDAV 密码");
    }
    Ok(())
}

fn normalize_chat_completion_cache_settings(value: Option<&Value>) -> Value {
    let src = value.and_then(|v| v.as_object());
    let get = |k: &str| src.and_then(|m| m.get(k));
    json!({
        "enabled": normalize_bool(get("enabled"), true),
        "ttl_seconds": normalize_positive_int(get("ttl_seconds"), 60, 0),
        "max_entries": normalize_positive_int(get("max_entries"), 256, 1),
        "dedupe_inflight": normalize_bool(get("dedupe_inflight"), true),
        "stream_cache": normalize_bool(get("stream_cache"), true),
        "normalize_messages": normalize_bool(get("normalize_messages"), true),
        "drop_adjacent_duplicates": normalize_bool(get("drop_adjacent_duplicates"), true),
        "drop_assistant_history": normalize_bool(get("drop_assistant_history"), false),
    })
}

/// Recursively walk files under `dir`.
fn walk_files(dir: &Path) -> Vec<std::fs::DirEntry> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(walk_files(&path));
            } else {
                out.push(entry);
            }
        }
    }
    out
}
