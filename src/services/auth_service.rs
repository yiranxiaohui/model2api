//! Port of `services/auth_service.py` — management of API auth keys (admin/user
//! roles), with SHA-256 key hashing and constant-time comparison.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::config::Config;
use crate::services::storage::StorageBackend;

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

fn hash_key(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

fn clean(value: &Value) -> String {
    match value {
        Value::String(s) => s.trim().to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn uuid12() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

/// A stored auth key (internal representation; `key_hash` is never exposed).
#[derive(Debug, Clone)]
struct AuthKeyItem {
    id: String,
    name: String,
    role: String,
    key_hash: String,
    enabled: bool,
    created_at: String,
    last_used_at: Option<String>,
}

impl AuthKeyItem {
    fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "key_hash": self.key_hash,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        })
    }

    fn public(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        })
    }
}

fn default_name(role: &str) -> String {
    if role.trim().to_lowercase() == "admin" {
        "管理员密钥".to_string()
    } else {
        "普通用户".to_string()
    }
}

fn normalize_item(raw: &Value) -> Option<AuthKeyItem> {
    let obj = raw.as_object()?;
    let role = clean(obj.get("role").unwrap_or(&Value::Null)).to_lowercase();
    if role != "admin" && role != "user" {
        return None;
    }
    let key_hash = clean(obj.get("key_hash").unwrap_or(&Value::Null));
    if key_hash.is_empty() {
        return None;
    }
    let item_id = {
        let v = clean(obj.get("id").unwrap_or(&Value::Null));
        if v.is_empty() {
            uuid12()
        } else {
            v
        }
    };
    let name = {
        let v = clean(obj.get("name").unwrap_or(&Value::Null));
        if v.is_empty() {
            default_name(&role)
        } else {
            v
        }
    };
    let created_at = {
        let v = clean(obj.get("created_at").unwrap_or(&Value::Null));
        if v.is_empty() {
            now_iso()
        } else {
            v
        }
    };
    let last_used_at = {
        let v = clean(obj.get("last_used_at").unwrap_or(&Value::Null));
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    };
    let enabled = obj.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
    Some(AuthKeyItem {
        id: item_id,
        name,
        role,
        key_hash,
        enabled,
        created_at,
        last_used_at,
    })
}

/// The auth-key service. Holds an in-memory copy of normalized keys, reloaded
/// from storage on mutating operations (mirrors the Python `_reload_locked`).
pub struct AuthService {
    storage: Arc<dyn StorageBackend>,
    config: Config,
    inner: Mutex<Inner>,
}

struct Inner {
    items: Vec<AuthKeyItem>,
    last_used_flush_at: HashMap<String, chrono::DateTime<chrono::Utc>>,
}

impl AuthService {
    pub fn new(storage: Arc<dyn StorageBackend>, config: Config) -> Self {
        let items = Self::load(&storage);
        Self {
            storage,
            config,
            inner: Mutex::new(Inner {
                items,
                last_used_flush_at: HashMap::new(),
            }),
        }
    }

    fn load(storage: &Arc<dyn StorageBackend>) -> Vec<AuthKeyItem> {
        storage
            .load_auth_keys()
            .iter()
            .filter_map(normalize_item)
            .collect()
    }

    fn save(&self, items: &[AuthKeyItem]) -> anyhow::Result<()> {
        let values: Vec<Value> = items.iter().map(|i| i.to_value()).collect();
        self.storage.save_auth_keys(&values)
    }

    fn has_key_hash(items: &[AuthKeyItem], key_hash: &str, exclude_id: &str) -> bool {
        items.iter().any(|item| {
            (exclude_id.is_empty() || item.id != exclude_id)
                && !item.key_hash.is_empty()
                && ct_eq(&item.key_hash, key_hash)
        })
    }

    fn build_key_hash(&self, items: &[AuthKeyItem], raw_key: &str, exclude_id: &str) -> Result<String, String> {
        let candidate = raw_key.trim();
        if candidate.is_empty() {
            return Err("请输入新的专用密钥".to_string());
        }
        let admin_key = self.config.auth_key();
        let admin_key = admin_key.trim();
        if !admin_key.is_empty() && ct_eq(candidate, admin_key) {
            return Err("这个密钥和管理员密钥冲突了，请换一个新的密钥".to_string());
        }
        let key_hash = hash_key(candidate);
        if Self::has_key_hash(items, &key_hash, exclude_id) {
            return Err("这个专用密钥已经存在，请换一个新的密钥".to_string());
        }
        Ok(key_hash)
    }

    fn has_name(items: &[AuthKeyItem], name: &str, role: Option<&str>, exclude_id: &str) -> bool {
        let candidate = name.trim();
        if candidate.is_empty() {
            return false;
        }
        items.iter().any(|item| {
            (exclude_id.is_empty() || item.id != exclude_id)
                && role.map_or(true, |r| item.role == r)
                && item.name == candidate
        })
    }

    fn build_default_name(items: &[AuthKeyItem], role: &str, exclude_id: &str) -> String {
        let base_name = default_name(role);
        if !Self::has_name(items, &base_name, Some(role), exclude_id) {
            return base_name;
        }
        let mut suffix = 2;
        loop {
            let candidate = format!("{base_name} {suffix}");
            if !Self::has_name(items, &candidate, Some(role), exclude_id) {
                return candidate;
            }
            suffix += 1;
        }
    }

    fn build_name(items: &[AuthKeyItem], name: &str, role: &str, exclude_id: &str) -> Result<String, String> {
        let candidate = name.trim();
        if candidate.is_empty() {
            return Ok(Self::build_default_name(items, role, exclude_id));
        }
        if Self::has_name(items, candidate, Some(role), exclude_id) {
            return Err("这个名称已经在使用中了，换一个更容易区分的名称吧".to_string());
        }
        Ok(candidate.to_string())
    }

    /// List public key records, optionally filtered by role.
    pub fn list_keys(&self, role: Option<&str>) -> Vec<Value> {
        let mut guard = self.inner.lock();
        guard.items = Self::load(&self.storage);
        guard
            .items
            .iter()
            .filter(|item| role.map_or(true, |r| item.role == r))
            .map(|item| item.public())
            .collect()
    }

    /// Create a new key; returns `(public_item, raw_key)`.
    pub fn create_key(&self, role: &str, name: &str) -> Result<(Value, String), String> {
        let mut guard = self.inner.lock();
        guard.items = Self::load(&self.storage);
        let normalized_name = Self::build_name(&guard.items, name, role, "")?;
        let (raw_key, key_hash) = loop {
            let raw_key = format!("sk-{}", token_urlsafe(24));
            match self.build_key_hash(&guard.items, &raw_key, "") {
                Ok(h) => break (raw_key, h),
                Err(_) => continue,
            }
        };
        let item = AuthKeyItem {
            id: uuid12(),
            name: normalized_name,
            role: role.to_string(),
            key_hash,
            enabled: true,
            created_at: now_iso(),
            last_used_at: None,
        };
        guard.items.push(item.clone());
        self.save(&guard.items).map_err(|e| e.to_string())?;
        Ok((item.public(), raw_key))
    }

    /// Update name/enabled/key on a key. Returns the updated public item.
    pub fn update_key(&self, key_id: &str, updates: &Value, role: Option<&str>) -> Option<Value> {
        let normalized_id = key_id.trim();
        if normalized_id.is_empty() {
            return None;
        }
        let mut guard = self.inner.lock();
        guard.items = Self::load(&self.storage);
        let idx = guard.items.iter().position(|i| i.id == normalized_id)?;
        if let Some(r) = role {
            if guard.items[idx].role != r {
                return None;
            }
        }
        let mut next = guard.items[idx].clone();
        let next_role = if next.role.trim().to_lowercase() == "admin" {
            "admin"
        } else {
            "user"
        };
        if let Some(name) = updates.get("name") {
            if !name.is_null() {
                let name_str = clean(name);
                match Self::build_name(&guard.items, &name_str, next_role, normalized_id) {
                    Ok(n) => next.name = n,
                    Err(_) => return None,
                }
            }
        }
        if let Some(enabled) = updates.get("enabled") {
            if !enabled.is_null() {
                next.enabled = enabled.as_bool().unwrap_or(next.enabled);
            }
        }
        if let Some(key) = updates.get("key") {
            if !key.is_null() {
                let key_str = clean(key);
                match self.build_key_hash(&guard.items, &key_str, normalized_id) {
                    Ok(h) => next.key_hash = h,
                    Err(_) => return None,
                }
            }
        }
        guard.items[idx] = next.clone();
        self.save(&guard.items).ok()?;
        Some(next.public())
    }

    /// Delete a key by id (optionally constrained to a role).
    pub fn delete_key(&self, key_id: &str, role: Option<&str>) -> bool {
        let normalized_id = key_id.trim();
        if normalized_id.is_empty() {
            return false;
        }
        let mut guard = self.inner.lock();
        guard.items = Self::load(&self.storage);
        let before = guard.items.len();
        guard
            .items
            .retain(|item| !(item.id == normalized_id && role.map_or(true, |r| item.role == r)));
        if guard.items.len() == before {
            return false;
        }
        self.save(&guard.items).is_ok()
    }

    /// Authenticate a raw key. Returns the public item on success, recording
    /// `last_used_at` (flushed to storage at most once per 60s per key).
    pub fn authenticate(&self, raw_key: &str) -> Option<Value> {
        let candidate = raw_key.trim();
        if candidate.is_empty() {
            return None;
        }
        let candidate_hash = hash_key(candidate);
        let mut guard = self.inner.lock();
        let idx = guard.items.iter().position(|item| {
            item.enabled && !item.key_hash.is_empty() && ct_eq(&item.key_hash, &candidate_hash)
        })?;
        let now = chrono::Utc::now();
        guard.items[idx].last_used_at = Some(now.to_rfc3339_opts(chrono::SecondsFormat::Micros, true));
        let item_id = guard.items[idx].id.clone();
        let public = guard.items[idx].public();
        let should_flush = guard
            .last_used_flush_at
            .get(&item_id)
            .map_or(true, |last| (now - *last).num_seconds() >= 60);
        if should_flush {
            let snapshot = guard.items.clone();
            if self.save(&snapshot).is_ok() {
                guard.last_used_flush_at.insert(item_id, now);
            }
        }
        Some(public)
    }
}

/// `secrets.token_urlsafe(n)` equivalent: n random bytes, URL-safe base64, no pad.
fn token_urlsafe(n: usize) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use rand::RngCore;
    let mut bytes = vec![0u8; n];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}
