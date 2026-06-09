//! Port of `services/storage/json_storage.py`.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::StorageBackend;

pub struct JsonStorageBackend {
    file_path: PathBuf,
    auth_keys_path: PathBuf,
}

impl JsonStorageBackend {
    pub fn new(file_path: PathBuf, auth_keys_path: PathBuf) -> Self {
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Some(parent) = auth_keys_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        Self {
            file_path,
            auth_keys_path,
        }
    }

    fn load_json_list(path: &Path) -> Vec<Value> {
        if !path.exists() {
            return Vec::new();
        }
        match std::fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(Value::Array(items)) => items,
                _ => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    }

    fn save_json_list(path: &Path, items: &[Value]) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&items)? + "\n";
        std::fs::write(path, text)?;
        Ok(())
    }
}

impl StorageBackend for JsonStorageBackend {
    fn load_accounts(&self) -> Vec<Value> {
        Self::load_json_list(&self.file_path)
    }

    fn save_accounts(&self, accounts: &[Value]) -> anyhow::Result<()> {
        Self::save_json_list(&self.file_path, accounts)
    }

    fn load_auth_keys(&self) -> Vec<Value> {
        if !self.auth_keys_path.exists() {
            return Vec::new();
        }
        let data: Value = match std::fs::read_to_string(&self.auth_keys_path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or(Value::Null),
            Err(_) => return Vec::new(),
        };
        match data {
            // `{"items": [...]}` wrapper form
            Value::Object(map) => match map.get("items") {
                Some(Value::Array(items)) => items.clone(),
                _ => Vec::new(),
            },
            Value::Array(items) => items,
            _ => Vec::new(),
        }
    }

    fn save_auth_keys(&self, auth_keys: &[Value]) -> anyhow::Result<()> {
        if let Some(parent) = self.auth_keys_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = json!({ "items": auth_keys });
        let text = serde_json::to_string_pretty(&payload)? + "\n";
        std::fs::write(&self.auth_keys_path, text)?;
        Ok(())
    }

    fn health_check(&self) -> Value {
        if self.file_path.exists() {
            if let Err(e) = std::fs::read_to_string(&self.file_path) {
                return json!({
                    "status": "unhealthy",
                    "backend": "json",
                    "error": e.to_string(),
                });
            }
        }
        json!({
            "status": "healthy",
            "backend": "json",
            "file_exists": self.file_path.exists(),
            "file_path": self.file_path.display().to_string(),
            "auth_keys_file_exists": self.auth_keys_path.exists(),
            "auth_keys_file_path": self.auth_keys_path.display().to_string(),
        })
    }

    fn get_backend_info(&self) -> Value {
        json!({
            "type": "json",
            "description": "本地 JSON 文件存储",
            "file_path": self.file_path.display().to_string(),
            "file_exists": self.file_path.exists(),
            "auth_keys_file_path": self.auth_keys_path.display().to_string(),
            "auth_keys_file_exists": self.auth_keys_path.exists(),
        })
    }
}
