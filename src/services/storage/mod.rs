//! Port of `services/storage/` — pluggable persistence for accounts and auth
//! keys. The JSON backend is implemented here; the database (sqlx) and git
//! (git2) backends are wired up in a later phase.

mod database_storage;
mod git_storage;
mod json_storage;

pub use database_storage::DatabaseStorageBackend;
pub use git_storage::GitStorageBackend;
pub use json_storage::JsonStorageBackend;

use serde_json::Value;
use std::path::Path;

/// Abstract storage backend (port of `StorageBackend` ABC). Methods are
/// blocking; callers on async paths should wrap in `spawn_blocking`.
pub trait StorageBackend: Send + Sync {
    fn load_accounts(&self) -> Vec<Value>;
    fn save_accounts(&self, accounts: &[Value]) -> anyhow::Result<()>;
    fn load_auth_keys(&self) -> Vec<Value>;
    fn save_auth_keys(&self, auth_keys: &[Value]) -> anyhow::Result<()>;
    fn health_check(&self) -> Value;
    fn get_backend_info(&self) -> Value;
}

/// Port of `services/storage/factory.create_storage_backend`. Chooses a backend
/// from the `STORAGE_BACKEND` environment variable.
pub fn create_storage_backend(data_dir: &Path) -> anyhow::Result<Box<dyn StorageBackend>> {
    let backend_type = std::env::var("STORAGE_BACKEND")
        .unwrap_or_else(|_| "json".to_string())
        .trim()
        .to_ascii_lowercase();

    tracing::info!("[storage] Initializing storage backend: {backend_type}");

    match backend_type.as_str() {
        "json" => {
            let file_path = data_dir.join("accounts.json");
            let auth_keys_path = data_dir.join("auth_keys.json");
            tracing::info!("[storage] Using JSON storage: {}", file_path.display());
            Ok(Box::new(JsonStorageBackend::new(file_path, auth_keys_path)))
        }
        "sqlite" | "postgres" | "postgresql" | "mysql" | "database" => {
            let mut database_url = std::env::var("DATABASE_URL").unwrap_or_default().trim().to_string();
            if database_url.is_empty() {
                database_url = format!("sqlite:///{}", data_dir.join("accounts.db").display());
                tracing::info!("[storage] No DATABASE_URL provided, using local SQLite: {database_url}");
            } else {
                tracing::info!("[storage] Using database storage");
            }
            Ok(Box::new(DatabaseStorageBackend::new(&database_url)?))
        }
        "git" => {
            let repo_url = std::env::var("GIT_REPO_URL").unwrap_or_default().trim().to_string();
            let token = std::env::var("GIT_TOKEN").unwrap_or_default().trim().to_string();
            let branch = {
                let b = std::env::var("GIT_BRANCH").unwrap_or_default().trim().to_string();
                if b.is_empty() { "main".to_string() } else { b }
            };
            let file_path = {
                let f = std::env::var("GIT_FILE_PATH").unwrap_or_default().trim().to_string();
                if f.is_empty() { "accounts.json".to_string() } else { f }
            };
            let auth_keys_file_path = {
                let f = std::env::var("GIT_AUTH_KEYS_FILE_PATH").unwrap_or_default().trim().to_string();
                if f.is_empty() { "auth_keys.json".to_string() } else { f }
            };
            if repo_url.is_empty() {
                anyhow::bail!("GIT_REPO_URL is required when using git storage backend.");
            }
            tracing::info!("[storage] Using Git storage, branch: {branch}, file: {file_path}");
            let cache_dir = data_dir.join("git_cache");
            Ok(Box::new(GitStorageBackend::new(
                &repo_url,
                &token,
                &branch,
                &file_path,
                &auth_keys_file_path,
                cache_dir,
            )?))
        }
        other => anyhow::bail!(
            "Unknown storage backend: {other}. Supported backends: json, sqlite, postgres, git"
        ),
    }
}
