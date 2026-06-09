//! Port of `services/storage/database_storage.py` — SQLite database backend
//! via `rusqlite`. (Postgres/MySQL from the Python version are not built in this
//! port; the JSON, SQLite and Git backends cover the supported deployments.)

use std::sync::Mutex;

use rusqlite::Connection;
use serde_json::{json, Value};

use super::StorageBackend;

pub struct DatabaseStorageBackend {
    database_url: String,
    conn: Mutex<Connection>,
}

fn sqlite_path(url: &str) -> String {
    // Accept `sqlite:///abs/path`, `sqlite://path`, or a bare path.
    let trimmed = url.trim();
    let rest = trimmed
        .strip_prefix("sqlite:///")
        .or_else(|| trimmed.strip_prefix("sqlite://"))
        .unwrap_or(trimmed);
    // On Windows an absolute path may arrive as `/D:/...`; strip the leading slash.
    if rest.len() >= 3 && rest.starts_with('/') && rest.as_bytes()[2] == b':' {
        rest[1..].to_string()
    } else {
        rest.to_string()
    }
}

impl DatabaseStorageBackend {
    pub fn new(database_url: &str) -> anyhow::Result<Self> {
        let lower = database_url.to_lowercase();
        if lower.contains("postgres") || lower.contains("mysql") {
            anyhow::bail!(
                "database backend '{database_url}' (postgres/mysql) is not built in this port; \
                 use STORAGE_BACKEND=json, sqlite, or git"
            );
        }
        let path = sqlite_path(database_url);
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(&path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_token TEXT UNIQUE NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT UNIQUE NOT NULL,
                data TEXT NOT NULL
            );",
        )?;
        Ok(Self {
            database_url: database_url.to_string(),
            conn: Mutex::new(conn),
        })
    }

    fn load_rows(&self, table: &str) -> Vec<Value> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(&format!("SELECT data FROM {table}")) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let rows = stmt.query_map([], |row| row.get::<_, String>(0));
        let Ok(rows) = rows else { return vec![] };
        rows.filter_map(|r| r.ok())
            .filter_map(|data| serde_json::from_str::<Value>(&data).ok())
            .filter(|v| v.is_object())
            .collect()
    }

    fn save_rows(&self, table: &str, key_col: &str, source_key: &str, items: &[Value]) -> anyhow::Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute(&format!("DELETE FROM {table}"), [])?;
        for item in items {
            let Some(obj) = item.as_object() else { continue };
            let key_value = obj
                .get(source_key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if key_value.is_empty() {
                continue;
            }
            let data = serde_json::to_string(item)?;
            tx.execute(
                &format!("INSERT OR REPLACE INTO {table} ({key_col}, data) VALUES (?1, ?2)"),
                rusqlite::params![key_value, data],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}

impl StorageBackend for DatabaseStorageBackend {
    fn load_accounts(&self) -> Vec<Value> {
        self.load_rows("accounts")
    }

    fn save_accounts(&self, accounts: &[Value]) -> anyhow::Result<()> {
        self.save_rows("accounts", "access_token", "access_token", accounts)
    }

    fn load_auth_keys(&self) -> Vec<Value> {
        self.load_rows("auth_keys")
    }

    fn save_auth_keys(&self, auth_keys: &[Value]) -> anyhow::Result<()> {
        self.save_rows("auth_keys", "key_id", "id", auth_keys)
    }

    fn health_check(&self) -> Value {
        let conn = self.conn.lock().unwrap();
        let account_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM accounts", [], |r| r.get(0)).unwrap_or(0);
        let auth_key_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM auth_keys", [], |r| r.get(0)).unwrap_or(0);
        json!({
            "status": "healthy",
            "backend": "database",
            "database_url": mask_password(&self.database_url),
            "account_count": account_count,
            "auth_key_count": auth_key_count,
        })
    }

    fn get_backend_info(&self) -> Value {
        json!({
            "type": "database",
            "db_type": "sqlite",
            "description": "数据库存储 (sqlite)",
            "database_url": mask_password(&self.database_url),
        })
    }
}

fn mask_password(url: &str) -> String {
    if !url.contains("://") {
        return url.to_string();
    }
    let Some((protocol, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    if let Some((credentials, host)) = rest.split_once('@') {
        if let Some((username, _)) = credentials.split_once(':') {
            return format!("{protocol}://{username}:****@{host}");
        }
    }
    url.to_string()
}
