//! Port of the `LogService` core in `services/log_service.py` — append-only
//! JSONL log of calls and account events. The `LoggedCall` request wrapper
//! (FastAPI-specific) is ported in Phase 7.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

pub const LOG_TYPE_CALL: &str = "call";
pub const LOG_TYPE_ACCOUNT: &str = "account";

/// Cloneable handle to the JSONL log file.
#[derive(Clone)]
pub struct LogService {
    inner: Arc<Inner>,
}

struct Inner {
    path: PathBuf,
    lock: Mutex<()>,
}

fn now_str() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn serialize_item(item: &Value) -> String {
    serde_json::to_string(item).unwrap_or_default()
}

impl LogService {
    pub fn new(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self {
            inner: Arc::new(Inner {
                path,
                lock: Mutex::new(()),
            }),
        }
    }

    /// Append a log item (`type` + summary + detail).
    pub fn add(&self, log_type: &str, summary: &str, detail: Value) {
        let item = json!({
            "id": uuid::Uuid::new_v4().simple().to_string(),
            "time": now_str(),
            "type": log_type,
            "summary": summary,
            "detail": detail,
        });
        let _guard = self.inner.lock.lock();
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.inner.path)
        {
            let _ = writeln!(file, "{}", serialize_item(&item));
        }
    }

    fn legacy_id(raw_line: &str, line_number: usize) -> String {
        use sha1::{Digest, Sha1};
        let payload = format!("{line_number}:{raw_line}");
        hex::encode(Sha1::digest(payload.as_bytes()))[..24].to_string()
    }

    fn parse_line(raw_line: &str, line_number: usize) -> Option<Value> {
        let mut item: Value = serde_json::from_str(raw_line).ok()?;
        if !item.is_object() {
            return None;
        }
        let has_id = item.get("id").and_then(|v| v.as_str()).map_or(false, |s| !s.is_empty());
        if !has_id {
            item["id"] = Value::String(Self::legacy_id(raw_line, line_number));
        }
        Some(item)
    }

    fn matches_filters(item: &Value, log_type: &str, start_date: &str, end_date: &str) -> bool {
        let t = item.get("time").and_then(|v| v.as_str()).unwrap_or("");
        let day = if t.len() >= 10 { &t[..10] } else { t };
        if !log_type.is_empty() && item.get("type").and_then(|v| v.as_str()) != Some(log_type) {
            return false;
        }
        if !start_date.is_empty() && day < start_date {
            return false;
        }
        if !end_date.is_empty() && day > end_date {
            return false;
        }
        true
    }

    /// List the most recent entries (newest first), filtered and limited.
    pub fn list(&self, log_type: &str, start_date: &str, end_date: &str, limit: usize) -> Vec<Value> {
        let _guard = self.inner.lock.lock();
        let content = match std::fs::read_to_string(&self.inner.path) {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let lines: Vec<&str> = content.lines().collect();
        let mut items = Vec::new();
        for line_number in (0..lines.len()).rev() {
            let Some(item) = Self::parse_line(lines[line_number], line_number) else {
                continue;
            };
            if !Self::matches_filters(&item, log_type, start_date, end_date) {
                continue;
            }
            items.push(item);
            if items.len() >= limit {
                break;
            }
        }
        items
    }

    /// Delete entries by id. Returns the number removed.
    pub fn delete(&self, ids: &[String]) -> usize {
        let target: std::collections::HashSet<String> =
            ids.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        if target.is_empty() {
            return 0;
        }
        let _guard = self.inner.lock.lock();
        let content = match std::fs::read_to_string(&self.inner.path) {
            Ok(c) => c,
            Err(_) => return 0,
        };
        let lines: Vec<&str> = content.lines().collect();
        let mut kept: Vec<String> = Vec::new();
        let mut removed = 0;
        for (line_number, raw_line) in lines.iter().enumerate() {
            match Self::parse_line(raw_line, line_number) {
                None => kept.push(raw_line.to_string()),
                Some(item) => {
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    if target.contains(id) {
                        removed += 1;
                    } else {
                        kept.push(serialize_item(&item));
                    }
                }
            }
        }
        let mut out = kept.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        let _ = std::fs::write(&self.inner.path, out);
        removed
    }
}
