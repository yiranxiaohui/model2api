//! Port of `services/editable_file_task_service.py` — async PPT/PSD export task
//! tracking.
//!
//! NOTE: the underlying engine editable-file export (PPT/PSD) is deferred (see
//! the project plan — the ~600-line `_export_editable_file_zip` flow in
//! `openai_backend_api.py`). This service therefore tracks task records but
//! reports submissions as unsupported until the engine methods land. The task
//! bookkeeping/listing API matches the Python so the routes stay faithful.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::config::Config;

#[derive(Clone)]
pub struct EditableFileTaskService {
    inner: Arc<Inner>,
}

struct Inner {
    #[allow(dead_code)]
    config: Config,
    tasks: Mutex<Vec<Value>>,
}

impl EditableFileTaskService {
    pub fn new(config: Config) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                tasks: Mutex::new(Vec::new()),
            }),
        }
    }

    fn owner_id(identity: &Value) -> String {
        identity.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string()
    }

    /// List tasks owned by `identity`, optionally filtered to `task_ids`.
    pub fn list_tasks(&self, identity: &Value, task_ids: &[String]) -> Value {
        let owner = Self::owner_id(identity);
        let tasks = self.inner.tasks.lock();
        let items: Vec<Value> = tasks
            .iter()
            .filter(|t| t.get("owner_id").and_then(|v| v.as_str()).unwrap_or("") == owner)
            .filter(|t| {
                task_ids.is_empty()
                    || task_ids.iter().any(|id| t.get("task_id").and_then(|v| v.as_str()) == Some(id.as_str()))
            })
            .cloned()
            .collect();
        json!({ "items": items })
    }

    fn submit_unsupported(&self, kind: &str) -> Value {
        json!({
            "error": format!("{kind} export is not available in this build (engine editable-file export pending)"),
        })
    }

    pub fn submit_ppt(
        &self,
        _identity: &Value,
        _client_task_id: &str,
        _prompt: &str,
        _base64_images: &[String],
        _base_url: &str,
    ) -> Value {
        self.submit_unsupported("PPT")
    }

    pub fn submit_psd(
        &self,
        _identity: &Value,
        _client_task_id: &str,
        _prompt: &str,
        _base64_images: &[String],
        _base_url: &str,
    ) -> Value {
        self.submit_unsupported("PSD")
    }

    /// Resolve a public file path for download (no files produced yet).
    pub fn public_file_path(&self, _file_path: &str) -> Option<std::path::PathBuf> {
        None
    }
}
