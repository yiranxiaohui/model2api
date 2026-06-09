//! Port of `services/image_task_service.py` — async image-generation task
//! tracker. A client submits a generation/edit task (identified by a
//! caller-supplied `client_task_id` scoped to the auth-key owner); the service
//! records a `queued` task, runs the image flow in the background via
//! `tokio::spawn`, and transitions the task through `running` → `success`/`error`
//! while persisting the full task map to `data_dir/image_tasks.json` (atomic
//! tmp-then-rename, exactly like the Python original).
//!
//! Adaptation notes vs. the Python source:
//!   * The Python `_run_task` called the `openai_v1_image_generations.handle` /
//!     `openai_v1_image_edit.handle` handlers. In Rust the image flow lives in
//!     `protocol::conversation`, so a [`ConversationRequest`] is built and run
//!     through [`stream_image_outputs_with_pool`] + [`collect_image_outputs`].
//!   * Granular per-step progress (Python's `progress_callback` fed by the
//!     streaming handler) is not surfaced by `collect_image_outputs`, which
//!     drains the whole channel into a final value. Progress is therefore
//!     coarse: the task is marked `running` (with a `started_ts` so
//!     `elapsed_secs` works) and the per-step `progress` field is not updated.
//!   * `usage` is not part of the `collect_image_outputs` result, so successful
//!     tasks do not carry a `usage` field (Python copied it from the handler).
//!   * `resume_poll` (continuing to poll a timed-out `conversation_id`) is NOT
//!     ported: it relied on private engine helpers (`_poll_image_results`,
//!     `download_image_bytes`, `format_image_result`) that the Rust engine does
//!     not expose — consistent with the `// TODO(engine-internal)` notes in
//!     `conversation.rs`.
//!   * Tasks are kept in a `parking_lot::Mutex<HashMap<..>>` of `serde_json`
//!     objects (Python used a `threading.RLock` + dict of dicts). Because the
//!     Rust mutex is not reentrant, the locked helpers operate on the guarded
//!     map directly rather than re-locking.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{Local, TimeZone};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use crate::services::log_service::{LogService, LOG_TYPE_CALL};
use crate::services::protocol::conversation::{
    collect_image_outputs, stream_image_outputs_with_pool, ConvDeps, ConversationRequest,
};

const TASK_STATUS_QUEUED: &str = "queued";
const TASK_STATUS_RUNNING: &str = "running";
const TASK_STATUS_SUCCESS: &str = "success";
const TASK_STATUS_ERROR: &str = "error";

const VALID_STATUSES: [&str; 4] = [
    TASK_STATUS_QUEUED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_ERROR,
];

/// Default message when the pool produced no image and no upstream message
/// (mirrors the Chinese fallback in the Python source).
const NO_ACCOUNT_MESSAGE: &str =
    "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）";

const RECOVER_MESSAGE: &str = "服务已重启，未完成的图片任务已中断";

// ---------------------------------------------------------------------------
// small helpers (ports of the module-level Python functions)
// ---------------------------------------------------------------------------

fn now_iso() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn fmt_unix(ts: f64) -> String {
    match Local.timestamp_opt(ts as i64, 0) {
        chrono::LocalResult::Single(dt) => dt.format("%Y-%m-%d %H:%M:%S").to_string(),
        _ => now_iso(),
    }
}

/// `str(value or default).strip()` for JSON values.
fn clean_value(value: Option<&Value>, default: &str) -> String {
    let raw = match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => default.to_string(),
        Some(Value::Bool(false)) => default.to_string(),
        Some(Value::Number(n)) if n.as_f64() == Some(0.0) => default.to_string(),
        Some(other) => other.to_string(),
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        default.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

fn clean_str(value: &str) -> String {
    value.trim().to_string()
}

fn owner_id(identity: &Value) -> String {
    let id = clean_value(identity.get("id"), "");
    if id.is_empty() {
        "anonymous".to_string()
    } else {
        id
    }
}

fn task_key(owner_id: &str, task_id: &str) -> String {
    format!("{owner_id}:{task_id}")
}

fn truthy_str(value: &Value) -> bool {
    value.as_str().map(|s| !s.is_empty()).unwrap_or(false)
}

fn num(value: Option<&Value>) -> f64 {
    value.and_then(|v| v.as_f64()).unwrap_or(0.0)
}

/// Parse a stored timestamp string into epoch seconds (for retention cleanup).
fn parse_timestamp(value: Option<&Value>) -> f64 {
    let s = match value {
        Some(Value::String(s)) => s.trim().to_string(),
        _ => return 0.0,
    };
    if s.is_empty() {
        return 0.0;
    }
    let head: String = s.chars().take(26).collect();
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S"] {
        let probe = if fmt == "%Y-%m-%d %H:%M:%S" {
            head.chars().take(19).collect::<String>()
        } else {
            head.clone()
        };
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&probe, fmt) {
            if let chrono::LocalResult::Single(local) = Local.from_local_datetime(&dt) {
                return local.timestamp() as f64;
            }
        }
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s.replace('Z', "+00:00")) {
        return dt.timestamp() as f64;
    }
    0.0
}

fn collect_image_urls(data: &[Value]) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    for item in data {
        if let Some(url) = item.get("url").and_then(|v| v.as_str()) {
            if !url.is_empty() {
                urls.push(url.to_string());
            }
        }
    }
    urls
}

/// Deduplicate preserving order (port of `list(dict.fromkeys(...))`).
fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for v in values {
        if seen.insert(v.clone()) {
            out.push(v);
        }
    }
    out
}

/// Build the public (client-facing) view of a task (port of `_public_task`).
fn public_task(task: &Value) -> Value {
    let mut item = Map::new();
    for key in ["id", "status", "mode", "model", "size", "quality", "created_at", "updated_at"] {
        item.insert(key.to_string(), task.get(key).cloned().unwrap_or(Value::Null));
    }
    if let Some(v) = task.get("conversation_id") {
        if truthy_str(v) {
            item.insert("conversation_id".into(), v.clone());
        }
    }
    if let Some(v) = task.get("data") {
        if !v.is_null() {
            item.insert("data".into(), v.clone());
        }
    }
    if let Some(v) = task.get("usage") {
        if !v.is_null() {
            item.insert("usage".into(), v.clone());
        }
    }
    if let Some(v) = task.get("error") {
        if truthy_str(v) {
            item.insert("error".into(), v.clone());
        }
    }
    if let Some(v) = task.get("progress") {
        if truthy_str(v) {
            item.insert("progress".into(), v.clone());
        }
    }
    if let Some(v) = task.get("duration_ms") {
        if !v.is_null() {
            item.insert("duration_ms".into(), v.clone());
        }
    }
    let status = task.get("status").and_then(|v| v.as_str()).unwrap_or("");
    if status == TASK_STATUS_RUNNING || status == TASK_STATUS_QUEUED {
        let base_ts = if status == TASK_STATUS_RUNNING {
            num(task.get("started_ts"))
        } else {
            let created = num(task.get("created_ts"));
            if created > 0.0 {
                created
            } else {
                num(task.get("updated_ts"))
            }
        };
        if base_ts > 0.0 {
            let elapsed = ((now_unix() - base_ts) * 10.0).round() / 10.0;
            item.insert("elapsed_secs".into(), json!(elapsed));
        }
    }
    Value::Object(item)
}

// ---------------------------------------------------------------------------
// locked map helpers (operate on the guarded HashMap directly)
// ---------------------------------------------------------------------------

fn load_tasks(path: &Path) -> HashMap<String, Value> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let raw: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let raw_items = if raw.is_object() {
        raw.get("tasks").cloned()
    } else {
        Some(raw)
    };
    let arr = match raw_items {
        Some(Value::Array(a)) => a,
        _ => return HashMap::new(),
    };
    let mut tasks: HashMap<String, Value> = HashMap::new();
    for item in arr {
        if !item.is_object() {
            continue;
        }
        let task_id = clean_value(item.get("id"), "");
        let owner = clean_value(item.get("owner_id"), "");
        if task_id.is_empty() || owner.is_empty() {
            continue;
        }
        let mut status = clean_value(item.get("status"), "");
        if !VALID_STATUSES.contains(&status.as_str()) {
            status = TASK_STATUS_ERROR.to_string();
        }
        let mode = if item.get("mode").and_then(|v| v.as_str()) == Some("edit") {
            "edit"
        } else {
            "generate"
        };
        let now = now_iso();
        let created_at = clean_value(item.get("created_at"), &now);
        let updated_at = clean_value(item.get("updated_at"), &created_at);
        let mut task = json!({
            "id": task_id,
            "owner_id": owner,
            "status": status,
            "mode": mode,
            "model": clean_value(item.get("model"), "gpt-image-2"),
            "size": clean_value(item.get("size"), ""),
            "quality": clean_value(item.get("quality"), "auto"),
            "created_at": created_at,
            "updated_at": updated_at,
            "created_ts": item.get("created_ts").cloned().unwrap_or(Value::Null),
            "updated_ts": item.get("updated_ts").cloned().unwrap_or(Value::Null),
            "started_ts": item.get("started_ts").cloned().unwrap_or(Value::Null),
            "duration_ms": item.get("duration_ms").cloned().unwrap_or(Value::Null),
        });
        if let Some(data) = item.get("data") {
            if data.is_array() {
                task["data"] = data.clone();
            }
        }
        if let Some(usage) = item.get("usage") {
            if usage.is_object() {
                task["usage"] = usage.clone();
            }
        }
        let error = clean_value(item.get("error"), "");
        if !error.is_empty() {
            task["error"] = json!(error);
        }
        tasks.insert(task_key(&owner, &task_id), task);
    }
    tasks
}

fn save_tasks(path: &Path, tasks: &HashMap<String, Value>) {
    let mut items: Vec<Value> = tasks.values().cloned().collect();
    items.sort_by(|a, b| {
        let av = a.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
        let bv = b.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
        bv.cmp(av) // descending (newest first)
    });
    let payload = json!({ "tasks": items });
    let text = serde_json::to_string_pretty(&payload).unwrap_or_default() + "\n";
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, text.as_bytes()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn recover_unfinished(tasks: &mut HashMap<String, Value>) -> bool {
    let mut changed = false;
    for task in tasks.values_mut() {
        let status = task.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if status == TASK_STATUS_QUEUED || status == TASK_STATUS_RUNNING {
            if let Some(obj) = task.as_object_mut() {
                obj.insert("status".into(), json!(TASK_STATUS_ERROR));
                obj.insert("error".into(), json!(RECOVER_MESSAGE));
                obj.insert("updated_at".into(), json!(now_iso()));
            }
            changed = true;
        }
    }
    changed
}

fn cleanup_tasks(tasks: &mut HashMap<String, Value>, retention_days: i64) -> bool {
    let retention = retention_days.max(1);
    let cutoff = now_unix() - (retention as f64) * 86400.0;
    let remove: Vec<String> = tasks
        .iter()
        .filter(|(_, task)| {
            let status = task.get("status").and_then(|v| v.as_str()).unwrap_or("");
            (status == TASK_STATUS_SUCCESS || status == TASK_STATUS_ERROR)
                && parse_timestamp(task.get("updated_at")) < cutoff
        })
        .map(|(k, _)| k.clone())
        .collect();
    for key in &remove {
        tasks.remove(key);
    }
    !remove.is_empty()
}

// ---------------------------------------------------------------------------
// ImageTaskService
// ---------------------------------------------------------------------------

/// Cheaply-cloneable handle (inner `Arc`, like `LogService`).
#[derive(Clone)]
pub struct ImageTaskService {
    inner: Arc<Inner>,
}

struct Inner {
    deps: ConvDeps,
    log: LogService,
    path: PathBuf,
    tasks: Mutex<HashMap<String, Value>>,
}

impl ImageTaskService {
    /// Build the service, load the persisted task map, fail any unfinished tasks
    /// (server restarted mid-run), prune expired ones, and persist if changed.
    pub fn new(deps: ConvDeps) -> Self {
        let data_dir = deps.config.data_dir().to_path_buf();
        let path = data_dir.join("image_tasks.json");
        let log = LogService::new(data_dir.join("logs.jsonl"));
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let retention = deps.config.image_retention_days();
        let svc = Self {
            inner: Arc::new(Inner {
                deps,
                log,
                path,
                tasks: Mutex::new(HashMap::new()),
            }),
        };
        {
            let mut tasks = svc.inner.tasks.lock();
            *tasks = load_tasks(&svc.inner.path);
            let mut changed = recover_unfinished(&mut tasks);
            changed = cleanup_tasks(&mut tasks, retention) || changed;
            if changed {
                save_tasks(&svc.inner.path, &tasks);
            }
        }
        svc
    }

    fn retention(&self) -> i64 {
        self.inner.deps.config.image_retention_days()
    }

    // ---- public API ----

    /// Submit a text-to-image generation task.
    pub async fn submit_generation(
        &self,
        identity: &Value,
        client_task_id: &str,
        prompt: &str,
        model: &str,
        size: Option<&str>,
        quality: &str,
        base_url: &str,
    ) -> Value {
        let request = ConversationRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            images: None,
            n: 1,
            size: size
                .map(|s| s.to_string())
                .filter(|s| !s.trim().is_empty()),
            quality: if quality.trim().is_empty() {
                "auto".to_string()
            } else {
                quality.to_string()
            },
            response_format: "url".to_string(),
            base_url: if base_url.trim().is_empty() {
                None
            } else {
                Some(base_url.to_string())
            },
            ..Default::default()
        };
        self.submit(identity, client_task_id, "generate", request).await
    }

    /// Submit an image-edit (image-to-image) task. `images` are the input images
    /// in whatever encoding the engine accepts (base64 / data-URL strings).
    pub async fn submit_edit(
        &self,
        identity: &Value,
        client_task_id: &str,
        prompt: &str,
        model: &str,
        size: Option<&str>,
        quality: &str,
        base_url: &str,
        images: Vec<String>,
    ) -> Value {
        let request = ConversationRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            images: Some(images),
            n: 1,
            size: size
                .map(|s| s.to_string())
                .filter(|s| !s.trim().is_empty()),
            quality: if quality.trim().is_empty() {
                "auto".to_string()
            } else {
                quality.to_string()
            },
            response_format: "url".to_string(),
            base_url: if base_url.trim().is_empty() {
                None
            } else {
                Some(base_url.to_string())
            },
            ..Default::default()
        };
        self.submit(identity, client_task_id, "edit", request).await
    }

    /// Query tasks for `identity`. With explicit `task_ids`, return the matching
    /// public tasks plus a `missing_ids` list; with an empty list, return all of
    /// the owner's tasks (newest first). Returns `{ "items": [...], "missing_ids": [...] }`.
    pub fn list_tasks(&self, identity: &Value, task_ids: &[String]) -> Value {
        let owner = owner_id(identity);
        let requested: Vec<String> = task_ids
            .iter()
            .map(|t| clean_str(t))
            .filter(|s| !s.is_empty())
            .collect();
        let mut tasks = self.inner.tasks.lock();
        if cleanup_tasks(&mut tasks, self.retention()) {
            save_tasks(&self.inner.path, &tasks);
        }
        let mut items: Vec<Value> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        if requested.is_empty() {
            for task in tasks.values() {
                if task.get("owner_id").and_then(|v| v.as_str()) == Some(owner.as_str()) {
                    items.push(public_task(task));
                }
            }
            items.sort_by(|a, b| {
                let av = a.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
                let bv = b.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
                bv.cmp(av)
            });
        } else {
            for task_id in &requested {
                match tasks.get(&task_key(&owner, task_id)) {
                    Some(task) => items.push(public_task(task)),
                    None => missing.push(task_id.clone()),
                }
            }
        }
        json!({ "items": items, "missing_ids": missing })
    }

    /// Convenience single-task lookup. Returns the public task or `null`.
    pub fn get_task(&self, identity: &Value, task_id: &str) -> Value {
        let owner = owner_id(identity);
        let key = task_key(&owner, &clean_str(task_id));
        let tasks = self.inner.tasks.lock();
        match tasks.get(&key) {
            Some(task) => public_task(task),
            None => Value::Null,
        }
    }

    /// Prune expired terminal tasks now. Returns `{ "removed": <bool> }`.
    pub fn cleanup(&self) -> Value {
        let mut tasks = self.inner.tasks.lock();
        let removed = cleanup_tasks(&mut tasks, self.retention());
        if removed {
            save_tasks(&self.inner.path, &tasks);
        }
        json!({ "removed": removed })
    }

    // ---- internal ----

    async fn submit(
        &self,
        identity: &Value,
        client_task_id: &str,
        mode: &str,
        request: ConversationRequest,
    ) -> Value {
        let task_id = clean_str(client_task_id);
        if task_id.is_empty() {
            return json!({ "error": "client_task_id is required" });
        }
        let owner = owner_id(identity);
        let key = task_key(&owner, &task_id);
        let now = now_iso();

        let model = if request.model.trim().is_empty() {
            "gpt-image-2".to_string()
        } else {
            request.model.trim().to_string()
        };
        let size = request.size.clone().unwrap_or_default().trim().to_string();
        let quality = if request.quality.trim().is_empty() {
            "auto".to_string()
        } else {
            request.quality.trim().to_string()
        };

        let snapshot;
        {
            let mut tasks = self.inner.tasks.lock();
            let cleaned = cleanup_tasks(&mut tasks, self.retention());
            if let Some(existing) = tasks.get(&key) {
                let public = public_task(existing);
                if cleaned {
                    save_tasks(&self.inner.path, &tasks);
                }
                return public;
            }
            let task = json!({
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": model.clone(),
                "size": size,
                "quality": quality,
                "created_at": now,
                "updated_at": now,
                "created_ts": now_unix(),
            });
            snapshot = public_task(&task);
            tasks.insert(key.clone(), task);
            save_tasks(&self.inner.path, &tasks);
        }

        // Run the generation in the background.
        let svc = self.clone();
        let identity_owned = identity.clone();
        let mode_owned = mode.to_string();
        let model_owned = model;
        let key_owned = key;
        tokio::spawn(async move {
            svc.run_task(key_owned, mode_owned, request, identity_owned, model_owned)
                .await;
        });

        snapshot
    }

    async fn run_task(
        &self,
        key: String,
        mode: String,
        request: ConversationRequest,
        identity: Value,
        model: String,
    ) {
        let started = now_unix();
        self.update_task(
            &key,
            vec![
                ("status", json!(TASK_STATUS_RUNNING)),
                ("error", json!("")),
                ("started_ts", json!(started)),
                ("progress", json!("running")),
            ],
        );

        let prompt = request.prompt.clone();
        let rx = stream_image_outputs_with_pool(self.inner.deps.clone(), request);

        match collect_image_outputs(rx).await {
            Ok(result) => {
                let data: Vec<Value> = result
                    .get("data")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let account_email = clean_value(result.get("_account_email"), "");
                if data.is_empty() {
                    let upstream = clean_value(result.get("message"), "");
                    let message = if upstream.is_empty() {
                        NO_ACCOUNT_MESSAGE.to_string()
                    } else {
                        upstream
                    };
                    let duration_ms = ((now_unix() - started) * 1000.0) as i64;
                    self.update_task(
                        &key,
                        vec![
                            ("status", json!(TASK_STATUS_ERROR)),
                            ("error", json!(message.clone())),
                            ("data", json!([])),
                            ("duration_ms", json!(duration_ms)),
                        ],
                    );
                    self.log_call(
                        &identity, &mode, &model, started, "调用失败", &prompt, "failed",
                        &message, &[], &account_email,
                    );
                } else {
                    let duration_ms = ((now_unix() - started) * 1000.0) as i64;
                    let urls = collect_image_urls(&data);
                    self.update_task(
                        &key,
                        vec![
                            ("status", json!(TASK_STATUS_SUCCESS)),
                            ("data", json!(data)),
                            ("error", json!("")),
                            ("duration_ms", json!(duration_ms)),
                        ],
                    );
                    self.log_call(
                        &identity, &mode, &model, started, "调用完成", &prompt, "success", "",
                        &urls, &account_email,
                    );
                }
            }
            Err(err) => {
                let message = if err.message.trim().is_empty() {
                    "image task failed".to_string()
                } else {
                    err.message.clone()
                };
                let account_email = err.account_email.trim().to_string();
                let conversation_id = err.conversation_id.trim().to_string();
                let duration_ms = ((now_unix() - started) * 1000.0) as i64;
                let mut updates = vec![
                    ("status", json!(TASK_STATUS_ERROR)),
                    ("error", json!(message.clone())),
                    ("data", json!([])),
                    ("duration_ms", json!(duration_ms)),
                ];
                if !conversation_id.is_empty() {
                    updates.push(("conversation_id", json!(conversation_id)));
                }
                self.update_task(&key, updates);
                self.log_call(
                    &identity, &mode, &model, started, "调用失败", &prompt, "failed", &message,
                    &[], &account_email,
                );
            }
        }
    }

    fn update_task(&self, key: &str, updates: Vec<(&str, Value)>) {
        let mut tasks = self.inner.tasks.lock();
        let mut found = false;
        if let Some(obj) = tasks.get_mut(key).and_then(|t| t.as_object_mut()) {
            for (k, v) in updates {
                obj.insert(k.to_string(), v);
            }
            obj.insert("updated_at".into(), json!(now_iso()));
            obj.insert("updated_ts".into(), json!(now_unix()));
            found = true;
        }
        if found {
            save_tasks(&self.inner.path, &tasks);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn log_call(
        &self,
        identity: &Value,
        mode: &str,
        model: &str,
        started: f64,
        suffix: &str,
        request_preview: &str,
        status: &str,
        error: &str,
        urls: &[String],
        account_email: &str,
    ) {
        let endpoint = if mode == "edit" {
            "/v1/images/edits"
        } else {
            "/v1/images/generations"
        };
        let summary_prefix = if mode == "edit" { "图生图" } else { "文生图" };
        let duration_ms = ((now_unix() - started) * 1000.0) as i64;
        let mut detail = Map::new();
        detail.insert("key_id".into(), identity.get("id").cloned().unwrap_or(Value::Null));
        detail.insert("key_name".into(), identity.get("name").cloned().unwrap_or(Value::Null));
        detail.insert("role".into(), identity.get("role").cloned().unwrap_or(Value::Null));
        detail.insert("endpoint".into(), json!(endpoint));
        detail.insert("model".into(), json!(model));
        detail.insert("started_at".into(), json!(fmt_unix(started)));
        detail.insert("ended_at".into(), json!(now_iso()));
        detail.insert("duration_ms".into(), json!(duration_ms));
        detail.insert("status".into(), json!(status));
        let preview = request_preview.trim();
        if !preview.is_empty() {
            detail.insert("request_text".into(), json!(preview));
        }
        if !error.is_empty() {
            detail.insert("error".into(), json!(error));
        }
        if !account_email.is_empty() {
            detail.insert("account_email".into(), json!(account_email));
        }
        if !urls.is_empty() {
            detail.insert("urls".into(), json!(dedupe(urls.to_vec())));
        }
        self.inner.log.add(
            LOG_TYPE_CALL,
            &format!("{summary_prefix}{suffix}"),
            Value::Object(detail),
        );
    }
}
