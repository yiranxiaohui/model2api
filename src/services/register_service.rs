//! Port of `services/register_service.py` — the registration *orchestrator* that
//! drives many [`openai_register::register_one`] calls toward a target, persists
//! each successful account, and exposes live progress + logs to the web UI.
//!
//! The Python original kept its state in `data/register.json`, ran the worker
//! pool on a `ThreadPoolExecutor` inside a daemon `threading.Thread`, and
//! reported progress via a SSE-polled `get()` snapshot. This port mirrors that:
//!
//! * State (the normalized config dict + an in-memory ring of the last 300 log
//!   lines + a `running` guard) lives behind a `parking_lot::Mutex` inside an
//!   `Arc`, so [`RegisterService`] is cheaply cloneable and the spawned
//!   `'static` runner can drive both the pool and the [`AccountService`].
//! * `start()` spawns a `tokio` task ([`RegisterService::run`]) that fills a
//!   `futures::stream::FuturesUnordered` up to `threads` in flight (the analog
//!   of `wait(FIRST_COMPLETED)`), re-evaluating the stop condition each time a
//!   registration completes. `stop()` flips the cooperative `enabled` flag.
//! * Three run *modes* gate submission: `total` (register exactly `total`
//!   accounts then finish), `quota` (keep the pool's normal-account quota at or
//!   above `target_quota`), and `available` (keep the count of normal accounts
//!   at or above `target_available`). The quota/available modes never finish on
//!   their own — they idle for `check_interval` seconds and re-poll, refilling
//!   as accounts get consumed, until `stop()` is called.
//!
//! Deviations from the Python original:
//! * The requirement suggested `buffer_unordered`; that combinator drives a
//!   *fixed* input stream and cannot express the dynamic stop-condition / idle
//!   re-poll loop of the quota/available modes, so a manually-driven
//!   `FuturesUnordered` (also from `futures`) is used instead.
//! * The registrar client prefers the register config's own `proxy` field
//!   (Python's `PlatformRegistrar(config["proxy"])`), falling back to the global
//!   [`Config::proxy_setting`] only when it is empty. The register proxy is also
//!   injected into the `mail` section (`inject_proxy_to_mail`).
//! * Python's module-global `openai_register.stats` / `stats_lock` sync is
//!   dropped — the per-run counters live entirely in this service's `stats`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::config::Config;
use crate::services::account_service::AccountService;
use crate::services::register::openai_register;

const LOG_LIMIT: usize = 300;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// UTC timestamp in RFC3339 (matches Python's `datetime.now(timezone.utc).isoformat()`).
fn now() -> String {
    Utc::now().to_rfc3339()
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Port of Python `int(value or fallback)`: parse to int, but treat a falsy
/// value (missing / null / empty string / 0) as `fallback`.
fn falsy_int(value: Option<&Value>, fallback: i64) -> i64 {
    let parsed = match value {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<i64>().ok()
            }
        }
        _ => None,
    };
    match parsed {
        None | Some(0) => fallback,
        Some(x) => x,
    }
}

/// Shallow `{**dst, **src}` for JSON objects.
fn merge_shallow(dst: &mut Value, src: &Value) {
    if let (Some(d), Some(s)) = (dst.as_object_mut(), src.as_object()) {
        for (k, v) in s {
            d.insert(k.clone(), v.clone());
        }
    }
}

/// Port of `_default_config()["stats"]`.
fn default_stats(threads: i64) -> Value {
    json!({
        "success": 0,
        "fail": 0,
        "done": 0,
        "running": 0,
        "threads": threads,
        "elapsed_seconds": 0,
        "avg_seconds": 0,
        "success_rate": 0,
        "current_quota": 0,
        "current_available": 0,
    })
}

/// Port of `_default_config()` (merges `openai_register.config`'s defaults).
fn default_config() -> Value {
    json!({
        "mail": {
            "request_timeout": 30,
            "wait_timeout": 30,
            "wait_interval": 2,
            "providers": [],
        },
        "proxy": "",
        "flaresolverr": {
            "enabled": false,
            "url": "",
            "max_timeout": 60000,
        },
        "total": 10,
        "threads": 3,
        "mode": "total",
        "target_quota": 100,
        "target_available": 10,
        "check_interval": 5,
        "enabled": false,
        "stats": default_stats(3),
    })
}

/// Port of `_normalize` — start from defaults, overlay the persisted config
/// (minus `stats`/`logs`), then clamp/coerce every field.
fn normalize(raw: &Value) -> Value {
    let mut cfg = default_config();
    if let (Some(c), Some(obj)) = (cfg.as_object_mut(), raw.as_object()) {
        for (k, v) in obj {
            if k == "stats" || k == "logs" {
                continue;
            }
            c.insert(k.clone(), v.clone());
        }
    }

    let total = falsy_int(cfg.get("total"), 1).max(1);
    let threads = falsy_int(cfg.get("threads"), 1).max(1);
    let target_quota = falsy_int(cfg.get("target_quota"), 1).max(1);
    let target_available = falsy_int(cfg.get("target_available"), 1).max(1);
    let check_interval = falsy_int(cfg.get("check_interval"), 5).max(1);

    let mode = {
        let m = cfg
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("total")
            .trim()
            .to_string();
        if matches!(m.as_str(), "total" | "quota" | "available") {
            m
        } else {
            "total".to_string()
        }
    };

    let proxy = match cfg.get("proxy") {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };

    let enabled = match cfg.get("enabled") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    };

    // Re-derive stats from the persisted stats (if any), pinning `threads`.
    let mut stats = default_stats(threads);
    if let (Some(s), Some(Value::Object(raw_stats))) =
        (stats.as_object_mut(), raw.get("stats"))
    {
        for (k, v) in raw_stats {
            s.insert(k.clone(), v.clone());
        }
    }
    stats["threads"] = json!(threads);

    if let Some(c) = cfg.as_object_mut() {
        c.insert("total".into(), json!(total));
        c.insert("threads".into(), json!(threads));
        c.insert("mode".into(), json!(mode));
        c.insert("target_quota".into(), json!(target_quota));
        c.insert("target_available".into(), json!(target_available));
        c.insert("check_interval".into(), json!(check_interval));
        c.insert("proxy".into(), json!(proxy));
        c.insert("enabled".into(), json!(enabled));
        c.insert("stats".into(), stats);
    }
    cfg
}

// ---------------------------------------------------------------------------
// service
// ---------------------------------------------------------------------------

struct Inner {
    store_file: PathBuf,
    /// The normalized register config (also holds `stats`).
    config: Value,
    /// In-memory ring of the last [`LOG_LIMIT`] log lines.
    logs: Vec<Value>,
    /// Whether a runner task is currently active (guards double-start).
    running: bool,
}

/// Registration orchestrator (port of `RegisterService`). Cheaply cloneable.
#[derive(Clone)]
pub struct RegisterService {
    config: Config,
    accounts: Arc<AccountService>,
    inner: Arc<Mutex<Inner>>,
}

impl RegisterService {
    /// Build the service, loading `data/register.json`. If the persisted config
    /// has `enabled = true` and a tokio runtime is active, the run resumes
    /// automatically (port of the `__init__` auto-start).
    pub fn new(config: Config, accounts: Arc<AccountService>) -> Self {
        let store_file = config.data_dir().join("register.json");
        let cfg = Self::load(&store_file);
        let svc = Self {
            config,
            accounts,
            inner: Arc::new(Mutex::new(Inner {
                store_file,
                config: cfg,
                logs: Vec::new(),
                running: false,
            })),
        };
        let enabled = svc
            .inner
            .lock()
            .config
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if enabled {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let this = svc.clone();
                handle.spawn(async move {
                    this.start().await;
                });
            }
        }
        svc
    }

    // ---- persistence ----

    fn load(store_file: &Path) -> Value {
        let raw = std::fs::read_to_string(store_file)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .unwrap_or_else(|| json!({}));
        normalize(&raw)
    }

    fn save_locked(inner: &Inner) {
        if let Some(parent) = inner.store_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(&inner.config) {
            let _ = std::fs::write(&inner.store_file, text + "\n");
        }
    }

    // ---- snapshot / logs ----

    /// Port of `get()` — config plus the last [`LOG_LIMIT`] log lines.
    fn snapshot(inner: &Inner) -> Value {
        let mut out = inner.config.clone();
        let len = inner.logs.len();
        let start = len.saturating_sub(LOG_LIMIT);
        out["logs"] = Value::Array(inner.logs[start..].to_vec());
        out
    }

    pub fn get(&self) -> Value {
        Self::snapshot(&self.inner.lock())
    }

    fn append_log(&self, text: &str, color: &str) {
        let mut inner = self.inner.lock();
        inner.logs.push(json!({
            "time": now(),
            "text": text,
            "level": if color.is_empty() { "info" } else { color },
        }));
        let len = inner.logs.len();
        if len > LOG_LIMIT {
            inner.logs.drain(0..len - LOG_LIMIT);
        }
    }

    // ---- config mutation ----

    fn inject_proxy_to_mail(config: &mut Value) {
        let proxy = config
            .get("proxy")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if proxy.is_empty() {
            return;
        }
        if let Some(mail) = config.get_mut("mail") {
            if mail.is_object() {
                mail["proxy"] = Value::String(proxy);
            }
        }
    }

    /// Port of `update()` — merge `updates`, re-normalize, inject proxy, save.
    pub fn update(&self, updates: Value) -> Value {
        let mut inner = self.inner.lock();
        let mut merged = inner.config.clone();
        merge_shallow(&mut merged, &updates);
        inner.config = normalize(&merged);
        Self::inject_proxy_to_mail(&mut inner.config);
        Self::save_locked(&inner);
        Self::snapshot(&inner)
    }

    // ---- pool metrics ----

    /// Port of `_pool_metrics` — normal-account count and summed quota.
    fn pool_metrics(&self) -> Value {
        let items = self.accounts.list_accounts();
        let mut current_quota: i64 = 0;
        let mut current_available: i64 = 0;
        for item in &items {
            if item.get("status").and_then(|v| v.as_str()) == Some("正常") {
                current_available += 1;
                if !item
                    .get("image_quota_unknown")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    current_quota += item.get("quota").and_then(|v| v.as_i64()).unwrap_or(0);
                }
            }
        }
        json!({ "current_quota": current_quota, "current_available": current_available })
    }

    /// Port of `_bump` — merge `updates` into `stats`, recompute the derived
    /// timing metrics from `started_at`, stamp `updated_at`, and persist.
    fn bump(&self, updates: Value) {
        let mut inner = self.inner.lock();
        if !inner.config.get("stats").map(|v| v.is_object()).unwrap_or(false) {
            inner.config["stats"] = json!({});
        }
        let stats = inner.config["stats"].as_object_mut().unwrap();
        if let Some(u) = updates.as_object() {
            for (k, v) in u {
                stats.insert(k.clone(), v.clone());
            }
        }
        let started_at = stats
            .get("started_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !started_at.is_empty() {
            if let Ok(start) = DateTime::parse_from_rfc3339(&started_at) {
                let elapsed = ((Utc::now() - start.with_timezone(&Utc)).num_milliseconds()
                    as f64
                    / 1000.0)
                    .max(0.0);
                let success = stats.get("success").and_then(|v| v.as_i64()).unwrap_or(0);
                let fail = stats.get("fail").and_then(|v| v.as_i64()).unwrap_or(0);
                stats.insert("elapsed_seconds".into(), json!(round1(elapsed)));
                stats.insert(
                    "avg_seconds".into(),
                    if success > 0 {
                        json!(round1(elapsed / success as f64))
                    } else {
                        json!(0)
                    },
                );
                stats.insert(
                    "success_rate".into(),
                    json!(round1(success as f64 * 100.0 / (success + fail).max(1) as f64)),
                );
            }
        }
        stats.insert("updated_at".into(), json!(now()));
        Self::save_locked(&inner);
    }

    // ---- lifecycle ----

    /// Port of `start()` — (re)enable the run, reset stats/logs, spawn the runner.
    pub async fn start(&self) -> Value {
        let (mode, threads) = {
            let mut inner = self.inner.lock();
            if inner.running {
                inner.config["enabled"] = json!(true);
                Self::save_locked(&inner);
                return Self::snapshot(&inner);
            }
            inner.running = true;
            inner.config["enabled"] = json!(true);
            Self::inject_proxy_to_mail(&mut inner.config);
            inner.logs.clear();
            let threads = inner
                .config
                .get("threads")
                .and_then(|v| v.as_i64())
                .unwrap_or(1)
                .max(1);
            let metrics = self.pool_metrics();
            inner.config["stats"] = json!({
                "job_id": uuid::Uuid::new_v4().simple().to_string(),
                "success": 0,
                "fail": 0,
                "done": 0,
                "running": 0,
                "threads": threads,
                "current_quota": metrics.get("current_quota").cloned().unwrap_or(json!(0)),
                "current_available": metrics.get("current_available").cloned().unwrap_or(json!(0)),
                "started_at": now(),
                "updated_at": now(),
            });
            Self::save_locked(&inner);
            let mode = inner
                .config
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("total")
                .to_string();
            (mode, threads)
        };
        self.append_log(
            &format!("注册任务启动，模式={mode}，线程数={threads}"),
            "yellow",
        );
        let this = self.clone();
        tokio::spawn(async move {
            this.run().await;
        });
        self.get()
    }

    /// Port of `stop()` — flip the cooperative `enabled` flag.
    pub fn stop(&self) -> Value {
        {
            let mut inner = self.inner.lock();
            inner.config["enabled"] = json!(false);
            inner.config["stats"]["updated_at"] = json!(now());
            Self::save_locked(&inner);
        }
        self.append_log("已请求停止注册任务，正在等待当前运行任务结束", "yellow");
        self.get()
    }

    /// Port of `reset()` — clear logs and stats (only meaningful when idle).
    pub fn reset(&self) -> Value {
        {
            let mut inner = self.inner.lock();
            inner.logs.clear();
            let threads = inner
                .config
                .get("threads")
                .and_then(|v| v.as_i64())
                .unwrap_or(1)
                .max(1);
            let metrics = self.pool_metrics();
            inner.config["stats"] = json!({
                "success": 0,
                "fail": 0,
                "done": 0,
                "running": 0,
                "threads": threads,
                "elapsed_seconds": 0,
                "avg_seconds": 0,
                "success_rate": 0,
                "current_quota": metrics.get("current_quota").cloned().unwrap_or(json!(0)),
                "current_available": metrics.get("current_available").cloned().unwrap_or(json!(0)),
                "updated_at": now(),
            });
            Self::save_locked(&inner);
        }
        self.get()
    }

    fn is_enabled(&self) -> bool {
        self.inner
            .lock()
            .config
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn config_snapshot(&self) -> Value {
        self.inner.lock().config.clone()
    }

    // ---- target / stop conditions ----

    /// Port of `_target_reached` — also bumps the live pool metrics and (for the
    /// quota/available modes) logs a per-check decision line.
    fn target_reached(&self, cfg: &Value, submitted: i64) -> bool {
        let mode = cfg.get("mode").and_then(|v| v.as_str()).unwrap_or("total");
        let metrics = self.pool_metrics();
        self.bump(metrics.clone());
        let current_quota = metrics.get("current_quota").and_then(|v| v.as_i64()).unwrap_or(0);
        let current_available = metrics
            .get("current_available")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        match mode {
            "quota" => {
                let target = cfg.get("target_quota").and_then(|v| v.as_i64()).unwrap_or(1);
                let reached = current_quota >= target;
                self.append_log(
                    &format!(
                        "检查号池：当前正常账号={current_available}，当前剩余额度={current_quota}，目标额度={target}，{}",
                        if reached { "跳过注册" } else { "继续注册" }
                    ),
                    "yellow",
                );
                reached
            }
            "available" => {
                let target = cfg
                    .get("target_available")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(1);
                let reached = current_available >= target;
                self.append_log(
                    &format!(
                        "检查号池：当前正常账号={current_available}，目标账号={target}，当前剩余额度={current_quota}，{}",
                        if reached { "跳过注册" } else { "继续注册" }
                    ),
                    "yellow",
                );
                reached
            }
            _ => submitted >= cfg.get("total").and_then(|v| v.as_i64()).unwrap_or(1),
        }
    }

    // ---- worker ----

    /// Port of the success branch of `worker`: register one account and, on
    /// success, persist it to the pool and refresh its status. Returns the
    /// `register_one` envelope so the caller can tally success/fail.
    async fn register_and_persist(
        &self,
        index: i64,
        mail_config: Value,
        proxy: String,
        flaresolverr_config: Value,
    ) -> Value {
        let result =
            openai_register::register_one(&self.config, &mail_config, index, &proxy, &flaresolverr_config)
                .await;
        if result.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            let access_token = result
                .get("access_token")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            // Persist the account item (drop the orchestration-only fields).
            let mut item = result.clone();
            if let Some(o) = item.as_object_mut() {
                o.remove("ok");
                o.remove("index");
            }
            self.accounts.add_account_items(&[item]);
            if !access_token.is_empty() {
                let refresh = self
                    .accounts
                    .refresh_accounts(&[access_token], None, true)
                    .await;
                let has_errors = refresh
                    .get("errors")
                    .and_then(|v| v.as_array())
                    .map(|a| !a.is_empty())
                    .unwrap_or(false);
                if has_errors {
                    self.append_log(
                        &format!(
                            "[任务{index}] 账号已保存，刷新状态暂未成功，稍后可重试: {}",
                            refresh.get("errors").cloned().unwrap_or(Value::Null)
                        ),
                        "yellow",
                    );
                }
            }
        }
        result
    }

    /// Port of `_run` — the producer/consumer loop. Fills up to `threads`
    /// registrations in flight, awaiting one at a time (the analog of
    /// `wait(FIRST_COMPLETED)`), tallying results and re-evaluating the stop
    /// condition after each completion.
    async fn run(self) {
        use futures::stream::{FuturesUnordered, StreamExt};

        let threads = self
            .config_snapshot()
            .get("threads")
            .and_then(|v| v.as_i64())
            .unwrap_or(1)
            .max(1) as usize;

        let mut submitted: i64 = 0;
        let mut done: i64 = 0;
        let mut success: i64 = 0;
        let mut fail: i64 = 0;
        let mut futs = FuturesUnordered::new();

        loop {
            let cfg = self.config_snapshot();
            while self.is_enabled()
                && !self.target_reached(&cfg, submitted)
                && futs.len() < threads
            {
                submitted += 1;
                let idx = submitted;
                let this = self.clone();
                let mail_config = cfg.get("mail").cloned().unwrap_or_else(|| json!({}));
                let flaresolverr_config = cfg.get("flaresolverr").cloned().unwrap_or_else(|| json!({}));
                // Prefer the register config's own `proxy` (Python parity); the
                // registrar falls back to the global config proxy when empty.
                let reg_proxy = cfg
                    .get("proxy")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                futs.push(async move {
                    let t0 = std::time::Instant::now();
                    let r = this
                        .register_and_persist(idx, mail_config, reg_proxy, flaresolverr_config)
                        .await;
                    (r, t0.elapsed().as_secs_f64())
                });
            }

            self.bump(json!({
                "running": futs.len(),
                "done": done,
                "success": success,
                "fail": fail,
            }));

            let mode = cfg.get("mode").and_then(|v| v.as_str()).unwrap_or("total");
            if futs.is_empty() && (!self.is_enabled() || mode == "total") {
                break;
            }
            if futs.is_empty() {
                let interval = cfg
                    .get("check_interval")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(5)
                    .max(1);
                tokio::time::sleep(Duration::from_secs(interval as u64)).await;
                continue;
            }

            if let Some((result, cost)) = futs.next().await {
                done += 1;
                let idx = result.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                if result.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                    success += 1;
                    // Recompute avg_seconds (elapsed / success) before reporting it.
                    self.bump(json!({"done": done, "success": success, "fail": fail}));
                    let avg = self
                        .config_snapshot()
                        .get("stats")
                        .and_then(|s| s.get("avg_seconds"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let email = result.get("email").and_then(|v| v.as_str()).unwrap_or("");
                    self.append_log(
                        &format!(
                            "[任务{idx}] {email} 注册成功，本次耗时{cost:.1}s，全局平均每个号{avg:.1}s"
                        ),
                        "green",
                    );
                } else {
                    fail += 1;
                    let reason = result
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("未知错误");
                    self.append_log(
                        &format!("[任务{idx}] 注册失败，本次耗时{cost:.1}s，原因: {reason}"),
                        "red",
                    );
                }
            }
        }

        self.bump(json!({
            "running": 0,
            "done": done,
            "success": success,
            "fail": fail,
            "finished_at": now(),
        }));
        {
            let mut inner = self.inner.lock();
            inner.config["enabled"] = json!(false);
            inner.running = false;
            Self::save_locked(&inner);
        }
        self.append_log(&format!("注册任务结束，成功{success}，失败{fail}"), "yellow");
    }
}
