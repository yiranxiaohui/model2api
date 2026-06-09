//! Port of `services/protocol/chat_completion_cache.py` — TTL response/stream
//! cache for chat completions, with in-flight de-duplication.
//!
//! Async port: the Python `threading.Condition` waiters become
//! `tokio::sync::Notify`; compute closures are async. Cache values are
//! `serde_json::Value`; errors are carried as `String` so they can be shared
//! with in-flight waiters.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;

const CACHEABLE_TEXT_KEYS: &[&str] = &[
    "frequency_penalty",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "model",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "user",
];

struct CacheEntry {
    expires_at: f64,
    value: Value,
}

#[derive(Default)]
struct InflightState {
    done: bool,
    value: Option<Value>,
    error: Option<String>,
}

struct Inflight {
    notify: tokio::sync::Notify,
    state: Mutex<InflightState>,
}

fn now() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}

/// Canonicalized request body used for the cache key.
pub fn canonical_body(body: &Value, messages: &[Value], stream: bool) -> Value {
    let mut payload = Map::new();
    if let Some(obj) = body.as_object() {
        for key in CACHEABLE_TEXT_KEYS {
            if let Some(v) = obj.get(*key) {
                payload.insert((*key).to_string(), v.clone());
            }
        }
    }
    payload.insert("messages".into(), Value::Array(messages.to_vec()));
    payload.insert("stream".into(), Value::Bool(stream));
    Value::Object(payload)
}

/// Recursively replace byte-ish values; here all values are JSON already, so we
/// just produce a stable canonical JSON string with sorted keys.
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", serde_json::to_string(k).unwrap_or_default(), canonical_json(&m[k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(canonical_json).collect();
            format!("[{}]", parts.join(","))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

pub fn cache_key(body: &Value, messages: &[Value], stream: bool) -> String {
    let canonical = canonical_json(&canonical_body(body, messages, stream));
    hex::encode(Sha256::digest(canonical.as_bytes()))
}

fn message_signature(message: &Value) -> String {
    canonical_json(message)
}

/// Normalize chat messages per the cache settings (drop assistant history /
/// adjacent duplicates).
pub fn normalize_text_messages(config: &Config, messages: &[Value]) -> Vec<Value> {
    let settings = config.get_chat_completion_cache_settings();
    if !settings.get("normalize_messages").and_then(|v| v.as_bool()).unwrap_or(false) {
        return messages.to_vec();
    }
    let drop_assistant = settings.get("drop_assistant_history").and_then(|v| v.as_bool()).unwrap_or(false);
    let drop_dupes = settings.get("drop_adjacent_duplicates").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut normalized = Vec::new();
    let mut previous = String::new();
    for message in messages {
        if drop_assistant && message.get("role").and_then(|v| v.as_str()) == Some("assistant") {
            continue;
        }
        let sig = message_signature(message);
        if drop_dupes && sig == previous {
            continue;
        }
        normalized.push(message.clone());
        previous = sig;
    }
    normalized
}

pub struct ChatCompletionCache {
    config: Config,
    inner: Mutex<Inner>,
}

struct Inner {
    entries: HashMap<String, CacheEntry>,
    inflight: HashMap<String, Arc<Inflight>>,
}

impl ChatCompletionCache {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                inflight: HashMap::new(),
            }),
        }
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.inflight.clear();
    }

    fn settings(&self) -> Value {
        self.config.get_chat_completion_cache_settings()
    }

    fn prune_locked(inner: &mut Inner, now: f64, max_entries: usize) {
        inner.entries.retain(|_, e| e.expires_at > now);
        while inner.entries.len() > max_entries {
            if let Some(oldest) = inner
                .entries
                .iter()
                .min_by(|a, b| a.1.expires_at.partial_cmp(&b.1.expires_at).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.clone())
            {
                inner.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }

    /// Cache a single response, de-duplicating concurrent identical requests.
    pub async fn get_or_compute_response<F, Fut>(&self, key: &str, compute: F) -> Result<Value, String>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Value, String>>,
    {
        let settings = self.settings();
        let enabled = settings.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
        let ttl = settings.get("ttl_seconds").and_then(|v| v.as_i64()).unwrap_or(0);
        if !enabled || ttl <= 0 {
            return compute().await;
        }
        let max_entries = settings.get("max_entries").and_then(|v| v.as_i64()).unwrap_or(1).max(1) as usize;
        let dedupe = settings.get("dedupe_inflight").and_then(|v| v.as_bool()).unwrap_or(false);

        let (owner, inflight) = {
            let mut inner = self.inner.lock();
            Self::prune_locked(&mut inner, now(), max_entries);
            if let Some(entry) = inner.entries.get(key) {
                if entry.expires_at > now() {
                    return Ok(entry.value.clone());
                }
            }
            if dedupe {
                if let Some(existing) = inner.inflight.get(key) {
                    (false, Arc::clone(existing))
                } else {
                    let inflight = Arc::new(Inflight {
                        notify: tokio::sync::Notify::new(),
                        state: Mutex::new(InflightState::default()),
                    });
                    inner.inflight.insert(key.to_string(), Arc::clone(&inflight));
                    (true, inflight)
                }
            } else {
                (
                    true,
                    Arc::new(Inflight {
                        notify: tokio::sync::Notify::new(),
                        state: Mutex::new(InflightState::default()),
                    }),
                )
            }
        };

        if !owner {
            loop {
                {
                    let state = inflight.state.lock();
                    if state.done {
                        if let Some(err) = &state.error {
                            return Err(err.clone());
                        }
                        return Ok(state.value.clone().unwrap_or(Value::Null));
                    }
                }
                inflight.notify.notified().await;
            }
        }

        let result = compute().await;
        match &result {
            Ok(value) => {
                let mut inner = self.inner.lock();
                inner.entries.insert(
                    key.to_string(),
                    CacheEntry {
                        expires_at: now() + ttl as f64,
                        value: value.clone(),
                    },
                );
                Self::prune_locked(&mut inner, now(), max_entries);
                inner.inflight.remove(key);
                drop(inner);
                {
                    let mut state = inflight.state.lock();
                    state.value = Some(value.clone());
                    state.done = true;
                }
                inflight.notify.notify_waiters();
            }
            Err(err) => {
                self.inner.lock().inflight.remove(key);
                {
                    let mut state = inflight.state.lock();
                    state.error = Some(err.clone());
                    state.done = true;
                }
                inflight.notify.notify_waiters();
            }
        }
        result
    }

    /// Cache a streamed response (as a vector of chunks). De-dups concurrent
    /// identical requests; returns the full chunk list.
    pub async fn get_or_compute_stream<F, Fut>(&self, key: &str, compute: F) -> Result<Vec<Value>, String>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Vec<Value>, String>>,
    {
        let settings = self.settings();
        let enabled = settings.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
        let stream_cache = settings.get("stream_cache").and_then(|v| v.as_bool()).unwrap_or(false);
        let ttl = settings.get("ttl_seconds").and_then(|v| v.as_i64()).unwrap_or(0);
        if !enabled || !stream_cache || ttl <= 0 {
            return compute().await;
        }
        // Delegate to the response path by packing the chunk list into a Value.
        let packed = self
            .get_or_compute_response(key, || async {
                compute().await.map(Value::Array)
            })
            .await?;
        Ok(packed.as_array().cloned().unwrap_or_default())
    }
}

/// Build a `{"__bytes_sha256__":..,"length":..}` descriptor (parity helper for
/// hashing binary parts that may appear in cache bodies).
pub fn bytes_descriptor(data: &[u8]) -> Value {
    json!({
        "__bytes_sha256__": hex::encode(Sha256::digest(data)),
        "length": data.len(),
    })
}
