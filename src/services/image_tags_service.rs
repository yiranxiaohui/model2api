//! Port of `services/image_tags_service.py` — per-image tag storage backed by a
//! single JSON file (`image_tags.json`) under the data dir.
//!
//! The Python module exposed free functions that re-read/re-write the file on
//! every call. Here they become snake_case methods on a cheaply cloneable
//! `ImageTagsService` (inner `Arc`, like `log_service.rs`). A `parking_lot::Mutex`
//! guards the in-memory cache and serializes file access; tag data is carried as
//! `serde_json::Value` arrays to mirror the dynamic `dict[str, list[str]]` shape.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::config::Config;

/// Cloneable handle to the image-tags JSON store.
#[derive(Clone)]
pub struct ImageTagsService {
    inner: Arc<Inner>,
}

struct Inner {
    path: PathBuf,
    /// In-memory cache of the tag map, guarded by the same lock that serializes
    /// file access.
    cache: Mutex<Map<String, Value>>,
}

/// Ensure the file's parent dir exists and the file itself exists (seeded with
/// `{}`). Mirrors `_ensure_file`.
fn ensure_file(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if !path.exists() {
        let _ = std::fs::write(path, "{}");
    }
}

/// Read the tag map from disk. Mirrors `load_tags` — non-object / unreadable
/// content collapses to an empty map.
fn read_file(path: &Path) -> Map<String, Value> {
    ensure_file(path);
    match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        },
        Err(_) => Map::new(),
    }
}

/// Write the tag map to disk. Mirrors `save_tags` (pretty, non-ASCII-escaped,
/// trailing newline).
fn write_file(path: &Path, data: &Map<String, Value>) {
    ensure_file(path);
    if let Ok(text) = serde_json::to_string_pretty(&Value::Object(data.clone())) {
        let _ = std::fs::write(path, text + "\n");
    }
}

/// Extract the string tags from a value (a JSON array of strings).
fn value_to_tags(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

impl ImageTagsService {
    /// Build a service rooted at `<data_dir>/image_tags.json`, priming the cache
    /// from disk.
    pub fn new(config: Config) -> Self {
        let path = config.data_dir().join("image_tags.json");
        let cache = read_file(&path);
        Self {
            inner: Arc::new(Inner {
                path,
                cache: Mutex::new(cache),
            }),
        }
    }

    /// Load the full tag map (`image_rel -> [tags]`). Mirrors `load_tags`.
    pub fn load_tags(&self) -> Map<String, Value> {
        let mut guard = self.inner.cache.lock();
        *guard = read_file(&self.inner.path);
        guard.clone()
    }

    /// Persist the full tag map. Mirrors `save_tags`.
    pub fn save_tags(&self, data: Map<String, Value>) {
        let mut guard = self.inner.cache.lock();
        write_file(&self.inner.path, &data);
        *guard = data;
    }

    /// Tags for a single image (empty if none). Mirrors `get_tags`.
    pub fn get_tags(&self, image_rel: &str) -> Vec<String> {
        let mut guard = self.inner.cache.lock();
        *guard = read_file(&self.inner.path);
        value_to_tags(guard.get(image_rel))
    }

    /// Replace the tags for an image, de-duplicating (order-preserving) and
    /// dropping blank entries. Removes the key entirely when the cleaned set is
    /// empty. Returns the cleaned tags. Mirrors `set_tags`.
    pub fn set_tags(&self, image_rel: &str, tags: &[String]) -> Vec<String> {
        let mut cleaned: Vec<String> = Vec::new();
        for t in tags {
            let trimmed = t.trim();
            if !trimmed.is_empty() && !cleaned.iter().any(|c| c == trimmed) {
                cleaned.push(trimmed.to_string());
            }
        }

        let mut guard = self.inner.cache.lock();
        let mut data = read_file(&self.inner.path);
        if cleaned.is_empty() {
            data.remove(image_rel);
        } else {
            data.insert(
                image_rel.to_string(),
                Value::Array(cleaned.iter().cloned().map(Value::String).collect()),
            );
        }
        write_file(&self.inner.path, &data);
        *guard = data;
        cleaned
    }

    /// Remove all tags for an image (only rewrites the file if the key existed).
    /// Mirrors `remove_tags`.
    pub fn remove_tags(&self, image_rel: &str) {
        let mut guard = self.inner.cache.lock();
        let mut data = read_file(&self.inner.path);
        if data.remove(image_rel).is_some() {
            write_file(&self.inner.path, &data);
        }
        *guard = data;
    }

    /// Delete a tag from every image; drops images left tagless. Returns the
    /// number of affected images. Mirrors `delete_tag`.
    pub fn delete_tag(&self, tag: &str) -> usize {
        let mut guard = self.inner.cache.lock();
        let mut data = read_file(&self.inner.path);
        let mut count = 0usize;
        let keys: Vec<String> = data.keys().cloned().collect();
        for rel in keys {
            let current = value_to_tags(data.get(&rel));
            if current.iter().any(|t| t == tag) {
                let remaining: Vec<String> =
                    current.into_iter().filter(|t| t != tag).collect();
                if remaining.is_empty() {
                    data.remove(&rel);
                } else {
                    data.insert(
                        rel,
                        Value::Array(remaining.into_iter().map(Value::String).collect()),
                    );
                }
                count += 1;
            }
        }
        if count > 0 {
            write_file(&self.inner.path, &data);
        }
        *guard = data;
        count
    }

    /// All distinct tags across every image, in first-seen order. Mirrors
    /// `get_all_tags`.
    pub fn get_all_tags(&self) -> Vec<String> {
        let mut guard = self.inner.cache.lock();
        *guard = read_file(&self.inner.path);
        let mut result: Vec<String> = Vec::new();
        for value in guard.values() {
            for t in value_to_tags(Some(value)) {
                if !result.iter().any(|r| r == &t) {
                    result.push(t);
                }
            }
        }
        result
    }
}
