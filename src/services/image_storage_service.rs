//! Port of `services/image_storage_service.py` — local-disk + WebDAV image
//! storage with a JSON index file (`image_index.json`) guarded by a lock.
//!
//! Images are written either to the local `images/` directory under the data
//! dir, to a remote WebDAV server, or to both ("local" / "webdav" / "both"
//! modes, taken from `config.get_image_storage_settings()`). Every stored image
//! is recorded in `image_index.json` so listing/cleanup can reconcile the
//! on-disk tree with the remote store.
//!
//! Differences from the Python original are intentional and noted inline:
//!   * filenames are `sha256`-based with the extension derived from the image's
//!     magic bytes (header-only, via the `imagesize` crate) rather than the
//!     Python `md5` + fixed `.png`;
//!   * image dimensions come from `imagesize::blob_size` (header-only) instead
//!     of Pillow;
//!   * the WebDAV transport uses the async `wreq` client instead of curl_cffi,
//!     so the I/O methods are `async` and the parking_lot index lock is never
//!     held across an `.await`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::error::AppError;

const IMAGE_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "webp"];

/// Public, cheaply-cloneable result of a `save()` — what the caller embeds in
/// API responses.
#[derive(Clone, Debug)]
pub struct StoredImage {
    pub rel: String,
    pub url: String,
    pub storage: String,
    pub size: usize,
}

/// Cloneable handle to the image storage backend.
#[derive(Clone)]
pub struct ImageStorageService {
    inner: Arc<Inner>,
}

struct Inner {
    config: Config,
    index_file: PathBuf,
    index_lock: Mutex<()>,
}

// ---------------------------------------------------------------------------
// free helpers (mirror the module-level helpers in the Python file)
// ---------------------------------------------------------------------------

fn now_iso() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn setting_str(settings: &Value, key: &str) -> String {
    settings
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

/// `_safe_relative_path` — reject empty / traversal paths, normalise to posix.
fn safe_relative_path(path: &str) -> Result<String, AppError> {
    let value = path.trim().replace('\\', "/");
    let value = value.trim_start_matches('/');
    if value.is_empty() {
        return Err(AppError::not_found("image not found"));
    }
    let parts: Vec<&str> = value.split('/').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return Err(AppError::not_found("image not found"));
    }
    if parts.iter().any(|p| *p == "." || *p == "..") {
        return Err(AppError::not_found("image not found"));
    }
    Ok(parts.join("/"))
}

/// Lower-cased file suffix (without the dot) of a posix path/name, or "".
fn suffix_lower(name: &str) -> String {
    let file = name.rsplit('/').next().unwrap_or(name);
    match file.rsplit_once('.') {
        Some((_, ext)) if !ext.is_empty() => ext.to_ascii_lowercase(),
        _ => String::new(),
    }
}

fn is_image_rel(path: &str) -> bool {
    match safe_relative_path(path) {
        Ok(safe) => IMAGE_EXTENSIONS.contains(&suffix_lower(&safe).as_str()),
        Err(_) => false,
    }
}

/// Extension chosen from the image's magic bytes; constrained to the allowed
/// extension set (falls back to `png`).
fn ext_from_type(data: &[u8]) -> &'static str {
    match imagesize::image_type(data) {
        Ok(imagesize::ImageType::Png) => "png",
        Ok(imagesize::ImageType::Jpeg) => "jpg",
        Ok(imagesize::ImageType::Webp) => "webp",
        _ => "png",
    }
}

fn read_json_object(path: &Path) -> Map<String, Value> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Map::new(),
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

fn write_json_object(path: &Path, data: &Value) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(".tmp");
        PathBuf::from(s)
    };
    let body = serde_json::to_string_pretty(data).unwrap_or_else(|_| "{}".to_string()) + "\n";
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Recursively collect every regular file under `root`.
fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

fn rel_posix(root: &Path, path: &Path) -> Option<String> {
    let r = path.strip_prefix(root).ok()?;
    Some(
        r.components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

fn mtime(path: &Path) -> Option<DateTime<Local>> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .map(DateTime::<Local>::from)
}

fn dims_into(item: &mut Map<String, Value>, data: &[u8]) {
    if let Ok(sz) = imagesize::blob_size(data) {
        item.insert("width".into(), json!(sz.width));
        item.insert("height".into(), json!(sz.height));
    }
}

fn b64_basic_auth(user: &str, pass: &str) -> Option<String> {
    if user.is_empty() && pass.is_empty() {
        return None;
    }
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    Some(format!("Basic {}", STANDARD.encode(format!("{user}:{pass}"))))
}

/// Percent-encode a single path segment (Python `quote(part, safe="")`).
fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// WebDAV transport (async port of WebDAVClient)
// ---------------------------------------------------------------------------

struct WebDavClient {
    url: String,
    username: String,
    password: String,
    root_path: String,
    client: wreq::Client,
}

impl WebDavClient {
    fn new(settings: &Value) -> Result<Self, AppError> {
        let client = wreq::Client::builder()
            .build()
            .map_err(|e| AppError::upstream(format!("WebDAV client build: {e}")))?;
        Ok(Self {
            url: setting_str(settings, "webdav_url").trim_end_matches('/').to_string(),
            username: setting_str(settings, "webdav_username"),
            password: setting_str(settings, "webdav_password"),
            root_path: setting_str(settings, "webdav_root_path").trim_matches('/').to_string(),
            client,
        })
    }

    async fn do_request(
        &self,
        method: &str,
        url: &str,
        body: Option<Vec<u8>>,
        content_type: Option<&str>,
    ) -> Result<wreq::Response, AppError> {
        let m = wreq::Method::from_bytes(method.as_bytes())
            .map_err(|e| AppError::upstream(format!("WebDAV bad method {method}: {e}")))?;
        let mut rb = self.client.request(m, url).timeout(Duration::from_secs(30));
        if let Some(auth) = b64_basic_auth(&self.username, &self.password) {
            rb = rb.header("Authorization", auth);
        }
        if let Some(ct) = content_type {
            rb = rb.header("Content-Type", ct);
        }
        if let Some(b) = body {
            rb = rb.body(b);
        }
        rb.send()
            .await
            .map_err(|e| AppError::upstream(format!("WebDAV {method} error: {e}")))
    }

    /// `_request` — raise on HTTP >= 400 (MKCOL 405 tolerated).
    async fn request_checked(
        &self,
        method: &str,
        url: &str,
        body: Option<Vec<u8>>,
        content_type: Option<&str>,
    ) -> Result<wreq::Response, AppError> {
        let resp = self.do_request(method, url, body, content_type).await?;
        let status = resp.status().as_u16();
        if status >= 400 && !(method == "MKCOL" && status == 405) {
            return Err(AppError::upstream(format!(
                "WebDAV {method} failed: HTTP {status}"
            )));
        }
        Ok(resp)
    }

    fn remote_url(&self, rel: &str) -> Result<String, AppError> {
        let mut parts: Vec<String> = Vec::new();
        if !self.root_path.is_empty() {
            parts.push(self.root_path.clone());
        }
        if !rel.is_empty() {
            parts.push(safe_relative_path(rel)?);
        }
        let encoded: Vec<String> = parts
            .iter()
            .flat_map(|item| item.split('/'))
            .filter(|p| !p.is_empty())
            .map(encode_segment)
            .collect();
        if encoded.is_empty() {
            Ok(self.url.clone())
        } else {
            Ok(format!("{}/{}", self.url, encoded.join("/")))
        }
    }

    async fn ensure_dirs(&self, rel: &str) -> Result<(), AppError> {
        let safe = safe_relative_path(rel)?;
        // parent dir of the relative path, posix
        let parent = match safe.rsplit_once('/') {
            Some((dir, _)) => dir.to_string(),
            None => String::new(),
        };
        let mut parts: Vec<String> = Vec::new();
        if !self.root_path.is_empty() && self.root_path != "." {
            parts.push(self.root_path.clone());
        }
        if !parent.is_empty() && parent != "." {
            parts.push(parent);
        }

        let mut current = self.url.clone();
        for item in parts.join("/").split('/') {
            if item.is_empty() {
                continue;
            }
            current = format!("{current}/{}", encode_segment(item));
            let resp = self.do_request("MKCOL", &current, None, None).await?;
            let status = resp.status().as_u16();
            if status == 201 || status == 405 {
                continue;
            }
            if status >= 400 {
                return Err(AppError::upstream(format!(
                    "WebDAV MKCOL failed: HTTP {status}"
                )));
            }
        }
        Ok(())
    }

    async fn put(&self, rel: &str, payload: &[u8], content_type: &str) -> Result<String, AppError> {
        self.ensure_dirs(rel).await?;
        let url = self.remote_url(rel)?;
        self.request_checked("PUT", &url, Some(payload.to_vec()), Some(content_type))
            .await?;
        Ok(url)
    }

    async fn get(&self, rel: &str) -> Result<Vec<u8>, AppError> {
        let url = self.remote_url(rel)?;
        let resp = self.request_checked("GET", &url, None, None).await?;
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| AppError::upstream(format!("WebDAV GET body error: {e}")))?;
        Ok(bytes.to_vec())
    }

    async fn delete(&self, rel: &str) -> Result<bool, AppError> {
        let url = self.remote_url(rel)?;
        let resp = self.do_request("DELETE", &url, None, None).await?;
        let status = resp.status().as_u16();
        if matches!(status, 200 | 202 | 204 | 404) {
            return Ok(status != 404);
        }
        Err(AppError::upstream(format!(
            "WebDAV DELETE failed: HTTP {status}"
        )))
    }

    async fn test(&self) -> Value {
        if self.url.is_empty() {
            return json!({"ok": false, "status": 0, "error": "WebDAV URL is required"});
        }
        let scheme_ok = self.url.starts_with("http://") || self.url.starts_with("https://");
        if !scheme_ok {
            return json!({"ok": false, "status": 0, "error": "invalid WebDAV URL"});
        }
        let test_rel = ".chatgpt2api_webdav_test.txt";
        let result = async {
            self.put(test_rel, b"chatgpt2api webdav test\n", "text/plain").await?;
            self.delete(test_rel).await?;
            Ok::<(), AppError>(())
        }
        .await;
        match result {
            Ok(()) => json!({"ok": true, "status": 200, "error": Value::Null}),
            Err(e) => json!({"ok": false, "status": 0, "error": e.to_string()}),
        }
    }
}

// ---------------------------------------------------------------------------
// ImageStorageService
// ---------------------------------------------------------------------------

impl ImageStorageService {
    pub fn new(config: Config) -> Self {
        let index_file = config.data_dir().join("image_index.json");
        Self {
            inner: Arc::new(Inner {
                config,
                index_file,
                index_lock: Mutex::new(()),
            }),
        }
    }

    fn settings(&self) -> Value {
        self.inner.config.get_image_storage_settings()
    }

    /// Effective storage mode (`local` / `webdav` / `both`).
    pub fn mode(&self) -> String {
        let m = setting_str(&self.settings(), "mode");
        if m.is_empty() {
            "local".to_string()
        } else {
            m
        }
    }

    fn images_root(&self) -> PathBuf {
        self.inner.config.images_dir()
    }

    fn local_image_path(&self, rel: &str) -> Result<PathBuf, AppError> {
        let safe = safe_relative_path(rel)?;
        Ok(self.images_root().join(safe))
    }

    fn load_index(&self) -> Map<String, Value> {
        let raw = read_json_object(&self.inner.index_file);
        match raw.get("items") {
            Some(Value::Object(items)) => items
                .iter()
                .filter(|(_, v)| v.is_object())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            _ => Map::new(),
        }
    }

    fn load_clean_index(&self) -> Map<String, Value> {
        self.load_index()
            .into_iter()
            .filter(|(rel, _)| is_image_rel(rel))
            .collect()
    }

    fn save_index(&self, items: &Map<String, Value>) {
        write_json_object(&self.inner.index_file, &json!({ "items": items }));
    }

    fn public_url(&self, rel: &str, base_url: Option<&str>) -> Result<String, AppError> {
        let safe = safe_relative_path(rel)?;
        let settings = self.settings();
        let public_base_url = setting_str(&settings, "public_base_url");
        if !public_base_url.is_empty() {
            return Ok(format!("{}/{}", public_base_url.trim_end_matches('/'), safe));
        }
        let base = match base_url {
            Some(b) if !b.trim().is_empty() => b.trim().trim_end_matches('/').to_string(),
            _ => self.inner.config.base_url(),
        };
        Ok(format!("{}/images/{}", base.trim_end_matches('/'), safe))
    }

    /// Build the `YYYY/MM/DD/{unix}_{sha256}.{ext}` relative path for a payload.
    pub fn make_relative_path(&self, image_data: &[u8]) -> String {
        let hash = hex::encode(Sha256::digest(image_data));
        let ext = ext_from_type(image_data);
        let now = Local::now();
        let dir = now.format("%Y/%m/%d");
        format!("{dir}/{}_{hash}.{ext}", now.timestamp())
    }

    /// Store `data`, returning where it landed and its public URL.
    ///
    /// Faithful to the Python `save()` except the signature returns
    /// `StoredImage` directly (per the caller contract): a WebDAV failure is
    /// logged and the corresponding `webdav` flag is left false rather than
    /// raising.
    pub async fn save(&self, data: &[u8], base_url: Option<&str>) -> StoredImage {
        self.inner.config.cleanup_old_images();
        let rel = self.make_relative_path(data);
        let mut mode = self.mode();
        if !matches!(mode.as_str(), "local" | "webdav" | "both") {
            mode = "local".to_string();
        }

        let mut stored_local = false;
        let mut stored_webdav = false;
        let mut remote_url = String::new();

        if mode == "local" || mode == "both" {
            match self.local_image_path(&rel) {
                Ok(path) => {
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if std::fs::write(&path, data).is_ok() {
                        stored_local = true;
                    } else {
                        tracing::warn!("[image-storage] local write failed: {}", rel);
                    }
                }
                Err(e) => tracing::warn!("[image-storage] local path rejected: {e}"),
            }
        }

        if mode == "webdav" || mode == "both" {
            // WebDAV runs before taking the index lock (matches Python ordering).
            match WebDavClient::new(&self.settings()) {
                Ok(client) => match client.put(&rel, data, "image/png").await {
                    Ok(url) => {
                        remote_url = url;
                        stored_webdav = true;
                    }
                    Err(e) => tracing::warn!("[image-storage] WebDAV upload failed: {e}"),
                },
                Err(e) => tracing::warn!("[image-storage] WebDAV client init failed: {e}"),
            }
        }

        let storage = if stored_local && stored_webdav {
            "both"
        } else if stored_webdav {
            "webdav"
        } else {
            "local"
        };

        let mut item = Map::new();
        item.insert("rel".into(), json!(rel));
        item.insert("path".into(), json!(rel));
        item.insert("name".into(), json!(file_name(&rel)));
        item.insert("date".into(), json!(date_prefix(&rel)));
        item.insert("size".into(), json!(data.len()));
        item.insert("created_at".into(), json!(now_iso()));
        item.insert("storage".into(), json!(storage));
        item.insert("local".into(), json!(stored_local));
        item.insert("webdav".into(), json!(stored_webdav));
        item.insert("remote_url".into(), json!(remote_url));
        dims_into(&mut item, data);

        {
            let _guard = self.inner.index_lock.lock();
            let mut items = self.load_clean_index();
            items.insert(rel.clone(), Value::Object(item));
            self.save_index(&items);
        }

        let url = self.public_url(&rel, base_url).unwrap_or_default();
        StoredImage {
            rel,
            url,
            storage: storage.to_string(),
            size: data.len(),
        }
    }

    /// Read the bytes of a stored image, falling back to WebDAV.
    pub async fn get_bytes(&self, rel: &str) -> Result<Vec<u8>, AppError> {
        let safe = safe_relative_path(rel)?;
        if !is_image_rel(&safe) {
            return Err(AppError::not_found("image not found"));
        }
        let path = self.local_image_path(&safe)?;
        if path.is_file() {
            return std::fs::read(&path).map_err(|_| AppError::not_found("image not found"));
        }
        let is_webdav = self
            .load_clean_index()
            .get(&safe)
            .and_then(|i| i.get("webdav"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if is_webdav {
            return WebDavClient::new(&self.settings())?.get(&safe).await;
        }
        Err(AppError::not_found("image not found"))
    }

    pub fn exists(&self, rel: &str) -> bool {
        let Ok(safe) = safe_relative_path(rel) else {
            return false;
        };
        if !is_image_rel(&safe) {
            return false;
        }
        if self.local_image_path(&safe).map(|p| p.is_file()).unwrap_or(false) {
            return true;
        }
        self.load_clean_index()
            .get(&safe)
            .and_then(|i| i.get("webdav"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    pub fn has_local(&self, rel: &str) -> bool {
        let Ok(safe) = safe_relative_path(rel) else {
            return false;
        };
        is_image_rel(&safe)
            && self.local_image_path(&safe).map(|p| p.is_file()).unwrap_or(false)
    }

    /// Reconcile the index with the on-disk tree and return the display list
    /// (newest first), filtered by `[start_date, end_date]` on the `date` field.
    pub fn list_items(&self, base_url: &str, start_date: &str, end_date: &str) -> Vec<Value> {
        let _guard = self.inner.index_lock.lock();
        let mut indexed = self.load_clean_index();
        let root = self.images_root();
        let mut changed = false;

        for path in walk_files(&root) {
            let name = file_name_path(&path);
            if !is_image_rel(&name) {
                continue;
            }
            let Some(rel) = rel_posix(&root, &path) else {
                continue;
            };
            if indexed.contains_key(&rel) {
                continue;
            }
            let data = std::fs::read(&path).unwrap_or_default();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mt = mtime(&path);
            let segs = rel.split('/').count();
            let date = if segs >= 4 {
                date_prefix(&rel)
            } else {
                mt.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()
            };
            let created_at = mt
                .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_default();
            let mut item = Map::new();
            item.insert("rel".into(), json!(rel));
            item.insert("path".into(), json!(rel));
            item.insert("name".into(), json!(name));
            item.insert("date".into(), json!(date));
            item.insert("size".into(), json!(size));
            item.insert("created_at".into(), json!(created_at));
            item.insert("storage".into(), json!("local"));
            item.insert("local".into(), json!(true));
            item.insert("webdav".into(), json!(false));
            if !data.is_empty() {
                dims_into(&mut item, &data);
            }
            indexed.insert(rel, Value::Object(item));
            changed = true;
        }

        let mut items: Vec<Value> = Vec::new();
        let keys: Vec<String> = indexed.keys().cloned().collect();
        for rel in keys {
            if !is_image_rel(&rel) {
                indexed.remove(&rel);
                changed = true;
                continue;
            }
            let local = self.local_image_path(&rel).map(|p| p.is_file()).unwrap_or(false);
            let webdav = indexed
                .get(&rel)
                .and_then(|i| i.get("webdav"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !local && !webdav {
                indexed.remove(&rel);
                changed = true;
                continue;
            }
            let storage = if local && webdav {
                "both"
            } else if webdav {
                "webdav"
            } else {
                "local"
            };

            // reflect local/storage drift back into the index
            {
                let item = indexed.get_mut(&rel).unwrap();
                let obj = item.as_object_mut().unwrap();
                let prev_local = obj.get("local").and_then(|v| v.as_bool());
                let prev_storage = obj.get("storage").and_then(|v| v.as_str());
                if prev_local != Some(local) || prev_storage != Some(storage) {
                    obj.insert("local".into(), json!(local));
                    obj.insert("storage".into(), json!(storage));
                    changed = true;
                }
            }

            let item = indexed.get(&rel).cloned().unwrap();
            let day = item.get("date").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if !start_date.is_empty() && day.as_str() < start_date {
                continue;
            }
            if !end_date.is_empty() && day.as_str() > end_date {
                continue;
            }

            let mut display = item.as_object().unwrap().clone();
            display.insert("rel".into(), json!(rel));
            display.insert("path".into(), json!(rel));
            display.insert(
                "url".into(),
                json!(self.public_url(&rel, Some(base_url)).unwrap_or_default()),
            );
            items.push(Value::Object(display));
        }

        if changed {
            self.save_index(&indexed);
        }
        drop(_guard);

        items.sort_by(|a, b| {
            let ka = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            let kb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            kb.cmp(ka)
        });
        items
    }

    /// Delete an image locally (and remotely if the index records WebDAV).
    pub async fn delete(&self, rel: &str) -> Result<bool, AppError> {
        let safe = safe_relative_path(rel)?;
        let mut removed = false;
        if let Ok(path) = self.local_image_path(&safe) {
            if path.is_file() && std::fs::remove_file(&path).is_ok() {
                removed = true;
            }
        }

        let is_webdav = {
            let _guard = self.inner.index_lock.lock();
            self.load_clean_index()
                .get(&safe)
                .and_then(|i| i.get("webdav"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        };

        if is_webdav {
            match WebDavClient::new(&self.settings()) {
                Ok(client) => match client.delete(&safe).await {
                    Ok(r) => removed = r || removed,
                    Err(e) => {
                        if !removed {
                            return Err(e);
                        }
                    }
                },
                Err(e) => {
                    if !removed {
                        return Err(e);
                    }
                }
            }
        }

        {
            let _guard = self.inner.index_lock.lock();
            let mut items = self.load_clean_index();
            if items.remove(&safe).is_some() {
                self.save_index(&items);
            }
        }
        Ok(removed)
    }

    /// Upload every local image not yet marked as stored on WebDAV.
    pub async fn sync_all(&self) -> Result<Value, AppError> {
        if !matches!(self.mode().as_str(), "webdav" | "both") {
            return Err(AppError::upstream("WebDAV 图片存储未启用"));
        }
        let client = WebDavClient::new(&self.settings())?;
        let mut items = {
            let _guard = self.inner.index_lock.lock();
            self.load_clean_index()
        };

        let mut uploaded = 0i64;
        let mut skipped = 0i64;
        let mut failed = 0i64;

        let root = self.images_root();
        let mut files = walk_files(&root);
        files.sort();
        for path in files {
            let name = file_name_path(&path);
            if !is_image_rel(&name) {
                continue;
            }
            let Some(rel) = rel_posix(&root, &path) else {
                continue;
            };
            let already = items
                .get(&rel)
                .and_then(|i| i.get("webdav"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if already {
                skipped += 1;
                continue;
            }
            let payload = match std::fs::read(&path) {
                Ok(p) => p,
                Err(_) => {
                    failed += 1;
                    continue;
                }
            };
            match client.put(&rel, &payload, "image/png").await {
                Ok(remote_url) => {
                    let mt = mtime(&path);
                    let segs = rel.split('/').count();
                    let date = if segs >= 4 {
                        date_prefix(&rel)
                    } else {
                        mt.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()
                    };
                    let prev_created = items
                        .get(&rel)
                        .and_then(|i| i.get("created_at"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let created_at = prev_created.unwrap_or_else(|| {
                        mt.map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                            .unwrap_or_default()
                    });
                    let mut item = items
                        .get(&rel)
                        .and_then(|v| v.as_object().cloned())
                        .unwrap_or_default();
                    item.insert("rel".into(), json!(rel));
                    item.insert("path".into(), json!(rel));
                    item.insert("name".into(), json!(name));
                    item.insert("date".into(), json!(date));
                    item.insert("size".into(), json!(payload.len()));
                    item.insert("created_at".into(), json!(created_at));
                    item.insert("storage".into(), json!("both"));
                    item.insert("local".into(), json!(true));
                    item.insert("webdav".into(), json!(true));
                    item.insert("remote_url".into(), json!(remote_url));
                    dims_into(&mut item, &payload);
                    items.insert(rel, Value::Object(item));
                    uploaded += 1;
                }
                Err(_) => failed += 1,
            }
        }

        {
            let _guard = self.inner.index_lock.lock();
            self.save_index(&items);
        }
        Ok(json!({ "uploaded": uploaded, "skipped": skipped, "failed": failed }))
    }

    /// Probe the configured WebDAV server (PUT + DELETE a tiny file).
    pub async fn test_webdav(&self) -> Value {
        match WebDavClient::new(&self.settings()) {
            Ok(client) => client.test().await,
            Err(e) => json!({"ok": false, "status": 0, "error": e.to_string()}),
        }
    }
}

fn file_name(rel: &str) -> String {
    rel.rsplit('/').next().unwrap_or(rel).to_string()
}

fn file_name_path(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// `"-".join(rel.split("/")[:3])` — the `YYYY-MM-DD` date prefix.
fn date_prefix(rel: &str) -> String {
    rel.split('/').take(3).collect::<Vec<_>>().join("-")
}
