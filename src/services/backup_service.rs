//! Port of `services/backup_service.py` — Cloudflare R2 backup service.
//!
//! Builds a `tar.gz` archive of selected application data (respecting the
//! `backup.include` options), optionally encrypts it with AES-256-CBC in the
//! OpenSSL "Salted__" / PBKDF2 envelope, uploads it to Cloudflare R2 via a
//! hand-written AWS SigV4 (HMAC-SHA256) signer, and applies backup rotation.
//!
//! Differences from the Python original are intentional and documented inline:
//!   * Encryption is performed with the `aes`/`cbc` crates instead of shelling
//!     out to the `openssl` CLI, but the on-disk format is byte-compatible with
//!     `openssl enc -aes-256-cbc -pbkdf2 -salt -md sha256` (so archives encrypted
//!     here can be decrypted by `openssl` and vice-versa).
//!   * The background scheduler thread (`start`/`stop`/`_run`) is not ported;
//!     the async `run_scheduled_backup_if_needed` entry point is meant to be
//!     driven by the lifespan scheduler (Phase 10).
//!   * `load_backup_state` / `save_backup_state` (defined on the Python config
//!     module) are ported here as private helpers backed by
//!     `<data_dir>/backup_state.json`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aes::Aes256;
use anyhow::{anyhow, bail, Result};
use cbc::cipher::block_padding::Pkcs7;
use cbc::cipher::generic_array::GenericArray;
use cbc::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use rand::RngCore;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tar::{Archive, Builder, Header};

use crate::config::Config;
use crate::services::storage::{create_storage_backend, StorageBackend};

type HmacSha256 = Hmac<Sha256>;
type Aes256CbcEnc = cbc::Encryptor<Aes256>;
type Aes256CbcDec = cbc::Decryptor<Aes256>;

/// OpenSSL `-pbkdf2` default iteration count.
const PBKDF2_ITERATIONS: u32 = 10_000;
/// OpenSSL salted-envelope magic prefix.
const OPENSSL_MAGIC: &[u8] = b"Salted__";

// ---------------------------------------------------------------------------
// Small string / hashing helpers (ports of the module-level functions).
// ---------------------------------------------------------------------------

fn utc_now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

/// `_iso_now` — UTC timestamp truncated to seconds with a trailing `Z`.
fn iso_now() -> String {
    utc_now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// `_clean` — stringify (treating null/missing as empty) and trim.
fn clean(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string().trim().to_string(),
    }
}

fn sval(v: &Value, k: &str) -> String {
    clean(v.get(k))
}

fn bval(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn ival(v: &Value, k: &str) -> i64 {
    match v.get(k) {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(0),
        _ => 0,
    }
}

fn sha256_hex(value: &[u8]) -> String {
    hex::encode(Sha256::digest(value))
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().to_vec()
}

/// `_is_backup_object` — name (after the last `/`) looks like one of our backups.
fn is_backup_object(key: &str) -> bool {
    let name = key.rsplit('/').next().unwrap_or("");
    name.starts_with("backup-") && (name.ends_with(".tar.gz") || name.ends_with(".tar.gz.enc"))
}

/// `_guess_content_type`.
fn guess_content_type(name: &str) -> &'static str {
    if name.ends_with(".json") {
        "application/json"
    } else if name.ends_with(".jsonl") {
        "application/x-ndjson"
    } else if name.ends_with(".tar.gz") || name.ends_with(".gz") {
        "application/gzip"
    } else {
        "application/octet-stream"
    }
}

/// `_json_bytes` — pretty JSON (2-space indent, non-ASCII preserved).
fn json_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec_pretty(value).unwrap_or_else(|_| b"null".to_vec())
}

/// `_count_items` — len of a list/dict, else 0.
fn count_items(value: &Value) -> usize {
    match value {
        Value::Array(a) => a.len(),
        Value::Object(o) => o.len(),
        _ => 0,
    }
}

/// Percent-encode following Python's `quote` rules. Always-safe set is
/// unreserved (`A-Za-z0-9_.-~`); `/` is additionally kept when `keep_slash`.
fn percent_encode(s: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let safe = matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~')
            || (keep_slash && b == b'/');
        if safe {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `urlencode(sorted(items))` — sorted by key, value/key percent-encoded.
fn encode_query(query: &[(String, String)]) -> String {
    let mut items: Vec<&(String, String)> = query.iter().collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k, false), percent_encode(v, false)))
        .collect::<Vec<_>>()
        .join("&")
}

// ---------------------------------------------------------------------------
// PBKDF2 + AES-256-CBC (OpenSSL "Salted__" envelope).
// ---------------------------------------------------------------------------

/// PBKDF2-HMAC-SHA256 (implemented locally to avoid an extra crate).
fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iterations: u32, dk_len: usize) -> Vec<u8> {
    let h_len = 32usize;
    let blocks = (dk_len + h_len - 1) / h_len;
    let mut out = Vec::with_capacity(blocks * h_len);
    for block_index in 1..=blocks as u32 {
        let mut salt_block = salt.to_vec();
        salt_block.extend_from_slice(&block_index.to_be_bytes());
        let mut u = hmac_sha256(password, &salt_block);
        let mut t = u.clone();
        for _ in 1..iterations {
            u = hmac_sha256(password, &u);
            for (acc, byte) in t.iter_mut().zip(u.iter()) {
                *acc ^= *byte;
            }
        }
        out.extend_from_slice(&t);
    }
    out.truncate(dk_len);
    out
}

/// `_openssl_encrypt` — AES-256-CBC, PBKDF2(sha256), random 8-byte salt, output
/// is `b"Salted__" + salt + ciphertext` (matches `openssl enc -aes-256-cbc -pbkdf2 -salt -md sha256`).
fn encrypt_openssl(data: &[u8], passphrase: &str) -> Vec<u8> {
    let mut salt = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut salt);
    let derived = pbkdf2_hmac_sha256(passphrase.as_bytes(), &salt, PBKDF2_ITERATIONS, 48);
    let (key, iv) = derived.split_at(32);
    let ciphertext = Aes256CbcEnc::new(GenericArray::from_slice(key), GenericArray::from_slice(iv))
        .encrypt_padded_vec_mut::<Pkcs7>(data);
    let mut out = Vec::with_capacity(OPENSSL_MAGIC.len() + 8 + ciphertext.len());
    out.extend_from_slice(OPENSSL_MAGIC);
    out.extend_from_slice(&salt);
    out.extend_from_slice(&ciphertext);
    out
}

/// `_openssl_decrypt` — inverse of [`encrypt_openssl`].
fn decrypt_openssl(data: &[u8], passphrase: &str) -> Result<Vec<u8>> {
    if data.len() < 16 || &data[..8] != OPENSSL_MAGIC {
        bail!("解密备份失败：数据不是有效的加密备份（缺少 Salted__ 头）");
    }
    let salt = &data[8..16];
    let ciphertext = &data[16..];
    let derived = pbkdf2_hmac_sha256(passphrase.as_bytes(), salt, PBKDF2_ITERATIONS, 48);
    let (key, iv) = derived.split_at(32);
    Aes256CbcDec::new(GenericArray::from_slice(key), GenericArray::from_slice(iv))
        .decrypt_padded_vec_mut::<Pkcs7>(ciphertext)
        .map_err(|_| anyhow!("解密备份失败：口令错误或数据已损坏"))
}

// ---------------------------------------------------------------------------
// Cloudflare R2 client (port of `CloudflareR2Client`).
// ---------------------------------------------------------------------------

struct R2Client {
    account_id: String,
    access_key_id: String,
    secret_access_key: String,
    bucket: String,
    prefix: String,
    http: wreq::Client,
}

impl R2Client {
    fn new(settings: &Value, proxy: &str) -> Result<Self> {
        let prefix = {
            let p = sval(settings, "prefix");
            if p.is_empty() {
                "backups".to_string()
            } else {
                p
            }
        };
        let mut builder = wreq::Client::builder().emulation(wreq_util::Emulation::Chrome137);
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        let http = builder.build().map_err(|e| anyhow!("构建 R2 HTTP 客户端失败：{e}"))?;
        Ok(Self {
            account_id: sval(settings, "account_id"),
            access_key_id: sval(settings, "access_key_id"),
            secret_access_key: sval(settings, "secret_access_key"),
            bucket: sval(settings, "bucket"),
            prefix,
            http,
        })
    }

    fn validate(&self) -> Result<()> {
        let mut missing = Vec::new();
        if self.account_id.is_empty() {
            missing.push("Account ID");
        }
        if self.access_key_id.is_empty() {
            missing.push("Access Key ID");
        }
        if self.secret_access_key.is_empty() {
            missing.push("Secret Access Key");
        }
        if self.bucket.is_empty() {
            missing.push("Bucket");
        }
        if !missing.is_empty() {
            bail!("R2 配置不完整：缺少 {}", missing.join("、"));
        }
        Ok(())
    }

    fn host(&self) -> String {
        format!("{}.r2.cloudflarestorage.com", self.account_id)
    }

    fn endpoint(&self) -> String {
        format!("https://{}", self.host())
    }

    /// `_aws_v4_headers` — returns `(encoded_query, headers)`.
    fn aws_v4_headers(
        &self,
        method: &str,
        path: &str,
        query: &[(String, String)],
        body: &[u8],
        extra_headers: &[(String, String)],
    ) -> (String, Vec<(String, String)>) {
        let now = utc_now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();
        let encoded_query = encode_query(query);
        let payload_hash = sha256_hex(body);
        let host = self.host();

        // Build the header set (lower-cased keys), mirroring the Python dict.
        let mut headers: Vec<(String, String)> = vec![
            ("host".to_string(), host),
            ("x-amz-content-sha256".to_string(), payload_hash.clone()),
            ("x-amz-date".to_string(), amz_date.clone()),
        ];
        for (k, v) in extra_headers {
            headers.push((k.to_ascii_lowercase(), v.trim().to_string()));
        }

        // Canonical (sorted, whitespace-collapsed) view used only for signing.
        let mut sorted_items: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| (k.clone(), v.split_whitespace().collect::<Vec<_>>().join(" ")))
            .collect();
        sorted_items.sort_by(|a, b| a.0.cmp(&b.0));

        let canonical_headers: String =
            sorted_items.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
        let signed_headers = sorted_items
            .iter()
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_request = [
            method.to_uppercase(),
            path.to_string(),
            encoded_query.clone(),
            canonical_headers,
            signed_headers.clone(),
            payload_hash,
        ]
        .join("\n");

        let credential_scope = format!("{date_stamp}/auto/s3/aws4_request");
        let string_to_sign = [
            "AWS4-HMAC-SHA256".to_string(),
            amz_date,
            credential_scope.clone(),
            sha256_hex(canonical_request.as_bytes()),
        ]
        .join("\n");

        let k_date = hmac_sha256(
            format!("AWS4{}", self.secret_access_key).as_bytes(),
            date_stamp.as_bytes(),
        );
        let k_region = hmac_sha256(&k_date, b"auto");
        let k_service = hmac_sha256(&k_region, b"s3");
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.access_key_id, credential_scope, signed_headers, signature
        );
        headers.push(("authorization".to_string(), authorization));
        (encoded_query, headers)
    }

    /// `_request`.
    async fn request(
        &self,
        method: &str,
        key: &str,
        query: &[(String, String)],
        body: Vec<u8>,
        extra_headers: &[(String, String)],
        timeout: Duration,
    ) -> Result<wreq::Response> {
        let mut object_path = format!("/{}", self.bucket);
        if !key.is_empty() {
            object_path.push('/');
            object_path.push_str(&percent_encode(key.trim_start_matches('/'), true));
        }
        let (encoded_query, headers) =
            self.aws_v4_headers(method, &object_path, query, &body, extra_headers);
        let mut url = format!("{}{}", self.endpoint(), object_path);
        if !encoded_query.is_empty() {
            url.push('?');
            url.push_str(&encoded_query);
        }

        let mut header_map = wreq::header::HeaderMap::new();
        for (k, v) in &headers {
            if let (Ok(name), Ok(val)) = (
                wreq::header::HeaderName::from_bytes(k.as_bytes()),
                wreq::header::HeaderValue::from_str(v),
            ) {
                header_map.insert(name, val);
            }
        }

        let http_method = wreq::Method::from_bytes(method.to_uppercase().as_bytes())
            .map_err(|e| anyhow!("非法 HTTP 方法：{e}"))?;
        let resp = self
            .http
            .request(http_method, url)
            .headers(header_map)
            .body(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| anyhow!("R2 请求失败：{e}"))?;
        Ok(resp)
    }

    /// `test_connection`.
    async fn test_connection(&self) -> Result<Value> {
        self.validate()?;
        let query = vec![
            ("list-type".to_string(), "2".to_string()),
            ("max-keys".to_string(), "1".to_string()),
        ];
        let resp = self
            .request("GET", "", &query, Vec::new(), &[], Duration::from_secs(30))
            .await?;
        let status = resp.status().as_u16();
        if status >= 400 {
            bail!("连接 R2 失败：HTTP {status}");
        }
        Ok(json!({ "ok": true, "status": status }))
    }

    /// `upload_bytes`.
    async fn upload_bytes(
        &self,
        key: &str,
        payload: Vec<u8>,
        content_type: &str,
        metadata: &[(String, String)],
    ) -> Result<Value> {
        let mut extra = vec![("content-type".to_string(), content_type.to_string())];
        for (k, v) in metadata {
            extra.push((format!("x-amz-meta-{k}"), v.clone()));
        }
        let resp = self
            .request("PUT", key, &[], payload, &extra, Duration::from_secs(60))
            .await?;
        let status = resp.status().as_u16();
        if status >= 400 {
            bail!("上传备份失败：HTTP {status}");
        }
        let etag = resp
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .trim_matches('"')
            .to_string();
        Ok(json!({ "key": key, "etag": etag }))
    }

    /// `delete_object`.
    async fn delete_object(&self, key: &str) -> Result<()> {
        let resp = self
            .request("DELETE", key, &[], Vec::new(), &[], Duration::from_secs(30))
            .await?;
        let status = resp.status().as_u16();
        if status >= 400 && status != 404 {
            bail!("删除备份失败：HTTP {status}");
        }
        Ok(())
    }

    /// `download_bytes`.
    async fn download_bytes(&self, key: &str) -> Result<Vec<u8>> {
        let resp = self
            .request("GET", key, &[], Vec::new(), &[], Duration::from_secs(60))
            .await?;
        let status = resp.status().as_u16();
        if status >= 400 {
            bail!("读取备份失败：HTTP {status}");
        }
        let bytes = resp.bytes().await.map_err(|e| anyhow!("读取备份内容失败：{e}"))?;
        Ok(bytes.to_vec())
    }

    /// `list_objects` — paginated, sorted newest-first. Parses the S3 XML with
    /// the same naive string-splitting strategy as the Python original.
    async fn list_objects(&self) -> Result<Vec<Value>> {
        let mut items: Vec<Value> = Vec::new();
        let mut continuation = String::new();
        loop {
            let mut query = vec![
                ("list-type".to_string(), "2".to_string()),
                ("prefix".to_string(), format!("{}/", self.prefix.trim_end_matches('/'))),
                ("max-keys".to_string(), "1000".to_string()),
            ];
            if !continuation.is_empty() {
                query.push(("continuation-token".to_string(), continuation.clone()));
            }
            let resp = self
                .request("GET", "", &query, Vec::new(), &[], Duration::from_secs(30))
                .await?;
            let status = resp.status().as_u16();
            if status >= 400 {
                bail!("获取备份列表失败：HTTP {status}");
            }
            let text = resp.text().await.map_err(|e| anyhow!("解析备份列表失败：{e}"))?;

            for block in text.split("<Contents>").skip(1) {
                let key = extract_between(block, "<Key>", "</Key>");
                if key.is_empty() {
                    continue;
                }
                let size_text = extract_between(block, "<Size>", "</Size>");
                let updated = extract_between(block, "<LastModified>", "</LastModified>");
                items.push(json!({
                    "key": key,
                    "size": size_text.trim().parse::<i64>().unwrap_or(0),
                    "updated_at": updated,
                }));
            }

            let truncated = text.contains("<IsTruncated>true</IsTruncated>");
            if !truncated || !text.contains("<NextContinuationToken>") {
                break;
            }
            continuation =
                extract_between(&text, "<NextContinuationToken>", "</NextContinuationToken>");
            if continuation.is_empty() {
                break;
            }
        }
        items.sort_by(|a, b| sval(b, "updated_at").cmp(&sval(a, "updated_at")));
        Ok(items)
    }
}

/// Extract the (trimmed) text between the first `start` and the following `end`.
fn extract_between(haystack: &str, start: &str, end: &str) -> String {
    if let Some(s) = haystack.find(start) {
        let rest = &haystack[s + start.len()..];
        if let Some(e) = rest.find(end) {
            return rest[..e].trim().to_string();
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Backup state persistence (port of config.load_backup_state/save_backup_state).
// ---------------------------------------------------------------------------

fn normalize_backup_state(value: &Value) -> Value {
    let opt_str = |k: &str| {
        let s = sval(value, k);
        if s.is_empty() {
            Value::Null
        } else {
            Value::String(s)
        }
    };
    let last_status = {
        let s = sval(value, "last_status");
        if s.is_empty() {
            "idle".to_string()
        } else {
            s
        }
    };
    json!({
        "last_started_at": opt_str("last_started_at"),
        "last_finished_at": opt_str("last_finished_at"),
        "last_status": last_status,
        "last_error": opt_str("last_error"),
        "last_object_key": opt_str("last_object_key"),
    })
}

fn backup_state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("backup_state.json")
}

fn load_backup_state(data_dir: &Path) -> Value {
    let path = backup_state_path(data_dir);
    let raw = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or(Value::Null);
    normalize_backup_state(&raw)
}

fn save_backup_state(data_dir: &Path, state: &Value) -> Value {
    let normalized = normalize_backup_state(state);
    let text = serde_json::to_string_pretty(&normalized).unwrap_or_else(|_| "{}".to_string()) + "\n";
    let _ = std::fs::write(backup_state_path(data_dir), text);
    normalized
}

// ---------------------------------------------------------------------------
// BackupService.
// ---------------------------------------------------------------------------

/// Result of [`BackupService::download_backup`]. Carries the raw bytes, which do
/// not fit cleanly in `serde_json::Value`.
pub struct DownloadedBackup {
    pub key: String,
    pub name: String,
    pub content_type: String,
    pub payload: Vec<u8>,
    pub size: usize,
}

struct Inner {
    config: Config,
    running: Mutex<bool>,
    storage: Mutex<Option<Arc<dyn StorageBackend>>>,
}

/// Cloudflare R2 backup service (port of `BackupService`). Cheaply cloneable.
#[derive(Clone)]
pub struct BackupService {
    inner: Arc<Inner>,
}

impl BackupService {
    pub fn new(config: Config) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                running: Mutex::new(false),
                storage: Mutex::new(None),
            }),
        }
    }

    fn data_dir(&self) -> &Path {
        self.inner.config.data_dir()
    }

    /// Lazily-created, cached storage backend (mirrors `config.get_storage_backend`).
    fn storage_backend(&self) -> Result<Arc<dyn StorageBackend>> {
        let mut guard = self.inner.storage.lock();
        if let Some(existing) = guard.as_ref() {
            return Ok(existing.clone());
        }
        let boxed = create_storage_backend(self.data_dir())?;
        let arc: Arc<dyn StorageBackend> = Arc::from(boxed);
        *guard = Some(arc.clone());
        Ok(arc)
    }

    fn make_client(&self, settings: &Value) -> Result<R2Client> {
        R2Client::new(settings, &self.inner.config.proxy_setting())
    }

    // ---- status / configuration --------------------------------------------

    /// `get_status` — persisted backup state merged with the live `running` flag.
    pub fn get_status(&self) -> Value {
        let mut state = load_backup_state(self.data_dir());
        if let Some(obj) = state.as_object_mut() {
            obj.insert("running".to_string(), Value::Bool(*self.inner.running.lock()));
        }
        state
    }

    /// `is_configured` — all four required R2 credentials present.
    pub fn is_configured(&self) -> bool {
        let s = self.inner.config.get_backup_settings();
        !sval(&s, "account_id").is_empty()
            && !sval(&s, "access_key_id").is_empty()
            && !sval(&s, "secret_access_key").is_empty()
            && !sval(&s, "bucket").is_empty()
    }

    /// `get_settings` — settings with secrets masked.
    pub fn get_settings(&self) -> Value {
        let mut settings = self.inner.config.get_backup_settings();
        if let Some(obj) = settings.as_object_mut() {
            let mask = |obj: &mut Map<String, Value>, key: &str| {
                let has = obj.get(key).map(|v| !clean(Some(v)).is_empty()).unwrap_or(false);
                obj.insert(key.to_string(), Value::String(if has { "********" } else { "" }.into()));
            };
            mask(obj, "secret_access_key");
            mask(obj, "passphrase");
        }
        settings
    }

    /// `update_settings` — merge a patch, preserving masked secrets, then persist.
    pub fn update_settings(&self, payload: Value) -> Result<Value> {
        let current = self.inner.config.get_backup_settings();
        let payload_obj = payload.as_object().cloned().unwrap_or_default();

        let mut merged = current.as_object().cloned().unwrap_or_default();
        for (k, v) in &payload_obj {
            merged.insert(k.clone(), v.clone());
        }

        // Deep-merge the `include` sub-map.
        if let Some(Value::Object(patch_include)) = payload_obj.get("include") {
            let mut include = current
                .get("include")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            for (k, v) in patch_include {
                include.insert(k.clone(), v.clone());
            }
            merged.insert("include".to_string(), Value::Object(include));
        }

        // Keep existing secrets when the masked sentinel is sent back unchanged.
        for key in ["secret_access_key", "passphrase"] {
            if payload_obj.get(key).and_then(|v| v.as_str()) == Some("********") {
                let prev = current.get(key).cloned().unwrap_or(Value::String(String::new()));
                merged.insert(key.to_string(), prev);
            }
        }

        let updated = self
            .inner
            .config
            .update(json!({ "backup": Value::Object(merged) }))?;
        Ok(updated.get("backup").cloned().unwrap_or_else(|| json!({})))
    }

    // ---- remote operations -------------------------------------------------

    /// `test_connection`.
    pub async fn test_connection(&self) -> Result<Value> {
        let settings = self.inner.config.get_backup_settings();
        let client = self.make_client(&settings)?;
        client.test_connection().await
    }

    /// `list_backups`.
    pub async fn list_backups(&self) -> Result<Value> {
        if !self.is_configured() {
            return Ok(json!([]));
        }
        let settings = self.inner.config.get_backup_settings();
        let client = self.make_client(&settings)?;
        let items = client.list_objects().await?;
        let parsed: Vec<Value> = items
            .iter()
            .map(|item| {
                let key = sval(item, "key");
                let name = key.rsplit('/').next().unwrap_or("").to_string();
                let encrypted = name.ends_with(".enc");
                json!({
                    "key": key,
                    "name": name,
                    "size": ival(item, "size"),
                    "updated_at": item.get("updated_at").cloned().unwrap_or(Value::Null),
                    "encrypted": encrypted,
                })
            })
            .collect();
        Ok(Value::Array(parsed))
    }

    /// `delete_backup`.
    pub async fn delete_backup(&self, key: &str) -> Result<()> {
        let candidate = key.trim();
        if candidate.is_empty() {
            bail!("备份对象 key 不能为空");
        }
        let settings = self.inner.config.get_backup_settings();
        let client = self.make_client(&settings)?;
        client.delete_object(candidate).await
    }

    /// `download_backup` — fetch and (if needed) decrypt for download.
    pub async fn download_backup(&self, key: &str) -> Result<DownloadedBackup> {
        let candidate = key.trim().to_string();
        if candidate.is_empty() {
            bail!("备份对象 key 不能为空");
        }
        let settings = self.inner.config.get_backup_settings();
        let client = self.make_client(&settings)?;
        let mut payload = client.download_bytes(&candidate).await?;
        let mut name = {
            let n = candidate.rsplit('/').next().unwrap_or("");
            if n.is_empty() {
                "backup.bin".to_string()
            } else {
                n.to_string()
            }
        };
        if candidate.ends_with(".enc") {
            let passphrase = sval(&settings, "passphrase");
            if passphrase.is_empty() {
                bail!("当前未配置加密口令，无法下载并解密已加密备份");
            }
            payload = decrypt_openssl(&payload, &passphrase)?;
            if name.ends_with(".enc") {
                name = name[..name.len() - 4].to_string();
                if name.is_empty() {
                    name = "backup.tar.gz".to_string();
                }
            }
        }
        let content_type = guess_content_type(&name).to_string();
        let size = payload.len();
        Ok(DownloadedBackup { key: candidate, name, content_type, payload, size })
    }

    /// `get_backup_detail` — fetch, decrypt if needed, and summarize the archive.
    pub async fn get_backup_detail(&self, key: &str) -> Result<Value> {
        let candidate = key.trim().to_string();
        if candidate.is_empty() {
            bail!("备份对象 key 不能为空");
        }
        let settings = self.inner.config.get_backup_settings();
        let client = self.make_client(&settings)?;
        let payload = client.download_bytes(&candidate).await?;

        let mut decoded = payload;
        if candidate.ends_with(".enc") {
            let passphrase = sval(&settings, "passphrase");
            if passphrase.is_empty() {
                bail!("当前未配置加密口令，无法查看已加密备份");
            }
            decoded = decrypt_openssl(&decoded, &passphrase)?;
        }
        let mut detail = decode_archive_detail(&decoded)?;
        if let Some(obj) = detail.as_object_mut() {
            obj.insert("key".to_string(), Value::String(candidate.clone()));
            obj.insert(
                "name".to_string(),
                Value::String(candidate.rsplit('/').next().unwrap_or("").to_string()),
            );
            obj.insert("encrypted".to_string(), Value::Bool(candidate.ends_with(".enc")));
        }
        Ok(detail)
    }

    // ---- scheduling --------------------------------------------------------

    /// `run_scheduled_backup_if_needed` — run a backup if enabled and the
    /// configured interval has elapsed since the last successful run.
    pub async fn run_scheduled_backup_if_needed(&self) -> Result<()> {
        let settings = self.inner.config.get_backup_settings();
        if !bval(&settings, "enabled") {
            return Ok(());
        }
        if *self.inner.running.lock() {
            return Ok(());
        }
        let state = load_backup_state(self.data_dir());
        let interval_minutes = {
            let v = ival(&settings, "interval_minutes");
            if v <= 0 {
                360
            } else {
                v
            }
        };
        let last_finished_raw = sval(&state, "last_finished_at");
        if !last_finished_raw.is_empty() {
            let parse = last_finished_raw.replace('Z', "+00:00");
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&parse) {
                let elapsed = utc_now()
                    .signed_duration_since(dt.with_timezone(&chrono::Utc))
                    .num_seconds();
                if elapsed < interval_minutes * 60 {
                    return Ok(());
                }
            }
        }
        self.run_backup("schedule").await.map(|_| ())
    }

    /// `run_backup` — orchestrate a single backup, recording status throughout.
    pub async fn run_backup(&self, trigger: &str) -> Result<Value> {
        let data_dir = self.data_dir().to_path_buf();
        let current = load_backup_state(&data_dir);
        let started_at = iso_now();

        {
            let mut running = self.inner.running.lock();
            if *running {
                bail!("当前已有备份任务正在执行");
            }
            *running = true;
        }
        save_backup_state(
            &data_dir,
            &json!({
                "last_started_at": started_at,
                "last_finished_at": current.get("last_finished_at").cloned().unwrap_or(Value::Null),
                "last_status": "idle",
                "last_error": Value::Null,
                "last_object_key": current.get("last_object_key").cloned().unwrap_or(Value::Null),
            }),
        );

        let outcome = self.run_backup_once(trigger).await;
        *self.inner.running.lock() = false;

        match outcome {
            Ok(result) => {
                save_backup_state(
                    &data_dir,
                    &json!({
                        "last_started_at": started_at,
                        "last_finished_at": iso_now(),
                        "last_status": "success",
                        "last_error": Value::Null,
                        "last_object_key": result.get("key").cloned().unwrap_or(Value::Null),
                    }),
                );
                Ok(result)
            }
            Err(err) => {
                save_backup_state(
                    &data_dir,
                    &json!({
                        "last_started_at": started_at,
                        "last_finished_at": iso_now(),
                        "last_status": "error",
                        "last_error": err.to_string(),
                        "last_object_key": current.get("last_object_key").cloned().unwrap_or(Value::Null),
                    }),
                );
                Err(err)
            }
        }
    }

    /// `_run_backup_once`.
    async fn run_backup_once(&self, trigger: &str) -> Result<Value> {
        let settings = self.inner.config.get_backup_settings();
        let client = self.make_client(&settings)?;
        client.validate()?;

        let payload_raw = self.build_backup_archive(&settings, trigger)?;
        let encrypted = bval(&settings, "encrypt");
        let (payload, suffix) = if encrypted {
            let passphrase = sval(&settings, "passphrase");
            if passphrase.is_empty() {
                bail!("已启用备份加密，但未设置加密口令");
            }
            (encrypt_openssl(&payload_raw, &passphrase), ".tar.gz.enc")
        } else {
            (payload_raw, ".tar.gz")
        };

        let timestamp = utc_now().format("%Y%m%dT%H%M%SZ").to_string();
        let random_tag = format!("{:04x}", rand::thread_rng().next_u32() & 0xFFFF);
        let object_key =
            format!("{}/backup-{timestamp}-{random_tag}{suffix}", client.prefix.trim_end_matches('/'));
        let metadata = vec![
            ("created-at".to_string(), iso_now()),
            ("encrypted".to_string(), if encrypted { "true" } else { "false" }.to_string()),
            ("trigger".to_string(), trigger.to_string()),
        ];

        let size = payload.len();
        let result = client
            .upload_bytes(&object_key, payload, "application/octet-stream", &metadata)
            .await?;
        self.apply_rotation(&client, ival(&settings, "rotation_keep")).await?;

        Ok(json!({
            "key": result.get("key").cloned().unwrap_or(Value::String(object_key)),
            "size": size,
            "encrypted": encrypted,
        }))
    }

    /// `_apply_rotation`.
    async fn apply_rotation(&self, client: &R2Client, keep: i64) -> Result<()> {
        if keep <= 0 {
            return Ok(());
        }
        let items: Vec<Value> = client
            .list_objects()
            .await?
            .into_iter()
            .filter(|item| is_backup_object(&sval(item, "key")))
            .collect();
        if items.len() as i64 <= keep {
            return Ok(());
        }
        for item in items.iter().skip(keep as usize) {
            let key = sval(item, "key");
            if !key.is_empty() {
                client.delete_object(&key).await?;
            }
        }
        Ok(())
    }

    /// `_build_backup_archive` — assemble the tar.gz in memory.
    fn build_backup_archive(&self, settings: &Value, trigger: &str) -> Result<Vec<u8>> {
        let include = settings.get("include").cloned().unwrap_or_else(|| json!({}));
        let inc = |k: &str| bval(&include, k);

        let backend = self.storage_backend()?;
        let metadata = json!({
            "version": 2,
            "created_at": iso_now(),
            "trigger": trigger,
            "app_version": self.inner.config.app_version(),
            "storage_backend": backend.get_backend_info(),
        });

        let data_dir = self.data_dir();
        let mtime = utc_now().timestamp() as u64;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut tar = Builder::new(&mut encoder);

            add_bytes(&mut tar, "backup-metadata.json", &json_bytes(&metadata), mtime)?;

            if inc("config") {
                // CONFIG_FILE lives next to the data dir (BASE_DIR/config.json).
                let config_path = data_dir
                    .parent()
                    .map(|p| p.join("config.json"))
                    .unwrap_or_else(|| PathBuf::from("config.json"));
                add_file(&mut tar, &config_path, "config.json")?;
            }
            if inc("register") {
                add_file(&mut tar, &data_dir.join("register.json"), "data/register.json")?;
            }
            if inc("cpa") {
                add_file(&mut tar, &data_dir.join("cpa_config.json"), "data/cpa_config.json")?;
            }
            if inc("sub2api") {
                add_file(&mut tar, &data_dir.join("sub2api_config.json"), "data/sub2api_config.json")?;
            }
            if inc("logs") {
                add_file(&mut tar, &data_dir.join("logs.jsonl"), "data/logs.jsonl")?;
            }
            if inc("image_tasks") {
                add_file(&mut tar, &data_dir.join("image_tasks.json"), "data/image_tasks.json")?;
                add_file(&mut tar, &data_dir.join("image_index.json"), "data/image_index.json")?;
            }
            if inc("accounts_snapshot") {
                let snapshot = Value::Array(backend.load_accounts());
                add_bytes(&mut tar, "snapshots/accounts.json", &json_bytes(&snapshot), mtime)?;
            }
            if inc("auth_keys_snapshot") {
                let snapshot = Value::Array(backend.load_auth_keys());
                add_bytes(&mut tar, "snapshots/auth_keys.json", &json_bytes(&snapshot), mtime)?;
            }
            if inc("images") {
                add_file(&mut tar, &data_dir.join("image_tags.json"), "data/image_tags.json")?;
                add_directory(&mut tar, &self.inner.config.images_dir(), "data/images")?;
            }

            tar.finish().map_err(|e| anyhow!("打包备份失败：{e}"))?;
        }
        encoder.finish().map_err(|e| anyhow!("压缩备份失败：{e}"))
    }
}

// ---------------------------------------------------------------------------
// tar helpers.
// ---------------------------------------------------------------------------

fn add_bytes<W: std::io::Write>(
    tar: &mut Builder<W>,
    name: &str,
    payload: &[u8],
    mtime: u64,
) -> Result<()> {
    let mut header = Header::new_gnu();
    header.set_size(payload.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(mtime);
    header.set_cksum();
    tar.append_data(&mut header, name, payload)
        .map_err(|e| anyhow!("写入归档项 {name} 失败：{e}"))
}

fn add_file<W: std::io::Write>(tar: &mut Builder<W>, source: &Path, arcname: &str) -> Result<()> {
    if !source.is_file() {
        return Ok(());
    }
    tar.append_path_with_name(source, arcname)
        .map_err(|e| anyhow!("写入文件 {arcname} 失败：{e}"))
}

fn add_directory<W: std::io::Write>(
    tar: &mut Builder<W>,
    source_dir: &Path,
    arcname_root: &str,
) -> Result<()> {
    if !source_dir.is_dir() {
        return Ok(());
    }
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(source_dir, &mut files);
    files.sort();
    for path in files {
        if let Ok(rel) = path.strip_prefix(source_dir) {
            let rel_posix = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/");
            let arcname = format!("{arcname_root}/{rel_posix}");
            add_file(tar, &path, &arcname)?;
        }
    }
    Ok(())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, out);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
}

/// `_decode_archive_detail` — summarize a decoded tar.gz payload.
fn decode_archive_detail(payload: &[u8]) -> Result<Value> {
    let mut files: Vec<Value> = Vec::new();
    let mut snapshots: Vec<Value> = Vec::new();
    let mut metadata = Map::new();

    let decoder = GzDecoder::new(payload);
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|_| anyhow!("解析备份压缩包失败，备份可能已损坏"))?;

    for entry in entries {
        let mut entry = entry.map_err(|_| anyhow!("解析备份压缩包失败，备份可能已损坏"))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry
            .path()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        let mut raw = Vec::new();
        if entry.read_to_end(&mut raw).is_err() {
            continue;
        }

        if name == "backup-metadata.json" {
            if let Ok(Value::Object(parsed)) = serde_json::from_slice::<Value>(&raw) {
                metadata = parsed;
            }
            continue;
        }
        if name.starts_with("snapshots/") && name.ends_with(".json") {
            let count = serde_json::from_slice::<Value>(&raw)
                .map(|v| count_items(&v))
                .unwrap_or(0);
            let short = name
                .strip_prefix("snapshots/")
                .and_then(|s| s.strip_suffix(".json"))
                .unwrap_or(&name)
                .to_string();
            snapshots.push(json!({ "name": short, "count": count }));
            continue;
        }
        files.push(json!({
            "name": name,
            "exists": true,
            "content_type": guess_content_type(&name),
            "size": raw.len(),
            "sha256": sha256_hex(&raw),
        }));
    }

    files.sort_by(|a, b| sval(a, "name").cmp(&sval(b, "name")));
    snapshots.sort_by(|a, b| sval(a, "name").cmp(&sval(b, "name")));

    let meta_get = |k: &str| metadata.get(k).cloned().unwrap_or(Value::Null);
    Ok(json!({
        "created_at": meta_get("created_at"),
        "trigger": meta_get("trigger"),
        "app_version": meta_get("app_version"),
        "storage_backend": meta_get("storage_backend"),
        "files": files,
        "snapshots": snapshots,
    }))
}
