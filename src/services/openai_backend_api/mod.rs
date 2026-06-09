//! Port of `services/openai_backend_api.py` — the reverse-engineered ChatGPT web
//! backend client. Holds a `wreq` client with an Edge TLS/HTTP2 fingerprint
//! (`Emulation::Edge101`, matching curl_cffi's `edge101`), performs the PoW /
//! sentinel / turnstile handshake, and drives the conversation, image-generation,
//! codex and search flows.
//!
//! Async port: SSE iteration is exposed as [`SseStream`] (call
//! [`SseStream::next_payload`] in a loop). Concurrency that Python did with a
//! `ThreadPoolExecutor` (see [`OpenAIBackendAPI::get_user_info`]) uses
//! `tokio::join!`.
//!
//! The editable-file export (PPT/PSD) methods are added in Phase 8.

use std::time::Duration;

use serde_json::{json, Map, Value};
use wreq::header::{HeaderMap, HeaderName, HeaderValue};
use wreq::{Client, Proxy, Response};
use wreq_util::Emulation;

use crate::config::Config;
use crate::utils::helper::{new_uuid, UpstreamHttpError};
use crate::utils::pow::{build_legacy_requirements_token, build_proof_token, parse_pow_resources};
use crate::utils::turnstile::solve_turnstile_token;

// ---- Constants (mirror the Python module constants) ----

pub const DEFAULT_CLIENT_VERSION: &str = "prod-a194cd50d4416d3c0b47c740f206b12ce60f5887";
pub const DEFAULT_CLIENT_BUILD_NUMBER: &str = "6708908";
pub const DEFAULT_POW_SCRIPT: &str = "https://chatgpt.com/backend-api/sentinel/sdk.js";
pub const CODEX_IMAGE_MODEL: &str = "codex-gpt-image-2";
pub const CODEX_RESPONSES_MODEL: &str = "gpt-5.5";
pub const SEARCH_MODEL: &str = "gpt-5-5";
pub const SEARCH_TIMEOUT_SECS: f64 = 300.0;
pub const SEARCH_POLL_INTERVAL_SECS: f64 = 3.0;

const CODEX_RESPONSES_INSTRUCTIONS: &str =
    "Use the image_generation tool to create exactly one image for the user's request. \
     Return the generated image result.";

const DEFAULT_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0";

/// Content-policy violation keywords (upstream refusals); mirror
/// `_CONTENT_POLICY_KEYWORDS`.
const CONTENT_POLICY_KEYWORDS: &[&str] = &[
    "内容政策",
    "防护限制",
    "违反",
    "moderation",
    "policy",
    "blocked",
    "不能生成",
    "无法生成",
    "不能帮助",
    "无法帮助",
    "裸体",
    "裸露",
    "色情",
    "性内容",
    "未成年",
    "抱歉，我不能",
];

fn is_content_policy_error(error_msg: &str) -> bool {
    if error_msg.is_empty() {
        return false;
    }
    let lower = error_msg.to_lowercase();
    CONTENT_POLICY_KEYWORDS.iter().any(|kw| lower.contains(&kw.to_lowercase()))
}

// ---- Errors ----

/// Engine-level error type — mirrors the Python exception hierarchy plus the
/// generic upstream/other cases.
#[derive(Debug)]
pub enum EngineError {
    /// `InvalidAccessTokenError` — the upstream returned 401 for this token.
    InvalidAccessToken(String),
    /// `ImagePollTimeoutError` — polling for image results exceeded the budget.
    ImagePollTimeout {
        message: String,
        conversation_id: String,
        task_error: Option<String>,
    },
    /// `ImageContentPolicyError` — upstream refused generation on policy grounds.
    ImageContentPolicy(String),
    /// `UpstreamHTTPError` — non-2xx response carrying status/body/retry-after.
    Upstream(UpstreamHttpError),
    /// Any other runtime failure (network, parse, invalid state).
    Other(String),
}

pub type EngineResult<T> = Result<T, EngineError>;

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidAccessToken(s) => write!(f, "invalid access token: {s}"),
            EngineError::ImagePollTimeout { message, .. } => write!(f, "{message}"),
            EngineError::ImageContentPolicy(s) => write!(f, "{s}"),
            EngineError::Upstream(e) => write!(f, "{e}"),
            EngineError::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<UpstreamHttpError> for EngineError {
    fn from(e: UpstreamHttpError) -> Self {
        EngineError::Upstream(e)
    }
}

impl EngineError {
    fn other(msg: impl Into<String>) -> Self {
        EngineError::Other(msg.into())
    }

    fn from_wreq(e: wreq::Error) -> Self {
        EngineError::Other(e.to_string())
    }
}

// ---- ChatRequirements ----

/// Sentinel tokens required for a single conversation request
/// (port of the `ChatRequirements` dataclass).
#[derive(Debug, Clone, Default)]
pub struct ChatRequirements {
    pub token: String,
    pub proof_token: String,
    pub turnstile_token: String,
    pub so_token: String,
    pub raw_finalize: Value,
}

// ---- SSE stream ----

/// Async reader over a streaming SSE response. Yields the trimmed payload after
/// each `data:` line (mirrors `iter_sse_payloads`).
pub struct SseStream {
    resp: Response,
    buf: Vec<u8>,
    done: bool,
}

impl SseStream {
    fn new(resp: Response) -> Self {
        Self {
            resp,
            buf: Vec::new(),
            done: false,
        }
    }

    fn take_line(&mut self) -> Option<Vec<u8>> {
        let pos = self.buf.iter().position(|&b| b == b'\n')?;
        let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
        line.pop(); // drop '\n'
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Some(line)
    }

    fn payload_from_line(line: &[u8]) -> Option<String> {
        if line.is_empty() {
            return None;
        }
        let s = String::from_utf8_lossy(line);
        let s = s.trim();
        let rest = s.strip_prefix("data:")?;
        let payload = rest.trim();
        if payload.is_empty() {
            None
        } else {
            Some(payload.to_string())
        }
    }

    /// Returns the next SSE `data:` payload, or `None` at end of stream.
    pub async fn next_payload(&mut self) -> EngineResult<Option<String>> {
        loop {
            while let Some(line) = self.take_line() {
                if let Some(payload) = Self::payload_from_line(&line) {
                    return Ok(Some(payload));
                }
            }
            if self.done {
                if !self.buf.is_empty() {
                    let line = std::mem::take(&mut self.buf);
                    if let Some(payload) = Self::payload_from_line(&line) {
                        return Ok(Some(payload));
                    }
                }
                return Ok(None);
            }
            match self.resp.chunk().await.map_err(EngineError::from_wreq)? {
                Some(chunk) => self.buf.extend_from_slice(&chunk),
                None => self.done = true,
            }
        }
    }
}

// ---- Fingerprint ----

#[derive(Debug, Clone)]
struct Fingerprint {
    user_agent: String,
    impersonate: String,
    device_id: String,
    session_id: String,
    sec_ch_ua: String,
    sec_ch_ua_mobile: String,
    sec_ch_ua_platform: String,
}

fn account_str(account: &Value, key: &str) -> String {
    account.get(key).and_then(|v| v.as_str()).unwrap_or("").trim().to_string()
}

fn build_fingerprint(account: &Value) -> Fingerprint {
    // Lower-cased `fp` sub-dict, then top-level overrides.
    let mut map: Map<String, Value> = Map::new();
    if let Some(raw_fp) = account.get("fp").and_then(|v| v.as_object()) {
        for (k, v) in raw_fp {
            let val = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            map.insert(k.to_lowercase(), Value::String(val));
        }
    }
    for key in [
        "user-agent",
        "impersonate",
        "oai-device-id",
        "oai-session-id",
        "sec-ch-ua",
        "sec-ch-ua-mobile",
        "sec-ch-ua-platform",
    ] {
        let value = account_str(account, key);
        if !value.is_empty() {
            map.insert(key.to_string(), Value::String(value));
        }
    }
    let get = |m: &Map<String, Value>, k: &str| m.get(k).and_then(|v| v.as_str()).map(String::from);

    Fingerprint {
        user_agent: get(&map, "user-agent").unwrap_or_else(|| DEFAULT_USER_AGENT.to_string()),
        impersonate: get(&map, "impersonate").unwrap_or_else(|| "edge101".to_string()),
        device_id: get(&map, "oai-device-id").unwrap_or_else(new_uuid),
        session_id: get(&map, "oai-session-id").unwrap_or_else(new_uuid),
        sec_ch_ua: get(&map, "sec-ch-ua").unwrap_or_else(|| {
            "\"Microsoft Edge\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"".to_string()
        }),
        sec_ch_ua_mobile: get(&map, "sec-ch-ua-mobile").unwrap_or_else(|| "?0".to_string()),
        sec_ch_ua_platform: get(&map, "sec-ch-ua-platform").unwrap_or_else(|| "\"Windows\"".to_string()),
    }
}

fn emulation_for(impersonate: &str) -> Emulation {
    // curl_cffi target names → wreq-util Emulation. We default to Edge101 (the
    // Python default) and map a couple of newer Edge targets if configured.
    match impersonate.trim().to_lowercase().as_str() {
        "edge131" => Emulation::Edge131,
        "edge134" => Emulation::Edge134,
        _ => Emulation::Edge101,
    }
}

// ---- Header helpers ----

fn header_name(name: &str) -> Option<HeaderName> {
    HeaderName::from_bytes(name.as_bytes()).ok()
}

fn header_value(value: &str) -> Option<HeaderValue> {
    HeaderValue::from_str(value).ok()
}

fn insert_header(map: &mut HeaderMap, name: &str, value: &str) {
    if let (Some(n), Some(v)) = (header_name(name), header_value(value)) {
        map.insert(n, v);
    }
}

// ---- Engine ----

pub struct OpenAIBackendAPI {
    base_url: String,
    client_version: String,
    client_build_number: String,
    access_token: String,
    account: Value,
    config: Config,
    fp: Fingerprint,
    user_agent: String,
    device_id: String,
    session_id: String,
    pow_script_sources: Vec<String>,
    pow_data_build: String,
    client: Client,
}

impl OpenAIBackendAPI {
    /// Build a client for the given account. `access_token` empty → anon flow.
    ///
    /// Unlike the Python version, which reached into the `account_service` and
    /// `proxy_settings` globals, the caller passes the resolved `account` dict
    /// and the shared [`Config`] so the engine stays decoupled (avoids the
    /// account-service ↔ engine cycle).
    pub fn new(config: Config, access_token: String, account: Value) -> EngineResult<Self> {
        let account = if account.is_object() { account } else { json!({}) };
        let fp = build_fingerprint(&account);

        let base_url = "https://chatgpt.com".to_string();
        let base_headers = Self::build_base_headers(&fp, &base_url, &access_token);

        let proxy = {
            let account_proxy = account_str(&account, "proxy");
            if !account_proxy.is_empty() {
                account_proxy
            } else {
                config.proxy_setting()
            }
        };

        let mut builder = Client::builder()
            .emulation(emulation_for(&fp.impersonate))
            .default_headers(base_headers)
            .cookie_store(true);
        if !proxy.trim().is_empty() {
            match Proxy::all(proxy.trim()) {
                Ok(p) => builder = builder.proxy(p),
                Err(e) => tracing::warn!("invalid proxy {proxy:?}: {e}"),
            }
        }
        let client = builder.build().map_err(EngineError::from_wreq)?;

        Ok(Self {
            base_url,
            client_version: DEFAULT_CLIENT_VERSION.to_string(),
            client_build_number: DEFAULT_CLIENT_BUILD_NUMBER.to_string(),
            access_token,
            account,
            config,
            user_agent: fp.user_agent.clone(),
            device_id: fp.device_id.clone(),
            session_id: fp.session_id.clone(),
            fp,
            pow_script_sources: Vec::new(),
            pow_data_build: String::new(),
            client,
        })
    }

    fn build_base_headers(fp: &Fingerprint, base_url: &str, access_token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        let pairs: [(&str, &str); 24] = [
            ("User-Agent", &fp.user_agent),
            ("Origin", base_url),
            ("Referer", &format!("{base_url}/")),
            ("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7"),
            ("Cache-Control", "no-cache"),
            ("Pragma", "no-cache"),
            ("Priority", "u=1, i"),
            ("Sec-Ch-Ua", &fp.sec_ch_ua),
            ("Sec-Ch-Ua-Arch", "\"x86\""),
            ("Sec-Ch-Ua-Bitness", "\"64\""),
            ("Sec-Ch-Ua-Full-Version", "\"143.0.3650.96\""),
            (
                "Sec-Ch-Ua-Full-Version-List",
                "\"Microsoft Edge\";v=\"143.0.3650.96\", \"Chromium\";v=\"143.0.7499.147\", \"Not A(Brand\";v=\"24.0.0.0\"",
            ),
            ("Sec-Ch-Ua-Mobile", &fp.sec_ch_ua_mobile),
            ("Sec-Ch-Ua-Model", "\"\""),
            ("Sec-Ch-Ua-Platform", &fp.sec_ch_ua_platform),
            ("Sec-Ch-Ua-Platform-Version", "\"19.0.0\""),
            ("Sec-Fetch-Dest", "empty"),
            ("Sec-Fetch-Mode", "cors"),
            ("Sec-Fetch-Site", "same-origin"),
            ("OAI-Device-Id", &fp.device_id),
            ("OAI-Session-Id", &fp.session_id),
            ("OAI-Language", "zh-CN"),
            ("OAI-Client-Version", DEFAULT_CLIENT_VERSION),
            ("OAI-Client-Build-Number", DEFAULT_CLIENT_BUILD_NUMBER),
        ];
        for (k, v) in pairs {
            insert_header(&mut h, k, v);
        }
        if !access_token.is_empty() {
            insert_header(&mut h, "Authorization", &format!("Bearer {access_token}"));
        }
        h
    }

    /// Per-request header overrides: the web target path/route plus any extra
    /// entries. These are merged over the client's default headers.
    fn req_headers(&self, path: &str, extra: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        insert_header(&mut h, "X-OpenAI-Target-Path", path);
        insert_header(&mut h, "X-OpenAI-Target-Route", path);
        for (k, v) in extra {
            insert_header(&mut h, k, v);
        }
        h
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    // ---- error handling ----

    fn raise_on_error(response_status: u16, path: &str) -> EngineError {
        if response_status == 401 {
            EngineError::InvalidAccessToken(format!("token invalidated ({path})"))
        } else {
            EngineError::other(format!("{path} failed: HTTP {response_status}"))
        }
    }

    /// Mirror of `ensure_ok`: returns the response on 2xx, else reads the body
    /// and raises an [`UpstreamHttpError`].
    async fn ensure_ok(resp: Response, context: &str) -> EngineResult<Response> {
        let status = resp.status().as_u16();
        if (200..300).contains(&status) {
            return Ok(resp);
        }
        let retry_after = resp
            .headers()
            .get("Retry-After")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok());
        let text = resp.text().await.unwrap_or_default();
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Err(EngineError::Upstream(UpstreamHttpError::new(
            context, status, body, retry_after,
        )))
    }

    async fn get_json(&self, path: &str, extra: &[(&str, &str)], timeout: u64) -> EngineResult<Value> {
        let resp = self
            .client
            .get(self.url(path))
            .headers(self.req_headers(path, extra))
            .timeout(Duration::from_secs(timeout))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(Self::raise_on_error(status, path));
        }
        resp.json::<Value>().await.map_err(EngineError::from_wreq)
    }

    // ---- bootstrap + chat-requirements ----

    fn bootstrap_headers(&self) -> Vec<(&'static str, String)> {
        vec![
            ("User-Agent", self.user_agent.clone()),
            (
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
                    .to_string(),
            ),
            ("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8".to_string()),
            ("Sec-Ch-Ua", self.fp.sec_ch_ua.clone()),
            ("Sec-Ch-Ua-Mobile", self.fp.sec_ch_ua_mobile.clone()),
            ("Sec-Ch-Ua-Platform", self.fp.sec_ch_ua_platform.clone()),
            ("Sec-Fetch-Dest", "document".to_string()),
            ("Sec-Fetch-Mode", "navigate".to_string()),
            ("Sec-Fetch-Site", "none".to_string()),
            ("Sec-Fetch-User", "?1".to_string()),
            ("Upgrade-Insecure-Requests", "1".to_string()),
        ]
    }

    /// Warm up the home page and extract PoW script references.
    pub async fn bootstrap(&mut self) -> EngineResult<()> {
        let mut h = HeaderMap::new();
        for (k, v) in self.bootstrap_headers() {
            insert_header(&mut h, k, &v);
        }
        let resp = self
            .client
            .get(self.url("/"))
            .headers(h)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, "bootstrap").await?;
        let text = resp.text().await.map_err(EngineError::from_wreq)?;
        let (sources, data_build) = parse_pow_resources(&text);
        self.pow_script_sources = sources;
        self.pow_data_build = data_build;
        if self.pow_script_sources.is_empty() {
            self.pow_script_sources = vec![DEFAULT_POW_SCRIPT.to_string()];
        }
        Ok(())
    }

    fn build_requirements(&self, data: Value, source_p: &str) -> EngineResult<ChatRequirements> {
        if data
            .get("arkose")
            .and_then(|v| v.get("required"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return Err(EngineError::other(
                "chat requirements requires arkose token, which is not implemented",
            ));
        }

        let mut proof_token = String::new();
        if let Some(proof_info) = data.get("proofofwork").and_then(|v| v.as_object()) {
            if proof_info.get("required").and_then(|v| v.as_bool()).unwrap_or(false) {
                let seed = proof_info.get("seed").and_then(|v| v.as_str()).unwrap_or("");
                let difficulty = proof_info.get("difficulty").and_then(|v| v.as_str()).unwrap_or("");
                proof_token = build_proof_token(
                    seed,
                    difficulty,
                    &self.user_agent,
                    &self.pow_script_sources,
                    &self.pow_data_build,
                )
                .map_err(|e| EngineError::other(e.to_string()))?;
            }
        }

        let mut turnstile_token = String::new();
        if let Some(turnstile_info) = data.get("turnstile").and_then(|v| v.as_object()) {
            let required = turnstile_info.get("required").and_then(|v| v.as_bool()).unwrap_or(false);
            let dx = turnstile_info.get("dx").and_then(|v| v.as_str()).unwrap_or("");
            if required && !dx.is_empty() {
                turnstile_token = solve_turnstile_token(dx, source_p).unwrap_or_default();
            }
        }

        Ok(ChatRequirements {
            token: data.get("token").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            proof_token,
            turnstile_token,
            so_token: data.get("so_token").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            raw_finalize: data,
        })
    }

    /// Acquire sentinel tokens for the current (auth/anon) mode.
    pub async fn get_chat_requirements(&self) -> EngineResult<ChatRequirements> {
        let path = if self.access_token.is_empty() {
            "/backend-anon/sentinel/chat-requirements"
        } else {
            "/backend-api/sentinel/chat-requirements"
        };
        let context = if self.access_token.is_empty() {
            "noauth_chat_requirements"
        } else {
            "auth_chat_requirements"
        };
        let p = build_legacy_requirements_token(&self.user_agent, &self.pow_script_sources, &self.pow_data_build);
        let body = json!({ "p": p });
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.req_headers(path, &[("Content-Type", "application/json")]))
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, context).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        let source_p = if self.access_token.is_empty() { p.as_str() } else { "" };
        let requirements = self.build_requirements(data, source_p)?;
        if requirements.token.is_empty() {
            let message = if self.access_token.is_empty() {
                "missing chat requirements token"
            } else {
                "missing auth chat requirements token"
            };
            return Err(EngineError::other(format!(
                "{message}: {}",
                requirements.raw_finalize
            )));
        }
        Ok(requirements)
    }

    fn chat_target(&self) -> (&'static str, &'static str) {
        if self.access_token.is_empty() {
            ("/backend-anon/conversation", "America/Los_Angeles")
        } else {
            ("/backend-api/conversation", "Asia/Shanghai")
        }
    }

    fn conversation_headers(&self, path: &str, requirements: &ChatRequirements) -> HeaderMap {
        let mut extra: Vec<(&str, &str)> = vec![
            ("Accept", "text/event-stream"),
            ("Content-Type", "application/json"),
            ("OpenAI-Sentinel-Chat-Requirements-Token", &requirements.token),
        ];
        if !requirements.proof_token.is_empty() {
            extra.push(("OpenAI-Sentinel-Proof-Token", &requirements.proof_token));
        }
        if !requirements.turnstile_token.is_empty() {
            extra.push(("OpenAI-Sentinel-Turnstile-Token", &requirements.turnstile_token));
        }
        if !requirements.so_token.is_empty() {
            extra.push(("OpenAI-Sentinel-SO-Token", &requirements.so_token));
        }
        self.req_headers(path, &extra)
    }

    // ---- conversation payload ----

    fn api_messages_to_conversation_messages(&self, messages: &[Value]) -> EngineResult<Vec<Value>> {
        let mut out = Vec::new();
        for item in messages {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = item.get("content").cloned().unwrap_or(Value::Null);
            if let Some(text) = content.as_str() {
                out.push(json!({
                    "id": new_uuid(),
                    "author": {"role": role},
                    "content": {"content_type": "text", "parts": [text]},
                }));
                continue;
            }
            let Some(parts_arr) = content.as_array() else {
                return Err(EngineError::other("only string or list message content is supported"));
            };
            let mut text_parts: Vec<String> = Vec::new();
            // NB: binary image inputs in conversation messages require an upload
            // round-trip (handled in the image flow). Plain chat passes text only.
            for part in parts_arr {
                let Some(obj) = part.as_object() else { continue };
                let part_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if part_type == "text" {
                    text_parts.push(obj.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string());
                }
            }
            out.push(json!({
                "id": new_uuid(),
                "author": {"role": role},
                "content": {"content_type": "text", "parts": [text_parts.join("")]},
            }));
        }
        Ok(out)
    }

    fn conversation_payload(&self, messages: &[Value], model: &str, timezone: &str) -> EngineResult<Value> {
        Ok(json!({
            "action": "next",
            "messages": self.api_messages_to_conversation_messages(messages)?,
            "model": model,
            "parent_message_id": new_uuid(),
            "conversation_mode": {"kind": "primary_assistant"},
            "conversation_origin": Value::Null,
            "force_paragen": false,
            "force_paragen_model_slug": "",
            "force_rate_limit": false,
            "force_use_sse": true,
            "history_and_training_disabled": true,
            "reset_rate_limits": false,
            "suggestions": [],
            "supported_encodings": [],
            "system_hints": [],
            "timezone": timezone,
            "timezone_offset_min": -480,
            "variant_purpose": "comparison_implicit",
            "websocket_request_id": new_uuid(),
            "client_contextual_info": {
                "is_dark_mode": false,
                "time_since_loaded": 120,
                "page_height": 900,
                "page_width": 1400,
                "pixel_ratio": 2,
                "screen_height": 1440,
                "screen_width": 2560,
            },
        }))
    }

    /// Stream a plain (text) conversation. Returns an [`SseStream`].
    pub async fn stream_conversation(
        &mut self,
        messages: Option<Vec<Value>>,
        model: &str,
        prompt: &str,
    ) -> EngineResult<SseStream> {
        let normalized = messages.unwrap_or_else(|| vec![json!({"role": "user", "content": prompt})]);
        self.bootstrap().await?;
        let requirements = self.get_chat_requirements().await?;
        let (path, timezone) = self.chat_target();
        let payload = self.conversation_payload(&normalized, model, timezone)?;
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.conversation_headers(path, &requirements))
            .json(&payload)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, path).await?;
        Ok(SseStream::new(resp))
    }

    // ---- models ----

    /// List available models (OpenAI `/v1/models` shape).
    pub async fn list_models(&mut self) -> EngineResult<Value> {
        self.bootstrap().await?;
        let (path, route, context) = if self.access_token.is_empty() {
            (
                "/backend-anon/models?iim=false&is_gizmo=false",
                "/backend-anon/models",
                "anon_models",
            )
        } else {
            (
                "/backend-api/models?history_and_training_disabled=false",
                "/backend-api/models",
                "auth_models",
            )
        };
        let resp = self
            .client
            .get(self.url(path))
            .headers(self.req_headers(route, &[]))
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, context).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        let mut out: Vec<Value> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        if let Some(models) = data.get("models").and_then(|v| v.as_array()) {
            for item in models {
                let Some(obj) = item.as_object() else { continue };
                let slug = obj.get("slug").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                if slug.is_empty() || !seen.insert(slug.clone()) {
                    continue;
                }
                out.push(json!({
                    "id": slug,
                    "object": "model",
                    "created": obj.get("created").and_then(|v| v.as_i64()).unwrap_or(0),
                    "owned_by": obj.get("owned_by").and_then(|v| v.as_str()).unwrap_or("chatgpt"),
                    "permission": [],
                    "root": slug,
                    "parent": Value::Null,
                }));
            }
        }
        out.sort_by(|a, b| {
            a.get("id").and_then(|v| v.as_str()).unwrap_or("")
                .cmp(b.get("id").and_then(|v| v.as_str()).unwrap_or(""))
        });
        Ok(json!({ "object": "list", "data": out }))
    }

    // ---- user info ----

    async fn get_me(&self) -> EngineResult<Value> {
        self.get_json("/backend-api/me", &[], 20).await
    }

    async fn get_conversation_init(&self) -> EngineResult<Value> {
        let path = "/backend-api/conversation/init";
        let body = json!({
            "gizmo_id": Value::Null,
            "requested_default_model": Value::Null,
            "conversation_id": Value::Null,
            "timezone_offset_min": -480,
        });
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.req_headers(path, &[("Content-Type", "application/json")]))
            .json(&body)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(Self::raise_on_error(status, path));
        }
        resp.json::<Value>().await.map_err(EngineError::from_wreq)
    }

    async fn get_default_account(&self) -> EngineResult<Value> {
        let path = "/backend-api/accounts/check/v4-2023-04-27";
        let full = format!("{path}?timezone_offset_min=-480");
        let resp = self
            .client
            .get(self.url(&full))
            .headers(self.req_headers(path, &[]))
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(Self::raise_on_error(status, path));
        }
        let payload = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        let default_account = payload
            .get("accounts")
            .and_then(|v| v.get("default"))
            .and_then(|v| v.get("account"))
            .cloned()
            .unwrap_or(json!({}));
        Ok(default_account)
    }

    fn extract_quota_and_restore_at(limits_progress: &[Value]) -> (i64, Option<String>, bool) {
        for item in limits_progress {
            if item.get("feature_name").and_then(|v| v.as_str()) == Some("image_gen") {
                let remaining = item.get("remaining").and_then(|v| v.as_i64()).unwrap_or(0);
                let reset_after = item
                    .get("reset_after")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(String::from);
                return (remaining, reset_after, false);
            }
        }
        (0, None, true)
    }

    /// Fetch account info for the current token (runs the three upstream calls
    /// concurrently, mirroring the Python `ThreadPoolExecutor`).
    pub async fn get_user_info(&self) -> EngineResult<Value> {
        if self.access_token.is_empty() {
            return Err(EngineError::other("access_token is required"));
        }
        let (me, init, account) =
            tokio::join!(self.get_me(), self.get_conversation_init(), self.get_default_account());
        let me_payload = me?;
        let init_payload = init?;
        let default_account = account?;

        let plan_type = default_account
            .get("plan_type")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("free")
            .to_string();

        let limits_progress: Vec<Value> = init_payload
            .get("limits_progress")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let (quota, restore_at, image_quota_unknown) = Self::extract_quota_and_restore_at(&limits_progress);

        let status = if image_quota_unknown && plan_type.to_lowercase() != "free" {
            "正常"
        } else if quota == 0 {
            "限流"
        } else {
            "正常"
        };

        Ok(json!({
            "email": me_payload.get("email").cloned().unwrap_or(Value::Null),
            "user_id": me_payload.get("id").cloned().unwrap_or(Value::Null),
            "type": plan_type,
            "quota": quota,
            "image_quota_unknown": image_quota_unknown,
            "limits_progress": limits_progress,
            "default_model_slug": init_payload.get("default_model_slug").cloned().unwrap_or(Value::Null),
            "restore_at": restore_at,
            "status": status,
        }))
    }

    // ---- accessors ----

    pub fn access_token(&self) -> &str {
        &self.access_token
    }

    pub fn account(&self) -> &Value {
        &self.account
    }
}

// Image generation, codex, search and editable-file flows live in submodules.
mod image;
