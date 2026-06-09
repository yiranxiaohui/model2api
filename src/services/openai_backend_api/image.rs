//! Image-generation, codex, and search flows for [`OpenAIBackendAPI`]
//! (port of the corresponding sections of `openai_backend_api.py`).
//!
//! This is a child module so it can extend the engine via `impl super::...` and
//! reach the parent's private fields.

use std::path::Path as FsPath;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use wreq::header::HeaderMap;
use wreq::Response;

use super::{
    insert_header, is_content_policy_error, EngineError, EngineResult, OpenAIBackendAPI, SseStream,
    CODEX_IMAGE_MODEL, CODEX_RESPONSES_INSTRUCTIONS, CODEX_RESPONSES_MODEL, SEARCH_MODEL,
    SEARCH_POLL_INTERVAL_SECS, SEARCH_TIMEOUT_SECS,
};
use crate::utils::helper::{new_uuid, split_image_model};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;

static FILE_SERVICE_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"file-service://([A-Za-z0-9_-]+)").unwrap());
static REAL_IMAGE_FILE_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bfile_00000000[a-f0-9]{24}\b").unwrap());
static SEDIMENT_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"sediment://([A-Za-z0-9_-]+)").unwrap());
static SEARCH_CONVERSATION_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#""conversation_id"\s*:\s*"([^"]+)""#).unwrap());
static SEARCH_URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"https?://[^\s"'<>）)\]}]+"#).unwrap());

const SEARCH_DONE_STATUS: &[&str] = &["finished_successfully", "finished_partial_completion"];

fn now_epoch() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}

/// Detect an image's mime type from its magic bytes (header-only).
fn detect_mime(data: &[u8]) -> &'static str {
    match imagesize::image_type(data) {
        Ok(imagesize::ImageType::Png) => "image/png",
        Ok(imagesize::ImageType::Jpeg) => "image/jpeg",
        Ok(imagesize::ImageType::Gif) => "image/gif",
        Ok(imagesize::ImageType::Webp) => "image/webp",
        Ok(imagesize::ImageType::Bmp) => "image/bmp",
        _ => "image/png",
    }
}

fn image_dims(data: &[u8]) -> (i64, i64) {
    match imagesize::blob_size(data) {
        Ok(sz) => (sz.width as i64, sz.height as i64),
        Err(_) => (0, 0),
    }
}

fn add_unique(values: &mut Vec<String>, candidates: impl IntoIterator<Item = String>) {
    for c in candidates {
        if !c.is_empty() && !values.contains(&c) {
            values.push(c);
        }
    }
}

impl OpenAIBackendAPI {
    fn image_model_slug(&self, model: &str) -> String {
        let (_, base_model) = split_image_model(&Value::String(model.to_string()));
        match base_model.as_deref() {
            None => "auto".to_string(),
            Some("gpt-image-2") => "gpt-5-3".to_string(),
            Some(CODEX_IMAGE_MODEL) => CODEX_IMAGE_MODEL.to_string(),
            Some(_) => "auto".to_string(),
        }
    }

    fn image_headers(
        &self,
        path: &str,
        requirements: &super::ChatRequirements,
        conduit_token: &str,
        accept: &str,
    ) -> HeaderMap {
        let turn_trace = new_uuid();
        let mut extra: Vec<(&str, &str)> = vec![
            ("Content-Type", "application/json"),
            ("Accept", accept),
            ("OpenAI-Sentinel-Chat-Requirements-Token", &requirements.token),
        ];
        if !requirements.proof_token.is_empty() {
            extra.push(("OpenAI-Sentinel-Proof-Token", &requirements.proof_token));
        }
        if !conduit_token.is_empty() {
            extra.push(("X-Conduit-Token", conduit_token));
        }
        if accept == "text/event-stream" {
            extra.push(("X-Oai-Turn-Trace-Id", &turn_trace));
        }
        self.req_headers(path, &extra)
    }

    fn decode_image_base64(&self, image: &str) -> EngineResult<Vec<u8>> {
        // Local file path heuristic (mirrors the Python check).
        if !image.is_empty()
            && image.len() < 512
            && !image.starts_with("data:")
            && !image.contains('\n')
            && !image.contains('\r')
        {
            let p = FsPath::new(image);
            if p.is_file() {
                return std::fs::read(p).map_err(|e| EngineError::other(format!("read file: {e}")));
            }
        }
        let payload = if image.starts_with("data:") && image.contains(',') {
            image.splitn(2, ',').nth(1).unwrap_or("")
        } else {
            image
        };
        BASE64_STANDARD
            .decode(payload.trim())
            .map_err(|e| EngineError::other(format!("invalid base64 image: {e}")))
    }

    /// Upload one base64 image and return its backend file metadata.
    async fn upload_image(&self, image: &str, file_name: &str) -> EngineResult<Value> {
        let data = self.decode_image_base64(image)?;
        let mut file_name = file_name.to_string();
        if !image.is_empty()
            && image.len() < 512
            && !image.starts_with("data:")
            && !image.contains('\n')
            && !image.contains('\r')
        {
            let p = FsPath::new(image);
            if p.is_file() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    file_name = name.to_string();
                }
            }
        }
        let (width, height) = image_dims(&data);
        let mime_type = detect_mime(&data);

        let path = "/backend-api/files";
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.req_headers(path, &[("Content-Type", "application/json"), ("Accept", "application/json")]))
            .json(&json!({
                "file_name": file_name,
                "file_size": data.len(),
                "use_case": "multimodal",
                "width": width,
                "height": height,
            }))
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, path).await?;
        let upload_meta = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        let upload_url = upload_meta.get("upload_url").and_then(|v| v.as_str()).unwrap_or("");
        let file_id = upload_meta.get("file_id").and_then(|v| v.as_str()).unwrap_or("");
        if upload_url.is_empty() || file_id.is_empty() {
            return Err(EngineError::other(format!("invalid upload response: {upload_meta}")));
        }

        // Azure blob PUT.
        let mut put_headers = HeaderMap::new();
        for (k, v) in [
            ("Content-Type", mime_type),
            ("x-ms-blob-type", "BlockBlob"),
            ("x-ms-version", "2020-04-08"),
            ("Origin", self.base_url.as_str()),
            ("Accept", "application/json, text/plain, */*"),
            ("Accept-Language", "en-US,en;q=0.8"),
            ("User-Agent", self.user_agent.as_str()),
        ] {
            insert_header(&mut put_headers, k, v);
        }
        insert_header(&mut put_headers, "Referer", &format!("{}/", self.base_url));
        let resp = self
            .client
            .put(upload_url)
            .headers(put_headers)
            .body(data.clone())
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        Self::ensure_ok(resp, "image_upload").await?;

        let path = format!("/backend-api/files/{file_id}/uploaded");
        let resp = self
            .client
            .post(self.url(&path))
            .headers(self.req_headers(&path, &[("Content-Type", "application/json"), ("Accept", "application/json")]))
            .body("{}")
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        Self::ensure_ok(resp, &path).await?;

        Ok(json!({
            "file_id": file_id,
            "file_name": file_name,
            "file_size": data.len(),
            "mime_type": mime_type,
            "width": width,
            "height": height,
        }))
    }

    async fn prepare_image_conversation(
        &self,
        prompt: &str,
        requirements: &super::ChatRequirements,
        model: &str,
    ) -> EngineResult<String> {
        let path = "/backend-api/f/conversation/prepare";
        let payload = json!({
            "action": "next",
            "fork_from_shared_post": false,
            "parent_message_id": new_uuid(),
            "model": self.image_model_slug(model),
            "client_prepare_state": "success",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": ["picture_v2"],
            "partial_query": {
                "id": new_uuid(),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
            },
            "supports_buffering": true,
            "supported_encodings": ["v1"],
            "client_contextual_info": {"app_name": "chatgpt.com"},
        });
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.image_headers(path, requirements, "", "*/*"))
            .json(&payload)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, path).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        Ok(data.get("conduit_token").and_then(|v| v.as_str()).unwrap_or("").to_string())
    }

    async fn start_image_generation(
        &self,
        prompt: &str,
        requirements: &super::ChatRequirements,
        conduit_token: &str,
        model: &str,
        references: &[Value],
    ) -> EngineResult<Response> {
        let mut parts: Vec<Value> = references
            .iter()
            .map(|item| {
                json!({
                    "content_type": "image_asset_pointer",
                    "asset_pointer": format!("file-service://{}", item.get("file_id").and_then(|v| v.as_str()).unwrap_or("")),
                    "width": item.get("width").cloned().unwrap_or(json!(0)),
                    "height": item.get("height").cloned().unwrap_or(json!(0)),
                    "size_bytes": item.get("file_size").cloned().unwrap_or(json!(0)),
                })
            })
            .collect();
        parts.push(Value::String(prompt.to_string()));
        let content = if references.is_empty() {
            json!({"content_type": "text", "parts": [prompt]})
        } else {
            json!({"content_type": "multimodal_text", "parts": parts})
        };
        let mut metadata = json!({
            "developer_mode_connector_ids": [],
            "selected_github_repos": [],
            "selected_all_github_repos": false,
            "system_hints": ["picture_v2"],
            "serialization_metadata": {"custom_symbol_offsets": []},
        });
        if !references.is_empty() {
            let attachments: Vec<Value> = references
                .iter()
                .map(|item| {
                    json!({
                        "id": item.get("file_id").cloned().unwrap_or(Value::Null),
                        "mimeType": item.get("mime_type").cloned().unwrap_or(Value::Null),
                        "name": item.get("file_name").cloned().unwrap_or(Value::Null),
                        "size": item.get("file_size").cloned().unwrap_or(Value::Null),
                        "width": item.get("width").cloned().unwrap_or(Value::Null),
                        "height": item.get("height").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect();
            metadata["attachments"] = Value::Array(attachments);
        }
        let payload = json!({
            "action": "next",
            "messages": [{
                "id": new_uuid(),
                "author": {"role": "user"},
                "create_time": now_epoch(),
                "content": content,
                "metadata": metadata,
            }],
            "parent_message_id": new_uuid(),
            "model": self.image_model_slug(model),
            "client_prepare_state": "sent",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": true,
            "system_hints": ["picture_v2"],
            "supports_buffering": true,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "is_dark_mode": false,
                "time_since_loaded": 1200,
                "page_height": 1072,
                "page_width": 1724,
                "pixel_ratio": 1.2,
                "screen_height": 1440,
                "screen_width": 2560,
                "app_name": "chatgpt.com",
            },
            "paragen_cot_summary_display_override": "allow",
            "force_parallel_switch": "auto",
        });
        let path = "/backend-api/f/conversation";
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.image_headers(path, requirements, conduit_token, "text/event-stream"))
            .json(&payload)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        Self::ensure_ok(resp, path).await
    }

    /// The `picture_v2` streaming flow: upload references, bootstrap, get
    /// requirements, prepare, then return the generation SSE stream.
    pub async fn stream_picture_conversation(
        &mut self,
        prompt: &str,
        model: &str,
        images: &[String],
    ) -> EngineResult<SseStream> {
        if self.access_token.is_empty() {
            return Err(EngineError::other("access_token is required for image endpoints"));
        }
        let mut references = Vec::new();
        for (idx, image) in images.iter().enumerate() {
            references.push(self.upload_image(image, &format!("image_{}.png", idx + 1)).await?);
        }
        self.bootstrap().await?;
        let requirements = self.get_chat_requirements().await?;
        let conduit_token = self.prepare_image_conversation(prompt, &requirements, model).await?;
        let resp = self
            .start_image_generation(prompt, &requirements, &conduit_token, model, &references)
            .await?;
        Ok(SseStream::new(resp))
    }

    // ---- conversation document + tasks ----

    async fn get_conversation(&self, conversation_id: &str) -> EngineResult<Value> {
        let path = format!("/backend-api/conversation/{conversation_id}");
        let resp = self
            .client
            .get(self.url(&path))
            .headers(self.req_headers(&path, &[("Accept", "application/json")]))
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, &path).await?;
        resp.json::<Value>().await.map_err(EngineError::from_wreq)
    }

    async fn query_backend_tasks(&self, conversation_id: &str, timeout_secs: u64) -> EngineResult<Vec<Value>> {
        let path = "/backend-api/tasks";
        let resp = self
            .client
            .get(self.url(path))
            .headers(self.req_headers(path, &[("Accept", "application/json")]))
            .timeout(Duration::from_secs(timeout_secs))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, path).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        let Some(tasks) = data.get("tasks").and_then(|v| v.as_array()) else {
            return Ok(vec![]);
        };
        let mut out: Vec<Value> = Vec::new();
        for t in tasks {
            if !conversation_id.is_empty() {
                let cid = t.get("conversation_id").and_then(|v| v.as_str());
                let ocid = t.get("original_conversation_id").and_then(|v| v.as_str());
                if cid != Some(conversation_id) && ocid != Some(conversation_id) {
                    continue;
                }
            }
            out.push(t.clone());
        }
        Ok(out)
    }

    /// Inspect a task for a structured error. Returns `(is_error, error_msg)`.
    fn check_task_error(task: &Value) -> (bool, String) {
        let Some(img_msg) = task.get("image_gen_message").filter(|v| v.is_object()) else {
            return (false, String::new());
        };
        let metadata = img_msg.get("metadata").cloned().unwrap_or(json!({}));
        let content = img_msg.get("content").cloned().unwrap_or(json!({}));
        let is_error = metadata.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
        let is_text_only = content.get("content_type").and_then(|v| v.as_str()) == Some("text");
        let mut error_msg = String::new();
        if is_error && is_text_only {
            if let Some(parts) = content.get("parts").and_then(|v| v.as_array()) {
                error_msg = parts.iter().filter_map(|p| p.as_str()).collect::<String>();
            }
        }
        (is_error, error_msg)
    }

    fn extract_image_reference_ids(payload: &Value) -> (Vec<String>, Vec<String>) {
        let mut file_ids = Vec::new();
        let mut sediment_ids = Vec::new();
        fn walk(value: &Value, file_ids: &mut Vec<String>, sediment_ids: &mut Vec<String>) {
            match value {
                Value::String(s) => {
                    add_unique(file_ids, FILE_SERVICE_ID_RE.captures_iter(s).map(|c| c[1].to_string()));
                    add_unique(file_ids, REAL_IMAGE_FILE_ID_RE.find_iter(s).map(|m| m.as_str().to_string()));
                    add_unique(sediment_ids, SEDIMENT_ID_RE.captures_iter(s).map(|c| c[1].to_string()));
                }
                Value::Object(m) => {
                    for v in m.values() {
                        walk(v, file_ids, sediment_ids);
                    }
                }
                Value::Array(a) => {
                    for v in a {
                        walk(v, file_ids, sediment_ids);
                    }
                }
                _ => {}
            }
        }
        walk(payload, &mut file_ids, &mut sediment_ids);
        (file_ids, sediment_ids)
    }

    fn has_image_asset_pointer(payload: &Value) -> bool {
        match payload {
            Value::Object(m) => {
                if m.get("content_type").and_then(|v| v.as_str()) == Some("image_asset_pointer") {
                    return true;
                }
                if let Some(ap) = m.get("asset_pointer").and_then(|v| v.as_str()) {
                    if ap.starts_with("file-service://") || ap.starts_with("sediment://") {
                        return true;
                    }
                }
                m.values().any(Self::has_image_asset_pointer)
            }
            Value::Array(a) => a.iter().any(Self::has_image_asset_pointer),
            _ => false,
        }
    }

    fn extract_image_tool_records(data: &Value) -> Vec<Value> {
        let mut records: Vec<Value> = Vec::new();
        let Some(mapping) = data.get("mapping").and_then(|v| v.as_object()) else {
            return records;
        };
        for (message_id, node) in mapping {
            let message = node.get("message").cloned().unwrap_or(json!({}));
            let role = message
                .get("author")
                .and_then(|a| a.get("role"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_lowercase();
            if role != "tool" && role != "assistant" {
                continue;
            }
            let metadata = message.get("metadata").cloned().unwrap_or(json!({}));
            let content = message.get("content").cloned().unwrap_or(json!({}));
            let is_image_gen = metadata.get("async_task_type").and_then(|v| v.as_str()) == Some("image_gen");
            let has_asset_pointer =
                Self::has_image_asset_pointer(&content) || Self::has_image_asset_pointer(&metadata);
            if role == "assistant" && !(is_image_gen || has_asset_pointer) {
                continue;
            }
            let (file_ids, sediment_ids) =
                Self::extract_image_reference_ids(&json!({"content": content, "metadata": metadata}));
            if !is_image_gen && !has_asset_pointer && file_ids.is_empty() && sediment_ids.is_empty() {
                continue;
            }
            records.push(json!({
                "message_id": message_id,
                "create_time": message.get("create_time").cloned().unwrap_or(json!(0)),
                "file_ids": file_ids,
                "sediment_ids": sediment_ids,
            }));
        }
        records.sort_by(|a, b| {
            let ca = a.get("create_time").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let cb = b.get("create_time").and_then(|v| v.as_f64()).unwrap_or(0.0);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        records
    }

    fn find_content_policy_error_in_conversation(data: &Value) -> String {
        let Some(mapping) = data.get("mapping").and_then(|v| v.as_object()) else {
            return String::new();
        };
        for node in mapping.values() {
            let message = node.get("message").cloned().unwrap_or(json!({}));
            let role = message
                .get("author")
                .and_then(|a| a.get("role"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_lowercase();
            if role != "assistant" && role != "tool" {
                continue;
            }
            let content = message.get("content").cloned().unwrap_or(json!({}));
            let mut text_parts: Vec<String> = Vec::new();
            if let Some(obj) = content.as_object() {
                if let Some(parts) = obj.get("parts").and_then(|v| v.as_array()) {
                    for part in parts {
                        if let Some(s) = part.as_str() {
                            if !s.trim().is_empty() {
                                text_parts.push(s.trim().to_string());
                            }
                        }
                    }
                }
                if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                    if !t.trim().is_empty() {
                        text_parts.push(t.trim().to_string());
                    }
                }
            } else if let Some(s) = content.as_str() {
                if !s.trim().is_empty() {
                    text_parts.push(s.trim().to_string());
                }
            }
            let msg_text = text_parts.join("\n");
            if !msg_text.is_empty() && is_content_policy_error(&msg_text) {
                return msg_text.chars().take(500).collect();
            }
        }
        String::new()
    }

    // ---- polling ----

    async fn poll_image_results(
        &self,
        conversation_id: &str,
        timeout_secs: f64,
        initial_file_ids: &[String],
        initial_sediment_ids: &[String],
    ) -> EngineResult<(Vec<String>, Vec<String>)> {
        let start = Instant::now();
        let mut attempt = 0u32;
        let interval = self.config.image_poll_interval_secs();
        let initial_wait = self.config.image_poll_initial_wait_secs();
        let mut file_ids: Vec<String> = Vec::new();
        let mut sediment_ids: Vec<String> = Vec::new();
        add_unique(&mut file_ids, initial_file_ids.iter().cloned());
        add_unique(&mut sediment_ids, initial_sediment_ids.iter().cloned());
        let has_initial_ids = !file_ids.is_empty() || !sediment_ids.is_empty();
        let mut last_hit_key: Option<(Vec<String>, Vec<String>)> = if has_initial_ids {
            Some((file_ids.clone(), sediment_ids.clone()))
        } else {
            None
        };

        let remaining = |start: Instant| timeout_secs - start.elapsed().as_secs_f64();

        if has_initial_ids && self.config.image_settle_enabled() {
            let settle_for = self.config.image_settle_secs().min(remaining(start).max(0.0));
            if settle_for > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(settle_for)).await;
            }
        } else if initial_wait > 0.0 {
            let jitter = rand::random::<f64>() * (2.0_f64).min(initial_wait * 0.2);
            let sleep_for = (initial_wait + jitter).min(remaining(start).max(0.0));
            if sleep_for > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(sleep_for)).await;
            }
        }

        let mut last_task_error = String::new();
        while remaining(start) > 0.0 {
            attempt += 1;
            // Best-effort task error scan (non-blocking).
            last_task_error.clear();
            if let Ok(tasks) = self.query_backend_tasks(conversation_id, 5).await {
                for task in &tasks {
                    let (is_error, error_msg) = Self::check_task_error(task);
                    if is_error && !error_msg.is_empty() {
                        last_task_error = error_msg;
                    }
                }
            }

            let conversation = match self.get_conversation(conversation_id).await {
                Ok(c) => c,
                Err(EngineError::Upstream(e)) if matches!(e.status_code, 429 | 500 | 502 | 503 | 504) => {
                    let base = e.retry_after.map(|r| r as f64).unwrap_or_else(|| {
                        (2u64.pow(attempt.min(4)) as f64).min(16.0)
                    });
                    let backoff = base + rand::random::<f64>() * 0.5;
                    let rem = remaining(start);
                    if rem <= 0.0 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_secs_f64(backoff.min(rem))).await;
                    continue;
                }
                Err(e) => return Err(e),
            };

            for record in Self::extract_image_tool_records(&conversation) {
                if let Some(fids) = record.get("file_ids").and_then(|v| v.as_array()) {
                    for fid in fids {
                        if let Some(s) = fid.as_str() {
                            if !file_ids.iter().any(|x| x == s) {
                                file_ids.push(s.to_string());
                            }
                        }
                    }
                }
                if let Some(sids) = record.get("sediment_ids").and_then(|v| v.as_array()) {
                    for sid in sids {
                        if let Some(s) = sid.as_str() {
                            if !sediment_ids.iter().any(|x| x == s) {
                                sediment_ids.push(s.to_string());
                            }
                        }
                    }
                }
            }

            if file_ids.is_empty() && sediment_ids.is_empty() {
                let policy_msg = Self::find_content_policy_error_in_conversation(&conversation);
                if !policy_msg.is_empty() {
                    return Err(EngineError::ImageContentPolicy(policy_msg));
                }
            }

            if !file_ids.is_empty() || !sediment_ids.is_empty() {
                if !self.config.image_check_before_hit_enabled() {
                    return Ok((file_ids, sediment_ids));
                }
                let hit_key = (file_ids.clone(), sediment_ids.clone());
                if last_hit_key.as_ref() == Some(&hit_key) {
                    return Ok((file_ids, sediment_ids));
                }
                last_hit_key = Some(hit_key);
                if !self.config.image_settle_enabled() {
                    return Ok((file_ids, sediment_ids));
                }
                let wait = self.config.image_settle_secs().min(remaining(start).max(0.0));
                if wait > 0.0 {
                    tokio::time::sleep(Duration::from_secs_f64(wait)).await;
                    continue;
                }
                return Ok((file_ids, sediment_ids));
            }

            let wait = interval.min(remaining(start).max(0.0));
            if wait > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            }
        }

        Err(EngineError::ImagePollTimeout {
            message: format!(
                "ChatGPT 生图超时（已等待 {timeout_secs} 秒）。当前超时阈值可在 config.json 中调大 image_poll_timeout_secs，也可能是账号被限流或生图队列拥堵导致。"
            ),
            conversation_id: conversation_id.to_string(),
            task_error: if last_task_error.is_empty() { None } else { Some(last_task_error) },
        })
    }

    async fn get_file_download_url(&self, file_id: &str) -> EngineResult<String> {
        let path = format!("/backend-api/files/{file_id}/download");
        let resp = self
            .client
            .get(self.url(&path))
            .headers(self.req_headers(&path, &[("Accept", "application/json")]))
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, &path).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        Ok(data
            .get("download_url")
            .or_else(|| data.get("url"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string())
    }

    async fn get_attachment_download_url(&self, conversation_id: &str, attachment_id: &str) -> EngineResult<String> {
        let path = format!("/backend-api/conversation/{conversation_id}/attachment/{attachment_id}/download");
        let resp = self
            .client
            .get(self.url(&path))
            .headers(self.req_headers(&path, &[("Accept", "application/json")]))
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, &path).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        Ok(data
            .get("download_url")
            .or_else(|| data.get("url"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string())
    }

    async fn resolve_image_urls(
        &self,
        conversation_id: &str,
        file_ids: &[String],
        sediment_ids: &[String],
    ) -> Vec<String> {
        let mut urls: Vec<String> = Vec::new();
        for file_id in file_ids {
            if file_id == "file_upload" {
                continue;
            }
            if let Ok(url) = self.get_file_download_url(file_id).await {
                if !url.is_empty() && !urls.contains(&url) {
                    urls.push(url);
                }
            }
        }
        if conversation_id.is_empty() || sediment_ids.is_empty() {
            return urls;
        }
        for sediment_id in sediment_ids {
            if let Ok(url) = self.get_attachment_download_url(conversation_id, sediment_id).await {
                if !url.is_empty() && !urls.contains(&url) {
                    urls.push(url);
                }
            }
        }
        urls
    }

    /// Resolve image result ids into downloadable URLs, polling if requested.
    pub async fn resolve_conversation_image_urls(
        &self,
        conversation_id: &str,
        file_ids: Vec<String>,
        sediment_ids: Vec<String>,
        poll: bool,
        poll_timeout_secs: Option<f64>,
    ) -> EngineResult<Vec<String>> {
        let mut file_ids: Vec<String> = file_ids.into_iter().filter(|x| x != "file_upload").collect();
        let mut sediment_ids = sediment_ids;
        let timeout = poll_timeout_secs.unwrap_or_else(|| self.config.image_poll_timeout_secs() as f64);
        let have_ids = !file_ids.is_empty() || !sediment_ids.is_empty();

        if poll && !conversation_id.is_empty() && have_ids {
            if !self.config.image_check_before_hit_enabled() && !self.config.image_settle_enabled() {
                return Ok(self.resolve_image_urls(conversation_id, &file_ids, &sediment_ids).await);
            }
        }
        if poll && !conversation_id.is_empty() {
            match self
                .poll_image_results(conversation_id, timeout, &file_ids, &sediment_ids)
                .await
            {
                Ok((pf, ps)) => {
                    for item in pf {
                        if !item.is_empty() && !file_ids.contains(&item) {
                            file_ids.push(item);
                        }
                    }
                    for item in ps {
                        if !item.is_empty() && !sediment_ids.contains(&item) {
                            sediment_ids.push(item);
                        }
                    }
                }
                Err(EngineError::ImagePollTimeout { task_error, .. }) if !have_ids => {
                    if let Some(te) = task_error {
                        return Err(EngineError::ImageContentPolicy(te));
                    }
                    return Err(EngineError::ImagePollTimeout {
                        message: "image poll timeout".to_string(),
                        conversation_id: conversation_id.to_string(),
                        task_error: None,
                    });
                }
                Err(EngineError::ImageContentPolicy(m)) if !have_ids => {
                    return Err(EngineError::ImageContentPolicy(m));
                }
                Err(e) if !have_ids => return Err(e),
                Err(_) => { /* partial: keep what we already have */ }
            }
        }
        Ok(self.resolve_image_urls(conversation_id, &file_ids, &sediment_ids).await)
    }

    /// Download raw image bytes for each URL (deduped).
    pub async fn download_image_bytes(&self, urls: &[String]) -> EngineResult<Vec<Vec<u8>>> {
        let mut images: Vec<Vec<u8>> = Vec::new();
        for url in urls {
            let resp = self
                .client
                .get(url)
                .timeout(Duration::from_secs(120))
                .send()
                .await
                .map_err(EngineError::from_wreq)?;
            let resp = Self::ensure_ok(resp, "image_download").await?;
            let bytes = resp.bytes().await.map_err(EngineError::from_wreq)?.to_vec();
            if !images.contains(&bytes) {
                images.push(bytes);
            }
        }
        Ok(images)
    }

    // ---- codex ----

    fn codex_responses_headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        insert_header(&mut h, "Authorization", &format!("Bearer {}", self.access_token));
        insert_header(&mut h, "Content-Type", "application/json");
        h
    }

    fn codex_image_input(prompt: &str, images: &[String]) -> Vec<Value> {
        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        for image in images {
            let payload = if image.starts_with("data:image/") {
                image.clone()
            } else {
                format!("data:image/png;base64,{image}")
            };
            content.push(json!({"type": "input_image", "image_url": payload}));
        }
        vec![json!({"role": "user", "content": content})]
    }

    /// Codex image responses — returns the raw event list (parsed from the
    /// JSON or SSE response body).
    pub async fn codex_image_response_events(
        &self,
        prompt: &str,
        images: &[String],
        size: Option<&str>,
        quality: &str,
    ) -> EngineResult<Vec<Value>> {
        if self.access_token.is_empty() {
            return Err(EngineError::other("access_token is required for codex image endpoints"));
        }
        let source_type = self
            .account
            .get("source_type")
            .and_then(|v| v.as_str())
            .unwrap_or("web")
            .trim()
            .to_lowercase();
        if source_type != "codex" {
            return Err(EngineError::other("codex responses endpoint requires a codex source account"));
        }
        let path = "/backend-api/codex/responses";
        let payload = json!({
            "model": CODEX_RESPONSES_MODEL,
            "instructions": CODEX_RESPONSES_INSTRUCTIONS,
            "store": false,
            "input": Self::codex_image_input(prompt, images),
            "tools": [{
                "type": "image_generation",
                "model": "gpt-image-2",
                "action": if images.is_empty() { "generate" } else { "edit" },
                "size": size.unwrap_or("1024x1024"),
                "quality": if quality.is_empty() { "auto" } else { quality },
                "output_format": "png",
            }],
            "tool_choice": {"type": "image_generation"},
            "stream": true,
        });
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.codex_responses_headers())
            .json(&payload)
            .timeout(Duration::from_secs(1200))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        if !(200..300).contains(&status) {
            let retry_after = resp
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            let text = resp.text().await.unwrap_or_default();
            let body: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
            return Err(EngineError::Upstream(crate::utils::helper::UpstreamHttpError::new(
                path, status, body, retry_after,
            )));
        }
        let text = resp.text().await.map_err(EngineError::from_wreq)?;
        Ok(Self::parse_codex_events(&content_type, &text))
    }

    fn parse_codex_events(content_type: &str, text: &str) -> Vec<Value> {
        let mut events: Vec<Value> = Vec::new();
        if content_type.contains("application/json") {
            if let Ok(data) = serde_json::from_str::<Value>(text) {
                if data.is_object() {
                    events.push(data);
                }
            }
            return events;
        }
        let mut lines: Vec<String> = Vec::new();
        let mut all: Vec<&str> = text.lines().collect();
        all.push("");
        for line in all {
            if line.is_empty() {
                if !lines.is_empty() {
                    let payload_text = lines.join("\n");
                    let payload_text = payload_text.trim();
                    if !payload_text.is_empty() && payload_text != "[DONE]" {
                        if let Ok(data) = serde_json::from_str::<Value>(payload_text) {
                            if data.is_object() {
                                events.push(data);
                            }
                        }
                    }
                    lines.clear();
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                lines.push(rest.trim_start().to_string());
            }
        }
        events
    }

    // ---- search ----

    fn find_value(payload: &Value, key: &str) -> String {
        match payload {
            Value::String(s) => {
                if key == "conversation_id" {
                    if let Some(c) = SEARCH_CONVERSATION_ID_RE.captures(s) {
                        return c[1].to_string();
                    }
                }
                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                    return Self::find_value(&parsed, key);
                }
                String::new()
            }
            Value::Object(m) => {
                if let Some(v) = m.get(key).and_then(|v| v.as_str()) {
                    if !v.is_empty() {
                        return v.to_string();
                    }
                }
                for v in m.values() {
                    let found = Self::find_value(v, key);
                    if !found.is_empty() {
                        return found;
                    }
                }
                String::new()
            }
            Value::Array(a) => {
                for v in a {
                    let found = Self::find_value(v, key);
                    if !found.is_empty() {
                        return found;
                    }
                }
                String::new()
            }
            _ => String::new(),
        }
    }

    fn walk_dicts<'a>(payload: &'a Value, out: &mut Vec<&'a Value>) {
        match payload {
            Value::Object(m) => {
                out.push(payload);
                for v in m.values() {
                    Self::walk_dicts(v, out);
                }
            }
            Value::Array(a) => {
                for v in a {
                    Self::walk_dicts(v, out);
                }
            }
            _ => {}
        }
    }

    fn clean_search_url(value: &str) -> String {
        value.trim().trim_end_matches(['.', ',', ';', '，', '。', '；']).to_string()
    }

    fn search_message_text(message: &Value) -> String {
        let content = message.get("content").cloned().unwrap_or(json!({}));
        let mut parts: Vec<String> = Vec::new();
        if let Some(obj) = content.as_object() {
            if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                parts.push(t.to_string());
            }
            if let Some(arr) = obj.get("parts").and_then(|v| v.as_array()) {
                for part in arr {
                    if let Some(s) = part.as_str() {
                        parts.push(s.to_string());
                    } else if let Some(o) = part.as_object() {
                        for key in ["text", "summary", "content"] {
                            if let Some(s) = o.get(key).and_then(|v| v.as_str()) {
                                if !s.is_empty() {
                                    parts.push(s.to_string());
                                }
                            }
                        }
                    }
                }
            }
        } else if let Some(s) = content.as_str() {
            parts.push(s.to_string());
        }
        parts
            .iter()
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    }

    fn extract_search_sources(payload: &Value) -> Vec<Value> {
        let mut dicts: Vec<&Value> = Vec::new();
        Self::walk_dicts(payload, &mut dicts);
        let mut sources: Vec<Value> = Vec::new();
        for obj in dicts {
            let metadata = obj.get("metadata").cloned().unwrap_or(json!({}));
            let raw_url = obj
                .get("url")
                .or_else(|| obj.get("link"))
                .or_else(|| obj.get("source_url"))
                .or_else(|| metadata.get("url"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let url = Self::clean_search_url(raw_url);
            if url.is_empty() {
                continue;
            }
            if sources.iter().any(|s| s.get("url").and_then(|v| v.as_str()) == Some(url.as_str())) {
                continue;
            }
            let getx = |keys: &[&str]| -> String {
                for k in keys {
                    if let Some(s) = obj.get(*k).and_then(|v| v.as_str()) {
                        if !s.trim().is_empty() {
                            return s.trim().to_string();
                        }
                    }
                }
                String::new()
            };
            sources.push(json!({
                "title": getx(&["title", "name", "source"]),
                "url": url,
                "snippet": getx(&["snippet", "text", "description"]),
                "source_type": getx(&["type", "source_type"]),
            }));
        }
        sources
    }

    fn extract_search_result(conversation_id: &str, conversation: &Value) -> Value {
        let mut messages: Vec<Value> = Vec::new();
        if let Some(mapping) = conversation.get("mapping").and_then(|v| v.as_object()) {
            for node in mapping.values() {
                let message = node.get("message").cloned().unwrap_or(json!({}));
                if message.get("author").and_then(|a| a.get("role")).and_then(|v| v.as_str()) == Some("assistant") {
                    messages.push(message);
                }
            }
        }
        let message = messages
            .into_iter()
            .max_by(|a, b| {
                let ca = a.get("create_time").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let cb = b.get("create_time").and_then(|v| v.as_f64()).unwrap_or(0.0);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(json!({}));
        let metadata = message.get("metadata").cloned().unwrap_or(json!({}));
        let finish_details = metadata.get("finish_details").cloned().unwrap_or(json!({}));
        let answer = Self::search_message_text(&message);
        let mut sources = Self::extract_search_sources(&message);
        for m in SEARCH_URL_RE.find_iter(&answer) {
            let url = Self::clean_search_url(m.as_str());
            if !url.is_empty()
                && !sources.iter().any(|s| s.get("url").and_then(|v| v.as_str()) == Some(url.as_str()))
            {
                sources.push(json!({"title": "", "url": url, "snippet": "", "source_type": ""}));
            }
        }
        let status = finish_details
            .get("type")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| metadata.get("status").and_then(|v| v.as_str()).map(String::from))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| Self::find_value(&message, "status"));
        json!({
            "conversation_id": conversation_id,
            "status": status.trim(),
            "answer": answer,
            "sources": sources,
            "assistant_message_id": message.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            "create_time": message.get("create_time").and_then(|v| v.as_f64()).unwrap_or(0.0),
        })
    }

    async fn prepare_search_conversation(&self, prompt: &str, model: &str) -> EngineResult<String> {
        let path = "/backend-api/f/conversation/prepare";
        let payload = json!({
            "action": "next",
            "fork_from_shared_post": false,
            "parent_message_id": "client-created-root",
            "model": model,
            "client_prepare_state": "success",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": ["search"],
            "partial_query": {"id": new_uuid(), "author": {"role": "user"}, "content": {"content_type": "text", "parts": [prompt]}},
            "supports_buffering": true,
            "supported_encodings": ["v1"],
            "client_contextual_info": {"app_name": "chatgpt.com"},
        });
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.req_headers(path, &[("Accept", "*/*"), ("Content-Type", "application/json"), ("X-Conduit-Token", "no-token")]))
            .json(&payload)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, path).await?;
        let data = resp.json::<Value>().await.map_err(EngineError::from_wreq)?;
        let token = data.get("conduit_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if token.is_empty() {
            return Err(EngineError::other("missing conduit_token"));
        }
        Ok(token)
    }

    async fn run_search_conversation(&self, prompt: &str, conduit_token: &str, model: &str) -> EngineResult<String> {
        let requirements = self.get_chat_requirements().await?;
        let path = "/backend-api/f/conversation";
        let payload = json!({
            "action": "next",
            "messages": [{
                "id": new_uuid(),
                "author": {"role": "user"},
                "create_time": now_epoch(),
                "content": {"content_type": "text", "parts": [prompt]},
                "metadata": {
                    "developer_mode_connector_ids": [],
                    "selected_github_repos": [],
                    "selected_all_github_repos": false,
                    "system_hints": ["search"],
                    "serialization_metadata": {"custom_symbol_offsets": []},
                },
            }],
            "parent_message_id": "client-created-root",
            "model": model,
            "client_prepare_state": "success",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": true,
            "system_hints": [],
            "supports_buffering": true,
            "supported_encodings": ["v1"],
            "force_use_search": true,
            "client_reported_search_source": "conversation_composer_web_icon",
            "client_contextual_info": {"is_dark_mode": false, "time_since_loaded": 36, "page_height": 925, "page_width": 886, "pixel_ratio": 2, "screen_height": 1440, "screen_width": 2560, "app_name": "chatgpt.com"},
            "paragen_cot_summary_display_override": "allow",
            "force_parallel_switch": "auto",
        });
        let resp = self
            .client
            .post(self.url(path))
            .headers(self.image_headers(path, &requirements, conduit_token, "text/event-stream"))
            .json(&payload)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, path).await?;
        let mut stream = SseStream::new(resp);
        let mut conversation_id = String::new();
        while let Some(payload) = stream.next_payload().await? {
            if conversation_id.is_empty() {
                conversation_id = Self::find_value(&Value::String(payload.clone()), "conversation_id");
            }
            if payload == "[DONE]" {
                break;
            }
        }
        if conversation_id.is_empty() {
            return Err(EngineError::other("conversation_id not found in stream"));
        }
        Ok(conversation_id)
    }

    async fn get_search_conversation(&self, conversation_id: &str) -> EngineResult<Value> {
        let path = format!("/backend-api/conversation/{conversation_id}");
        let referer = format!("{}/c/{}", self.base_url, conversation_id);
        let resp = self
            .client
            .get(self.url(&path))
            .headers(self.req_headers(
                &path,
                &[
                    ("Accept", "*/*"),
                    ("Referer", &referer),
                    ("X-OpenAI-Target-Route", "/backend-api/conversation/{conversation_id}"),
                ],
            ))
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(EngineError::from_wreq)?;
        let resp = Self::ensure_ok(resp, &path).await?;
        resp.json::<Value>().await.map_err(EngineError::from_wreq)
    }

    async fn wait_search_result(
        &self,
        conversation_id: &str,
        timeout_secs: f64,
        poll_interval_secs: f64,
    ) -> EngineResult<Value> {
        let start = Instant::now();
        let mut last_result: Option<Value> = None;
        let mut last_answer = String::new();
        let mut stable_hits = 0u32;
        while start.elapsed().as_secs_f64() < timeout_secs {
            match self.get_search_conversation(conversation_id).await {
                Ok(conv) => {
                    last_result = Some(Self::extract_search_result(conversation_id, &conv));
                }
                Err(EngineError::Upstream(e)) if matches!(e.status_code, 404 | 409 | 423 | 429 | 500 | 502 | 503 | 504) => {}
                Err(e) => return Err(e),
            }
            if let Some(result) = &last_result {
                let answer = result.get("answer").and_then(|v| v.as_str()).unwrap_or("");
                if !answer.is_empty() {
                    let status = result.get("status").and_then(|v| v.as_str()).unwrap_or("");
                    if SEARCH_DONE_STATUS.contains(&status) {
                        return Ok(result.clone());
                    }
                    stable_hits = if answer == last_answer { stable_hits + 1 } else { 0 };
                    last_answer = answer.to_string();
                    if stable_hits >= 2 {
                        return Ok(result.clone());
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs_f64(poll_interval_secs)).await;
        }
        if let Some(result) = last_result {
            return Ok(result);
        }
        Err(EngineError::other(format!(
            "timed out waiting for search result: {conversation_id}"
        )))
    }

    /// Run a web-search conversation and return the structured result.
    pub async fn search(
        &mut self,
        prompt: &str,
        model: Option<&str>,
        timeout_secs: Option<f64>,
        poll_interval_secs: Option<f64>,
    ) -> EngineResult<Value> {
        if self.access_token.is_empty() {
            return Err(EngineError::other("access_token is required for search"));
        }
        let model = model.unwrap_or(SEARCH_MODEL);
        let conduit_token = self.prepare_search_conversation(prompt, model).await?;
        self.bootstrap().await?;
        let conversation_id = self.run_search_conversation(prompt, &conduit_token, model).await?;
        self.wait_search_result(
            &conversation_id,
            timeout_secs.unwrap_or(SEARCH_TIMEOUT_SECS),
            poll_interval_secs.unwrap_or(SEARCH_POLL_INTERVAL_SECS),
        )
        .await
    }
}
