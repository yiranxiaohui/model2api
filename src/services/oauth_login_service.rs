//! Port of `services/oauth_login_service.py` — manual OAuth + PKCE bridge.
//!
//! The user runs OpenAI's standard authorization-code+PKCE flow in their own
//! browser; the backend mints the `code_verifier`/`code_challenge`/`state`,
//! hands back the authorize URL, then later exchanges the returned `code` for
//! the `{access_token, refresh_token, id_token}` triple.

use std::collections::HashMap;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::config::Config;
use crate::services::register::constants::{
    common_headers, platform_oauth_redirect_uri, AUTH_BASE, PLATFORM_AUTH0_CLIENT, PLATFORM_BASE,
    PLATFORM_OAUTH_AUDIENCE, PLATFORM_OAUTH_CLIENT_ID, SEC_CH_UA, USER_AGENT,
};
use crate::utils::pkce::generate_pkce;

const SESSION_TTL_SECONDS: f64 = 10.0 * 60.0;
const MAX_SESSIONS: usize = 64;

/// Expected error in the OAuth bridge; the API layer maps it to HTTP 400.
#[derive(Debug)]
pub struct OAuthLoginError(pub String);

impl std::fmt::Display for OAuthLoginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for OAuthLoginError {}

fn err(msg: impl Into<String>) -> OAuthLoginError {
    OAuthLoginError(msg.into())
}

#[derive(Clone)]
struct Session {
    code_verifier: String,
    state: String,
    created_at: f64,
    redirect_uri: String,
}

fn now() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}

fn token_urlsafe(n: usize) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use rand::RngCore;
    let mut bytes = vec![0u8; n];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

pub struct OAuthLoginService {
    config: Config,
    sessions: Mutex<HashMap<String, Session>>,
}

impl OAuthLoginService {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    fn purge_expired(sessions: &mut HashMap<String, Session>) {
        let now = now();
        sessions.retain(|_, item| now - item.created_at <= SESSION_TTL_SECONDS);
        if sessions.len() > MAX_SESSIONS {
            let mut ordered: Vec<(String, f64)> =
                sessions.iter().map(|(k, v)| (k.clone(), v.created_at)).collect();
            ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let remove = sessions.len() - MAX_SESSIONS;
            for (sid, _) in ordered.into_iter().take(remove) {
                sessions.remove(&sid);
            }
        }
    }

    /// Register a new PKCE session; returns session_id + authorize_url.
    pub fn start(&self, email_hint: &str) -> Value {
        let (verifier, challenge) = generate_pkce();
        let nonce = token_urlsafe(32);
        let device_id = uuid::Uuid::new_v4().to_string();
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let state = format!("{session_id}.{}", token_urlsafe(16));
        let redirect_uri = platform_oauth_redirect_uri();

        let mut params: Vec<(&str, String)> = vec![
            ("issuer", AUTH_BASE.to_string()),
            ("client_id", PLATFORM_OAUTH_CLIENT_ID.to_string()),
            ("audience", PLATFORM_OAUTH_AUDIENCE.to_string()),
            ("redirect_uri", redirect_uri.clone()),
            ("device_id", device_id),
            ("screen_hint", "login_or_signup".to_string()),
            ("max_age", "0".to_string()),
            ("scope", "openid profile email offline_access".to_string()),
            ("response_type", "code".to_string()),
            ("response_mode", "query".to_string()),
            ("state", state.clone()),
            ("nonce", nonce),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256".to_string()),
            ("auth0Client", PLATFORM_AUTH0_CLIENT.to_string()),
        ];
        let email_hint = email_hint.trim();
        if !email_hint.is_empty() {
            params.push(("login_hint", email_hint.to_string()));
        }
        let query = urlencode(&params);
        let authorize_url = format!("{AUTH_BASE}/api/accounts/authorize?{query}");

        {
            let mut sessions = self.sessions.lock();
            Self::purge_expired(&mut sessions);
            sessions.insert(
                session_id.clone(),
                Session {
                    code_verifier: verifier,
                    state,
                    created_at: now(),
                    redirect_uri: redirect_uri.clone(),
                },
            );
        }

        json!({
            "session_id": session_id,
            "authorize_url": authorize_url,
            "expires_in": (SESSION_TTL_SECONDS as i64).to_string(),
            "redirect_uri_prefix": redirect_uri,
        })
    }

    /// Extract `(code, state)` from a callback URL or a raw code string.
    fn extract_code_from_callback(value: &str) -> Result<(String, String), OAuthLoginError> {
        let raw = value.trim();
        if raw.is_empty() {
            return Ok((String::new(), String::new()));
        }
        if raw.starts_with("http://") || raw.starts_with("https://") {
            let parsed = wreq::Url::parse(raw).map_err(|e| err(format!("无法解析 callback URL: {e}")))?;
            let mut code = String::new();
            let mut state = String::new();
            let mut error_desc = String::new();
            let mut error = String::new();
            for (k, v) in parsed.query_pairs() {
                match k.as_ref() {
                    "code" => code = v.trim().to_string(),
                    "state" => state = v.trim().to_string(),
                    "error_description" => error_desc = v.trim().to_string(),
                    "error" => error = v.trim().to_string(),
                    _ => {}
                }
            }
            if code.is_empty() {
                let detail = if !error_desc.is_empty() {
                    error_desc
                } else if !error.is_empty() {
                    error
                } else {
                    "callback URL 中没有 code 参数".to_string()
                };
                return Err(err(detail));
            }
            return Ok((code, state));
        }
        Ok((raw.to_string(), String::new()))
    }

    /// Exchange the callback's code for tokens, keyed by the matching session.
    pub async fn finish(&self, session_id: &str, callback: &str) -> Result<Value, OAuthLoginError> {
        let body_sid = session_id.trim().to_string();
        let (code, state) = Self::extract_code_from_callback(callback)?;
        if code.is_empty() {
            return Err(err("缺少 code 或 callback URL"));
        }

        let state_sid = if state.is_empty() {
            String::new()
        } else {
            state.split_once('.').map(|(a, _)| a.to_string()).unwrap_or_else(|| state.clone())
        };
        let candidate_sids: Vec<String> =
            [state_sid, body_sid].into_iter().filter(|s| !s.is_empty()).collect();
        if candidate_sids.is_empty() {
            return Err(err("既未提供 session_id，callback URL 中也未携带 state"));
        }

        let (picked_sid, session) = {
            let mut sessions = self.sessions.lock();
            Self::purge_expired(&mut sessions);
            let mut found = None;
            for sid in &candidate_sids {
                if let Some(cur) = sessions.get(sid) {
                    found = Some((sid.clone(), cur.clone()));
                    break;
                }
            }
            match found {
                Some(v) => v,
                None => {
                    return Err(err(
                        "OAuth 会话已过期或不存在，请回到导入对话框点\"重新生成\"再走一次",
                    ))
                }
            }
        };

        if !state.is_empty() && !session.state.is_empty() && state != session.state {
            return Err(err(
                "state 不匹配。常见原因：你点过两次\"打开授权页面\"，但浏览器里登录的还是前一次的窗口。请点\"重新生成\"重来。",
            ));
        }

        let tokens = self
            .exchange_code(&code, &session.code_verifier, &session.redirect_uri)
            .await?;
        self.sessions.lock().remove(&picked_sid);
        Ok(tokens)
    }

    async fn exchange_code(
        &self,
        code: &str,
        code_verifier: &str,
        redirect_uri: &str,
    ) -> Result<Value, OAuthLoginError> {
        let mut builder = wreq::Client::builder()
            .emulation(wreq_util::Emulation::Chrome137)
            .cert_verification(false);
        let proxy = self.config.proxy_setting();
        if !proxy.trim().is_empty() {
            if let Ok(p) = wreq::Proxy::all(proxy.trim()) {
                builder = builder.proxy(p);
            }
        }
        let client = builder.build().map_err(|e| err(format!("client build: {e}")))?;

        let mut headers = wreq::header::HeaderMap::new();
        for (k, v) in common_headers() {
            if let (Ok(name), Ok(val)) = (
                wreq::header::HeaderName::from_bytes(k.as_bytes()),
                wreq::header::HeaderValue::from_str(&v),
            ) {
                headers.insert(name, val);
            }
        }
        for (k, v) in [
            ("referer", format!("{PLATFORM_BASE}/")),
            ("origin", PLATFORM_BASE.to_string()),
            ("auth0-client", PLATFORM_AUTH0_CLIENT.to_string()),
            ("sec-ch-ua", SEC_CH_UA.to_string()),
            ("user-agent", USER_AGENT.to_string()),
        ] {
            if let (Ok(name), Ok(val)) = (
                wreq::header::HeaderName::from_bytes(k.as_bytes()),
                wreq::header::HeaderValue::from_str(&v),
            ) {
                headers.insert(name, val);
            }
        }

        let payload = json!({
            "client_id": PLATFORM_OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        });

        let resp = client
            .post(format!("{AUTH_BASE}/api/accounts/oauth/token"))
            .headers(headers)
            .json(&payload)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| err(format!("换 token 网络异常: {e}")))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let data: Value = serde_json::from_str(&text).unwrap_or(Value::Null);

        let access_token = data.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if status != 200 || !data.is_object() || access_token.is_empty() {
            let mut detail = String::new();
            if let Some(obj) = data.as_object() {
                detail = obj
                    .get("error_description")
                    .or_else(|| obj.get("error"))
                    .or_else(|| obj.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
            }
            if detail.is_empty() {
                detail = text.chars().take(300).collect();
            }
            tracing::warn!(
                "[oauth-login] /api/accounts/oauth/token rejected: status={status} detail={detail:?}"
            );
            return Err(err(format!(
                "OpenAI 拒绝换 token (HTTP {status}){}",
                if detail.is_empty() { String::new() } else { format!(": {detail}") }
            )));
        }

        let refresh_token = data.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let id_token = data.get("id_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

        if refresh_token.is_empty() {
            return Err(err(
                "OpenAI 没有返回 refresh_token（可能 scope 未包含 offline_access 或 code 已使用过）",
            ));
        }

        Ok(json!({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": id_token,
        }))
    }
}

/// Minimal `application/x-www-form-urlencoded` query serializer.
fn urlencode(params: &[(&str, String)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", encode_component(k), encode_component(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
