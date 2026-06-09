//! Port of `api/support.py` — auth identity helpers and static web-asset
//! resolution. `auth_service`-backed key authentication is added in Phase 5;
//! for now `require_identity` honours the legacy admin auth-key.

use std::path::{Path, PathBuf};

use axum::http::StatusCode;
use serde_json::{json, Value};

use crate::error::AppError;
use crate::state::AppState;

#[derive(Debug, Clone)]
pub struct Identity {
    pub id: String,
    pub name: String,
    pub role: String,
}

impl Identity {
    pub fn is_admin(&self) -> bool {
        self.role == "admin"
    }

    pub fn to_json(&self) -> Value {
        json!({ "id": self.id, "name": self.name, "role": self.role })
    }
}

pub fn extract_bearer_token(authorization: Option<&str>) -> String {
    let value = authorization.unwrap_or("");
    let (scheme, rest) = value.split_once(' ').unwrap_or(("", ""));
    if scheme.to_ascii_lowercase() != "bearer" || rest.trim().is_empty() {
        return String::new();
    }
    rest.trim().to_string()
}

fn legacy_admin_identity(state: &AppState, token: &str) -> Option<Identity> {
    let auth_key = state.config.auth_key();
    if !auth_key.is_empty() && token == auth_key {
        return Some(Identity {
            id: "admin".into(),
            name: "管理员".into(),
            role: "admin".into(),
        });
    }
    None
}

/// Authenticate a request, returning the resolved identity or a 401.
/// Honours the legacy admin auth-key, then consults `auth_service`.
pub fn require_identity(state: &AppState, authorization: Option<&str>) -> Result<Identity, AppError> {
    let token = extract_bearer_token(authorization);
    if let Some(identity) = legacy_admin_identity(state, &token) {
        return Ok(identity);
    }
    if let Some(value) = state.auth.authenticate(&token) {
        return Ok(Identity {
            id: value.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            name: value.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            role: value.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string(),
        });
    }
    Err(AppError::message(
        StatusCode::UNAUTHORIZED,
        "密钥无效或已失效，请重新登录",
    ))
}

/// Require an admin identity (403 otherwise).
pub fn require_admin(state: &AppState, authorization: Option<&str>) -> Result<Identity, AppError> {
    let identity = require_identity(state, authorization)?;
    if !identity.is_admin() {
        return Err(AppError::message(
            StatusCode::FORBIDDEN,
            "需要管理员权限才能执行这个操作",
        ));
    }
    Ok(identity)
}

/// Resolve the public base URL for image links: configured `base_url` or the
/// request's scheme+host.
pub fn resolve_image_base_url(state: &AppState, host: Option<&str>, scheme: &str) -> String {
    let configured = state.config.base_url();
    if !configured.trim().is_empty() {
        return configured;
    }
    format!("{scheme}://{}", host.unwrap_or("localhost"))
}

/// Map an image-quota error to 429, otherwise 502.
pub fn image_quota_error(message: &str) -> AppError {
    if message.to_lowercase().contains("no available image quota") {
        AppError::message(StatusCode::TOO_MANY_REQUESTS, "no available image quota")
    } else {
        AppError::message(StatusCode::BAD_GATEWAY, message)
    }
}

/// Strip the `secret_key` field from a CPA pool object.
pub fn sanitize_cpa_pool(pool: &Value) -> Option<Value> {
    let obj = pool.as_object()?;
    let mut out = obj.clone();
    out.remove("secret_key");
    Some(Value::Object(out))
}

pub fn sanitize_cpa_pools(pools: &[Value]) -> Vec<Value> {
    pools.iter().filter_map(sanitize_cpa_pool).collect()
}

/// Strip `password`/`api_key` from a sub2api server, adding `has_api_key`.
pub fn sanitize_sub2api_server(server: &Value) -> Option<Value> {
    let obj = server.as_object()?;
    let mut out = obj.clone();
    let has_api_key = obj.get("api_key").and_then(|v| v.as_str()).map_or(false, |s| !s.trim().is_empty());
    out.remove("password");
    out.remove("api_key");
    out.insert("has_api_key".into(), Value::Bool(has_api_key));
    Some(Value::Object(out))
}

pub fn sanitize_sub2api_servers(servers: &[Value]) -> Vec<Value> {
    servers.iter().filter_map(sanitize_sub2api_server).collect()
}

/// Port of `resolve_web_asset`: map a request path to a file inside `web_dist`,
/// guarding against path traversal. Returns the resolved file path if present.
pub fn resolve_web_asset(web_dist_dir: &Path, requested_path: &str) -> Option<PathBuf> {
    if !web_dist_dir.exists() {
        return None;
    }
    let base_dir = web_dist_dir.canonicalize().ok()?;
    let clean = requested_path.trim_matches('/');

    let candidates: Vec<PathBuf> = if clean.is_empty() {
        vec![base_dir.join("index.html")]
    } else {
        vec![
            base_dir.join(clean),
            base_dir.join(clean).join("index.html"),
            base_dir.join(format!("{clean}.html")),
        ]
    };

    for candidate in candidates {
        // Ensure the candidate stays within base_dir.
        if let Ok(resolved) = candidate.canonicalize() {
            if resolved.starts_with(&base_dir) && resolved.is_file() {
                return Some(resolved);
            }
        }
    }
    None
}
