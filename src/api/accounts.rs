//! Port of `api/accounts.py` — account-pool management routes (all under
//! `/api/...`): user-key CRUD, account list/add/delete/update, bulk
//! refresh/re-login with background progress, refresh-token keepalive (export),
//! manual OAuth login bridge, CLIProxyAPI (CPA) pools CRUD + import, and
//! sub2api servers CRUD + remote browse + import.
//!
//! Handlers authenticate via `require_admin` (every route in the Python source
//! is admin-only) and return `json_ok` JSON, except `export` which streams a
//! downloadable JSON/ZIP attachment. Background-import / progress endpoints
//! stage their work and return the job/progress value immediately.

use std::collections::HashSet;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Map, Value};

use crate::api::routes_common::{authorization, json_ok};
use crate::api::support::{
    require_admin, sanitize_cpa_pool, sanitize_cpa_pools, sanitize_sub2api_server,
    sanitize_sub2api_servers,
};
use crate::error::AppError;
use crate::state::SharedState;

/// Map a remote-fetch / service `Result<_, String>` error to a 502.
fn upstream(e: String) -> AppError {
    AppError::message(StatusCode::BAD_GATEWAY, e)
}

pub fn router() -> Router<SharedState> {
    Router::new()
        // ---- user keys ----
        .route("/api/auth/users", get(list_user_keys).post(create_user_key))
        .route(
            "/api/auth/users/{key_id}",
            post(update_user_key).delete(delete_user_key),
        )
        // ---- accounts ----
        .route(
            "/api/accounts",
            get(get_accounts).post(create_accounts).delete(delete_accounts),
        )
        .route("/api/accounts/refresh", post(refresh_accounts))
        .route(
            "/api/accounts/refresh/progress/{progress_id}",
            get(get_refresh_progress),
        )
        .route("/api/accounts/re-login", post(re_login_accounts))
        .route(
            "/api/accounts/re-login/progress/{progress_id}",
            get(get_relogin_progress),
        )
        .route("/api/accounts/export", post(export_accounts))
        .route("/api/accounts/update", post(update_account))
        .route("/api/accounts/oauth/start", post(start_oauth_login))
        .route("/api/accounts/oauth/finish", post(finish_oauth_login))
        // ---- CPA pools ----
        .route("/api/cpa/pools", get(list_cpa_pools).post(create_cpa_pool))
        .route(
            "/api/cpa/pools/{pool_id}",
            post(update_cpa_pool).delete(delete_cpa_pool),
        )
        .route("/api/cpa/pools/{pool_id}/files", get(cpa_pool_files))
        .route(
            "/api/cpa/pools/{pool_id}/import",
            post(cpa_pool_import).get(cpa_pool_import_progress),
        )
        // ---- sub2api servers ----
        .route(
            "/api/sub2api/servers",
            get(list_sub2api_servers).post(create_sub2api_server),
        )
        .route(
            "/api/sub2api/servers/{server_id}",
            post(update_sub2api_server).delete(delete_sub2api_server),
        )
        .route(
            "/api/sub2api/servers/{server_id}/groups",
            get(sub2api_server_groups),
        )
        .route(
            "/api/sub2api/servers/{server_id}/accounts",
            get(sub2api_server_accounts),
        )
        .route(
            "/api/sub2api/servers/{server_id}/import",
            post(sub2api_server_import).get(sub2api_server_import_progress),
        )
}

// ============================ helpers ============================

/// Read a string field (`str(value or "")` semantics): missing/null -> "".
fn field_str(body: &Value, key: &str) -> String {
    match body.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Collect a JSON array field into `Vec<String>` (`str(x)` per element).
fn str_array(body: &Value, key: &str) -> Vec<String> {
    body.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|x| match x {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Build an updates object from the given keys, keeping only present non-null
/// values (mirrors Pydantic `model_dump(exclude_none=True)` filtering).
fn build_updates(body: &Value, keys: &[&str]) -> Map<String, Value> {
    let mut out = Map::new();
    for key in keys {
        if let Some(v) = body.get(*key) {
            if !v.is_null() {
                out.insert((*key).to_string(), v.clone());
            }
        }
    }
    out
}

/// Port of `_account_payload_token`: access_token / accessToken, trimmed.
fn account_payload_token(item: &Value) -> String {
    let a = item.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    let b = item.get("accessToken").and_then(|v| v.as_str()).unwrap_or("");
    (if !a.is_empty() { a } else { b }).trim().to_string()
}

/// Port of `_unique_tokens`: ordered, de-duplicated, non-empty trimmed tokens.
fn unique_tokens(tokens: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for token in tokens {
        let trimmed = token.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.clone()) {
            out.push(trimmed);
        }
    }
    out
}

/// Port of `_download_timestamp`: local time `%Y%m%d-%H%M%S`.
fn download_timestamp() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

/// Port of `_safe_export_name`: collapse non `[A-Za-z0-9._-]` runs to `-`,
/// strip leading/trailing `-._`, fall back, then cap at 80 chars.
fn safe_export_name(value: &str, fallback: &str) -> String {
    let mut collapsed = String::new();
    let mut prev_dash = false;
    for ch in value.trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            collapsed.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            collapsed.push('-');
            prev_dash = true;
        }
    }
    let trimmed = collapsed.trim_matches(|c| c == '-' || c == '.' || c == '_');
    let clean = if trimmed.is_empty() { fallback } else { trimmed };
    clean.chars().take(80).collect()
}

/// Port of `_account_zip_bytes`: one pretty-printed `{name}.json` per item.
fn account_zip_bytes(items: &[Value]) -> Vec<u8> {
    use std::io::Write;
    let cursor = std::io::Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let mut used: HashSet<String> = HashSet::new();
    for (i, item) in items.iter().enumerate() {
        let index = i + 1;
        let fallback = format!("account-{index:03}");
        let raw_name = {
            let email = field_str(item, "email");
            let account_id = field_str(item, "account_id");
            if !email.is_empty() {
                email
            } else if !account_id.is_empty() {
                account_id
            } else {
                fallback.clone()
            }
        };
        let base = safe_export_name(&raw_name, &fallback);
        let mut name = base.clone();
        let mut suffix = 2;
        while used.contains(&name) {
            name = format!("{base}-{suffix}");
            suffix += 1;
        }
        used.insert(name.clone());
        let content = serde_json::to_string_pretty(item).unwrap_or_default() + "\n";
        if zip.start_file(format!("{name}.json"), options).is_ok() {
            let _ = zip.write_all(content.as_bytes());
        }
    }
    match zip.finish() {
        Ok(cursor) => cursor.into_inner(),
        Err(_) => Vec::new(),
    }
}

/// Build a downloadable attachment response.
fn attachment(content_type: &str, filename: &str, body: Body) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        )
        .body(body)
        .unwrap()
}

// ============================ user keys ============================

async fn list_user_keys(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({ "items": state.auth.list_keys(Some("user")) })))
}

async fn create_user_key(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let name = field_str(&body, "name");
    let (item, raw_key) = state
        .auth
        .create_key("user", &name)
        .map_err(AppError::bad_request)?;
    Ok(json_ok(json!({
        "item": item,
        "key": raw_key,
        "items": state.auth.list_keys(Some("user")),
    })))
}

async fn update_user_key(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let updates = build_updates(&body, &["name", "enabled", "key"]);
    if updates.is_empty() {
        return Err(AppError::bad_request("还没有检测到改动，请修改后再保存"));
    }
    match state.auth.update_key(&key_id, &Value::Object(updates), Some("user")) {
        Some(item) => Ok(json_ok(json!({
            "item": item,
            "items": state.auth.list_keys(Some("user")),
        }))),
        None => Err(AppError::not_found("这条用户密钥不存在，可能已经被删除")),
    }
}

async fn delete_user_key(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    if !state.auth.delete_key(&key_id, Some("user")) {
        return Err(AppError::not_found("这条用户密钥不存在，可能已经被删除"));
    }
    Ok(json_ok(json!({ "items": state.auth.list_keys(Some("user")) })))
}

// ============================ accounts ============================

async fn get_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({ "items": state.accounts.list_accounts() })))
}

async fn create_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;

    let account_payloads: Vec<Value> = body
        .get("accounts")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter(|x| x.is_object()).cloned().collect())
        .unwrap_or_default();
    let payload_tokens: Vec<String> =
        account_payloads.iter().map(account_payload_token).collect();

    let mut combined = str_array(&body, "tokens");
    combined.extend(payload_tokens.iter().cloned());
    let tokens = unique_tokens(&combined);
    if tokens.is_empty() {
        return Err(AppError::bad_request("tokens is required"));
    }

    let mut result = if !account_payloads.is_empty() {
        let mut r = state.accounts.add_account_items(&account_payloads);
        let payload_token_set: HashSet<String> = unique_tokens(&payload_tokens).into_iter().collect();
        let extra: Vec<String> = tokens
            .iter()
            .filter(|t| !payload_token_set.contains(*t))
            .cloned()
            .collect();
        if !extra.is_empty() {
            let extra_result = state.accounts.add_accounts(&extra, "web");
            let added = r.get("added").and_then(|v| v.as_i64()).unwrap_or(0)
                + extra_result.get("added").and_then(|v| v.as_i64()).unwrap_or(0);
            let skipped = r.get("skipped").and_then(|v| v.as_i64()).unwrap_or(0)
                + extra_result.get("skipped").and_then(|v| v.as_i64()).unwrap_or(0);
            r["added"] = json!(added);
            r["skipped"] = json!(skipped);
        }
        r
    } else {
        state.accounts.add_accounts(&tokens, "web")
    };

    let refresh_result = state.accounts.refresh_accounts(&tokens, None, false).await;
    let items = refresh_result
        .get("items")
        .cloned()
        .or_else(|| result.get("items").cloned())
        .unwrap_or_else(|| json!([]));
    result["refreshed"] = refresh_result.get("refreshed").cloned().unwrap_or_else(|| json!(0));
    result["errors"] = refresh_result.get("errors").cloned().unwrap_or_else(|| json!([]));
    result["items"] = items;
    Ok(json_ok(result))
}

async fn delete_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let tokens: Vec<String> = str_array(&body, "tokens")
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err(AppError::bad_request("tokens is required"));
    }
    Ok(json_ok(state.accounts.delete_accounts(&tokens)))
}

async fn refresh_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let mut access_tokens: Vec<String> = str_array(&body, "access_tokens")
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if access_tokens.is_empty() {
        access_tokens = state.accounts.list_tokens();
    }
    if access_tokens.is_empty() {
        return Err(AppError::bad_request("access_tokens is required"));
    }

    let progress_id = uuid::Uuid::new_v4().to_string();
    state.accounts.init_refresh_progress(&progress_id, access_tokens.len());

    let accounts = state.accounts.clone();
    let pid = progress_id.clone();
    tokio::spawn(async move {
        accounts.refresh_accounts(&access_tokens, Some(&pid), false).await;
    });

    Ok(json_ok(json!({ "progress_id": progress_id })))
}

async fn get_refresh_progress(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(progress_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    match state.accounts.get_refresh_progress(&progress_id) {
        Some(progress) => Ok(json_ok(progress)),
        None => Err(AppError::not_found("progress not found")),
    }
}

async fn re_login_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let access_tokens: Vec<String> = str_array(&body, "access_tokens")
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if access_tokens.is_empty() {
        return Err(AppError::bad_request("access_tokens is required"));
    }

    let progress_id = uuid::Uuid::new_v4().to_string();
    state.accounts.init_relogin_progress(&progress_id, access_tokens.len());

    let accounts = state.accounts.clone();
    let pid = progress_id.clone();
    tokio::task::spawn_blocking(move || {
        accounts.re_login_accounts(&access_tokens, Some(&pid));
    });

    Ok(json_ok(json!({ "progress_id": progress_id })))
}

async fn get_relogin_progress(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(progress_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    match state.accounts.get_relogin_progress(&progress_id) {
        Some(progress) => Ok(json_ok(progress)),
        None => Err(AppError::not_found("progress not found")),
    }
}

async fn export_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let access_tokens = unique_tokens(&str_array(&body, "access_tokens"));
    let items = state.accounts.build_export_items(Some(access_tokens.as_slice()));
    if items.is_empty() {
        return Err(AppError::bad_request(
            "没有可导出的完整账号，需要同时有 access_token、refresh_token 和 id_token",
        ));
    }

    let timestamp = download_timestamp();
    let format = field_str(&body, "format");
    if format == "zip" {
        let content = account_zip_bytes(&items);
        return Ok(attachment(
            "application/zip",
            &format!("codex-accounts-{timestamp}.zip"),
            Body::from(content),
        ));
    }

    let payload = if items.len() == 1 {
        items[0].clone()
    } else {
        Value::Array(items)
    };
    let text = serde_json::to_string_pretty(&payload).unwrap_or_default() + "\n";
    Ok(attachment(
        "application/json",
        &format!("codex-accounts-{timestamp}.json"),
        Body::from(text),
    ))
}

async fn update_account(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let access_token = field_str(&body, "access_token").trim().to_string();
    if access_token.is_empty() {
        return Err(AppError::bad_request("access_token is required"));
    }
    let updates = build_updates(&body, &["type", "status", "quota", "proxy"]);
    if updates.is_empty() {
        return Err(AppError::bad_request("还没有检测到改动，请修改后再保存"));
    }
    match state.accounts.update_account(&access_token, &Value::Object(updates), false) {
        Some(account) => Ok(json_ok(json!({
            "item": account,
            "items": state.accounts.list_accounts(),
        }))),
        None => Err(AppError::not_found("account not found")),
    }
}

async fn start_oauth_login(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let email_hint = field_str(&body, "email_hint");
    Ok(json_ok(state.oauth_login.start(&email_hint)))
}

async fn finish_oauth_login(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let session_id = field_str(&body, "session_id");
    let callback = field_str(&body, "callback");
    let sid_preview: String = session_id.chars().take(8).collect();
    tracing::info!("[oauth-login] finish called: session_id={sid_preview}...");

    let tokens = state
        .oauth_login
        .finish(&session_id, &callback)
        .await
        .map_err(|e| {
            tracing::warn!("[oauth-login] finish rejected: {}", e.0);
            AppError::bad_request(e.0)
        })?;

    let payload = json!({
        "access_token": tokens.get("access_token").cloned().unwrap_or_else(|| json!("")),
        "refresh_token": tokens.get("refresh_token").cloned().unwrap_or_else(|| json!("")),
        "id_token": tokens.get("id_token").cloned().unwrap_or_else(|| json!("")),
        "source_type": "oauth_login",
    });
    let mut add_result = state.accounts.add_account_items(&[payload]);
    let access_token = tokens
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let refresh_result = state
        .accounts
        .refresh_accounts(&[access_token], None, false)
        .await;
    let items = refresh_result
        .get("items")
        .cloned()
        .or_else(|| add_result.get("items").cloned())
        .unwrap_or_else(|| json!([]));
    add_result["refreshed"] = refresh_result.get("refreshed").cloned().unwrap_or_else(|| json!(0));
    add_result["errors"] = refresh_result.get("errors").cloned().unwrap_or_else(|| json!([]));
    add_result["items"] = items;
    Ok(json_ok(add_result))
}

// ============================ CPA pools ============================

async fn list_cpa_pools(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({
        "pools": sanitize_cpa_pools(&state.cpa.list_pools()),
    })))
}

async fn create_cpa_pool(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let name = field_str(&body, "name");
    let base_url = field_str(&body, "base_url");
    let secret_key = field_str(&body, "secret_key");
    if base_url.trim().is_empty() {
        return Err(AppError::bad_request("base_url is required"));
    }
    if secret_key.trim().is_empty() {
        return Err(AppError::bad_request("secret_key is required"));
    }
    let pool = state.cpa.add_pool(&name, &base_url, &secret_key);
    Ok(json_ok(json!({
        "pool": sanitize_cpa_pool(&pool).unwrap_or(Value::Null),
        "pools": sanitize_cpa_pools(&state.cpa.list_pools()),
    })))
}

async fn update_cpa_pool(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(pool_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    match state.cpa.update_pool(&pool_id, &body) {
        Some(pool) => Ok(json_ok(json!({
            "pool": sanitize_cpa_pool(&pool).unwrap_or(Value::Null),
            "pools": sanitize_cpa_pools(&state.cpa.list_pools()),
        }))),
        None => Err(AppError::not_found("pool not found")),
    }
}

async fn delete_cpa_pool(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(pool_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    if !state.cpa.delete_pool(&pool_id) {
        return Err(AppError::not_found("pool not found"));
    }
    Ok(json_ok(json!({
        "pools": sanitize_cpa_pools(&state.cpa.list_pools()),
    })))
}

async fn cpa_pool_files(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(pool_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let pool = state
        .cpa
        .get_pool(&pool_id)
        .ok_or_else(|| AppError::not_found("pool not found"))?;
    let files = state.cpa.list_remote_files(&pool).await.map_err(upstream)?;
    Ok(json_ok(json!({ "pool_id": pool_id, "files": files })))
}

async fn cpa_pool_import(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(pool_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let pool = state
        .cpa
        .get_pool(&pool_id)
        .ok_or_else(|| AppError::not_found("pool not found"))?;
    let names = str_array(&body, "names");
    let job = state
        .cpa
        .start_import(&pool, &names)
        .await
        .map_err(AppError::bad_request)?;
    Ok(json_ok(json!({ "import_job": job })))
}

async fn cpa_pool_import_progress(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(pool_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let pool = state
        .cpa
        .get_pool(&pool_id)
        .ok_or_else(|| AppError::not_found("pool not found"))?;
    Ok(json_ok(json!({
        "import_job": pool.get("import_job").cloned().unwrap_or(Value::Null),
    })))
}

// ============================ sub2api servers ============================

async fn list_sub2api_servers(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    Ok(json_ok(json!({
        "servers": sanitize_sub2api_servers(&state.sub2api.list_servers()),
    })))
}

async fn create_sub2api_server(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let name = field_str(&body, "name");
    let base_url = field_str(&body, "base_url");
    let email = field_str(&body, "email");
    let password = field_str(&body, "password");
    let api_key = field_str(&body, "api_key");
    let group_id = field_str(&body, "group_id");
    if base_url.trim().is_empty() {
        return Err(AppError::bad_request("base_url is required"));
    }
    let has_login = !email.trim().is_empty() && !password.trim().is_empty();
    let has_api_key = !api_key.trim().is_empty();
    if !has_login && !has_api_key {
        return Err(AppError::bad_request("email+password or api_key is required"));
    }
    let server = state
        .sub2api
        .add_server(&name, &base_url, &email, &password, &api_key, &group_id);
    Ok(json_ok(json!({
        "server": sanitize_sub2api_server(&server).unwrap_or(Value::Null),
        "servers": sanitize_sub2api_servers(&state.sub2api.list_servers()),
    })))
}

async fn update_sub2api_server(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(server_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    match state.sub2api.update_server(&server_id, &body) {
        Some(server) => Ok(json_ok(json!({
            "server": sanitize_sub2api_server(&server).unwrap_or(Value::Null),
            "servers": sanitize_sub2api_servers(&state.sub2api.list_servers()),
        }))),
        None => Err(AppError::not_found("server not found")),
    }
}

async fn delete_sub2api_server(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(server_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    if !state.sub2api.delete_server(&server_id) {
        return Err(AppError::not_found("server not found"));
    }
    Ok(json_ok(json!({
        "servers": sanitize_sub2api_servers(&state.sub2api.list_servers()),
    })))
}

async fn sub2api_server_groups(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(server_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let server = state
        .sub2api
        .get_server(&server_id)
        .ok_or_else(|| AppError::not_found("server not found"))?;
    let groups = state.sub2api.list_remote_groups(&server).await.map_err(upstream)?;
    Ok(json_ok(json!({ "server_id": server_id, "groups": groups })))
}

async fn sub2api_server_accounts(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(server_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let server = state
        .sub2api
        .get_server(&server_id)
        .ok_or_else(|| AppError::not_found("server not found"))?;
    let accounts = state.sub2api.list_remote_accounts(&server).await.map_err(upstream)?;
    Ok(json_ok(json!({ "server_id": server_id, "accounts": accounts })))
}

async fn sub2api_server_import(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(server_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let server = state
        .sub2api
        .get_server(&server_id)
        .ok_or_else(|| AppError::not_found("server not found"))?;
    let account_ids = str_array(&body, "account_ids");
    let job = state
        .sub2api
        .start_import(&server, &account_ids)
        .map_err(AppError::bad_request)?;
    Ok(json_ok(json!({ "import_job": job })))
}

async fn sub2api_server_import_progress(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(server_id): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, authorization(&headers).as_deref())?;
    let server = state
        .sub2api
        .get_server(&server_id)
        .ok_or_else(|| AppError::not_found("server not found"))?;
    Ok(json_ok(json!({
        "import_job": server.get("import_job").cloned().unwrap_or(Value::Null),
    })))
}
