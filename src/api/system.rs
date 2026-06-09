//! Port of `api/system.py` — system / configuration / image-gallery / log /
//! backup routes.
//!
//! Every route defined in the Python `system.create_router(app_version)` is
//! reproduced here with the same path, HTTP method and authentication
//! requirement (admin vs. identity). Handlers return `Result<Response, AppError>`
//! and lean on the already-ported service layer hung off [`AppState`].
//!
//! Intentional differences from the Python original (see the per-handler notes
//! and the porting report):
//!   * `GET /api/settings` returns `config.get_public()` (the Rust config only
//!     exposes the public/sanitised view) where Python returned `config.get()`.
//!   * `test_proxy` is called with the Python default 15s timeout.
//!   * The three image-storage *maintenance* endpoints (`/api/images/storage`,
//!     `.../compress`, `.../cleanup-to-target`) have no backing Rust service —
//!     `storage_stats` / `compress_images` / `delete_to_target` were explicitly
//!     left unported in `image_service.rs` (they rely on `shutil.disk_usage`).
//!     The routes are still registered but respond `501 Not Implemented`.
//!   * Auth-key CRUD lives in `accounts.py` (`/api/auth/users`), not in
//!     `system.py`, so it is *not* part of this module.

use std::collections::HashMap;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::api::routes_common::{header_str, json_ok, scheme};
use crate::api::support::{require_admin, require_identity, resolve_image_base_url};
use crate::error::AppError;
use crate::services::proxy_service::test_proxy;
use crate::state::SharedState;

/// Python `test_proxy(..., timeout=15.0)` default.
const PROXY_TEST_TIMEOUT_SECS: f64 = 15.0;
/// Python `log_service.list(..., limit=200)` default.
const LOG_LIST_LIMIT: usize = 200;

/// Read the `Authorization` header value as a string slice.
pub fn auth_header(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
}

pub fn router() -> Router<SharedState> {
    Router::new()
        // --- auth / version / settings -------------------------------------
        .route("/auth/login", post(login))
        .route("/version", get(version))
        .route("/api/settings", get(get_settings).post(save_settings))
        // --- image gallery --------------------------------------------------
        .route("/api/images", get(get_images))
        .route("/images/{*image_path}", get(get_image))
        .route("/image-thumbnails/{*image_path}", get(get_image_thumbnail))
        .route("/api/images/delete", post(delete_images_endpoint))
        .route("/api/images/download", post(download_images_endpoint))
        .route(
            "/api/images/download/{*image_path}",
            get(download_single_image_endpoint),
        )
        // --- image tags -----------------------------------------------------
        .route("/api/images/tags", get(list_image_tags).post(update_image_tags))
        .route("/api/images/tags/{tag}", delete(delete_image_tag))
        // --- image storage maintenance (not ported — 501) -------------------
        .route("/api/images/storage", get(get_image_storage))
        .route("/api/images/storage/compress", post(compress_all_images))
        .route(
            "/api/images/storage/cleanup-to-target",
            post(cleanup_to_target),
        )
        // --- logs -----------------------------------------------------------
        .route("/api/logs", get(get_logs))
        .route("/api/logs/delete", post(delete_logs))
        // --- proxy / storage info ------------------------------------------
        .route("/api/proxy/test", post(test_proxy_endpoint))
        .route("/api/storage/info", get(get_storage_info))
        // --- image storage (webdav) ----------------------------------------
        .route("/api/image-storage/test", post(test_image_storage_endpoint))
        .route("/api/image-storage/sync", post(sync_image_storage_endpoint))
        // --- backups --------------------------------------------------------
        .route("/api/backup/test", post(test_backup_connection))
        .route("/api/backups", get(get_backups))
        .route("/api/backups/run", post(run_backup_endpoint))
        .route("/api/backups/delete", post(delete_backup_endpoint))
        .route("/api/backups/detail", get(get_backup_detail))
        .route("/api/backups/download", get(download_backup_endpoint))
        // --- health dashboard ----------------------------------------------
        .route("/health", get(health_dashboard))
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// Map a `BackupError`-equivalent (`anyhow::Error`) to the Python `400` envelope.
fn backup_err(e: impl std::fmt::Display) -> AppError {
    AppError::message(StatusCode::BAD_REQUEST, e.to_string())
}

/// Trimmed string field from a JSON body object.
fn body_str(body: &Value, key: &str) -> String {
    body.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Bool field from a JSON body object (defaults to false).
fn body_bool(body: &Value, key: &str) -> bool {
    body.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

/// String array field from a JSON body object.
fn body_str_list(body: &Value, key: &str) -> Vec<String> {
    body.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Trimmed query param (empty string when absent).
fn query_str(params: &HashMap<String, String>, key: &str) -> String {
    params.get(key).map(|s| s.trim().to_string()).unwrap_or_default()
}

/// RFC 5987 percent-encoding for `Content-Disposition` `filename*` values.
fn encode_rfc5987(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Build a raw bytes response with a content-type and optional headers.
fn bytes_response(bytes: Vec<u8>, content_type: &str, extra: &[(&str, String)]) -> Response {
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type);
    for (k, v) in extra {
        builder = builder.header(*k, v);
    }
    builder.body(Body::from(bytes)).unwrap_or_else(|_| {
        AppError::internal("failed to build response").into_response()
    })
}

/// CORS headers applied by the Python image responses.
fn cors_image_headers() -> Vec<(&'static str, String)> {
    vec![
        ("Access-Control-Allow-Origin", "*".to_string()),
        ("Access-Control-Allow-Methods", "GET, OPTIONS".to_string()),
        ("Access-Control-Allow-Headers", "*".to_string()),
    ]
}

// ---------------------------------------------------------------------------
// auth / version / settings
// ---------------------------------------------------------------------------

/// `POST /auth/login` — legacy login: validate the bearer key and echo identity.
async fn login(State(state): State<SharedState>, headers: HeaderMap) -> Result<Response, AppError> {
    let identity = require_identity(&state, auth_header(&headers))?;
    Ok(json_ok(json!({
        "ok": true,
        "version": state.config.app_version(),
        "role": identity.role,
        "subject_id": identity.id,
        "name": identity.name,
    })))
}

/// `GET /version`.
async fn version(State(state): State<SharedState>) -> Response {
    json_ok(json!({ "version": state.config.app_version() }))
}

/// `GET /api/settings` (admin).
async fn get_settings(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    Ok(json_ok(json!({ "config": state.config.get_public() })))
}

/// `POST /api/settings` (admin) — merge a config patch.
async fn save_settings(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let patch = body.map(|Json(v)| v).unwrap_or(Value::Object(Default::default()));
    let updated = state
        .config
        .update(patch)
        .map_err(|e| AppError::message(StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(json_ok(json!({ "config": updated })))
}

// ---------------------------------------------------------------------------
// image gallery
// ---------------------------------------------------------------------------

/// `GET /api/images` (admin) — list gallery images grouped by date.
async fn get_images(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let host = header_str(&headers, "host");
    let base_url = resolve_image_base_url(&state, host.as_deref(), &scheme(&headers));
    let start = query_str(&params, "start_date");
    let end = query_str(&params, "end_date");
    Ok(json_ok(state.image_service.list_images(&base_url, &start, &end)))
}

/// `GET /images/{image_path}` (public) — serve an image inline.
async fn get_image(
    State(state): State<SharedState>,
    Path(image_path): Path<String>,
) -> Result<Response, AppError> {
    let file = state.image_service.get_image(&image_path).await?;
    Ok(bytes_response(file.bytes, &file.content_type, &cors_image_headers()))
}

/// `GET /image-thumbnails/{image_path}` (public) — serve a cached thumbnail.
async fn get_image_thumbnail(
    State(state): State<SharedState>,
    Path(image_path): Path<String>,
) -> Result<Response, AppError> {
    let file = state.image_service.get_thumbnail(&image_path).await?;
    Ok(bytes_response(file.bytes, &file.content_type, &cors_image_headers()))
}

/// `POST /api/images/delete` (admin).
async fn delete_images_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let body = body.map(|Json(v)| v).unwrap_or(Value::Null);
    let paths = body_str_list(&body, "paths");
    let start = body_str(&body, "start_date");
    let end = body_str(&body, "end_date");
    let all_matching = body_bool(&body, "all_matching");
    let result = state
        .image_service
        .delete_images(Some(paths), &start, &end, all_matching)
        .await?;
    Ok(json_ok(result))
}

/// `POST /api/images/download` (admin) — export selected images as a ZIP.
async fn download_images_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let body = body.map(|Json(v)| v).unwrap_or(Value::Null);
    let paths = body_str_list(&body, "paths");
    let bytes = state.image_service.download_images_zip(&paths).await?;
    Ok(bytes_response(
        bytes,
        "application/zip",
        &[("Content-Disposition", "attachment; filename=\"images.zip\"".to_string())],
    ))
}

/// `GET /api/images/download/{image_path}` (admin) — download a single image.
async fn download_single_image_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(image_path): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let file = state.image_service.get_image_download(&image_path).await?;
    let name = file.filename.clone().unwrap_or_else(|| "image".to_string());
    let mut extra = cors_image_headers();
    extra.push((
        "Content-Disposition",
        format!("attachment; filename=\"{name}\""),
    ));
    Ok(bytes_response(file.bytes, &file.content_type, &extra))
}

// ---------------------------------------------------------------------------
// image tags
// ---------------------------------------------------------------------------

/// `GET /api/images/tags` (admin).
async fn list_image_tags(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    Ok(json_ok(json!({ "tags": state.image_tags.get_all_tags() })))
}

/// `POST /api/images/tags` (admin).
async fn update_image_tags(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let body = body.map(|Json(v)| v).unwrap_or(Value::Null);
    let rel = body_str(&body, "path");
    let rel = rel.trim_start_matches('/').to_string();
    if rel.is_empty() {
        return Err(AppError::message(StatusCode::BAD_REQUEST, "path is required"));
    }
    let tags = body_str_list(&body, "tags");
    let saved = state.image_tags.set_tags(&rel, &tags);
    Ok(json_ok(json!({ "ok": true, "tags": saved })))
}

/// `DELETE /api/images/tags/{tag}` (admin).
async fn delete_image_tag(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(tag): Path<String>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let count = state.image_tags.delete_tag(&tag);
    Ok(json_ok(json!({ "ok": true, "removed_from": count })))
}

// ---------------------------------------------------------------------------
// image storage maintenance — not ported (see module docs)
// ---------------------------------------------------------------------------

fn not_implemented(what: &str) -> AppError {
    AppError::message(
        StatusCode::NOT_IMPLEMENTED,
        format!("{what} 尚未在 Rust 版本中实现"),
    )
}

/// `GET /api/images/storage` (admin) — disk/storage stats (`storage_stats`).
async fn get_image_storage(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    Err(not_implemented("图片存储统计"))
}

/// `POST /api/images/storage/compress` (admin) — recompress all images.
async fn compress_all_images(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    Err(not_implemented("图片压缩"))
}

/// `POST /api/images/storage/cleanup-to-target` (admin).
async fn cleanup_to_target(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(_params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    Err(not_implemented("按目标空间清理图片"))
}

// ---------------------------------------------------------------------------
// logs
// ---------------------------------------------------------------------------

/// `GET /api/logs` (admin).
async fn get_logs(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let log_type = query_str(&params, "type");
    let start = query_str(&params, "start_date");
    let end = query_str(&params, "end_date");
    let items = state.log.list(&log_type, &start, &end, LOG_LIST_LIMIT);
    Ok(json_ok(json!({ "items": items })))
}

/// `POST /api/logs/delete` (admin).
async fn delete_logs(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let body = body.map(|Json(v)| v).unwrap_or(Value::Null);
    let ids = body_str_list(&body, "ids");
    let removed = state.log.delete(&ids);
    Ok(json_ok(json!({ "removed": removed })))
}

// ---------------------------------------------------------------------------
// proxy / storage info
// ---------------------------------------------------------------------------

/// `POST /api/proxy/test` (admin).
async fn test_proxy_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let body = body.map(|Json(v)| v).unwrap_or(Value::Null);
    let mut candidate = body_str(&body, "url");
    if candidate.is_empty() {
        candidate = state.config.proxy_setting();
    }
    if candidate.trim().is_empty() {
        return Err(AppError::message(
            StatusCode::BAD_REQUEST,
            "proxy url is required",
        ));
    }
    let result = test_proxy(&candidate, PROXY_TEST_TIMEOUT_SECS).await;
    Ok(json_ok(json!({ "result": result })))
}

/// `GET /api/storage/info` (admin).
async fn get_storage_info(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    Ok(json_ok(json!({
        "backend": state.storage.get_backend_info(),
        "health": state.storage.health_check(),
    })))
}

// ---------------------------------------------------------------------------
// image storage (webdav)
// ---------------------------------------------------------------------------

/// `POST /api/image-storage/test` (admin).
async fn test_image_storage_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let result = state.image_storage.test_webdav().await;
    Ok(json_ok(json!({ "result": result })))
}

/// `POST /api/image-storage/sync` (admin).
async fn sync_image_storage_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let result = state.image_storage.sync_all().await?;
    Ok(json_ok(json!({ "result": result })))
}

// ---------------------------------------------------------------------------
// backups
// ---------------------------------------------------------------------------

/// `POST /api/backup/test` (admin).
async fn test_backup_connection(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let result = state.backup.test_connection().await.map_err(backup_err)?;
    Ok(json_ok(json!({ "result": result })))
}

/// `GET /api/backups` (admin) — list + state + settings.
async fn get_backups(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let items = state.backup.list_backups().await.map_err(backup_err)?;
    Ok(json_ok(json!({
        "items": items,
        "state": state.backup.get_status(),
        "settings": state.backup.get_settings(),
    })))
}

/// `POST /api/backups/run` (admin).
async fn run_backup_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let result = state.backup.run_backup("manual").await.map_err(backup_err)?;
    Ok(json_ok(json!({ "result": result })))
}

/// `POST /api/backups/delete` (admin).
async fn delete_backup_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let body = body.map(|Json(v)| v).unwrap_or(Value::Null);
    let key = body_str(&body, "key");
    state.backup.delete_backup(&key).await.map_err(backup_err)?;
    Ok(json_ok(json!({ "ok": true })))
}

/// `GET /api/backups/detail` (admin).
async fn get_backup_detail(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let key = query_str(&params, "key");
    let item = state.backup.get_backup_detail(&key).await.map_err(backup_err)?;
    Ok(json_ok(json!({ "item": item })))
}

/// `GET /api/backups/download` (admin) — fetch (and decrypt) a backup object.
async fn download_backup_endpoint(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    require_admin(&state, auth_header(&headers))?;
    let key = query_str(&params, "key");
    let item = state.backup.download_backup(&key).await.map_err(backup_err)?;
    let extra = [
        (
            "Content-Disposition",
            format!("attachment; filename*=UTF-8''{}", encode_rfc5987(&item.name)),
        ),
        ("Content-Length", item.size.to_string()),
    ];
    Ok(bytes_response(item.payload, &item.content_type, &extra))
}

// ---------------------------------------------------------------------------
// health dashboard
// ---------------------------------------------------------------------------

/// `GET /health` — JSON (`?format=json`) or an HTML dashboard.
async fn health_dashboard(
    State(state): State<SharedState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let format = params.get("format").map(String::as_str).unwrap_or("html");
    let stats = state.accounts.get_stats();
    let storage_health = state.storage.health_check();
    let backend = state.storage.get_backend_info();
    let version = state.config.app_version();

    let g = |k: &str| stats.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
    let active = g("active");
    let unlimited = g("unlimited_quota_count");
    let healthy = active > 0 || unlimited > 0;

    if format == "json" {
        return json_ok(json!({
            "status": if healthy { "ok" } else { "degraded" },
            "healthy": healthy,
            "version": version,
            "storage": { "backend": backend, "health": storage_health },
            "accounts": stats,
        }));
    }

    // by_type rows, sorted by type name (mirrors `sorted(stats['by_type'].items())`).
    let mut type_rows: Vec<(String, i64)> = stats
        .get("by_type")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0)))
                .collect()
        })
        .unwrap_or_default();
    type_rows.sort_by(|a, b| a.0.cmp(&b.0));
    let type_table: String = type_rows
        .iter()
        .map(|(t, c)| format!("<tr><td>{t}</td><td>{c}</td></tr>"))
        .collect();

    let status_dot = if healthy { "status-ok" } else { "status-degraded" };
    let pool_color = if healthy { "green" } else { "yellow" };
    let pool_text = if healthy { "正常" } else { "异常" };

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>号池健康监控 - chatgpt2api</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}}
.header{{background:#1a1d27;border-bottom:1px solid #2a2d3a;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}}
.header h1{{font-size:20px}}
.status-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}}
.status-ok{{background:#22c55e;box-shadow:0 0 8px #22c55e88}}
.status-degraded{{background:#f59e0b;box-shadow:0 0 8px #f59e0b88}}
.container{{max-width:960px;margin:0 auto;padding:24px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:16px}}
.card .value{{font-size:28px;font-weight:700;margin:4px 0}}
.card .label{{font-size:13px;color:#94a3b8}}
.green{{color:#22c55e}}.yellow{{color:#f59e0b}}.red{{color:#ef4444}}.blue{{color:#6c63ff}}
table{{width:100%;border-collapse:collapse;background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;overflow:hidden}}
th{{background:#242836;font-weight:600;text-align:left;padding:10px 12px;font-size:12px;color:#94a3b8;text-transform:uppercase}}
td{{padding:8px 12px;border-top:1px solid #2a2d3a;font-size:14px}}tr:hover td{{background:rgba(108,99,255,.05)}}
.api-url{{font-family:monospace;font-size:12px;color:#6c63ff}}
.refresh{{font-size:12px;color:#64748b;text-align:center;margin-top:24px}}
</style>
<meta http-equiv="refresh" content="30">
</head>
<body>
<div class="header">
<h1><span class="status-dot {status_dot}"></span>号池健康监控</h1>
<div style="font-size:13px;color:#94a3b8">v{version} · 30s 自动刷新</div>
</div>
<div class="container">
<div class="cards">
<div class="card"><div class="label">号池状态</div><div class="value {pool_color}">{pool_text}</div></div>
<div class="card"><div class="label">当前账号</div><div class="value blue">{total}</div></div>
<div class="card"><div class="label">累计入库</div><div class="value">{cumulative_total}</div></div>
<div class="card"><div class="label">可用账号</div><div class="value green">{active}</div></div>
<div class="card"><div class="label">无限额</div><div class="value">{unlimited}</div></div>
<div class="card"><div class="label">剩余额度</div><div class="value">{total_quota}</div></div>
<div class="card"><div class="label">限流</div><div class="value yellow">{limited}</div></div>
<div class="card"><div class="label">异常</div><div class="value red">{abnormal}</div></div>
<div class="card"><div class="label">禁用</div><div class="value">{disabled}</div></div>
<div class="card"><div class="label">成功/失败</div><div class="value">{total_success}<span style="font-size:18px;color:#94a3b8">/</span><span class="red">{total_fail}</span></div></div>
</div>
<h2 style="margin-bottom:12px;font-size:16px">账号类型分布</h2>
<table>
<tr><th>类型</th><th>数量</th></tr>
{type_table}
</table>
<div class="refresh">JSON: <span class="api-url">/health?format=json</span></div>
</div></body></html>"#,
        total = g("total"),
        cumulative_total = g("cumulative_total"),
        total_quota = g("total_quota"),
        limited = g("limited"),
        abnormal = g("abnormal"),
        disabled = g("disabled"),
        total_success = g("total_success"),
        total_fail = g("total_fail"),
    );

    Html(html).into_response()
}
