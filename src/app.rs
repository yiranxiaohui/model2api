//! Port of `api/app.py` — assembles the axum router: API routers, CORS, the
//! SPA/static fallback, and (in later phases) background-task lifespan.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use crate::api;
use crate::state::SharedState;

pub fn build_router(state: SharedState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .merge(api::ai::router())
        .merge(api::accounts::router())
        .merge(api::image_tasks::router())
        .merge(api::register::router())
        .merge(api::system::router())
        // Catch-all: serve web assets / SPA fallback.
        .fallback(serve_web)
        .layer(cors)
        .with_state(state)
}

/// Port of `serve_web`: resolve a static asset under `web_dist`, otherwise fall
/// back to `index.html` (except for `_next/*` which 404s).
async fn serve_web(State(state): State<SharedState>, uri: Uri) -> Response {
    let full_path = uri.path().trim_start_matches('/').to_string();

    if let Some(asset) = api::support::resolve_web_asset(&state.web_dist_dir, &full_path) {
        return file_response(&asset).await;
    }
    if full_path.trim_matches('/').starts_with("_next/") {
        return (StatusCode::NOT_FOUND, "Not Found").into_response();
    }
    match api::support::resolve_web_asset(&state.web_dist_dir, "") {
        Some(fallback) => file_response(&fallback).await,
        None => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}

async fn file_response(path: &std::path::Path) -> Response {
    match tokio::fs::read(path).await {
        Ok(bytes) => {
            let mime = mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string();
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime)
                .body(Body::from(bytes))
                .unwrap()
        }
        Err(_) => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}
