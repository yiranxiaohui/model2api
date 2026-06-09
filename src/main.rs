//! model2api — Rust port of chatgpt2api's FastAPI backend.
//!
//! Entry point: load configuration, build the axum router, and serve. Mirrors
//! `main.py` / `api/app.py` (uvicorn → tokio + axum).

mod api;
mod app;
mod config;
mod error;
mod scheduler;
mod services;
mod state;
mod utils;

use std::net::SocketAddr;
use std::path::PathBuf;

use config::Config;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let base_dir = base_dir();
    let config = match Config::load(&base_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let version = config.app_version();
    let state = AppState::new(base_dir, config)?;

    // Best-effort startup cleanup (mirrors lifespan startup).
    let removed = state.config.cleanup_old_images();
    if removed > 0 {
        tracing::info!("[startup] cleaned {removed} expired images");
    }

    let router = app::build_router(state.clone());

    // Start background tasks (account watcher, image cleanup, scheduled backup).
    scheduler::spawn_background_tasks(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000);
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr: SocketAddr = format!("{host}:{port}").parse()?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("model2api v{version} listening on http://{addr}");
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,model2api=info"));
    fmt().with_env_filter(filter).with_target(false).init();
}

/// Resolve the application base directory (where `config.json`/`VERSION` live).
/// Uses the current working directory; falls back to the executable's dir.
fn base_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("config.json").exists() || cwd.join("VERSION").exists() {
        return cwd;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            return parent.to_path_buf();
        }
    }
    cwd
}
