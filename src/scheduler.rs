//! Background scheduling (port of the daemon threads started in `api/app.py`'s
//! lifespan + `api/support.start_limited_account_watcher` +
//! `services/image_service.start_image_cleanup_scheduler` +
//! `backup_service.start`). Each runs as a detached tokio task.

use std::time::Duration;

use crate::state::SharedState;

/// Spawn all background tasks. They run for the lifetime of the process.
pub fn spawn_background_tasks(state: SharedState) {
    spawn_account_watcher(state.clone());
    spawn_image_cleanup(state.clone());
    spawn_backup_scheduler(state);
}

/// Refresh limited/expiring tokens and keepalive refresh tokens on an interval
/// (port of `start_limited_account_watcher`).
fn spawn_account_watcher(state: SharedState) {
    tokio::spawn(async move {
        loop {
            let interval_secs =
                (state.config.refresh_account_interval_minute().max(1) * 60) as u64;
            let limited = state.accounts.list_limited_tokens();
            let expiring = state.accounts.list_expiring_access_tokens();
            let keepalive = state.accounts.list_refresh_token_keepalive_tokens();

            // Union of limited + expiring (dedup, preserve order).
            let mut tokens: Vec<String> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for t in limited.iter().chain(expiring.iter()) {
                if seen.insert(t.clone()) {
                    tokens.push(t.clone());
                }
            }
            let expiring_set: std::collections::HashSet<&String> = expiring.iter().collect();
            let keepalive: Vec<String> =
                keepalive.into_iter().filter(|t| !expiring_set.contains(t)).collect();

            if !tokens.is_empty() {
                tracing::info!(
                    "[account-watcher] checking {} limited, {} expiring",
                    limited.len(),
                    expiring.len()
                );
                let _ = state.accounts.refresh_accounts(&tokens, None, true).await;
            }
            if !keepalive.is_empty() {
                tracing::info!("[account-watcher] keepalive {} refresh tokens", keepalive.len());
                let result = state.accounts.keepalive_refresh_tokens(&keepalive).await;
                if let Some(errors) = result.get("errors").and_then(|v| v.as_array()) {
                    if !errors.is_empty() {
                        tracing::warn!("[account-watcher] keepalive errors: {errors:?}");
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    });
}

/// Clean expired images + thumbnails every 30 minutes (port of
/// `_auto_cleanup_worker`).
fn spawn_image_cleanup(state: SharedState) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1800)).await;
            let removed = state.config.cleanup_old_images();
            let thumbs = state.image_service.cleanup_image_thumbnails();
            if removed > 0 || thumbs > 0 {
                tracing::info!("[image-cleanup] removed {removed} images, {thumbs} thumbnails");
            }
        }
    });
}

/// Run scheduled R2 backups when due (port of `backup_service.start`'s loop).
fn spawn_backup_scheduler(state: SharedState) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(300)).await;
            if let Err(e) = state.backup.run_scheduled_backup_if_needed().await {
                tracing::warn!("[backup] scheduled backup failed: {e}");
            }
        }
    });
}
