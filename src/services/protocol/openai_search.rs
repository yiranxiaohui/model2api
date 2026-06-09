//! Port of `services/protocol/openai_search.py` — the `/v1/search` endpoint.
//!
//! Picks a text account from the pool, runs a web-search conversation through
//! the backend engine, marks the account used, and tags the structured result
//! with the originating account email (`_account_email`).

#![allow(dead_code)]

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::error::AppError;
use crate::services::openai_backend_api::OpenAIBackendAPI;
pub use crate::services::openai_backend_api::SEARCH_MODEL as MODEL;
use crate::services::protocol::conversation::ConvDeps;

/// Run a web search for `body["prompt"]` and return the structured result.
pub async fn search(deps: ConvDeps, body: Value, _base_url: Option<String>) -> Result<Value, AppError> {
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let token = deps.accounts.get_text_access_token(&HashSet::new()).await;
    let account = deps.accounts.get_account(&token).unwrap_or_else(|| json!({}));

    let mut engine = OpenAIBackendAPI::new(deps.config.clone(), token.clone(), account.clone())
        .map_err(|e| AppError::upstream(e.to_string()))?;
    let mut result = engine
        .search(&prompt, None, None, None)
        .await
        .map_err(|e| AppError::upstream(e.to_string()))?;

    deps.accounts.mark_text_used(&token);

    let email = account.get("email").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(map) = result.as_object_mut() {
        map.insert("_account_email".to_string(), json!(email));
    }
    Ok(result)
}
