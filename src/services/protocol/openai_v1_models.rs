//! Port of `services/protocol/openai_v1_models.py` — the `/v1/models` listing.
//!
//! The engine's anonymous `list_models` returns the upstream model catalogue
//! (already deduped + sorted into the OpenAI `{object:"list", data:[...]}`
//! shape). On top of that, this layer appends the *dynamically available* image
//! models derived from the configured account pool: `gpt-image-2` when any
//! account exists, and the codex image models gated on the codex plan types
//! (`Plus`/`Team`/`Pro`) present in the pool.
//!
//! Adaptation: Python reached into `account_service._normalize_source_type` /
//! `_normalize_account_type`. Those are private module functions in the Rust
//! `account_service`, so the two normalizers are reimplemented here verbatim.
#![allow(dead_code)]

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::error::AppError;
use crate::services::openai_backend_api::OpenAIBackendAPI;
use crate::services::protocol::conversation::ConvDeps;
use crate::utils::helper::CODEX_IMAGE_MODEL;

/// `account_service.normalize_source_type` (verbatim): lower/trim, empty → "web".
fn normalize_source_type(value: Option<&Value>) -> String {
    let raw = match value {
        Some(Value::String(s)) => s.trim().to_lowercase(),
        None | Some(Value::Null) => String::new(),
        Some(other) => other.to_string().trim().to_lowercase(),
    };
    if raw.is_empty() {
        "web".to_string()
    } else {
        raw
    }
}

/// `account_service.normalize_account_type` (verbatim): map aliases to the
/// canonical plan label (`Plus`/`Team`/`Pro`/...), else echo the raw value.
fn normalize_account_type(value: Option<&Value>) -> Option<String> {
    let raw = match value {
        Some(Value::String(s)) => s.trim().to_string(),
        None | Some(Value::Null) => String::new(),
        Some(other) => other.to_string(),
    };
    if raw.is_empty() {
        return None;
    }
    let key = raw.to_lowercase().replace('-', "_").replace(' ', "_");
    let compact = key.replace('_', "");
    let aliases: &[(&str, &str)] = &[
        ("free", "free"),
        ("plus", "Plus"),
        ("pro", "Pro"),
        ("prolite", "ProLite"),
        ("team", "Team"),
        ("business", "Team"),
        ("enterprise", "Enterprise"),
    ];
    for (k, v) in aliases {
        if *k == compact {
            return Some(v.to_string());
        }
    }
    for (k, v) in aliases {
        if *k == key {
            return Some(v.to_string());
        }
    }
    Some(raw)
}

/// List models, augmenting the upstream catalogue with pool-derived image models.
pub async fn list_models(deps: ConvDeps) -> Result<Value, AppError> {
    // Python uses an anonymous backend (`OpenAIBackendAPI()`); mirror that with
    // an empty access token + empty account.
    let mut engine = OpenAIBackendAPI::new(deps.config.clone(), String::new(), json!({}))
        .map_err(|e| AppError::upstream(e.to_string()))?;
    let mut result = engine
        .list_models()
        .await
        .map_err(|e| AppError::upstream(e.to_string()))?;

    if !result.get("data").map_or(false, |v| v.is_array()) {
        return Ok(result);
    }

    let seen: HashSet<String> = result
        .get("data")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|item| item.get("id").and_then(|v| v.as_str()).map(|s| s.trim().to_string()))
                .collect()
        })
        .unwrap_or_default();

    let accounts = deps.accounts.list_accounts();
    let has_accounts = accounts.iter().any(|a| a.is_object());
    let mut codex_types: HashSet<String> = HashSet::new();
    for account in &accounts {
        if !account.is_object() {
            continue;
        }
        if normalize_source_type(account.get("source_type")) == "codex" {
            if let Some(t) = normalize_account_type(account.get("type")) {
                codex_types.insert(t);
            }
        }
    }

    let mut dynamic: Vec<String> = Vec::new();
    if has_accounts {
        dynamic.push("gpt-image-2".to_string());
    }
    if codex_types.contains("Plus") || codex_types.contains("Team") || codex_types.contains("Pro") {
        dynamic.push(CODEX_IMAGE_MODEL.to_string());
    }
    if codex_types.contains("Plus") {
        dynamic.push(format!("plus-{CODEX_IMAGE_MODEL}"));
    }
    if codex_types.contains("Team") {
        dynamic.push(format!("team-{CODEX_IMAGE_MODEL}"));
    }
    if codex_types.contains("Pro") {
        dynamic.push(format!("pro-{CODEX_IMAGE_MODEL}"));
    }
    dynamic.sort();
    dynamic.dedup();

    if let Some(data) = result.get_mut("data").and_then(|v| v.as_array_mut()) {
        for model in dynamic {
            if !seen.contains(&model) {
                data.push(json!({
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "chatgpt2api",
                    "permission": [],
                    "root": model,
                    "parent": Value::Null,
                }));
            }
        }
    }

    Ok(result)
}
