//! Port of `utils/pow.py` — ChatGPT sentinel proof-of-work token generation.
//!
//! The proof token is the base64 of an 18-element config array (with two slots
//! filled by the iteration counter) whose SHA3-512 digest, prefixed by `seed`,
//! falls under a difficulty target. The exact byte layout matches the Python
//! `_pow_generate` "static segment" construction so the produced JSON is
//! identical except for time/uuid/random fields (which the server does not
//! cross-check — only the hash validity matters).

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use chrono::{FixedOffset, Utc};
use rand::seq::SliceRandom;
use rand::Rng;
use serde_json::{json, Value};
use sha3::{Digest, Sha3_512};

pub const DEFAULT_POW_SCRIPT: &str = "https://chatgpt.com/backend-api/sentinel/sdk.js";

const CORES: [i64; 4] = [8, 16, 24, 32];
const DOCUMENT_KEYS: [&str; 2] = ["_reactListeningo743lnnpvdg", "location"];

const POW_LIMIT: usize = 500_000;

/// Parse the bootstrap HTML for PoW script sources and the `data-build` id.
pub fn parse_pow_resources(html: &str) -> (Vec<String>, String) {
    let mut sources = Vec::new();
    let mut data_build = String::new();

    // Extract <script src="..."> values.
    let src_re = regex::Regex::new(r#"<script[^>]*\ssrc=["']([^"']+)["']"#).unwrap();
    for cap in src_re.captures_iter(html) {
        let src = cap[1].to_string();
        // data_build pattern: c/<...>/_
        if data_build.is_empty() {
            if let Some(m) = regex::Regex::new(r"c/[^/]*/_").unwrap().find(&src) {
                data_build = m.as_str().to_string();
            }
        }
        sources.push(src);
    }
    if sources.is_empty() {
        sources.push(DEFAULT_POW_SCRIPT.to_string());
    }
    if data_build.is_empty() {
        if let Some(cap) = regex::Regex::new(r#"<html[^>]*data-build="([^"]*)""#)
            .unwrap()
            .captures(html)
        {
            data_build = cap[1].to_string();
        }
    }
    (sources, data_build)
}

fn legacy_parse_time() -> String {
    // Eastern (GMT-0500) — matches Python's fixed -5h offset.
    let offset = FixedOffset::west_opt(5 * 3600).unwrap();
    let now = Utc::now().with_timezone(&offset);
    format!(
        "{} GMT-0500 (Eastern Standard Time)",
        now.format("%a %b %d %Y %H:%M:%S")
    )
}

const NAVIGATOR_KEYS: &[&str] = &[
    "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
    "storage−[object StorageManager]",
    "locks−[object LockManager]",
    "appCodeName−Mozilla",
    "permissions−[object Permissions]",
    "share−function share() { [native code] }",
    "webdriver−false",
    "managed−[object NavigatorManagedData]",
    "canShare−function canShare() { [native code] }",
    "vendor−Google Inc.",
    "mediaDevices−[object MediaDevices]",
    "vibrate−function vibrate() { [native code] }",
    "storageBuckets−[object StorageBucketManager]",
    "mediaCapabilities−[object MediaCapabilities]",
    "cookieEnabled−true",
    "virtualKeyboard−[object VirtualKeyboard]",
    "product−Gecko",
    "presentation−[object Presentation]",
    "onLine−true",
    "mimeTypes−[object MimeTypeArray]",
    "credentials−[object CredentialsContainer]",
    "serviceWorker−[object ServiceWorkerContainer]",
    "keyboard−[object Keyboard]",
    "gpu−[object GPU]",
    "doNotTrack",
    "serial−[object Serial]",
    "pdfViewerEnabled−true",
    "language−zh-CN",
    "geolocation−[object Geolocation]",
    "userAgentData−[object NavigatorUAData]",
    "getUserMedia−function getUserMedia() { [native code] }",
    "sendBeacon−function sendBeacon() { [native code] }",
    "hardwareConcurrency−32",
    "windowControlsOverlay−[object WindowControlsOverlay]",
];

const WINDOW_KEYS: &[&str] = &[
    "0", "window", "self", "document", "name", "location", "customElements", "history",
    "navigation", "innerWidth", "innerHeight", "scrollX", "scrollY", "visualViewport", "screenX",
    "screenY", "outerWidth", "outerHeight", "devicePixelRatio", "screen", "chrome", "navigator",
    "onresize", "performance", "crypto", "indexedDB", "sessionStorage", "localStorage", "scheduler",
    "alert", "atob", "btoa", "fetch", "matchMedia", "postMessage", "queueMicrotask",
    "requestAnimationFrame", "setInterval", "setTimeout", "caches", "__NEXT_DATA__",
    "__BUILD_MANIFEST", "__NEXT_PRELOADREADY",
];

/// Build the 18-element PoW config array (indices 3 and 9 are placeholders that
/// `pow_generate` overwrites with the iteration counter).
pub fn build_pow_config(user_agent: &str, script_sources: &[String], data_build: &str) -> Vec<Value> {
    let mut rng = rand::thread_rng();
    let navigator_key = *NAVIGATOR_KEYS.choose(&mut rng).unwrap();
    let window_key = *WINDOW_KEYS.choose(&mut rng).unwrap();
    let document_key = *DOCUMENT_KEYS.choose(&mut rng).unwrap();
    let script_source = if script_sources.is_empty() {
        DEFAULT_POW_SCRIPT.to_string()
    } else {
        script_sources.choose(&mut rng).unwrap().clone()
    };
    let timeout = *[3000i64, 4000, 5000].choose(&mut rng).unwrap();
    let cores = *CORES.choose(&mut rng).unwrap();
    let perf_ms = perf_counter_ms();
    let clock_offset = wall_ms() - perf_ms;

    vec![
        json!(timeout),
        json!(legacy_parse_time()),
        json!(4294705152u64),
        json!(0),
        json!(user_agent),
        json!(script_source),
        json!(data_build),
        json!("en-US"),
        json!("en-US,es-US,en,es"),
        json!(0),
        json!(navigator_key),
        json!(document_key),
        json!(window_key),
        json!(perf_ms),
        json!(uuid::Uuid::new_v4().to_string()),
        json!(""),
        json!(cores),
        json!(clock_offset),
    ]
}

fn perf_counter_ms() -> f64 {
    // Monotonic-ish clock in milliseconds since an arbitrary epoch.
    use std::time::Instant;
    use once_cell::sync::Lazy;
    static START: Lazy<Instant> = Lazy::new(Instant::now);
    START.elapsed().as_secs_f64() * 1000.0 + 1.0
}

fn wall_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// Solve the PoW. Returns `(answer_base64, solved)`. `answer` is the base64 of
/// the final JSON payload.
pub fn pow_generate(seed: &str, difficulty: &str, config: &[Value]) -> (String, bool) {
    let target = hex::decode(difficulty).unwrap_or_default();
    let diff_len = difficulty.len() / 2;
    let seed_bytes = seed.as_bytes();

    // Static segments matching Python's slicing of the compact JSON.
    let head = serde_json::to_string(&config[0..3]).unwrap(); // [c0,c1,c2]
    let static_1 = format!("{},", &head[..head.len() - 1]); // [c0,c1,c2,
    let mid = serde_json::to_string(&config[4..9]).unwrap(); // [c4..c8]
    let static_2 = format!(",{},", &mid[1..mid.len() - 1]); // ,c4..c8,
    let tail = serde_json::to_string(&config[10..18]).unwrap(); // [c10..c17]
    let static_3 = format!(",{}", &tail[1..]); // ,c10..c17]

    let mut hasher = Sha3_512::new();
    for i in 0..POW_LIMIT {
        let mut buf = Vec::with_capacity(static_1.len() + static_2.len() + static_3.len() + 24);
        buf.extend_from_slice(static_1.as_bytes());
        buf.extend_from_slice(itoa(i).as_bytes());
        buf.extend_from_slice(static_2.as_bytes());
        buf.extend_from_slice(itoa(i >> 1).as_bytes());
        buf.extend_from_slice(static_3.as_bytes());

        let encoded = BASE64_STANDARD.encode(&buf);

        hasher.update(seed_bytes);
        hasher.update(encoded.as_bytes());
        let digest = hasher.finalize_reset();

        if digest[..diff_len] <= target[..] {
            return (encoded, true);
        }
    }
    let fallback = format!(
        "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D{}",
        BASE64_STANDARD.encode(format!("\"{seed}\""))
    );
    (fallback, false)
}

fn itoa(v: usize) -> String {
    v.to_string()
}

/// Build the legacy chat-requirements token (`gAAAAAC` + answer).
pub fn build_legacy_requirements_token(
    user_agent: &str,
    script_sources: &[String],
    data_build: &str,
) -> String {
    let seed = format!("{}", rand::thread_rng().gen::<f64>());
    let config = build_pow_config(user_agent, script_sources, data_build);
    let (answer, _) = pow_generate(&seed, "0fffff", &config);
    format!("gAAAAAC{answer}")
}

/// Build a full proof token (`gAAAAAB` + answer). Errors if unsolved.
pub fn build_proof_token(
    seed: &str,
    difficulty: &str,
    user_agent: &str,
    script_sources: &[String],
    data_build: &str,
) -> anyhow::Result<String> {
    let config = build_pow_config(user_agent, script_sources, data_build);
    let (answer, solved) = pow_generate(seed, difficulty, &config);
    if !solved {
        anyhow::bail!("failed to solve proof token: difficulty={difficulty}");
    }
    Ok(format!("gAAAAAB{answer}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_solves_easy_difficulty_and_is_self_consistent() {
        // A lenient target so it solves quickly; verify the returned base64,
        // re-hashed with the seed, actually meets the difficulty.
        let cfg = build_pow_config("UA/1.0", &[DEFAULT_POW_SCRIPT.to_string()], "c/x/_");
        let seed = "test-seed";
        let difficulty = "0f"; // 1 byte, <= 0x0f
        let (answer, solved) = pow_generate(seed, difficulty, &cfg);
        assert!(solved, "should solve 1-byte difficulty within limit");
        let decoded = BASE64_STANDARD.decode(answer.as_bytes()).unwrap();
        let mut h = Sha3_512::new();
        h.update(seed.as_bytes());
        h.update(BASE64_STANDARD.encode(&decoded).as_bytes());
        let digest = h.finalize();
        assert!(digest[0] <= 0x0f);
        // Decoded payload must be valid JSON (an 18-element array).
        let v: Value = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 18);
    }

    #[test]
    fn config_layout_matches_python_segments() {
        let cfg = build_pow_config("UA", &["s".to_string()], "b");
        // Reconstruct the full payload for i=7 and confirm it equals the array
        // with slots 3 and 9 overwritten.
        let head = serde_json::to_string(&cfg[0..3]).unwrap();
        let static_1 = format!("{},", &head[..head.len() - 1]);
        let mid = serde_json::to_string(&cfg[4..9]).unwrap();
        let static_2 = format!(",{},", &mid[1..mid.len() - 1]);
        let tail = serde_json::to_string(&cfg[10..18]).unwrap();
        let static_3 = format!(",{}", &tail[1..]);
        let payload = format!("{static_1}7{static_2}3{static_3}");
        let v: Value = serde_json::from_str(&payload).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 18);
        assert_eq!(arr[3], json!(7));
        assert_eq!(arr[9], json!(3));
        assert_eq!(arr[4], cfg[4]);
    }
}
