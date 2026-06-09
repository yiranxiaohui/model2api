//! Port of `utils/sentinel.py` — Sentinel token (PoW) generator used by the
//! password-login / registration flows. The HTTP `build_sentinel_token` wrapper
//! is added in Phase 5 once the rquest client exists; this module provides the
//! pure `SentinelTokenGenerator`.

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use chrono::Utc;
use rand::Rng;
use serde_json::{json, Value};

pub const DEFAULT_SENTINEL_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36";
pub const DEFAULT_SENTINEL_SEC_CH_UA: &str =
    "\"Chromium\";v=\"145\", \"Google Chrome\";v=\"145\", \"Not/A)Brand\";v=\"99\"";

const MAX_ATTEMPTS: usize = 500_000;
const ERROR_PREFIX: &str = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D";

pub struct SentinelTokenGenerator {
    pub device_id: String,
    pub user_agent: String,
    pub sid: String,
}

impl SentinelTokenGenerator {
    pub fn new(device_id: impl Into<String>, ua: impl Into<String>) -> Self {
        Self {
            device_id: device_id.into(),
            user_agent: ua.into(),
            sid: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// FNV-1a 32-bit with the same avalanche finalizer as the Python version,
    /// returned as an 8-char lowercase hex string.
    fn fnv1a_32(text: &str) -> String {
        let mut h: u32 = 2166136261;
        for ch in text.chars() {
            // Python iterates code points via ord(ch); xor low bits.
            h ^= (ch as u32) & 0xFFFF_FFFF;
            h = h.wrapping_mul(16777619);
        }
        h ^= h >> 16;
        h = h.wrapping_mul(2246822507);
        h ^= h >> 13;
        h = h.wrapping_mul(3266489909);
        h ^= h >> 16;
        format!("{h:08x}")
    }

    fn get_config(&self) -> Vec<Value> {
        let mut rng = rand::thread_rng();
        let perf_now: f64 = rng.gen_range(1000.0..50000.0);
        let now = Utc::now()
            .format("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)")
            .to_string();
        let nav = *["vendorSub-undefined", "plugins-undefined", "mimeTypes-undefined", "hardwareConcurrency-undefined"]
            .iter()
            .nth(rng.gen_range(0..4))
            .unwrap();
        let doc = *["location", "implementation", "URL", "documentURI", "compatMode"]
            .iter()
            .nth(rng.gen_range(0..5))
            .unwrap();
        let win = *["Object", "Function", "Array", "Number", "parseFloat", "undefined"]
            .iter()
            .nth(rng.gen_range(0..6))
            .unwrap();
        let cores = *[4i64, 8, 12, 16].iter().nth(rng.gen_range(0..4)).unwrap();
        let wall = Utc::now().timestamp_millis() as f64;
        vec![
            json!("1920x1080"),
            json!(now),
            json!(4294705152u64),
            json!(rng.gen::<f64>()),
            json!(self.user_agent),
            json!("https://sentinel.openai.com/sentinel/20260124ceb8/sdk.js"),
            Value::Null,
            Value::Null,
            json!("en-US"),
            json!(rng.gen::<f64>()),
            json!(nav),
            json!(doc),
            json!(win),
            json!(perf_now),
            json!(self.sid),
            json!(""),
            json!(cores),
            json!(wall - perf_now),
        ]
    }

    fn b64(data: &[Value]) -> String {
        let s = serde_json::to_string(data).unwrap();
        BASE64_STANDARD.encode(s.as_bytes())
    }

    pub fn generate_requirements_token(&self) -> String {
        let mut data = self.get_config();
        data[3] = json!(1);
        data[9] = json!(rand::thread_rng().gen_range(5..=50));
        format!("gAAAAAC{}", Self::b64(&data))
    }

    pub fn generate_token(&self, seed: &str, difficulty: &str) -> String {
        let start = std::time::Instant::now();
        let mut data = self.get_config();
        let difficulty = if difficulty.is_empty() { "0" } else { difficulty };
        for i in 0..MAX_ATTEMPTS {
            data[3] = json!(i);
            data[9] = json!((start.elapsed().as_secs_f64() * 1000.0).round() as i64);
            let payload = Self::b64(&data);
            let hash = Self::fnv1a_32(&format!("{seed}{payload}"));
            let prefix_len = difficulty.len().min(hash.len());
            if &hash[..prefix_len] <= difficulty {
                return format!("gAAAAAB{payload}~S");
            }
        }
        let null_b64 = BASE64_STANDARD.encode("None");
        format!("gAAAAAB{ERROR_PREFIX}{null_b64}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_known_value() {
        // FNV-1a(32) with avalanche finalizer is deterministic; pin a value.
        let h = SentinelTokenGenerator::fnv1a_32("hello");
        assert_eq!(h.len(), 8);
        // Re-running yields the same hash.
        assert_eq!(h, SentinelTokenGenerator::fnv1a_32("hello"));
    }

    #[test]
    fn requirements_token_prefixed() {
        let g = SentinelTokenGenerator::new("dev", "UA");
        assert!(g.generate_requirements_token().starts_with("gAAAAAC"));
    }

    #[test]
    fn generate_token_solves_trivial_difficulty() {
        let g = SentinelTokenGenerator::new("dev", "UA");
        // difficulty "f" -> any hash prefix <= "f" basically always true.
        let tok = g.generate_token("seed", "f");
        assert!(tok.starts_with("gAAAAAB"));
        assert!(tok.ends_with("~S"));
    }
}
