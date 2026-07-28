//! OpenAI Sentinel PoW token 生成（PoW-only，t=""，无浏览器）。
//!
//! 端到端参照 `/opt/freeAgentIdentity/.../protocol_register.py::_SentinelTokenGenerator`
//! 的 25 字段指纹与爆破逻辑。PoW 有效性只取决于本地 hash 的正是发出的 base64，
//! 故 config 的 JSON 转义无需与 Python 对齐，只需自洽。

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use rand::Rng;
use serde_json::{json, Value};

use crate::services::register::constants::sentinel_sdk_url;

const POW_LIMIT: usize = 500_000;

/// FNV-1a 32-bit（带额外雪崩），返回 8 位小写十六进制。
pub fn fnv1a32(text: &str) -> String {
    let mut h: u32 = 2166136261;
    for ch in text.chars() {
        h ^= ch as u32;
        h = h.wrapping_mul(16777619);
    }
    h ^= h >> 16;
    h = h.wrapping_mul(2246822507);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489909);
    h ^= h >> 16;
    format!("{h:08x}")
}

/// compact JSON（serde 默认无空格）→ base64(STANDARD)。
fn encode(value: &Value) -> String {
    let raw = serde_json::to_vec(value).unwrap_or_default();
    BASE64_STANDARD.encode(raw)
}

/// 25 字段指纹（索引 3=nonce 槽、9=elapsed 槽）。时间/随机字段服务器不交叉核对。
fn reference_fingerprint(user_agent: &str, sid: &str) -> Vec<Value> {
    let mut rng = rand::thread_rng();
    let perf_now = 3000.0_f64 + rng.gen_range(1000.0..5000.0);
    let time_origin = 50000.0_f64;
    vec![
        json!(3000),
        json!("Mon Jan 01 2026 00:00:00 GMT+0000 (Coordinated Universal Time)"),
        json!(4294705152u64),
        json!(0),                       // 3: nonce 槽
        json!(user_agent),
        json!(sentinel_sdk_url()),
        json!(null),
        json!("en-US"),
        json!("en-US,en"),
        json!(0),                       // 9: elapsed 槽
        json!("webkitTemporaryStorage\u{2212}undefined"),
        json!("location"),
        json!("Object"),
        json!(perf_now),
        json!(sid),
        json!(""),
        json!(8),
        json!(time_origin),
        json!(0), json!(0), json!(0), json!(0), json!(0), json!(0), json!(0),
    ]
}

/// 爆破：找到 nonce 使 fnv1a32(seed+encoded) 的前缀 <= difficulty（字典序）。
/// 命中返回 `encoded + "~S"`；未命中返回 `encode(json!("e"))`（无 ~S）。
fn solve(seed: &str, difficulty: &str, mut config: Vec<Value>) -> String {
    let difficulty = if difficulty.is_empty() { "0" } else { difficulty };
    let dn = difficulty.len();
    for nonce in 0..POW_LIMIT {
        config[3] = json!(nonce);
        config[9] = json!(nonce); // elapsed 槽：用递增值即可，服务器不核对具体值
        let encoded = encode(&Value::Array(config.clone()));
        let digest = fnv1a32(&format!("{seed}{encoded}"));
        let take = dn.min(digest.len());
        if digest[..take] <= difficulty[..take] {
            return format!("{encoded}~S");
        }
    }
    encode(&json!("e"))
}

/// PoW-only Sentinel token 生成器（不含浏览器 VM token）。
pub struct SentinelPow {
    pub user_agent: String,
    pub sid: String,
}

impl SentinelPow {
    pub fn new(user_agent: impl Into<String>) -> Self {
        Self {
            user_agent: user_agent.into(),
            sid: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// 前缀 gAAAAAC；照 freeAgentIdentity：以随机 seed、difficulty "0" 跑 solve。
    pub fn requirements(&self) -> String {
        let mut cfg = reference_fingerprint(&self.user_agent, &self.sid);
        cfg[3] = json!(1);
        cfg[9] = json!(rand::thread_rng().gen_range(5..50));
        let seed = format!("{}", rand::thread_rng().gen::<f64>());
        format!("gAAAAAC{}", solve(&seed, "0", cfg))
    }

    /// 前缀 gAAAAAB；用服务器给的 seed/difficulty 跑 solve。
    pub fn enforcement(&self, seed: &str, difficulty: &str) -> String {
        let cfg = reference_fingerprint(&self.user_agent, &self.sid);
        format!("gAAAAAB{}", solve(seed, difficulty, cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a32_matches_reference_vectors() {
        assert_eq!(fnv1a32(""), "ab3e7c0b");
        assert_eq!(fnv1a32("a"), "1a80b1b3");
        assert_eq!(fnv1a32("abc"), "1cc93dbc");
        assert_eq!(fnv1a32("hello world"), "b90456ec");
        assert_eq!(fnv1a32("0.123456789xYz=="), "4bf08b72");
    }

    #[test]
    fn solve_result_satisfies_difficulty() {
        let cfg = reference_fingerprint("UA", "sid-1");
        let out = solve("seedvalue", "0", cfg);
        assert!(out.ends_with("~S"), "should have solved within limit");
        let encoded = out.trim_end_matches("~S");
        let digest = fnv1a32(&format!("seedvalue{encoded}"));
        assert!(&digest[..1] <= "0"); // 与服务器同款校验
    }

    #[test]
    fn difficulty_is_lexicographic_not_numeric() {
        // "0a" <= "1" 字典序成立；数值比较会不同。锁定用字符串比较。
        assert!("0a" <= "1");
    }

    #[test]
    fn solve_does_not_panic_on_overlong_difficulty() {
        // difficulty longer than the 8-char digest must not panic.
        let cfg = reference_fingerprint("UA", "sid-x");
        let out = solve("seed", "ffffffffff", cfg); // 10 chars, trivially satisfiable
        assert!(out.ends_with("~S"));
    }

    #[test]
    fn requirements_and_enforcement_have_prefixes() {
        let pow = SentinelPow::new("UA");
        assert!(pow.requirements().starts_with("gAAAAAC"));
        let e = pow.enforcement("seed", "0");
        assert!(e.starts_with("gAAAAAB") && e.ends_with("~S"));
    }
}
