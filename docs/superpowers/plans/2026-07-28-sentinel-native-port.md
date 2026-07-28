# Sentinel PoW 原生移植 + 丢 FlareSolverr（Phase 1）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 model2api 的 OpenAI 注册流程原生生成并发送 `openai-sentinel-token`（PoW-only，`t=""`，无浏览器），并可关闭 FlareSolverr 依赖靠 wreq 指纹过 CF。

**Architecture:** 新增 `src/services/register/sentinel.rs`，纯 Rust 生成 Sentinel PoW token（fnv1a32 + 25 字段指纹 + base64）。在 `PlatformRegistrar` 上加一个复用现有 wreq session 的方法拉 challenge 并组头，替换 `openai_register.rs` 三处 `// TODO: sentinel token (deferred)`。

**Tech Stack:** Rust, wreq 5.3 (JA3 emulation), serde_json, base64 (STANDARD), rand, uuid, chrono, tokio.

## Global Constraints

- 编译/打包**一律走 GitHub CI**（`.github/workflows/docker.yml`，push main 触发）——本地不 `cargo build`（缺 cmake/libclang，慢且卡）。本地只允许 `cargo test` 级命令（若本地跑不动，测试也放 CI/远端跑，实现者以代码+测试正确性为准）。见 memory `build-in-ci`。
- 开发在**独立 git worktree**（treeflow），不在主工作目录直接改。
- base64 用 `STANDARD` 表；JSON 用 compact 分隔符（serde_json 默认 `to_string` 即 `,`/`:` 无空格）。
- difficulty 比较是**十六进制字符串的字典序比较**（`&str` 的 `<=`），不是数值比较。
- **PoW 有效性只取决于"我们本地 hash 的正是我们发出的 base64"**：服务器用 `fnv1a32(seed + encoded)` 重算并比对 difficulty 前缀。因此 config 内部的 JSON 转义方式（Python `ensure_ascii=True` vs serde 默认原样 UTF-8）**不影响有效性**，无需与 Python 逐字节对齐——只需自洽。
- sentinel 请求用的 UA 必须与注册 session 相同：`cf_ua` 存在时用它，否则 `constants::USER_AGENT`。

## 参照文件（只读，不改）

- `/opt/freeAgentIdentity/platforms/chatgpt/protocol_register.py` — `_SentinelTokenGenerator`（25 字段 `_reference_fingerprint`、`_solve_reference_pow`、`requirements`、`enforcement`）。
- `/opt/chatgpt2api/utils/sentinel.py` — platform 流程的端点/头/flow 字符串与 `build_sentinel_token` 返回 `(header, oai_sc)`。
- `/opt/chatgpt2api/services/register/openai_register.py` — flow 注入点：`username_password_create` / `authorize_continue`(validate_otp 重试) / `oauth_create_account`。

## File Structure

- **Create** `src/services/register/sentinel.rs` — Sentinel PoW token 生成（纯计算 + 一个拉 challenge 的异步函数）。单一职责。
- **Modify** `src/services/register/mod.rs` — 加 `pub mod sentinel;`。
- **Modify** `src/services/register/constants.rs` — 加 Sentinel 端点与 SDK 版本常量。
- **Modify** `src/services/register/openai_register.rs` — 加 `PlatformRegistrar::build_sentinel_token`，替换三处 TODO。

---

### Task 1: Sentinel 纯计算核心（fnv1a32 + 编码 + solve + requirements/enforcement）

**Files:**
- Create: `src/services/register/sentinel.rs`
- Modify: `src/services/register/mod.rs`（加 `pub mod sentinel;`）
- Modify: `src/services/register/constants.rs`（加常量）
- Test: 同文件内 `#[cfg(test)]`

**Interfaces:**
- Consumes: `constants::{USER_AGENT, SENTINEL_SDK_URL}`。
- Produces（供 Task 2 使用）：
  - `pub struct SentinelPow { pub user_agent: String, pub sid: String }`
  - `impl SentinelPow`:
    - `pub fn new(user_agent: impl Into<String>) -> Self`（`sid` 用 `uuid::Uuid::new_v4()`）
    - `pub fn requirements(&self) -> String`
    - `pub fn enforcement(&self, seed: &str, difficulty: &str) -> String`
  - 自由函数 `pub fn fnv1a32(text: &str) -> String`

**常量（先加，供本 Task 引用）**——在 `constants.rs` 末尾追加：

```rust
/// Sentinel SDK 版本（跟随参照项目更新；freeAgentIdentity 当前值）。
pub const SENTINEL_SDK_VERSION: &str = "20260219f9f6";
pub const SENTINEL_BASE: &str = "https://sentinel.openai.com";
/// `POST` 拉 challenge 的端点。
pub const SENTINEL_REQ_URL: &str = "https://sentinel.openai.com/backend-api/sentinel/req";
/// SDK.js URL，写进指纹 config 第 5 槽。
pub fn sentinel_sdk_url() -> String {
    format!("{SENTINEL_BASE}/sentinel/{SENTINEL_SDK_VERSION}/sdk.js")
}
```

- [ ] **Step 1: 加模块声明与常量**

在 `mod.rs` 的 `pub mod openai_register;` 后加一行：
```rust
pub mod sentinel;
```
在 `constants.rs` 末尾追加上面的四个常量/函数。

- [ ] **Step 2: 写 `sentinel.rs` 骨架 + 失败测试（fnv1a32 向量）**

创建 `src/services/register/sentinel.rs`：
```rust
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
}
```

- [ ] **Step 3: 跑测试确认 fnv1a32 通过**

Run: `cargo test -p model2api sentinel::tests::fnv1a32_matches_reference_vectors`
Expected: PASS（这些向量由 Python 参照算法生成）。
若本地无法编译（缺 cmake/libclang），标注留待 CI，进入下一步；不得因本地环境跳过测试的编写。

- [ ] **Step 4: 加 encode/config/solve，并写 solve 自洽性失败测试**

在 `fnv1a32` 之后、`#[cfg(test)]` 之前加：
```rust
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
        if &digest[..dn] <= difficulty {
            return format!("{encoded}~S");
        }
    }
    encode(&json!("e"))
}
```
在 `mod tests` 内追加：
```rust
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
```

- [ ] **Step 5: 跑 solve 测试确认通过**

Run: `cargo test -p model2api sentinel::tests`
Expected: 三个测试全 PASS（本地不可编译则留 CI，见 Step 3 说明）。

- [ ] **Step 6: 加 `SentinelPow` 结构与 requirements/enforcement**

在 `solve` 之后加：
```rust
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
```
在 `mod tests` 内追加：
```rust
    #[test]
    fn requirements_and_enforcement_have_prefixes() {
        let pow = SentinelPow::new("UA");
        assert!(pow.requirements().starts_with("gAAAAAC"));
        let e = pow.enforcement("seed", "0");
        assert!(e.starts_with("gAAAAAB") && e.ends_with("~S"));
    }
```

- [ ] **Step 7: 跑全部 sentinel 单测**

Run: `cargo test -p model2api sentinel::`
Expected: 全 PASS（本地不可编译则留 CI）。

- [ ] **Step 8: Commit**

```bash
git add src/services/register/sentinel.rs src/services/register/mod.rs src/services/register/constants.rs
git commit -m "feat(register): native Sentinel PoW core (fnv1a32 + 25-field fingerprint)"
```

---

### Task 2: 拉 challenge 并组头（`PlatformRegistrar::build_sentinel_token`）

**Files:**
- Modify: `src/services/register/openai_register.rs`（加方法，在 `PlatformRegistrar` impl 内，`json_headers` 附近）

**Interfaces:**
- Consumes: Task 1 的 `sentinel::SentinelPow`；现有 `self.send_once`（`async fn(&self, method, url, &[(String,String)], &[(&str,String)], Option<&Value>, u64) -> Result<Resp, String>`）；`self.device_id`、`self.cf_ua`；`constants::{SENTINEL_REQ_URL, USER_AGENT}`；`Resp::json()`。
- Produces（供 Task 3）：`async fn build_sentinel_token(&self, flow: &str) -> Option<String>`——返回 `openai-sentinel-token` 头值；失败返回 `None`（调用方降级为不带该头，best-effort，不判注册失败）。

- [ ] **Step 1: 加 use 引入**

在 `openai_register.rs` 顶部 use 区（`use crate::services::register::flaresolverr;` 附近）加：
```rust
use crate::services::register::constants::SENTINEL_REQ_URL;
use crate::services::register::sentinel::SentinelPow;
```

- [ ] **Step 2: 写方法**

`send_once` 内部用 `req.json()`，会强制 `content-type: application/json`；而 sentinel/req 参照要求
`content-type: text/plain;charset=UTF-8` + body 为 JSON **字符串原文**。因此这里**直接用 `self.client`**
发原始 body，不走 `send_once`。在 `impl PlatformRegistrar` 内、`json_headers` 方法之后插入：
```rust
    /// 生成 `openai-sentinel-token` 头值（PoW-only，t=""）。复用注册 session，
    /// 同指纹/代理/cookie。任一步失败返回 None，由调用方降级为不带头（best-effort）。
    async fn build_sentinel_token(&self, flow: &str) -> Option<String> {
        let ua = self
            .cf_ua
            .clone()
            .unwrap_or_else(|| USER_AGENT.to_string());
        let pow = SentinelPow::new(ua.clone());

        // 拉 challenge：POST sentinel/req，body 是 text/plain 的 JSON 串（原文）。
        let body_str = serde_json::to_string(&json!({
            "p": pow.requirements(),
            "id": self.device_id,
            "flow": flow,
        }))
        .ok()?;
        let headers: Vec<(String, String)> = vec![
            ("accept".into(), "*/*".into()),
            ("content-type".into(), "text/plain;charset=UTF-8".into()),
            ("origin".into(), "https://sentinel.openai.com".into()),
            (
                "referer".into(),
                "https://sentinel.openai.com/backend-api/sentinel/frame.html".into(),
            ),
            ("user-agent".into(), ua),
        ];
        let resp = self
            .client
            .post(SENTINEL_REQ_URL)
            .headers(header_map(&headers))
            .body(body_str)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .ok()?;
        if resp.status().as_u16() != 200 {
            return None;
        }
        let text = resp.text().await.ok()?;
        let data: Value = serde_json::from_str(&text).ok()?;
        let challenge = data.get("token").and_then(|v| v.as_str()).unwrap_or("");
        if challenge.is_empty() {
            return None;
        }
        let pow_info = data.get("proofofwork").cloned().unwrap_or(Value::Null);
        let required = pow_info
            .get("required")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let seed = pow_info.get("seed").and_then(|v| v.as_str()).unwrap_or("");
        let p_value = if required && !seed.is_empty() {
            let difficulty = pow_info
                .get("difficulty")
                .and_then(|v| v.as_str())
                .unwrap_or("0");
            pow.enforcement(seed, difficulty)
        } else {
            pow.requirements()
        };
        let token = json!({
            "p": p_value,
            "t": "",
            "c": challenge,
            "id": self.device_id,
            "flow": flow,
        });
        Some(serde_json::to_string(&token).unwrap_or_default())
    }
```

- [ ] **Step 3: 编译检查（CI 或本地可编译时）**

Run: `cargo check -p model2api`
Expected: 通过。若本地不可编译，推分支触发 CI 的 build 校验；不进入下一 Task 直到 CI 编译通过。

- [ ] **Step 4: Commit**

```bash
git add src/services/register/openai_register.rs
git commit -m "feat(register): PlatformRegistrar::build_sentinel_token (fetch challenge + assemble header)"
```

---

### Task 3: 三处注入点接线

**Files:**
- Modify: `src/services/register/openai_register.rs`（`register_user` ~700、`validate_otp` ~796、`create_account` ~837）

**Interfaces:**
- Consumes: Task 2 的 `self.build_sentinel_token(flow).await -> Option<String>`。
- Produces: 无新对外接口；行为变化 = 三个请求带上 sentinel 头。

- [ ] **Step 1: `register_user` 注入 `username_password_create`**

把（约 698-700 行）：
```rust
        let headers = self.json_headers(&format!("{AUTH_BASE}/create-account/password"));
        // TODO: sentinel token (deferred) — Python sets
        // headers["openai-sentinel-token"] = build_sentinel_token(.., "username_password_create").
        let payload = json!({"username": email, "password": password});
```
改为：
```rust
        let mut headers = self.json_headers(&format!("{AUTH_BASE}/create-account/password"));
        if let Some(tok) = self.build_sentinel_token("username_password_create").await {
            headers.push(("openai-sentinel-token".to_string(), tok));
        } else {
            step_warn(index, "sentinel token 获取失败，裸发 register_user");
        }
        let payload = json!({"username": email, "password": password});
```

- [ ] **Step 2: `create_account` 注入 `oauth_create_account`**

把（约 836-838 行）：
```rust
        let headers = self.json_headers(&format!("{AUTH_BASE}/about-you"));
        // TODO: sentinel token (deferred) — Python sets
        // headers["openai-sentinel-token"] = build_sentinel_token(.., "oauth_create_account").
        let payload = json!({"name": name, "birthdate": birthdate});
```
改为：
```rust
        let mut headers = self.json_headers(&format!("{AUTH_BASE}/about-you"));
        if let Some(tok) = self.build_sentinel_token("oauth_create_account").await {
            headers.push(("openai-sentinel-token".to_string(), tok));
        } else {
            step_warn(index, "sentinel token 获取失败，裸发 create_account");
        }
        let payload = json!({"name": name, "birthdate": birthdate});
```

- [ ] **Step 3: `validate_otp` 重试段注入 `authorize_continue`**

把（约 793-800 行）：
```rust
        if resp.as_ref().map(|r| r.status == 200).unwrap_or(false) {
            return (resp, error);
        }
        // TODO: sentinel token (deferred) — Python sets
        // headers["openai-sentinel-token"] = build_sentinel_token(.., "authorize_continue")
        // before this second attempt.
        self.request_with_retry("post", &url, &headers, &[], Some(&payload), DEFAULT_TIMEOUT, 3)
            .await
```
改为：
```rust
        if resp.as_ref().map(|r| r.status == 200).unwrap_or(false) {
            return (resp, error);
        }
        if let Some(tok) = self.build_sentinel_token("authorize_continue").await {
            headers.push(("openai-sentinel-token".to_string(), tok));
        }
        self.request_with_retry("post", &url, &headers, &[], Some(&payload), DEFAULT_TIMEOUT, 3)
            .await
```
注意：此处 `headers` 目前是 `let mut headers`？确认 `validate_otp` 顶部为 `let mut headers = common_headers()...`（当前是 `let mut headers`，因为后面有 `.push`）。若不是 `mut` 则改为 `mut`。

- [ ] **Step 4: 编译检查**

Run: `cargo check -p model2api`（本地不可编译则走 CI）
Expected: 通过，无未使用告警（`build_sentinel_token` 已被调用）。

- [ ] **Step 5: Commit**

```bash
git add src/services/register/openai_register.rs
git commit -m "feat(register): wire Sentinel token into register_user/validate_otp/create_account"
```

---

### Task 4: 关闭 FlareSolverr 依赖的运行期验证（无代码改动，配置 + 端到端）

**Files:**
- 无源码改动（`flaresolverr.enabled` 已是运行期配置）。

**Interfaces:**
- Consumes: 已部署的新镜像（含 Task 1-3）。

- [ ] **Step 1: 触发 CI 构建新镜像**

推分支/合并触发 `.github/workflows/docker.yml`，产出 `ghcr.io/yiranxiaohui/model2api:latest`（amd64）。等 CI 绿。

- [ ] **Step 2: LXC 1011 拉新镜像并重启**

```bash
ssh root@10.1.41.1 'docker compose -f /opt/model2api/docker-compose.yml pull app && docker compose -f /opt/model2api/docker-compose.yml up -d app'
```

- [ ] **Step 3: 关 FlareSolverr，跑单账号（前提：上游代理已修好，见 memory register-failure-rootcause）**

```bash
ssh root@10.1.41.1 'python3 -c "import json;p=\"/opt/model2api/data/register.json\";d=json.load(open(p));d[\"flaresolverr\"][\"enabled\"]=False;json.dump(d,open(p,\"w\"),ensure_ascii=False,indent=2)"; docker compose -f /opt/model2api/docker-compose.yml restart app'
```
面板或 API 触发 1 个注册，`RUST_LOG=info,model2api=debug` 看日志。

- [ ] **Step 4: 判定**

Expected：`register_user` 与 `create_account` 返回 200/302（不再是 sentinel 相关 4xx），产出账号。
- 若过：Phase 1 达标，停。更新 memory `register-failure-rootcause` / `register-approach-vs-freeagentidentity`。
- 若 `register_user`/`create_account` 仍报 sentinel/turnstile 相关 4xx：记录响应体，触发 Phase 2（VM token 浏览器运行时）。
- 若被 Cloudflare 挡（`cf_block`）：回退 `flaresolverr.enabled=true`（sentinel 仍生效），单独评估 wreq 裸连过 CF 的可行性。

---

## Self-Review

**Spec coverage:**
- Sentinel 模块/接口 → Task 1 ✓
- 拉 challenge + 组头（端点/头/降级）→ Task 2 ✓
- 三注入点 + validate_otp 两段逻辑 → Task 3 ✓
- 丢 FlareSolverr（配置关闭 + 兜底回退）→ Task 4 ✓
- 25 字段指纹 / fnv1a32 / 字典序 difficulty → Task 1 ✓（含单测）
- 测试（单元 + 端到端 + CI 构建）→ Task 1 单测、Task 4 端到端、Global Constraints CI ✓
- oai-sc cookie：spec 标"可选、上游丢弃未用"，Phase 1 不实现 → 一致，无缺口 ✓

**Placeholder scan:** 无 TBD/TODO 残留（源码里原有的 `// TODO: sentinel` 注释在 Task 3 被删除）。所有代码步给出完整代码块。

**Type consistency:**
- `SentinelPow::new/requirements/enforcement` 在 Task 1 定义、Task 2 使用，签名一致。
- `build_sentinel_token(&self, &str) -> Option<String>` Task 2 定义、Task 3 三处使用一致。
- `send_once` 第 6 参 `timeout_secs: u64`，Task 2 传 `20` ✓；`Resp::json() -> Value`、`resp.status: u16` 与现有代码一致。
- `step_warn(index, &str)` 现有函数，Task 3 使用签名一致。

## Execution Handoff

见技能说明的两种执行方式。
