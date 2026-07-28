# Sentinel PoW 原生移植 + 丢弃 FlareSolverr（Phase 1）

日期：2026-07-28
状态：设计已确认（25 字段指纹），待写实现计划

## 背景与问题

model2api（Rust，`src/services/register/openai_register.rs`）的 OpenAI 账号注册流程里，
`openai-sentinel-token` 头**从未发送**——三处注入点都标着 `// TODO: sentinel token (deferred)`，
请求裸发碰运气。这是注册成功率的根本上限。

过 Cloudflare 目前依赖 **FlareSolverr** 无头浏览器先 solve 拿 `cf_clearance`+UA 再由 `wreq` 复用。
FlareSolverr 引入了额外的常驻 Chrome 容器、对带认证代理静默失效等问题
（见 memory `register-failure-rootcause`、`register-approach-vs-freeagentidentity`）。

参照项目 `/opt/freeAgentIdentity`（Python）证明：靠 HTTP 客户端的 TLS/JA3 指纹伪装即可过 CF，
Sentinel 的 PoW 部分可纯代码生成，只有 VM token（`t`/`so`）那一步才需要真浏览器跑 SDK 的 JS。

## Phase 1 范围

**做：** 纯 Rust 生成 Sentinel PoW token（`t=""`，无浏览器）+ 关闭 FlareSolverr 依赖，靠 `wreq` 指纹过 CF。
**不做（留 Phase 2）：** VM token（`t`/`so`）的浏览器运行时。若实测 OpenAI 强制 turnstile/VM token 才启动 Phase 2。

判定标准：Phase 1 在 LXC 1011（关 FlareSolverr）能稳定走过 `register_user` 与 `create_account`
（两个 sentinel 强制点）并产出账号，即达标、停在 Phase 1。

## 权威参照

- `/opt/chatgpt2api/utils/sentinel.py` —— model2api 的直系 Python 上游，`build_sentinel_token` 是 platform
  流程的 1:1 参照（返回 `(header, oai_sc)`，`t=""` 的 PoW-only 路径）。**流程、端点、flow 字符串以它为准。**
- `/opt/freeAgentIdentity/platforms/chatgpt/protocol_register.py::_SentinelTokenGenerator` ——
  **25 字段指纹**（`_reference_fingerprint`）与 `_solve_reference_pow` 的实现以它为准（较新，注释标"当前 SDK 用"）。

## 架构

新增模块 `src/services/register/sentinel.rs`，单一职责：生成 Sentinel PoW token。

### 对外接口

```rust
pub struct SentinelBuilder<'a> {
    session: &'a wreq::Client,   // 复用注册用的同一 session（同指纹/代理/cookie jar）
    device_id: &'a str,
    user_agent: &'a str,
}

impl SentinelBuilder<'_> {
    /// 返回 (openai-sentinel-token 头值, oai-sc cookie 值)。
    /// oai-sc = "0" + challenge；调用方可选是否写入 cookie（上游包装层丢弃未用，Phase 1 可先不写）。
    pub async fn build_token(&self, flow: &str) -> Result<(String, String), RegisterError>;
}
```

### 内部步骤

1. **requirements token**（本地）：**照 freeAgentIdentity `requirements()`**——取 25 字段 config，
   置 `config[3]=1`、`config[9]=round(5..50)`，再以随机 seed、difficulty `"0"` 跑一遍第 3 步的 solve 循环
   （solve 会覆盖 `[3]=nonce`、`[9]=elapsed`），命中后取其编码值，前缀 `"gAAAAAC"`。
   注意这是 25 字段版与直系上游 `sentinel.py`（18 字段、requirements 不爆破）的关键差异——本设计以 25 字段版为准。
2. **拉 challenge**：`POST https://sentinel.openai.com/backend-api/sentinel/req`，
   body `{"p": <requirements>, "id": <device_id>, "flow": <flow>}`，
   头含 `content-type: text/plain;charset=UTF-8`、`origin: https://sentinel.openai.com`、
   `referer: https://sentinel.openai.com/backend-api/sentinel/frame.html`、UA/sec-ch-ua 与注册一致。
   解析出 `token`（即 challenge `c`）与 `proofofwork{required, seed, difficulty}`。
   - 非 200 或无 `token`：返回 `Err`（由调用方决定是否降级为裸发；参照 Python 在 JSON 解析失败时降级为 `c=""` 的 fallback）。
3. **solve 例程**（共用）：入参 `(seed, difficulty, config)`。循环 `nonce=0..500_000`，
   每轮置 `data[3]=nonce`、`data[9]=已耗毫秒`，`encoded=base64(compact-json(data))`，
   `digest=fnv1a32(seed + encoded)`，当 `digest[..difficulty.len()] <= difficulty`
   （**十六进制字符串的字典序比较，非数值**）命中，返回 `encoded + "~S"`。
   未命中返回 `base64(json "e")`（无 `~S`）。requirements（第 1 步）用前缀 `"gAAAAAC"`、difficulty `"0"`、随机 seed；
   **enforcement**（本步）：若 `proofofwork.required && seed`，用前缀 `"gAAAAAB"`、服务器给的 `seed`/`difficulty` 调 solve；
   否则（不要求 PoW）复用第 1 步的 requirements token 作为 `p`。
4. **组头**：`{"p": <p>, "t": "", "c": <challenge>, "id": <device_id>, "flow": <flow>}` 的 compact JSON。
   `oai_sc = "0" + challenge`。

### fnv1a32（需与 Python 逐位对齐）

```
h = 2166136261
for ch in text: h ^= ord(ch); h = (h * 16777619) & 0xFFFFFFFF
h ^= h >> 16; h = (h * 2246822507) & 0xFFFFFFFF
h ^= h >> 13; h = (h * 3266489909) & 0xFFFFFFFF
h ^= h >> 16
return format(h, "08x")   // 8 位小写十六进制，前导零补齐
```
注意 `ord(ch)` 按 Unicode 码点；config 里都是 ASCII/数字，Rust 按 `char as u32` 一致。
base64 用标准表（`STANDARD`），JSON 用 compact 分隔符 `(",", ":")`，与现有 `pow.rs` 一致。

### 25 字段 config（来自 `_reference_fingerprint`，索引 3/9 为可变槽）

```
[ 3000, "<now, str(datetime.now().astimezone())>", 4294705152, <slot3>, <user_agent>,
  "<SENTINEL_SDK_URL>", null, "en-US", "en-US,en", <slot9>,
  "webkitTemporaryStorage−undefined", "location", "Object", <perf_now>, <sid uuid>,
  "", 8, <time_origin>, 0, 0, 0, 0, 0, 0, 0 ]
```
时间/随机/uuid 字段服务器不交叉核对（只校验哈希有效性），但字段个数与静态值需与参照一致。
`SENTINEL_SDK_URL` = `https://sentinel.openai.com/sentinel/<SDK_VERSION>/sdk.js`，
`SDK_VERSION` 作为常量（初值取 freeAgentIdentity 的 `20260219f9f6`），置于 `constants.rs` 便于后续更新。

## 集成点（`openai_register.rs`）

替换三处 `// TODO: sentinel token (deferred)`，sentinel 请求复用当前注册 `wreq` session：

| 函数 | flow | 逻辑 |
|---|---|---|
| `register_user` | `username_password_create` | 请求前加头 |
| `validate_otp` | `authorize_continue` | 保持"先裸发；非 200 再带 sentinel 头重试"的现有两段逻辑，把重试段的头补上 |
| `create_account` | `oauth_create_account` | 请求前加头 |

`build_token` 失败时的降级：参照 Python，challenge 拉取失败可退化为 `c=""` 的 best-effort 头（保证不比现状更差），
并 `step_warn` 记录；不因 sentinel 拉取失败直接判注册失败。

## 丢弃 FlareSolverr

不删代码。运行期 `register.json` 设 `flaresolverr.enabled=false`，走 `wreq` `Emulation::Chrome137` 指纹裸连过 CF；
现有 `is_cloudflare_challenge` 兜底检测保留。sentinel 头与 FlareSolverr 正交——即使回退开 FlareSolverr，sentinel 照发。
UA 一致性：sentinel 请求用的 UA 必须与注册 session 的 UA 相同（`constants.rs::USER_AGENT` 或 FlareSolverr 回填的 `cf_ua`）。

## 测试

- **单元测试**（`sentinel.rs`）：
  - `fnv1a32` 对固定输入比对 Python 输出（几个已知串）。
  - base64/compact-json 编码对齐（固定 config → 固定串）。
  - PoW 爆破：构造小 difficulty（如 `"0"` 或 `"00"`）验证能命中且 `digest[..n] <= difficulty` 成立。
  - 字典序比较边界：验证用字符串比较而非数值。
- **端到端**（LXC 1011）：关 FlareSolverr、单账号跑，观察 `register_user`/`create_account` 是否走过；
  对照关/不关 sentinel 的成功率。
- **构建**：编译/打包一律走 GitHub CI（`.github/workflows/docker.yml`），本地不构建（见 memory `build-in-ci`）。

## 风险与回退

- wreq 裸连可能过不了 auth.openai.com 的 CF：回退开 FlareSolverr（配置项，无需改码），sentinel 仍生效。
- OpenAI 强制 turnstile/VM token：`register_user`/`create_account` 仍 4xx 且报 sentinel 相关错 → 触发 Phase 2（浏览器运行时生成 `t`/`so`）。
- SDK 版本 / 字段结构漂移：`SDK_VERSION` 与 config 集中在 `constants.rs`/`sentinel.rs`，便于跟随参照项目更新。

## 开发方式

按 treeflow：实现在独立 git worktree 进行；本设计文档提交进主仓库。
