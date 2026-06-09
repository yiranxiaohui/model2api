# model2api

A **Rust** port of [chatgpt2api](https://github.com/) — an OpenAI/Anthropic-compatible
proxy over ChatGPT's web backend for image generation and text. The Python/FastAPI
backend has been rewritten in Rust (axum + tokio); the existing Next.js admin
frontend is reused unchanged and served as static assets.

## Highlights

- **OpenAI-compatible** endpoints: `/v1/chat/completions`, `/v1/images/generations`,
  `/v1/images/edits`, `/v1/responses`, `/v1/models`, `/v1/search`.
- **Anthropic-compatible** `/v1/messages` (with the full SSE event sequence).
- **TLS/HTTP2 fingerprint impersonation** via [`wreq`](https://crates.io/crates/wreq)
  + `wreq-util` (BoringSSL, `Emulation::Edge101` — the curl_cffi `edge101` equivalent),
  which is what lets the reverse-engineered ChatGPT web flow work.
- Full account pool: round-robin selection, per-token image-concurrency slots,
  OAuth access-token refresh/rotation, statistics, import/export.
- Pluggable storage: **JSON files**, **SQLite** (`rusqlite`), **Git** repo (`git2`).
- Admin web panel, API-key management, logs, image storage (local + WebDAV),
  Cloudflare R2 backups, account registration, CPA / sub2api import.
- Background scheduling: account refresh/keepalive, image cleanup, scheduled backups.

## Tech stack

| Area | Crate |
|---|---|
| Web framework | `axum` 0.8 + `tower-http` |
| Runtime | `tokio` |
| HTTP (fingerprint) | `wreq` + `wreq-util` (BoringSSL) |
| Serialization | `serde` / `serde_json` (preserve_order) |
| Crypto | `sha2`, `sha3`, `hmac`, `subtle`, `aes`, `cbc` |
| Tokenizer | `tiktoken-rs` |
| Database | `rusqlite` (bundled SQLite) |
| Git storage | `git2` (vendored libgit2) |
| Images | `imagesize` (dims), `image` (thumbnails), `zip` |

## Build

Native build prerequisites (for the BoringSSL / libgit2 / sqlite C builds):
**Rust ≥ 1.85**, `cmake`, `nasm`, and LLVM/`libclang` (set `LIBCLANG_PATH`).

```bash
cargo build --release
# binary: target/release/model2api
```

Build the frontend (optional; only needed to serve the admin panel):

```bash
cd web && npm install && npm run build    # output → web/out
# copy/symlink web/out → ./web_dist
```

## Run

```bash
# config.json + VERSION must be in the working directory
./target/release/model2api
```

Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | bind host |
| `PORT` | `8000` | bind port |
| `STORAGE_BACKEND` | `json` | `json` \| `sqlite` \| `git` |
| `DATABASE_URL` | — | e.g. `sqlite:///app/data/accounts.db` |
| `GIT_REPO_URL` / `GIT_TOKEN` / `GIT_BRANCH` / `GIT_FILE_PATH` | — | Git backend |
| `CHATGPT2API_AUTH_KEY` | — | overrides `config.json` `auth-key` |
| `CHATGPT2API_BASE_URL` | — | public base URL for image links |

The default admin key is in `config.json` (`auth-key`). Authenticate API calls with
`Authorization: Bearer <key>`.

## Docker

```bash
docker compose up --build      # serves on http://localhost:3000
```

The multi-stage `Dockerfile` builds the Next.js frontend, compiles the Rust backend
(installing cmake/nasm/clang for the native deps), and ships a slim runtime image.

## Smoke test

```bash
curl localhost:8000/version
curl localhost:8000/health?format=json
curl localhost:8000/v1/models -H "Authorization: Bearer chatgpt2api"
```

## Notes / not yet ported

This is a faithful port of the Python backend. A few low-traffic pieces are
intentionally deferred and clearly marked in the code (`// TODO`):

- **Editable-file export** (PPT/PSD) engine flow and its task service (the
  ~600-line `_export_editable_file_zip` in the Python engine) — the routes exist
  but report the feature as unavailable.
- **Password re-login** for accounts (needs the sentinel-HTTP login handshake
  shared with the registration flow) — `re_login_accounts` currently marks
  targets abnormal instead.
- **PostgreSQL / MySQL** storage backends (the JSON, SQLite and Git backends are
  built; the database backend supports SQLite).
- Image-storage disk-usage maintenance routes (compress / cleanup-to-target)
  return `501`.

Everything else — the reverse-engineering engine (PoW, sentinel, turnstile),
account pool, protocol translators, and the full HTTP API — is ported.
