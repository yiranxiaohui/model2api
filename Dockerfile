# syntax=docker/dockerfile:1

# ── Stage 1: build the Next.js frontend → static export (web_dist) ────────────
FROM node:22-alpine AS web-build
WORKDIR /app/web
COPY web/package.json web/bun.lock* web/package-lock.json* ./
RUN npm install
COPY VERSION /app/VERSION
COPY CHANGELOG.md /app/CHANGELOG.md
COPY web ./
RUN NEXT_PUBLIC_APP_VERSION="$(cat /app/VERSION)" npm run build

# ── Stage 2: build the Rust backend ──────────────────────────────────────────
# wreq → boring-sys2 (BoringSSL) needs cmake + nasm + clang/libclang;
# git2 (vendored-libgit2) and rusqlite (bundled) need a C toolchain + perl.
FROM rust:1.85-bookworm AS rust-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake nasm clang libclang-dev perl pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Cache dependencies first.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release || true
COPY src ./src
RUN touch src/main.rs && cargo build --release

# ── Stage 3: runtime ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS app
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=rust-build /app/target/release/model2api /usr/local/bin/model2api
COPY config.json ./
COPY VERSION ./
COPY --from=web-build /app/web/out ./web_dist

ENV HOST=0.0.0.0 \
    PORT=80
EXPOSE 80
CMD ["model2api"]
