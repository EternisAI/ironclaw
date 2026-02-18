# Agent Rules

## Quick Reference

- **Language**: Rust (edition 2024, MSRV 1.92)
- **Database**: PostgreSQL 15+ with pgvector (default), libSQL/Turso (optional)
- **Runtime**: tokio async
- **WASM**: wasmtime 28 for sandboxed tools and channels

## Build & Test

```bash
cargo fmt                                                       # Format
cargo clippy --all --benches --tests --examples --all-features  # Lint
cargo test                                                      # All tests
cargo test test_name                                            # Specific test
RUST_LOG=ironclaw=debug cargo run                               # Run with logging

# Feature-flag testing (required when touching feature-gated code)
cargo check                                          # default features
cargo check --no-default-features --features libsql  # libsql only
cargo check --all-features                           # all features
```

## Code Conventions

- Use `thiserror` for error types; never `.unwrap()` or `.expect()` in production code
- Use `crate::` imports, not `super::`
- All I/O is async with tokio; use `Arc<T>` + `RwLock` for shared state
- Both database backends (postgres, libsql) must be supported for new persistence features
- Prefer strong types (enums, newtypes) over strings
- See `CLAUDE.md` for full conventions, project structure, and architecture details

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/agent/` | Core agent loop, scheduler, sessions, routines |
| `src/channels/` | Input channels (REPL, HTTP, web gateway, WASM) |
| `src/cli/` | CLI subcommands (config, memory, pairing, tool, status) |
| `src/llm/` | Multi-provider LLM integration (NEAR AI, OpenAI, Ollama via rig-core) |
| `src/tools/` | Tool system (built-in, WASM sandbox, MCP, dynamic builder) |
| `src/db/` | Database trait + PostgreSQL/libSQL backends |
| `src/safety/` | Prompt injection defense, leak detection |
| `src/workspace/` | Persistent memory with hybrid search (FTS + vector) |
| `src/extensions/` | Extension discovery, install, auth, activation |
| `src/setup/` | First-run onboarding wizard (has spec: `src/setup/README.md`) |
| `channels-src/` | WASM channel source code (Telegram, Slack, Discord, WhatsApp) |
| `tools-src/` | WASM tool source code (GitHub, Gmail, Google Suite, etc.) |

## Feature Parity Update Policy

- If you change implementation status for any feature tracked in `FEATURE_PARITY.md`, update that file in the same branch.
- Do not open a PR that changes feature behavior without checking `FEATURE_PARITY.md` for needed status updates (`❌`, `🚧`, `✅`, notes, and priorities).

## Module Specifications

Some modules have a `README.md` spec. Code follows spec; update both if changing behavior.

| Module | Spec File |
|--------|-----------|
| `src/setup/` | `src/setup/README.md` |
