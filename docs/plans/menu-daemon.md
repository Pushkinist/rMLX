# rMLX Menu Bar Daemon Plan

## Goal

Build a small local daemon and macOS menu bar utility for administering an
rMLX server from the menu bar. The utility should make the common operational
state obvious at a glance:

- whether rMLX is running;
- which model is resident, if any;
- whether memory is currently occupied by weights / KV cache;
- how much memory and cache space are in use;
- how to start, stop, restart, load, unload, and pick idle eviction policy.

The menu should be dense, native, and predictable. It is an administration
surface, not a second chat UI or a custom dashboard.

## Product Boundary

Use three separate responsibilities:

| Component | Responsibility |
|---|---|
| `rmlx serve` | Owns inference, MLX/Metal, model loading, generation, existing HTTP API. |
| `rmlxd` | Supervises `rmlx serve`, stores config, polls health/metrics, exposes local admin API. |
| `rmlx-menu` | macOS menu bar UI. Talks to `rmlxd` / local HTTP APIs only. |

The menu app must not load models directly or own an MLX/Metal context. That
keeps the existing single-MLX-process rule intact and makes the UI process
safe to restart independently.

## Comparable App Patterns

Use these ecosystem patterns as product references:

- Ollama: simple model residency semantics, explicit stop/unload, and
  `keep_alive` style idle policy.
- llama.cpp server/router: dynamic load, unload, and switch without requiring
  a full process restart for every model change.
- oMLX: admin-oriented model status, manual load/unload, per-model TTL,
  pinning, LRU / memory pressure handling, and SSD cache visibility.

rMLX already has much of the server-side lifecycle surface:

- `GET /health`
- `GET /v1/models`
- `POST /v1/models/{id}/load`
- `POST /v1/models/{id}/unload`
- `GET /v1/models/{id}/status`
- `GET /metrics/cache`

The new daemon should compose these rather than duplicating inference state.

## Native macOS Stack

Default stack:

- SwiftUI `MenuBarExtra` for the menu bar item.
- SwiftUI `Settings` scene for preferences.
- Apple `SMAppService` for launch-at-login / helper registration.
- Type-safe settings persistence via `UserDefaults` or the `Defaults` Swift
  package.

Fallbacks / optional packages:

- Use AppKit `NSStatusBar` / `NSMenu` if `MenuBarExtra` is too limiting for
  standard menu behavior.
- Add `KeyboardShortcuts` only if global hotkeys become useful.
- Add Sparkle only if the menu app is distributed independently and needs
  app-level auto-updates.

The menu should use standard macOS styling: ordinary rows, checkmarks,
disabled status text, separators, submenus, SF Symbols, and a normal Settings
window. Avoid a custom-rendered menu unless native controls cannot express a
required interaction.

## Menu Shape

The menu item title should communicate state in compact form, for example:

```text
rMLX: gemma-4-e4b  18.4 GB
```

Suggested icon states:

| State | Meaning |
|---|---|
| Gray | daemon stopped |
| Idle / hollow | daemon running, no model loaded |
| Green | model loaded |
| Amber | loading / unloading / restarting |
| Red | unhealthy, OOM, claim conflict, or crashed |

Suggested menu:

```text
rMLX
  Status: Running on :8080
  Model: gemma-4-e4b
  Memory: 18.4 GB Metal peak, 642 MB KV
  Cache: 3 hits / 8 misses, SSD 1.2 GB

Models
  ✓ gemma-4-e4b        Loaded
    qwen3.6-35b        Unloaded
    bonsai-8b          Unloaded

Actions
  Load Selected Model
  Unload Current Model
  Restart Server
  Stop Server

Keep Alive
  After Each Request
  5 minutes
  15 minutes ✓
  1 hour
  Keep Loaded

Cache
  Prompt RAM: 2 GB
  SSD KV: 20 GB
  Clear RAM Prompt Cache
  Open Cache Folder

Server
  Port: 8080
  Copy OpenAI Base URL
  Open Logs
  Open Config

Settings...
Quit
```

Keep the top-level menu short and operational. Anything that requires typing,
file picking, validation, or a restart should live in Settings.

## Settings Window

Settings should cover lower-frequency configuration:

- model path or registry path;
- host and port;
- default idle timeout;
- prompt-cache slots;
- prompt-cache RAM GiB;
- SSD KV cache GiB;
- SSD project namespace;
- max loaded models;
- default KV quant;
- launch at login;
- log location.

When a setting requires restarting `rmlx serve`, mark it as pending and expose
an explicit `Restart Server` action. Do not silently kill active inference.

## Daemon Responsibilities

`rmlxd` should be a small local supervisor, not a second inference server.

Responsibilities:

- start, stop, and restart `rmlx serve`;
- persist effective config;
- track child PID, port, uptime, logs, crash status, and last exit code;
- poll `/health`;
- poll `/v1/models` and `/v1/models/{id}/status`;
- poll `/metrics/cache`;
- detect `/tmp/rmlx.<port>.claim` conflicts and report the holder PID;
- normalize state into one compact admin status object;
- proxy model load/unload actions to `rmlx serve`;
- expose log tail and common filesystem locations to the menu app.

The daemon should not parse generation traffic or proxy OpenAI requests unless
a later product requirement explicitly needs that.

## Admin API Sketch

Expose a localhost-only API for the menu app:

```http
GET  /admin/status
POST /admin/server/start
POST /admin/server/stop
POST /admin/server/restart
POST /admin/models/{id}/load
POST /admin/models/{id}/unload
POST /admin/config
POST /admin/cache/clear-ram
GET  /admin/logs/tail
```

Example normalized `GET /admin/status` response from the daemon:

```json
{
  "server": {
    "running": true,
    "pid": 1234,
    "port": 8080,
    "healthy": true,
    "uptime_secs": 3600
  },
  "model": {
    "id": "gemma-4-e4b",
    "status": "loaded",
    "keep_alive_secs": null
  },
  "memory": {
    "rss_bytes": 0,
    "metal_peak_alloc_bytes": 4294967296,
    "kv_cache_bytes": 134217728
  },
  "cache": {
    "hits": 3,
    "misses": 8,
    "evictions": 1,
    "ssd_hits": 1,
    "bytes": 1048576
  },
  "claim": {
    "held": true,
    "holder_pid": 1234
  }
}
```

Use `keep_alive` semantics compatible with the existing model load endpoint:

- negative: keep loaded until explicit unload or process exit;
- `0`: unload after the next request completes;
- positive: unload after that many idle seconds.

The daemon normalizes the existing server responses. The current server
`GET /v1/models` response uses a per-model `loaded` boolean, and
`POST /v1/models/{id}/load` / `unload` return `{"ok": ...}` envelopes rather
than a standalone `"status"` field. Do not make the menu depend on a server
field that does not exist yet. Until the server exposes keep-alive policy in a
status route, `keep_alive_secs` remains `null`.

## Repo-Specific Integration Notes

The first implementation pass should align with the existing CLI/server
contract in [`docs/CLI.md`](../CLI.md), [`docs/SERVER.md`](../SERVER.md), and
the runtime-root rules in [`CLAUDE.md`](../../CLAUDE.md).

### Current operator commands

Use these commands as the baseline behavior the daemon composes, not as a new
parallel lifecycle:

```bash
# Dev checkout: keep runtime state inside the gitignored repo-local root.
export RMLX_HOME="$PWD/.rmlx"

# Build the current single binary.
cargo build --release --bin rmlx

# Run one model directly.
./target/release/rmlx serve \
  --model /abs/path/to/snapshot \
  --host 127.0.0.1 \
  --port 8080 \
  --idle-timeout-secs 15m \
  --prompt-cache-slots 4 \
  --prompt-cache-ram-gb 2.0

# Or run from an existing launch profile.
./target/release/rmlx serve --profile menu-default

# Probe live status without loading MLX.
./target/release/rmlx healthcheck --port 8080
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/metrics/cache
```

Model load/unload actions should proxy the server routes documented in
`docs/SERVER.md`:

```bash
curl -X POST \
  -H 'content-type: application/json' \
  -d '{"keep_alive": 900}' \
  http://127.0.0.1:8080/v1/models/gemma-4-e4b/load

curl -X POST http://127.0.0.1:8080/v1/models/gemma-4-e4b/unload
curl -s http://127.0.0.1:8080/v1/models/gemma-4-e4b/status
```

Stopping or restarting the server process should be graceful first. The current
server installs SIGINT/SIGTERM cleanup and removes `/tmp/rmlx.<port>.claim` on
normal shutdown. Manual claim-file removal is only a recovery step after
SIGKILL/crash; the daemon must never steal a live claim.

### Runtime and config placement

Use the existing `<RMLX_HOME>` resolution order:

1. `$RMLX_HOME`, when set to an absolute path.
2. `<workspace>/.rmlx/`, auto-detected from a checkout containing `Cargo.lock`.
3. `$HOME/.rmlx/`, for installed binaries.

Recommended file placement:

| File / directory | Status | Purpose |
|---|---|---|
| `<RMLX_HOME>/profiles.toml` | Existing | Named `rmlx serve` launch profiles. Prefer this for serve flags that the CLI already binds. |
| `<RMLX_HOME>/projects.toml` | Existing | Per-project SSD/RAM cache defaults. rMLX reads it on serve restart. |
| `<RMLX_HOME>/daemon.toml` | New, planned | `rmlxd` supervisor/menu preferences that are not valid `rmlx serve` profile fields. |
| `<RMLX_HOME>/logs/` | Existing | Per-run JSON logs. Menu `Open Logs` should open this directory. |
| `<RMLX_HOME>/metrics/runs.db` | Existing | Metrics SQLite DB. Do not truncate or rewrite from the daemon. |
| `<RMLX_HOME>/cache/kv/<namespace>/` | Existing | SSD KV cache blocks when SSD tiering is enabled. |
| `/tmp/rmlx.<port>.claim` | Existing | Single-MLX-process claim file held by `rmlx serve`. Inspect only unless performing crash recovery. |

Keep `profiles.toml` as the source for current serve arguments:

```toml
[profile.menu-default]
registry = "/abs/path/to/registry.json"
host = "127.0.0.1"
port = 8080
device = "gpu"
kv_quant = "auto"
max_ctx = 8192
idle_timeout_secs = "15m"
prompt_cache_slots = 4
max_loaded_models = 1
max_queue_depth = 64
max_timeout_secs = 600
```

Use `daemon.toml` only for daemon/menu-owned settings that are not accepted by
`rmlx profile list` today. Proposed shape:

```toml
[daemon]
profile = "menu-default"
admin_host = "127.0.0.1"
admin_port = 6276
server_host = "127.0.0.1"
server_port = 8080

[menu]
show_memory_in_title = true
launch_at_login = false
```

Do not mirror every `rmlx serve` flag into `daemon.toml`. Duplicating serve
configuration creates precedence drift with `docs/CLI.md`; if a field already
belongs in `<RMLX_HOME>/profiles.toml`, keep it there.

### Launch placement

For a user-level macOS install, register the daemon as a LaunchAgent rather
than a system LaunchDaemon:

```text
~/Library/LaunchAgents/org.rmlx.rmlxd.plist
```

Recommended environment in the launch configuration:

- `RMLX_HOME=$HOME/.rmlx` for an installed binary.
- `RMLX_HOME=/abs/path/to/rmlx/.rmlx` for a dev checkout.

Recommended arguments for the first implementation slice:

```text
rmlx daemon --config <RMLX_HOME>/daemon.toml
```

For the first development milestone, starting `rmlxd` manually from the repo is
enough; launch-at-login can stay behind the Settings toggle until the daemon
start/stop/restart flow is verified.

### Status composition

`GET /admin/status` should be a normalized view over existing facts:

- server liveness from child PID plus `GET /health`;
- model list from `GET /v1/models`;
- current model residency from `GET /v1/models/{id}/status`;
- prompt/KV/Metal/cache counters from `GET /metrics/cache`;
- claim holder from `/tmp/rmlx.<port>.claim` using the same live-PID rule
  documented in `docs/CLI.md` and `docs/SERVER.md`;
- effective serve profile/config from `<RMLX_HOME>/profiles.toml` plus pending
  daemon settings.

If a status field cannot be derived from these sources, leave it absent or
unknown until a real server/admin hook is implemented.

## Implementation Sequence

1. Add `rmlxd` as either a new binary or an `rmlx daemon` subcommand.
2. Implement config loading / saving and child-process supervision.
3. Add `GET /admin/status` by composing existing `/health`, model lifecycle,
   metrics, PID, config, and claim-file state.
4. Add start, stop, restart, load, and unload admin actions.
5. Add any missing server-side admin hooks only where existing endpoints cannot
   safely express the operation.
6. Build the first SwiftUI menu bar shell with live polling and standard menu
   rows.
7. Add the Settings window.
8. Add launch-at-login.
9. Package the menu utility and daemon/helper with clear install/uninstall
   behavior.

## Implementation Checklist

### Documentation / config examples

- [x] Anchor the plan to existing `rmlx serve`, `healthcheck`, model lifecycle,
  cache metrics, profile, project config, runtime root, and claim-file docs.
- [x] Recommend `<RMLX_HOME>/profiles.toml` for serve launch configuration.
- [x] Recommend `<RMLX_HOME>/daemon.toml` only for daemon/menu-owned settings.
- [x] Recommend user LaunchAgent placement for launch-at-login.
- [ ] Add a short operator runbook after the daemon command/API exists.

### Daemon

- [x] Use an `rmlx daemon` subcommand for the first implementation slice.
- [x] Load `<RMLX_HOME>/daemon.toml` when present, with `--config` for explicit
  paths and CLI overrides for local development.
- [x] Restrict daemon admin and managed server hosts to local loopback hosts.
- [x] Resolve `profile` to `<RMLX_HOME>/profiles.toml` and start
  `rmlx serve --profile <name>` without translating profile fields manually.
- [x] Track child PID and uptime for daemon-supervised `rmlx serve` children.
- [ ] Track exit code, stdout/stderr/log location, and last
  crash reason.
- [x] Stop via SIGTERM first and wait before escalating to SIGINT/kill.
- [x] Restart by stopping the daemon-owned child gracefully, then starting a new
  `rmlx serve`.
- [x] Poll `/health`, `/v1/models`, and `/metrics/cache`; tolerate the server
  being down.
- [ ] Poll `/v1/models/{id}/status` once keep-alive/status detail is exposed in
  a stable server response.
- [x] Inspect `/tmp/rmlx.<port>.claim` without stealing a live claim.
- [x] Expose `/admin/status`, server start/stop/restart, and model load/unload.
- [ ] Expose config update, RAM-cache clear if implemented, and log-tail
  endpoints.

### Server gaps to verify before coding

- [ ] Confirm whether RAM prompt-cache clearing already has a safe internal
  hook; if not, keep `POST /admin/cache/clear-ram` unimplemented.
- [ ] Confirm whether `GET /v1/models/{id}/status` should expose keep-alive
  policy and a string status. Today the daemon derives loaded/unloaded from
  `GET /v1/models`'s `loaded` boolean.
- [ ] Confirm whether all memory figures needed by the title/menu already exist
  in `/metrics/cache`; do not derive bench-grade metrics from display-only
  counters.

### Menu app

- [x] Poll the daemon admin API only; never load MLX or call model code from
  Swift.
- [x] Display disabled/unknown states while daemon or server status is
  unavailable.
- [x] Use native menu rows, separators, Settings, and SF Symbols.
- [x] Keep serve settings read-only or placeholder-only until `/admin/config`
  exists; do not create a second authoritative client-side config store.
- [ ] Mark restart-required settings as pending and require an explicit
  `Restart Server`.
- [ ] Keep launch-at-login registration behind a Settings toggle.

## Design Constraints

- Keep the menu fast and dense.
- Prefer native macOS controls over custom drawing.
- Keep rich or infrequent configuration in Settings.
- Make destructive or disruptive actions explicit.
- Never bypass the existing single-MLX-process claim rule.
- Do not store secrets in daemon config.
- Do not introduce Python or a runtime dependency stack for the menu system.
