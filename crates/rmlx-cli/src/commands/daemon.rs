// unsafe_code: POSIX libc FFI — libc::kill process-liveness probe for claim holder status.
// LOC-exempt: daemon command owns one cohesive local admin/supervisor surface;
// split after config, lifecycle, and status contracts settle.
#![allow(unsafe_code)]

//! Minimal local admin daemon skeleton.

use std::io::{Read as _, Write as _};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs as _};
use std::path::{Path as StdPath, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn};

const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(2);
const CHILD_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const DEFAULT_ADMIN_HOST: &str = "127.0.0.1";
const DEFAULT_ADMIN_PORT: u16 = 6276;
const DEFAULT_SERVER_HOST: &str = "127.0.0.1";
const DEFAULT_SERVER_PORT: u16 = 8080;
const DEFAULT_DAEMON_CONFIG_FILE: &str = "daemon.toml";

#[derive(Debug, Clone)]
pub(crate) struct DaemonConfig {
    pub admin_host: String,
    pub admin_port: u16,
    pub server_host: String,
    pub server_port: u16,
    pub serve_profile: Option<String>,
    server_host_override: bool,
    server_port_override: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DaemonConfigOverrides {
    pub admin_host: Option<String>,
    pub admin_port: Option<u16>,
    pub server_host: Option<String>,
    pub server_port: Option<u16>,
    pub serve_profile: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DaemonConfigFile {
    #[serde(default)]
    daemon: DaemonFileConfig,
    #[serde(default, rename = "menu")]
    _menu: Option<toml::Value>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DaemonFileConfig {
    admin_host: Option<String>,
    admin_port: Option<u16>,
    server_host: Option<String>,
    server_port: Option<u16>,
    profile: Option<String>,
}

#[derive(Debug)]
struct DaemonState {
    config: DaemonConfig,
    started_at: Instant,
    lifecycle: parking_lot::Mutex<()>,
    child: parking_lot::Mutex<Option<SupervisedChild>>,
}

#[derive(Debug)]
struct SupervisedChild {
    child: Child,
    started_at: Instant,
}

#[derive(Debug, Serialize)]
struct AdminStatus {
    daemon: DaemonInfo,
    config: AdminConfigInfo,
    server: ServerStatus,
    models: Vec<ModelListItem>,
    model: ModelStatus,
    memory: MemoryStatus,
    cache: CacheStatus,
    claim: ClaimStatus,
}

#[derive(Debug, Serialize)]
struct DaemonInfo {
    running: bool,
    pid: u32,
    uptime_secs: u64,
}

#[derive(Debug, Serialize)]
struct AdminConfigInfo {
    admin_host: String,
    admin_port: u16,
    server_host: String,
    server_port: u16,
    serve_profile: Option<String>,
}

#[derive(Debug, Serialize)]
struct ServerStatus {
    running: bool,
    pid: Option<u32>,
    port: u16,
    healthy: bool,
    supervised: bool,
    uptime_secs: Option<u64>,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelStatus {
    id: Option<String>,
    status: String,
    keep_alive_secs: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ModelListItem {
    id: String,
    loaded: bool,
}

#[derive(Debug, Serialize)]
#[allow(
    clippy::struct_field_names,
    reason = "wire shape follows docs/plans/menu-daemon.md admin status bytes fields"
)]
struct MemoryStatus {
    rss_bytes: Option<u64>,
    metal_peak_alloc_bytes: Option<u64>,
    kv_cache_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct CacheStatus {
    hits: Option<u64>,
    misses: Option<u64>,
    evictions: Option<u64>,
    ssd_hits: Option<u64>,
    bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ClaimStatus {
    held: bool,
    holder_pid: Option<u32>,
    holder_alive: bool,
    path: String,
    last_error: Option<String>,
}

#[derive(Debug)]
struct HttpJson {
    status_code: u16,
    body: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartPreflight {
    Clear,
    AlreadySupervised {
        pid: u32,
    },
    ConflictAlreadyRunning {
        healthy: bool,
        holder_pid: Option<u32>,
    },
}

pub(crate) fn run_daemon(config: DaemonConfig) -> Result<()> {
    validate_local_admin_host(&config.admin_host)?;
    validate_local_server_host(&config.server_host)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building daemon tokio runtime")?;

    rt.block_on(async move { serve_admin_api(config).await })
}

pub(crate) fn resolve_daemon_config(
    config_path: Option<&StdPath>,
    cli: DaemonConfigOverrides,
) -> Result<DaemonConfig> {
    let file = load_daemon_file_config(config_path)?;
    let config = merge_daemon_config(file, cli)?;
    validate_daemon_config(&config)?;
    Ok(config)
}

fn default_daemon_config_path() -> PathBuf {
    rmlx_core::paths::home().join(DEFAULT_DAEMON_CONFIG_FILE)
}

fn load_daemon_file_config(config_path: Option<&StdPath>) -> Result<DaemonFileConfig> {
    let explicit = config_path.is_some();
    let path = config_path.map_or_else(default_daemon_config_path, StdPath::to_path_buf);

    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(err) if !explicit && err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DaemonFileConfig::default());
        }
        Err(err) => {
            return Err(err).with_context(|| format!("reading daemon config {}", path.display()));
        }
    };

    let parsed: DaemonConfigFile = toml::from_str(&contents)
        .with_context(|| format!("parsing daemon config {}", path.display()))?;
    Ok(parsed.daemon)
}

fn merge_daemon_config(file: DaemonFileConfig, cli: DaemonConfigOverrides) -> Result<DaemonConfig> {
    let serve_profile = cli.serve_profile.or(file.profile);
    let profile = serve_profile
        .as_deref()
        .map(load_serve_profile)
        .transpose()?;

    let server_host_override = cli.server_host.is_some() || file.server_host.is_some();
    let server_port_override = cli.server_port.is_some() || file.server_port.is_some();

    Ok(DaemonConfig {
        admin_host: cli
            .admin_host
            .or(file.admin_host)
            .unwrap_or_else(|| DEFAULT_ADMIN_HOST.to_owned()),
        admin_port: cli
            .admin_port
            .or(file.admin_port)
            .unwrap_or(DEFAULT_ADMIN_PORT),
        server_host: cli
            .server_host
            .or(file.server_host)
            .or_else(|| profile.as_ref().and_then(|profile| profile.host.clone()))
            .unwrap_or_else(|| DEFAULT_SERVER_HOST.to_owned()),
        server_port: cli
            .server_port
            .or(file.server_port)
            .or_else(|| profile.as_ref().and_then(|profile| profile.port))
            .unwrap_or(DEFAULT_SERVER_PORT),
        serve_profile,
        server_host_override,
        server_port_override,
    })
}

fn validate_daemon_config(config: &DaemonConfig) -> Result<()> {
    if config.admin_port == 0 {
        return Err(anyhow::anyhow!("daemon admin_port must be non-zero"));
    }
    if config.server_port == 0 {
        return Err(anyhow::anyhow!("daemon server_port must be non-zero"));
    }
    validate_local_admin_host(&config.admin_host)?;
    validate_local_server_host(&config.server_host)?;
    Ok(())
}

fn load_serve_profile(profile: &str) -> Result<super::profile::ServeProfile> {
    Ok(super::profile::ProfilesFile::load()?.get(profile)?.clone())
}

async fn serve_admin_api(config: DaemonConfig) -> Result<()> {
    let addr = resolve_bind_addr(&config.admin_host, config.admin_port)?;
    let state = Arc::new(DaemonState {
        config,
        started_at: Instant::now(),
        lifecycle: parking_lot::Mutex::new(()),
        child: parking_lot::Mutex::new(None),
    });

    let app = Router::new()
        .route("/health", get(daemon_health))
        .route("/admin/status", get(admin_status))
        .route("/admin/models/{id}/load", post(admin_load_model))
        .route("/admin/models/{id}/unload", post(admin_unload_model))
        .route("/admin/server/start", post(admin_server_start))
        .route("/admin/server/restart", post(admin_server_restart))
        .route("/admin/server/stop", post(admin_server_stop))
        .with_state(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding daemon admin API on {addr}"))?;

    info!(address = %addr, "rmlx daemon admin API listening");
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serving daemon admin API");
    stop_supervised_child(&state, "daemon shutdown");
    serve_result
}

async fn daemon_health() -> Json<Value> {
    Json(serde_json::json!({ "ok": true }))
}

async fn admin_status(State(state): State<Arc<DaemonState>>) -> Json<AdminStatus> {
    Json(build_status(&state))
}

async fn admin_load_model(
    State(state): State<Arc<DaemonState>>,
    Path(id): Path<String>,
    body: Option<Json<Value>>,
) -> Response {
    proxy_model_mutation(
        &state.config,
        &id,
        "load",
        body.as_ref().map(|Json(value)| value),
    )
}

async fn admin_unload_model(
    State(state): State<Arc<DaemonState>>,
    Path(id): Path<String>,
) -> Response {
    proxy_model_mutation(&state.config, &id, "unload", None)
}

async fn admin_server_start(State(state): State<Arc<DaemonState>>) -> Response {
    let state_for_error = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || start_supervised_child(&state))
        .await
        .map_err(|err| anyhow::anyhow!("server start task failed: {err}"))
        .and_then(|inner| inner);
    match result {
        Ok(StartResult::Started { pid }) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "ok": true,
                "supervised": true,
                "pid": pid,
                "message": "rmlx serve started under daemon supervision",
            })),
        )
            .into_response(),
        Ok(StartResult::AlreadySupervised { pid }) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "supervised": true,
                "pid": pid,
                "message": "rmlx serve is already supervised by this daemon",
            })),
        )
            .into_response(),
        Ok(StartResult::ConflictAlreadyRunning {
            healthy,
            holder_pid,
        }) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "ok": false,
                "supervised": false,
                "message": "rmlx serve already appears to be running outside this daemon; refusing to spawn another server",
                "server": {
                    "host": state_for_error.config.server_host,
                    "port": state_for_error.config.server_port,
                    "healthy": healthy,
                    "claim_holder_pid": holder_pid,
                },
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "ok": false,
                "supervised": false,
                "message": format!("failed to start supervised rmlx serve: {err}"),
            })),
        )
            .into_response(),
    }
}

async fn admin_server_restart(State(state): State<Arc<DaemonState>>) -> Response {
    let state_for_error = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || restart_supervised_child(&state))
        .await
        .map_err(|err| anyhow::anyhow!("server restart task failed: {err}"));
    let (stop, start) = match result {
        Ok(pair) => pair,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "ok": false,
                    "supervised": false,
                    "stopped": false,
                    "message": format!("failed to restart supervised rmlx serve: {err}"),
                })),
            )
                .into_response();
        }
    };
    match start {
        Ok(StartResult::Started { pid }) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "ok": true,
                "supervised": true,
                "pid": pid,
                "stopped": stop.stopped,
                "message": "rmlx serve restarted under daemon supervision",
            })),
        )
            .into_response(),
        Ok(StartResult::AlreadySupervised { pid }) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "supervised": true,
                "pid": pid,
                "stopped": stop.stopped,
                "message": "rmlx serve remained supervised by this daemon",
            })),
        )
            .into_response(),
        Ok(StartResult::ConflictAlreadyRunning {
            healthy,
            holder_pid,
        }) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "ok": false,
                "supervised": false,
                "stopped": stop.stopped,
                "message": "server restart could not start a new daemon-owned child because a server or live claim is still present",
                "server": {
                    "host": state_for_error.config.server_host,
                    "port": state_for_error.config.server_port,
                    "healthy": healthy,
                    "claim_holder_pid": holder_pid,
                },
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "ok": false,
                "supervised": false,
                "stopped": stop.stopped,
                "message": format!("failed to restart supervised rmlx serve: {err}"),
            })),
        )
            .into_response(),
    }
}

fn restart_supervised_child(state: &DaemonState) -> (StopResult, Result<StartResult>) {
    let _lifecycle = state.lifecycle.lock();
    let stop = stop_supervised_child_locked(state, "admin restart");
    let start = start_supervised_child_locked(state);
    (stop, start)
}

async fn admin_server_stop(State(state): State<Arc<DaemonState>>) -> Response {
    let result = tokio::task::spawn_blocking(move || stop_supervised_child(&state, "admin stop"))
        .await
        .unwrap_or_else(|err| StopResult {
            supervised: false,
            stopped: false,
            pid: None,
            message: format!("server stop task failed: {err}"),
        });
    let status = if result.stopped {
        StatusCode::OK
    } else {
        StatusCode::CONFLICT
    };

    (
        status,
        Json(serde_json::json!({
            "ok": result.stopped,
            "supervised": result.supervised,
            "stopped": result.stopped,
            "pid": result.pid,
            "message": result.message,
        })),
    )
        .into_response()
}

#[derive(Debug, Clone, Copy)]
struct SupervisedStatus {
    pid: u32,
    started_at: Instant,
}

fn supervised_child_status(state: &DaemonState) -> Option<SupervisedStatus> {
    let mut child = state.child.lock();
    let supervised = child.as_mut()?;
    match supervised.child.try_wait() {
        Ok(Some(status)) => {
            info!(pid = supervised.child.id(), %status, "supervised rmlx serve exited");
            *child = None;
            None
        }
        Ok(None) => Some(SupervisedStatus {
            pid: supervised.child.id(),
            started_at: supervised.started_at,
        }),
        Err(err) => {
            warn!(pid = supervised.child.id(), error = %err, "failed to poll supervised rmlx serve");
            Some(SupervisedStatus {
                pid: supervised.child.id(),
                started_at: supervised.started_at,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum StartResult {
    Started {
        pid: u32,
    },
    AlreadySupervised {
        pid: u32,
    },
    ConflictAlreadyRunning {
        healthy: bool,
        holder_pid: Option<u32>,
    },
}

fn start_supervised_child(state: &DaemonState) -> Result<StartResult> {
    let _lifecycle = state.lifecycle.lock();
    start_supervised_child_locked(state)
}

fn start_supervised_child_locked(state: &DaemonState) -> Result<StartResult> {
    let supervised = supervised_child_status(state).map(|status| status.pid);
    let health = http_get_json(
        &state.config.server_host,
        state.config.server_port,
        "/health",
        DEFAULT_HTTP_TIMEOUT,
    );
    let healthy = health
        .as_ref()
        .is_ok_and(|resp| resp.status_code == 200 && json_bool(resp.body.as_ref(), &["ok"]));
    let claim = read_claim_status(state.config.server_port);

    match classify_start_preflight(supervised, healthy, &claim) {
        StartPreflight::AlreadySupervised { pid } => Ok(StartResult::AlreadySupervised { pid }),
        StartPreflight::ConflictAlreadyRunning {
            healthy,
            holder_pid,
        } => Ok(StartResult::ConflictAlreadyRunning {
            healthy,
            holder_pid,
        }),
        StartPreflight::Clear => {
            let mut command = supervised_serve_command(&state.config)?;
            let child = command.spawn().context("spawning rmlx serve child")?;
            let pid = child.id();
            *state.child.lock() = Some(SupervisedChild {
                child,
                started_at: Instant::now(),
            });
            info!(pid, "started supervised rmlx serve");
            Ok(StartResult::Started { pid })
        }
    }
}

fn classify_start_preflight(
    supervised_pid: Option<u32>,
    healthy: bool,
    claim: &ClaimStatus,
) -> StartPreflight {
    if let Some(pid) = supervised_pid {
        return StartPreflight::AlreadySupervised { pid };
    }
    if healthy || claim.holder_alive {
        return StartPreflight::ConflictAlreadyRunning {
            healthy,
            holder_pid: claim.holder_pid,
        };
    }
    StartPreflight::Clear
}

fn supervised_serve_command(config: &DaemonConfig) -> Result<Command> {
    let exe = std::env::current_exe().context("resolving current rmlx executable")?;
    let args = supervised_serve_args(config);
    let mut command = Command::new(exe);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    Ok(command)
}

fn supervised_serve_args(config: &DaemonConfig) -> Vec<String> {
    let mut args = vec!["serve".to_owned()];
    if let Some(profile) = &config.serve_profile {
        args.push("--profile".to_owned());
        args.push(profile.clone());
    }
    if config.serve_profile.is_none() || config.server_host_override {
        args.push("--host".to_owned());
        args.push(config.server_host.clone());
    }
    if config.serve_profile.is_none() || config.server_port_override {
        args.push("--port".to_owned());
        args.push(config.server_port.to_string());
    }
    args
}

#[derive(Debug)]
struct StopResult {
    supervised: bool,
    stopped: bool,
    pid: Option<u32>,
    message: String,
}

fn stop_supervised_child(state: &DaemonState, reason: &str) -> StopResult {
    let _lifecycle = state.lifecycle.lock();
    stop_supervised_child_locked(state, reason)
}

fn stop_supervised_child_locked(state: &DaemonState, reason: &str) -> StopResult {
    let Some(mut supervised) = state.child.lock().take() else {
        return StopResult {
            supervised: false,
            stopped: false,
            pid: None,
            message: "no daemon-supervised rmlx serve child is running; refusing to stop an external server or alter claim files".to_owned(),
        };
    };

    let pid = supervised.child.id();
    if let Some(result) = already_exited_stop_result(&mut supervised.child) {
        return result;
    }

    info!(pid, reason, "stopping supervised rmlx serve");
    if signal_and_wait(&mut supervised.child, libc::SIGTERM, "SIGTERM") {
        return StopResult {
            supervised: true,
            stopped: true,
            pid: Some(pid),
            message: "supervised rmlx serve stopped after SIGTERM".to_owned(),
        };
    }

    if signal_and_wait(&mut supervised.child, libc::SIGINT, "SIGINT") {
        return StopResult {
            supervised: true,
            stopped: true,
            pid: Some(pid),
            message: "supervised rmlx serve stopped after SIGINT".to_owned(),
        };
    }

    match supervised.child.kill() {
        Ok(()) => {
            let _ = supervised.child.wait();
            StopResult {
                supervised: true,
                stopped: true,
                pid: Some(pid),
                message: "supervised rmlx serve required forced kill after graceful signals"
                    .to_owned(),
            }
        }
        Err(err) => StopResult {
            supervised: true,
            stopped: false,
            pid: Some(pid),
            message: format!("failed to stop supervised rmlx serve: {err}"),
        },
    }
}

fn already_exited_stop_result(child: &mut Child) -> Option<StopResult> {
    let pid = child.id();
    match child.try_wait() {
        Ok(Some(status)) => Some(StopResult {
            supervised: true,
            stopped: true,
            pid: Some(pid),
            message: format!("supervised rmlx serve had already exited with {status}"),
        }),
        Ok(None) => None,
        Err(err) => {
            warn!(pid, error = %err, "failed to poll supervised child before stop");
            None
        }
    }
}

fn signal_and_wait(child: &mut Child, signal: libc::c_int, signal_name: &str) -> bool {
    let pid = child.id();
    if let Err(err) = terminate_process(pid, signal) {
        warn!(pid, error = %err, "failed to send {signal_name} to supervised rmlx serve");
    }
    wait_child_exit(child, CHILD_SHUTDOWN_GRACE / 2)
}

fn wait_child_exit(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) => {
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return false,
        }
    }
}

fn terminate_process(pid: u32, signal: libc::c_int) -> std::io::Result<()> {
    // SAFETY: kill(2) is called for a PID returned by std::process::Child and
    // a standard termination signal. Errors are surfaced to the caller.
    let ret = unsafe { libc::kill(pid as libc::pid_t, signal) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn build_status(state: &DaemonState) -> AdminStatus {
    let config = &state.config;
    let supervised = supervised_child_status(state);
    let claim = read_claim_status(config.server_port);
    let health = http_get_json(
        &config.server_host,
        config.server_port,
        "/health",
        DEFAULT_HTTP_TIMEOUT,
    );
    let models = http_get_json(
        &config.server_host,
        config.server_port,
        "/v1/models",
        DEFAULT_HTTP_TIMEOUT,
    );
    let cache_metrics = http_get_json(
        &config.server_host,
        config.server_port,
        "/metrics/cache",
        DEFAULT_HTTP_TIMEOUT,
    );

    let healthy = health
        .as_ref()
        .is_ok_and(|resp| resp.status_code == 200 && json_bool(resp.body.as_ref(), &["ok"]));
    let server_error = health.as_ref().err().map(ToString::to_string);
    let holder_pid = supervised
        .as_ref()
        .map(|child| child.pid)
        .or(claim.holder_pid);
    let running = supervised.is_some() || healthy || claim.holder_alive;

    AdminStatus {
        daemon: DaemonInfo {
            running: true,
            pid: std::process::id(),
            uptime_secs: state.started_at.elapsed().as_secs(),
        },
        config: AdminConfigInfo {
            admin_host: config.admin_host.clone(),
            admin_port: config.admin_port,
            server_host: config.server_host.clone(),
            server_port: config.server_port,
            serve_profile: config.serve_profile.clone(),
        },
        server: ServerStatus {
            running,
            pid: holder_pid,
            port: config.server_port,
            healthy,
            supervised: supervised.is_some(),
            uptime_secs: supervised.map(|child| child.started_at.elapsed().as_secs()),
            last_error: server_error,
        },
        models: model_list(models.as_ref().ok().and_then(|r| r.body.as_ref())),
        model: model_status(models.as_ref().ok().and_then(|r| r.body.as_ref())),
        memory: memory_status(
            cache_metrics.as_ref().ok().and_then(|r| r.body.as_ref()),
            models.as_ref().ok().and_then(|r| r.body.as_ref()),
        ),
        cache: cache_status(
            cache_metrics.as_ref().ok().and_then(|r| r.body.as_ref()),
            models.as_ref().ok().and_then(|r| r.body.as_ref()),
        ),
        claim,
    }
}

fn model_list(models: Option<&Value>) -> Vec<ModelListItem> {
    models
        .and_then(|body| body.get("data"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let id = item.get("id").and_then(Value::as_str)?.to_owned();
                    let loaded = item.get("loaded").and_then(Value::as_bool).unwrap_or(false);
                    Some(ModelListItem { id, loaded })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn proxy_model_mutation(
    config: &DaemonConfig,
    id: &str,
    action: &str,
    body: Option<&Value>,
) -> Response {
    let path = model_lifecycle_path(id, action);
    match http_post_json(
        &config.server_host,
        config.server_port,
        &path,
        body,
        DEFAULT_HTTP_TIMEOUT,
    ) {
        Ok(resp) => http_json_to_response(resp),
        Err(err) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "ok": false,
                "message": format!("failed to proxy model {action} to rmlx serve: {err}"),
                "server": {
                    "host": config.server_host,
                    "port": config.server_port,
                },
            })),
        )
            .into_response(),
    }
}

fn model_lifecycle_path(id: &str, action: &str) -> String {
    format!(
        "/v1/models/{}/{}",
        percent_encode_path_segment(id),
        percent_encode_path_segment(action)
    )
}

fn http_json_to_response(resp: HttpJson) -> Response {
    let status =
        StatusCode::from_u16(resp.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(resp.body.unwrap_or(Value::Null))).into_response()
}

fn model_status(models: Option<&Value>) -> ModelStatus {
    let loaded = models
        .and_then(|body| body.get("data"))
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                let id = item.get("id").and_then(Value::as_str)?;
                if item.get("loaded").and_then(Value::as_bool).unwrap_or(false) {
                    Some((id.to_owned(), "loaded".to_owned()))
                } else {
                    None
                }
            })
        });

    match loaded {
        Some((id, status)) => ModelStatus {
            id: Some(id),
            status,
            keep_alive_secs: None,
        },
        None => ModelStatus {
            id: None,
            status: if models.is_some() {
                "unloaded".to_owned()
            } else {
                "unknown".to_owned()
            },
            keep_alive_secs: None,
        },
    }
}

fn memory_status(metrics: Option<&Value>, models: Option<&Value>) -> MemoryStatus {
    let model = metrics_model_for_loaded(metrics, models);
    MemoryStatus {
        rss_bytes: json_u64(metrics, &["rss_bytes"]),
        metal_peak_alloc_bytes: json_u64(model, &["metal_peak_alloc_bytes"])
            .or_else(|| json_u64(metrics, &["metal_peak_alloc_bytes"])),
        kv_cache_bytes: json_u64(model, &["kv_cache_bytes"])
            .or_else(|| json_u64(metrics, &["kv_cache_bytes"])),
    }
}

fn cache_status(metrics: Option<&Value>, models: Option<&Value>) -> CacheStatus {
    let model = metrics_model_for_loaded(metrics, models);
    CacheStatus {
        hits: json_u64(model, &["hits"]).or_else(|| json_u64(metrics, &["prompt_cache_hits"])),
        misses: json_u64(model, &["misses"])
            .or_else(|| json_u64(metrics, &["prompt_cache_misses"])),
        evictions: json_u64(model, &["evictions"]),
        ssd_hits: json_u64(model, &["ssd_hits"]),
        bytes: json_u64(model, &["bytes"]).or_else(|| json_u64(model, &["cache_bytes"])),
    }
}

fn metrics_model_for_loaded<'a>(
    metrics: Option<&'a Value>,
    models: Option<&Value>,
) -> Option<&'a Value> {
    let metrics_models = metrics
        .and_then(|body| body.get("models"))
        .and_then(Value::as_array)?;
    let loaded_id = loaded_model_id(models)?;
    metrics_models
        .iter()
        .find(|item| item.get("model_id").and_then(Value::as_str) == Some(loaded_id))
}

fn loaded_model_id(models: Option<&Value>) -> Option<&str> {
    models
        .and_then(|body| body.get("data"))
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                if item.get("loaded").and_then(Value::as_bool).unwrap_or(false) {
                    item.get("id").and_then(Value::as_str)
                } else {
                    None
                }
            })
        })
}

fn read_claim_status(port: u16) -> ClaimStatus {
    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));
    let path_display = path.display().to_string();

    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return ClaimStatus {
                held: false,
                holder_pid: None,
                holder_alive: false,
                path: path_display,
                last_error: None,
            };
        }
        Err(err) => {
            return ClaimStatus {
                held: true,
                holder_pid: None,
                holder_alive: true,
                path: path_display,
                last_error: Some(err.to_string()),
            };
        }
    };

    let holder_pid = match contents.trim().parse::<u32>() {
        Ok(pid) => Some(pid),
        Err(err) => {
            return ClaimStatus {
                held: true,
                holder_pid: None,
                holder_alive: true,
                path: path_display,
                last_error: Some(format!("invalid claim PID: {err}")),
            };
        }
    };
    let holder_alive = holder_pid.is_some_and(pid_is_alive);

    ClaimStatus {
        held: true,
        holder_pid,
        holder_alive,
        path: path_display,
        last_error: None,
    }
}

fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return true;
    }
    // SAFETY: kill(2) with signal 0 is an existence/permission probe and does
    // not deliver a signal. EPERM means the process exists but belongs to
    // another user, so only ESRCH proves the holder is dead.
    let ret = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if ret == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

fn http_get_json(host: &str, port: u16, path: &str, timeout: Duration) -> Result<HttpJson> {
    let addr = resolve_first_addr(host, port)?;
    let mut stream =
        TcpStream::connect_timeout(&addr, timeout).with_context(|| format!("connect {addr}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .context("setting read timeout")?;
    stream
        .set_write_timeout(Some(timeout))
        .context("setting write timeout")?;

    let request = format!(
        "GET {path} HTTP/1.0\r\nHost: {host}:{port}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .with_context(|| format!("write GET {path}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .with_context(|| format!("read GET {path}"))?;
    parse_http_json_response(&response)
}

fn http_post_json(
    host: &str,
    port: u16,
    path: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<HttpJson> {
    let addr = resolve_first_addr(host, port)?;
    let mut stream =
        TcpStream::connect_timeout(&addr, timeout).with_context(|| format!("connect {addr}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .context("setting read timeout")?;
    stream
        .set_write_timeout(Some(timeout))
        .context("setting write timeout")?;

    let body = match body {
        Some(body) => serde_json::to_string(body).context("serializing JSON request body")?,
        None => String::new(),
    };
    let request = format!(
        "POST {path} HTTP/1.0\r\nHost: {host}:{port}\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .with_context(|| format!("write POST {path}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .with_context(|| format!("read POST {path}"))?;
    parse_http_json_response(&response)
}

fn parse_http_json_response(response: &str) -> Result<HttpJson> {
    let status_line = response
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty HTTP response"))?;
    let status_code = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("missing HTTP status code"))?
        .parse::<u16>()
        .context("parsing HTTP status code")?;

    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.trim())
        .filter(|body| !body.is_empty())
        .map(serde_json::from_str::<Value>)
        .transpose()
        .context("parsing JSON response body")?;

    Ok(HttpJson { status_code, body })
}

fn json_bool(body: Option<&Value>, path: &[&str]) -> bool {
    json_at(body, path)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn json_u64(body: Option<&Value>, path: &[&str]) -> Option<u64> {
    json_at(body, path).and_then(Value::as_u64)
}

fn json_at<'a>(body: Option<&'a Value>, path: &[&str]) -> Option<&'a Value> {
    let mut current = body?;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn resolve_first_addr(host: &str, port: u16) -> Result<SocketAddr> {
    if !is_allowed_loopback_host(host) {
        return Err(anyhow::anyhow!(
            "daemon host must be local-only; got {host}"
        ));
    }
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolving {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no socket address for {host}:{port}"))?;
    if !ip_is_loopback(addr.ip()) {
        return Err(anyhow::anyhow!(
            "daemon host must resolve to localhost; {host} resolved to {}",
            addr.ip()
        ));
    }
    Ok(addr)
}

fn resolve_bind_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let addr = resolve_first_addr(host, port)?;
    if !ip_is_loopback(addr.ip()) {
        return Err(anyhow::anyhow!(
            "daemon admin host must resolve to localhost, got {addr}"
        ));
    }
    Ok(addr)
}

fn validate_local_admin_host(host: &str) -> Result<()> {
    if is_allowed_loopback_host(host) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "daemon admin API is localhost-only; got {host}"
        ))
    }
}

fn validate_local_server_host(host: &str) -> Result<()> {
    if is_allowed_loopback_host(host) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "daemon may only administer a local rmlx serve; got {host}"
        ))
    }
}

fn is_allowed_loopback_host(host: &str) -> bool {
    let normalized = host
        .trim_matches(|c| c == '[' || c == ']')
        .to_ascii_lowercase();
    normalized == "localhost" || normalized == "::1" || normalized.starts_with("127.")
}

fn ip_is_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}

fn percent_encode_path_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(hex_digit(byte >> 4));
                encoded.push(hex_digit(byte & 0x0f));
            }
        }
    }
    encoded
}

fn hex_digit(n: u8) -> char {
    debug_assert!(n < 16);
    match n {
        0..=9 => char::from(b'0' + n),
        _ => char::from(b'A' + (n - 10)),
    }
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(signal) => signal,
        Err(err) => {
            warn!(error = %err, "failed to install SIGTERM handler; daemon shutdown limited to SIGINT");
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("received SIGINT; daemon shutting down");
        }
        _ = sigterm.recv() => {
            info!("received SIGTERM; daemon shutting down");
        }
    }
}

#[cfg(test)]
#[path = "daemon_tests.rs"]
mod tests;
