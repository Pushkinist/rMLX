use super::*;

#[test]
fn daemon_config_toml_parses_supported_section_shape() {
    let parsed: DaemonConfigFile = toml::from_str(
        r#"
[daemon]
admin_host = "127.0.0.1"
admin_port = 7001
server_host = "localhost"
server_port = 7002
profile = "menu"
"#,
    )
    .expect("parse daemon config");

    assert_eq!(
        parsed.daemon,
        DaemonFileConfig {
            admin_host: Some("127.0.0.1".to_owned()),
            admin_port: Some(7001),
            server_host: Some("localhost".to_owned()),
            server_port: Some(7002),
            profile: Some("menu".to_owned()),
        }
    );
}

#[test]
fn daemon_config_merge_uses_config_over_builtin_defaults() {
    let config = merge_daemon_config(
        DaemonFileConfig {
            admin_host: Some("localhost".to_owned()),
            admin_port: Some(7001),
            server_host: Some("127.0.0.1".to_owned()),
            server_port: Some(7002),
            profile: None,
        },
        DaemonConfigOverrides::default(),
    )
    .expect("merge daemon config");

    assert_eq!(config.admin_host, "localhost");
    assert_eq!(config.admin_port, 7001);
    assert_eq!(config.server_host, "127.0.0.1");
    assert_eq!(config.server_port, 7002);
    assert!(config.serve_profile.is_none());
}

#[test]
fn daemon_config_merge_cli_overrides_config() {
    let config = merge_daemon_config(
        DaemonFileConfig {
            admin_host: Some("localhost".to_owned()),
            admin_port: Some(7001),
            server_host: Some("127.0.0.1".to_owned()),
            server_port: Some(7002),
            profile: None,
        },
        DaemonConfigOverrides {
            admin_host: Some("127.0.0.1".to_owned()),
            admin_port: Some(8001),
            server_host: Some("localhost".to_owned()),
            server_port: Some(8002),
            serve_profile: None,
        },
    )
    .expect("merge daemon config");

    assert_eq!(config.admin_host, "127.0.0.1");
    assert_eq!(config.admin_port, 8001);
    assert_eq!(config.server_host, "localhost");
    assert_eq!(config.server_port, 8002);
    assert!(config.serve_profile.is_none());
}

#[test]
fn daemon_config_merge_falls_back_to_builtin_defaults() {
    let config = merge_daemon_config(
        DaemonFileConfig::default(),
        DaemonConfigOverrides::default(),
    )
    .expect("merge daemon config");

    assert_eq!(config.admin_host, DEFAULT_ADMIN_HOST);
    assert_eq!(config.admin_port, DEFAULT_ADMIN_PORT);
    assert_eq!(config.server_host, DEFAULT_SERVER_HOST);
    assert_eq!(config.server_port, DEFAULT_SERVER_PORT);
    assert!(config.serve_profile.is_none());
}

#[test]
fn daemon_config_resolves_from_explicit_file() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("daemon.toml");
    std::fs::write(
        &path,
        r#"
[daemon]
admin_port = 7001
server_port = 7002
"#,
    )
    .expect("write daemon config");

    let config = resolve_daemon_config(
        Some(&path),
        DaemonConfigOverrides {
            server_port: Some(9002),
            ..DaemonConfigOverrides::default()
        },
    )
    .expect("resolve daemon config");

    assert_eq!(config.admin_host, DEFAULT_ADMIN_HOST);
    assert_eq!(config.admin_port, 7001);
    assert_eq!(config.server_host, DEFAULT_SERVER_HOST);
    assert_eq!(config.server_port, 9002);
    assert!(config.serve_profile.is_none());
}

#[test]
fn daemon_config_rejects_zero_ports() {
    let err = validate_daemon_config(&DaemonConfig {
        admin_host: DEFAULT_ADMIN_HOST.to_owned(),
        admin_port: 0,
        server_host: DEFAULT_SERVER_HOST.to_owned(),
        server_port: DEFAULT_SERVER_PORT,
        serve_profile: None,
        server_host_override: false,
        server_port_override: false,
    })
    .expect_err("admin port zero must be rejected")
    .to_string();
    assert!(err.contains("admin_port"), "got: {err}");

    let err = validate_daemon_config(&DaemonConfig {
        admin_host: DEFAULT_ADMIN_HOST.to_owned(),
        admin_port: DEFAULT_ADMIN_PORT,
        server_host: DEFAULT_SERVER_HOST.to_owned(),
        server_port: 0,
        serve_profile: None,
        server_host_override: false,
        server_port_override: false,
    })
    .expect_err("server port zero must be rejected")
    .to_string();
    assert!(err.contains("server_port"), "got: {err}");
}

#[test]
fn daemon_hosts_are_literal_loopback_only() {
    assert!(is_allowed_loopback_host("127.0.0.1"));
    assert!(is_allowed_loopback_host("localhost"));
    assert!(is_allowed_loopback_host("[::1]"));
    assert!(!is_allowed_loopback_host("0.0.0.0"));
    assert!(!is_allowed_loopback_host("example.com"));
}

#[test]
fn model_lifecycle_path_percent_encodes_model_id() {
    assert_eq!(
        model_lifecycle_path("team/model a", "load"),
        "/v1/models/team%2Fmodel%20a/load"
    );
}

#[test]
fn parse_http_json_response_accepts_empty_body() {
    let parsed = parse_http_json_response("HTTP/1.1 204 No Content\r\n\r\n")
        .expect("parse empty HTTP JSON response");

    assert_eq!(parsed.status_code, 204);
    assert!(parsed.body.is_none());
}

#[test]
fn status_metrics_join_loaded_model_by_id() {
    let models = serde_json::json!({
        "data": [
            { "id": "a", "loaded": false },
            { "id": "b", "loaded": true }
        ]
    });
    let metrics = serde_json::json!({
        "models": [
            { "model_id": "a", "hits": 1, "misses": 2, "bytes": 3, "kv_cache_bytes": 4 },
            { "model_id": "b", "hits": 10, "misses": 20, "bytes": 30, "kv_cache_bytes": 40 }
        ]
    });

    let cache = cache_status(Some(&metrics), Some(&models));
    let memory = memory_status(Some(&metrics), Some(&models));

    assert_eq!(cache.hits, Some(10));
    assert_eq!(cache.misses, Some(20));
    assert_eq!(cache.bytes, Some(30));
    assert_eq!(memory.kv_cache_bytes, Some(40));
}

#[test]
fn model_list_normalizes_registry_entries() {
    let models = serde_json::json!({
        "data": [
            { "id": "a", "loaded": false },
            { "id": "b", "loaded": true }
        ]
    });

    let list = model_list(Some(&models));
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].id, "a");
    assert!(!list[0].loaded);
    assert_eq!(list[1].id, "b");
    assert!(list[1].loaded);
}

#[test]
fn pid_zero_is_conservatively_alive() {
    assert!(pid_is_alive(0));
}

#[test]
fn supervised_serve_args_force_configured_host_and_port_before_passthrough() {
    let config = DaemonConfig {
        admin_host: "127.0.0.1".to_owned(),
        admin_port: 6276,
        server_host: "127.0.0.1".to_owned(),
        server_port: 9001,
        serve_profile: Some("menu-default".to_owned()),
        server_host_override: true,
        server_port_override: true,
    };

    assert_eq!(
        supervised_serve_args(&config),
        [
            "serve",
            "--profile",
            "menu-default",
            "--host",
            "127.0.0.1",
            "--port",
            "9001"
        ]
        .map(str::to_owned)
    );
}

#[test]
fn supervised_serve_args_do_not_override_profile_host_and_port_by_default() {
    let config = DaemonConfig {
        admin_host: "127.0.0.1".to_owned(),
        admin_port: 6276,
        server_host: "127.0.0.1".to_owned(),
        server_port: 9001,
        serve_profile: Some("menu-default".to_owned()),
        server_host_override: false,
        server_port_override: false,
    };

    assert_eq!(
        supervised_serve_args(&config),
        ["serve", "--profile", "menu-default"].map(str::to_owned)
    );
}

#[test]
fn start_preflight_prefers_existing_supervised_child() {
    let claim = ClaimStatus {
        held: true,
        holder_pid: Some(456),
        holder_alive: true,
        path: "/tmp/rmlx.1.claim".to_owned(),
        last_error: None,
    };

    assert_eq!(
        classify_start_preflight(Some(123), true, &claim),
        StartPreflight::AlreadySupervised { pid: 123 }
    );
}

#[test]
fn start_preflight_rejects_external_healthy_server() {
    let claim = ClaimStatus {
        held: false,
        holder_pid: None,
        holder_alive: false,
        path: "/tmp/rmlx.1.claim".to_owned(),
        last_error: None,
    };

    assert_eq!(
        classify_start_preflight(None, true, &claim),
        StartPreflight::ConflictAlreadyRunning {
            healthy: true,
            holder_pid: None
        }
    );
}

#[test]
fn start_preflight_rejects_live_claim_holder_even_without_health() {
    let claim = ClaimStatus {
        held: true,
        holder_pid: Some(456),
        holder_alive: true,
        path: "/tmp/rmlx.1.claim".to_owned(),
        last_error: None,
    };

    assert_eq!(
        classify_start_preflight(None, false, &claim),
        StartPreflight::ConflictAlreadyRunning {
            healthy: false,
            holder_pid: Some(456)
        }
    );
}

#[test]
fn start_preflight_allows_clear_port_with_no_live_claim() {
    let claim = ClaimStatus {
        held: false,
        holder_pid: None,
        holder_alive: false,
        path: "/tmp/rmlx.1.claim".to_owned(),
        last_error: None,
    };

    assert_eq!(
        classify_start_preflight(None, false, &claim),
        StartPreflight::Clear
    );
}
