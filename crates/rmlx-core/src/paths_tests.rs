use super::*;

#[test]
fn env_home_must_be_absolute() {
    // Pure unit on the helper — exercises both branches without
    // touching the cached `home()` (which would poison cross-test state).
    let prev = std::env::var_os(ENV_HOME);

    std::env::set_var(ENV_HOME, "not/absolute");
    let refused = env_home().map_err(|e| e.to_string());
    assert_eq!(
        refused,
        Err("config: RMLX_HOME=not/absolute must be an absolute path".to_owned()),
        "a relative RMLX_HOME is refused, naming the value"
    );
    assert!(check_env_home().is_err());

    std::env::set_var(ENV_HOME, "/tmp/rmlx-test-abs");
    assert_eq!(
        env_home().ok(),
        Some(Some(PathBuf::from("/tmp/rmlx-test-abs"))),
        "absolute RMLX_HOME must round-trip"
    );

    std::env::remove_var(ENV_HOME);
    assert!(
        matches!(env_home(), Ok(None)),
        "an unset RMLX_HOME is no override"
    );

    match prev {
        Some(v) => std::env::set_var(ENV_HOME, v),
        None => std::env::remove_var(ENV_HOME),
    }
}

#[test]
fn workspace_root_finds_cargo_lock_or_returns_none() {
    // Inside the repo: walking up from any subdir must terminate at
    // the workspace root (or None if cwd is somewhere odd).
    if let Some(ws) = workspace_root() {
        assert!(
            ws.join("Cargo.lock").is_file(),
            "returned path must actually contain Cargo.lock"
        );
    }
}
