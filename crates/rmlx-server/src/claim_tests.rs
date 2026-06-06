use super::*;

/// Two try_claim calls on the same port: second must fail.
#[test]
fn double_claim_same_port_fails() {
    // Use a unique port to avoid collision with other test runs.
    let port: u16 = 59876;
    // Clean up any stale file from a previous crashed run.
    let _ = std::fs::remove_file(format!("/tmp/rmlx.{port}.claim"));

    let first = try_claim(port).expect("first claim must succeed");
    let second = try_claim(port);

    assert!(
        matches!(second, Err(ClaimError::AlreadyHeld { .. })),
        "second claim should fail with AlreadyHeld, got: {second:?}"
    );

    // Drop first so the lock file is removed.
    drop(first);

    // After release, a third claim must succeed.
    let third = try_claim(port);
    assert!(
        third.is_ok(),
        "claim after release must succeed, got: {third:?}"
    );
}

/// Claim file is removed when the guard is dropped.
#[test]
fn claim_file_removed_on_drop() {
    let port: u16 = 59877;
    let _ = std::fs::remove_file(format!("/tmp/rmlx.{port}.claim"));

    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));
    {
        let _claim = try_claim(port).expect("claim must succeed");
        assert!(path.exists(), "claim file must exist while held");
    }
    assert!(!path.exists(), "claim file must be removed after drop");
}
