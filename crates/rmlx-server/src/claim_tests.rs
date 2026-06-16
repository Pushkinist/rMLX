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

/// Write a claim file holding `pid` with no open fd (so no flock is held).
/// Mimics a process that died without running `Drop` (SIGTERM/SIGKILL/crash).
fn write_stale_claim(path: &PathBuf, pid: u32) {
    use std::io::Write as _;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("create stale claim file");
    write!(f, "{pid}").expect("write stale pid");
    f.flush().expect("flush stale pid");
    // f dropped here: fd closed, flock released, file persists on disk.
}

/// A stale claim whose holder PID is dead must be auto-reclaimed: `try_claim`
/// succeeds and the body is rewritten with the current PID.
#[test]
fn stale_dead_pid_is_reclaimed() {
    let port: u16 = 59880;
    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));
    let _ = std::fs::remove_file(&path);

    // 999999 is well above the typical macOS PID ceiling and not alive.
    assert!(
        !pid_is_alive(999_999),
        "test precondition: PID 999999 must be dead"
    );
    write_stale_claim(&path, 999_999);
    assert!(path.exists(), "stale file should exist before reclaim");

    let claim = try_claim(port).expect("stale dead-PID claim must be reclaimed");
    let body = std::fs::read_to_string(&path).expect("read reclaimed claim body");
    assert_eq!(
        body.trim(),
        std::process::id().to_string(),
        "reclaimed file must hold our PID, with no trailing stale digits"
    );

    drop(claim);
    assert!(!path.exists(), "reclaimed claim must be removed on drop");
}

/// A claim whose holder PID is ALIVE must never be stolen — the single-MLX
/// invariant. We use our own PID (guaranteed alive) as the holder.
#[test]
fn live_pid_claim_is_refused() {
    let port: u16 = 59881;
    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));
    let _ = std::fs::remove_file(&path);

    let self_pid = std::process::id();
    assert!(pid_is_alive(self_pid), "our own PID must read as alive");
    write_stale_claim(&path, self_pid);

    let res = try_claim(port);
    assert!(
        matches!(res, Err(ClaimError::AlreadyHeld { holder_pid, .. }) if holder_pid == self_pid),
        "claim with a live holder PID must be refused, got: {res:?}"
    );
    // The live-holder file must be left intact (not stolen, not removed).
    assert!(
        path.exists(),
        "live-holder claim file must be left in place"
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("read body").trim(),
        self_pid.to_string(),
        "live-holder claim body must be untouched"
    );

    let _ = std::fs::remove_file(&path);
}

/// A claim file whose body holds a DEAD PID but which still has a live
/// `flock(LOCK_EX)` held on a second fd must NOT be reclaimed: the flock is the
/// load-bearing gate, so `try_claim` refuses with `AlreadyHeld` even though the
/// PID probe reports the body PID as dead. This covers the "PID dead but flock
/// still held" safety branch that `write_stale_claim` cannot exercise (it
/// closes its fd, releasing the lock).
#[test]
fn flock_held_dead_pid_is_refused() {
    use std::os::unix::io::AsRawFd as _;

    let port: u16 = 59882;
    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));
    let _ = std::fs::remove_file(&path);

    // Write a dead PID into the body, but keep a fd open and an exclusive flock
    // held on it — mimics a live fd whose recorded PID no longer resolves.
    assert!(
        !pid_is_alive(999_999),
        "test precondition: PID 999999 must be dead"
    );
    let holder = {
        use std::io::Write as _;
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create flock-held claim file");
        write!(f, "999999").expect("write dead pid");
        f.flush().expect("flush dead pid");
        f
    };
    // SAFETY: holder.as_raw_fd() is a valid open fd for the lifetime of `holder`.
    let locked = unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        locked, 0,
        "test setup: must acquire flock on the claim file"
    );

    let res = try_claim(port);
    assert!(
        matches!(res, Err(ClaimError::AlreadyHeld { .. })),
        "flock-held claim must be refused even with a dead PID body, got: {res:?}"
    );
    // File must be left intact — not stolen, not removed.
    assert!(path.exists(), "flock-held claim file must be left in place");

    drop(holder); // release the flock + close fd
    let _ = std::fs::remove_file(&path);
}

/// `pid_is_alive(0)` is conservatively treated as alive so an unreadable /
/// empty claim body is never stolen.
#[test]
fn pid_zero_treated_as_alive() {
    assert!(
        pid_is_alive(0),
        "PID 0 must be treated as alive (conservative)"
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
