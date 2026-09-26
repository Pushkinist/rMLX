// These tests stand in a flock on a temp file for the Metal claim: they never
// name, create or lock the machine-wide claim file.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::*;
use std::fs::{File, TryLockError};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

fn sh(script: &str) -> Vec<OsString> {
    vec!["/bin/sh".into(), "-c".into(), script.into()]
}

fn temp_lock(dir: &Path) -> OwnedFd {
    let file = File::create(dir.join("lock")).unwrap();
    file.lock().unwrap();
    OwnedFd::from(file)
}

fn locked_elsewhere(dir: &Path) -> bool {
    let probe = File::open(dir.join("lock")).unwrap();
    match probe.try_lock() {
        Ok(()) => false,
        Err(TryLockError::WouldBlock) => true,
        Err(TryLockError::Error(e)) => panic!("lock probe: {e}"),
    }
}

fn wait_for(path: &Path, limit: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

/// The child reads its stdin to the end, which does not block, and holds the
/// lock after this process has let go of every copy of it.
#[test]
fn claim_run_child_holds_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready");
    let go = dir.path().join("go");
    let command = sh(&format!(
        "cat >/dev/null; touch {ready}; i=0; \
         while [ ! -e {go} ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done",
        ready = ready.display(),
        go = go.display(),
    ));
    let lock = temp_lock(dir.path());
    let (_sender, signals) = mpsc::channel();
    let runner = std::thread::spawn(move || run_holding(lock, &command, &signals));

    assert!(
        wait_for(&ready, Duration::from_secs(10)),
        "the child's stdin read blocked"
    );
    let held_by_child = locked_elsewhere(dir.path());
    File::create(&go).unwrap();
    assert_eq!(runner.join().unwrap().unwrap(), 0);
    assert!(held_by_child, "the child must hold the lock");
    assert!(
        !locked_elsewhere(dir.path()),
        "the lock must be free once the child exits"
    );
}

#[test]
fn claim_run_exits_with_child_status() {
    let dir = tempfile::tempdir().unwrap();
    let (_sender, signals) = mpsc::channel();
    assert_eq!(
        run_holding(temp_lock(dir.path()), &sh("exit 7"), &signals).unwrap(),
        7
    );
    assert_eq!(
        run_holding(temp_lock(dir.path()), &sh("kill -TERM $$"), &signals).unwrap(),
        128 + libc::SIGTERM,
        "a child killed by a signal exits 128 + the signal"
    );
}

#[test]
fn claim_run_refuses_an_empty_command() {
    let dir = tempfile::tempdir().unwrap();
    let (_sender, signals) = mpsc::channel();
    let err = run_holding(temp_lock(dir.path()), &[], &signals).unwrap_err();
    assert!(err.to_string().contains("no command"), "{err}");
}

/// The child is a shell that waits on a grandchild in the same process group.
/// SIGTERM reaches the grandchild only when the whole group is signalled.
#[test]
fn claim_run_forwards_sigterm() {
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready");
    let marker = dir.path().join("got-term");
    let grandchild = dir.path().join("grandchild.sh");
    std::fs::write(
        &grandchild,
        format!(
            "trap 'touch {marker}; exit 0' TERM\ntouch {ready}\ni=0\n\
             while [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done\n",
            marker = marker.display(),
            ready = ready.display(),
        ),
    )
    .unwrap();
    let command = sh(&format!(
        "/bin/sh {} 2>/dev/null & wait",
        grandchild.display()
    ));
    let lock = temp_lock(dir.path());
    let (sender, signals) = mpsc::channel();
    let runner = std::thread::spawn(move || run_holding(lock, &command, &signals));

    assert!(
        wait_for(&ready, Duration::from_secs(10)),
        "grandchild never started"
    );
    sender.send(libc::SIGTERM).unwrap();
    let code = runner.join().unwrap().unwrap();

    assert_eq!(code, 128 + libc::SIGTERM, "the child must die of SIGTERM");
    assert!(
        wait_for(&marker, Duration::from_secs(5)),
        "SIGTERM must reach the child's whole process group"
    );
}

/// Each forwarded signal sent to this process arrives on the channel instead
/// of acting on the process.
#[test]
fn claim_run_receives_sigterm_sigint_and_sighup() {
    let signals = forward_signals().unwrap();
    let pid = std::process::id().to_string();
    for (name, number) in [
        ("TERM", libc::SIGTERM),
        ("INT", libc::SIGINT),
        ("HUP", libc::SIGHUP),
    ] {
        let sent = Command::new("/bin/kill")
            .args([format!("-{name}"), pid.clone()])
            .status()
            .unwrap();
        assert!(sent.success(), "kill -{name} failed");
        assert_eq!(
            signals.recv_timeout(Duration::from_secs(5)),
            Ok(number),
            "SIG{name} must reach the channel"
        );
    }
}
