// These tests stand in a flock on a temp file for the Metal claim: they never
// name, create or lock the machine-wide claim file.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::*;
use std::fs::{File, TryLockError};
use std::io::{BufRead as _, BufReader};
use std::path::Path;
use std::process::{Child, ChildStdin};
use std::sync::mpsc;
use std::time::Instant;

fn sh(script: &str) -> Vec<OsString> {
    vec!["/bin/sh".into(), "-c".into(), script.into()]
}

fn temp_lock(dir: &Path) -> OwnedFd {
    let path = dir.join("lock");
    std::fs::write(&path, "4242 holder record").unwrap();
    let file = File::options().read(true).write(true).open(&path).unwrap();
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

/// The child holds the lock through its inherited fd after dropping its
/// stdin, and this process keeps no copy: the lock frees when the child
/// closes that fd, while the child still runs. The child's stdin is empty.
#[test]
fn claim_run_child_holds_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let step = |name: &str| dir.path().join(name);
    let lock = temp_lock(dir.path());
    let fd = lock.as_raw_fd();
    let command = sh(&format!(
        "cat >{stdin}; exec 0</dev/null; touch {ready}; \
         i=0; while [ ! -e {close} ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done; \
         exec {fd}<&-; touch {closed}; \
         i=0; while [ ! -e {done} ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done",
        stdin = step("stdin").display(),
        ready = step("ready").display(),
        close = step("close").display(),
        closed = step("closed").display(),
        done = step("done").display(),
    ));
    let (_sender, signals) = mpsc::channel();
    let runner = std::thread::spawn(move || run_holding(lock, &command, &signals));

    assert!(
        wait_for(&step("ready"), Duration::from_secs(10)),
        "the child's stdin read blocked"
    );
    let held_after_stdin_redirect = locked_elsewhere(dir.path());
    File::create(step("close")).unwrap();
    assert!(wait_for(&step("closed"), Duration::from_secs(10)));
    let held_after_child_closed = locked_elsewhere(dir.path());
    File::create(step("done")).unwrap();
    assert_eq!(runner.join().unwrap().unwrap(), 0);

    assert!(
        held_after_stdin_redirect,
        "the child must hold the lock after redirecting its stdin"
    );
    assert!(
        !held_after_child_closed,
        "this process must keep no copy of the lock while the child runs"
    );
    assert_eq!(
        std::fs::read(step("stdin")).unwrap(),
        b"",
        "the child's stdin must be empty, not the holder record"
    );
}

/// Only the claim-run child gets the lock fd across exec: this process's copy
/// stays close-on-exec, so a process it starts at the same time does not.
#[test]
fn claim_run_other_spawns_do_not_inherit_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let lock = temp_lock(dir.path());
    let probe = format!("test -e /dev/fd/{}", lock.as_raw_fd());
    let mut holder = spawn_holding(&sh(&probe), lock.as_fd()).unwrap();
    let other = Command::new("/bin/sh")
        .args(["-c", &probe])
        .status()
        .unwrap();
    assert!(
        holder.wait().unwrap().success(),
        "the claim-run child must inherit the lock fd"
    );
    assert!(
        !other.success(),
        "a process started beside it must not inherit the lock fd"
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

const FORWARDED: [(&str, i32); 3] = [
    ("TERM", libc::SIGTERM),
    ("INT", libc::SIGINT),
    ("HUP", libc::SIGHUP),
];

/// A test filter that matches no test. A child test runs its body only when
/// its argv carries it, so a hand run of the ignored child tests does nothing.
const CHILD_MARKER: &str = "started-by-a-claim-run-parent-test";

/// Start the ignored child test `name` of this binary with a pipe as its stdin
/// and return it with its stdin and the lines it prints after `tag`.
fn start_child(name: &str, tag: &'static str) -> (Child, ChildStdin, impl Iterator<Item = String>) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            &format!("commands::claim_run::tests::{name}"),
            CHILD_MARKER,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    let lines = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map_while(Result::ok)
        // libtest writes its `test <name> ... ` prefix on the child's first line.
        .filter_map(move |line| line.find(tag).map(|at| line[at + tag.len()..].to_owned()));
    (child, stdin, lines)
}

fn started_by_parent() -> bool {
    std::env::args().any(|arg| arg == CHILD_MARKER)
}

/// The command's stdin is `/dev/null` even when this process's stdin is a
/// pipe, so the check runs in a child of this test binary whose stdin is one.
#[test]
fn claim_run_child_stdin_is_dev_null() {
    let (mut child, stdin, mut lines) = start_child("null_stdin_child", "null-stdin-child ");
    assert_eq!(lines.next().as_deref(), Some("exit 0"));
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

#[test]
#[ignore = "child process of claim_run_child_stdin_is_dev_null; the parent test starts it"]
fn null_stdin_child() {
    if !started_by_parent() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_sender, signals) = mpsc::channel();
    let code = run_holding(
        temp_lock(dir.path()),
        &sh("[ /dev/stdin -ef /dev/null ]"),
        &signals,
    )
    .unwrap();
    println!("null-stdin-child exit {code}");
}

/// Each forwarded signal sent to a process running `forward_signals` arrives
/// on its channel instead of acting on the process. The handlers stay for the
/// life of a process, so they are installed in a child of this test binary.
#[test]
fn claim_run_receives_sigterm_sigint_and_sighup() {
    let (mut child, stdin, mut lines) = start_child("signal_receiver_child", "signal-child ");
    assert_eq!(lines.next().as_deref(), Some("ready"));
    let pid = child.id().to_string();
    for (name, number) in FORWARDED {
        let sent = Command::new("/bin/kill")
            .args([format!("-{name}"), pid.clone()])
            .status()
            .unwrap();
        assert!(sent.success(), "kill -{name} failed");
        assert_eq!(
            lines.next(),
            Some(format!("got {number}")),
            "SIG{name} must reach the channel"
        );
    }
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

/// The child half of the signal test: installs the handlers and reports each
/// signal it receives.
#[test]
#[ignore = "child of claim_run_receives_sigterm_sigint_and_sighup, which starts it"]
fn signal_receiver_child() {
    if !started_by_parent() {
        return;
    }
    let signals = forward_signals().unwrap();
    println!("signal-child ready");
    for _ in FORWARDED {
        let number = signals.recv_timeout(Duration::from_secs(5)).unwrap();
        println!("signal-child got {number}");
    }
}
