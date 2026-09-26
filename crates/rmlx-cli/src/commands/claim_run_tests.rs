// These tests stand in a flock on a temp file for the Metal claim: they never
// name, create or lock the machine-wide claim file.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::*;
use std::fs::{File, TryLockError};
use std::io::{BufRead as _, BufReader, IsTerminal as _, Write as _};
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
        "the child's stdin must be empty"
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

/// Each forwarded signal sent to a process running `forward_signals` arrives
/// on its channel instead of acting on the process. The handlers stay for the
/// life of a process, so they are installed in a child of this test binary.
#[test]
fn claim_run_receives_sigterm_sigint_and_sighup() {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "commands::claim_run::tests::signal_receiver_child",
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
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "go").unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map_while(Result::ok)
        // libtest writes its `test <name> ... ` prefix on the child's first line.
        .filter_map(|line| line.find("signal-child ").map(|at| line[at..].to_owned()));
    assert_eq!(lines.next().as_deref(), Some("signal-child ready"));
    let pid = child.id().to_string();
    for (name, number) in FORWARDED {
        let sent = Command::new("/bin/kill")
            .args([format!("-{name}"), pid.clone()])
            .status()
            .unwrap();
        assert!(sent.success(), "kill -{name} failed");
        assert_eq!(
            lines.next(),
            Some(format!("signal-child got {number}")),
            "SIG{name} must reach the channel"
        );
    }
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

/// The child half of the signal test: installs the handlers, reports each
/// signal it receives, and returns at once when run by hand from a terminal.
#[test]
#[ignore = "child process of claim_run_receives_sigterm_sigint_and_sighup; the parent test starts it"]
fn signal_receiver_child() {
    let stdin = io::stdin();
    if stdin.is_terminal() {
        return;
    }
    let mut go = String::new();
    if stdin.lock().read_line(&mut go).is_err() || go.trim() != "go" {
        return;
    }
    let signals = forward_signals().unwrap();
    println!("signal-child ready");
    for _ in FORWARDED {
        let number = signals.recv_timeout(Duration::from_secs(5)).unwrap();
        println!("signal-child got {number}");
    }
}
