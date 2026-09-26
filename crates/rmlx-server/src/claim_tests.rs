use super::*;
use std::io::{BufRead as _, BufReader, IsTerminal as _, Write as _};
use std::os::unix::fs::symlink;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Mutex, MutexGuard, PoisonError};
use tempfile::TempDir;

/// A child that a concurrent test forks holds a copy of every open claim fd
/// of this process until its exec. A test that needs its own claim released
/// at drop holds this lock, and so does every spawn.
static SPAWN: Mutex<()> = Mutex::new(());

fn no_spawn() -> MutexGuard<'static, ()> {
    SPAWN.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lock_in(dir: &TempDir) -> PathBuf {
    dir.path().join("lock")
}

fn is_held(result: &Result<MetalClaim, ClaimError>) -> bool {
    matches!(result, Err(ClaimError::AlreadyHeld { .. }))
}

/// A child process that holds the claim on a lock file. The child is this
/// test binary, run as `claim_holder_child`.
struct Holder {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl Holder {
    fn start(lock: &Path) -> Self {
        let spawn = no_spawn();
        let mut child = Command::new(std::env::current_exe().expect("current_exe"))
            .args([
                "claim::tests::claim_holder_child",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the holder child");
        drop(spawn);
        let mut stdin = child.stdin.take().expect("child stdin");
        writeln!(stdin, "{}", lock.display()).expect("send the lock path");
        let stdout = BufReader::new(child.stdout.take().expect("child stdout"));
        let held = stdout
            .lines()
            .map_while(Result::ok)
            .any(|line| line.ends_with(&format!(" held {}", child.id())));
        assert!(held, "the child did not report that it holds the claim");
        Holder {
            child,
            stdin: Some(stdin),
        }
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Close the child's stdin: the child calls `process::exit` with the
    /// claim still open, so no `Drop` runs.
    fn exit(mut self) {
        drop(self.stdin.take());
        let status = self.child.wait().expect("wait for the child");
        assert!(status.success(), "the child failed: {status}");
    }

    fn kill(mut self) {
        self.child.kill().expect("SIGKILL the child");
        self.child.wait().expect("wait for the child");
    }
}

/// The child half of the two-process tests. It reads a lock path on stdin,
/// holds the claim on it, prints `held <pid>`, and exits when stdin closes.
#[test]
#[ignore = "child process of the two-process claim tests; the parent test starts it"]
fn claim_holder_child() {
    let stdin = io::stdin();
    if stdin.is_terminal() {
        return;
    }
    let mut path = String::new();
    if stdin.lock().read_line(&mut path).is_err() || path.trim().is_empty() {
        return;
    }
    let _claim = claim_at(Path::new(path.trim())).expect("the child must get the claim");
    println!("held {}", std::process::id());
    io::stdout().flush().expect("flush stdout");
    let mut rest = Vec::new();
    let _ = stdin.lock().read_to_end(&mut rest);
    std::process::exit(0);
}

#[test]
fn production_claim_path_is_one_file_under_var_tmp() {
    let path = Path::new(CLAIM_PATH);
    assert_eq!(path.parent(), Some(Path::new("/var/tmp")));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("file name");
    assert!(
        !name.contains(|c: char| c.is_ascii_digit()),
        "the claim file name must not carry a port: {name}"
    );
}

/// flock locks belong to the open file, so a second open in this process
/// conflicts. A per-process lock (`fcntl`, `lockf`) would not.
#[test]
fn second_claim_in_same_process_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let first = claim_at(&lock).expect("first claim");
    let second = claim_at(&lock);
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = second else {
        panic!("the second claim must be refused, got {second:?}");
    };
    assert_eq!(holder_pid, Some(std::process::id()));
    drop(first);
}

#[test]
fn drop_keeps_the_file_and_the_next_claim_locks_it() {
    let _no_spawn = no_spawn();
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    drop(claim_at(&lock).expect("first claim"));
    assert!(lock.exists(), "the claim file must stay after drop");
    let next = claim_at(&lock).expect("the next claim after drop");
    assert!(
        is_held(&claim_at(&lock)),
        "the next claim must hold the lock"
    );
    drop(next);
}

#[test]
fn symlink_at_claim_path_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let target = dir.path().join("target");
    std::fs::write(&target, "keep").expect("write target");
    let lock = lock_in(&dir);
    symlink(&target, &lock).expect("plant a symlink");
    let result = claim_at(&lock);
    assert!(
        matches!(result, Err(ClaimError::Io { .. })),
        "a symlink must be refused, got {result:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&target).expect("read target"),
        "keep"
    );
}

#[test]
fn claim_file_mode_is_0666() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let _claim = claim_at(&lock).expect("claim");
    let mode = std::fs::metadata(&lock).expect("stat").permissions().mode();
    assert_eq!(mode & 0o777, 0o666);
}

/// A file this user cannot write, as another user's claim file is: the claim
/// takes the flock through a read-only open and writes no body.
#[test]
fn read_only_file_is_still_locked() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    std::fs::write(&lock, "").expect("create the file");
    std::fs::set_permissions(&lock, Permissions::from_mode(0o444)).expect("chmod 0444");
    let claim = claim_at(&lock).expect("a read-only file must still be claimed");
    assert!(
        is_held(&claim_at(&lock)),
        "the read-only claim must hold the lock"
    );
    assert_eq!(std::fs::read_to_string(&lock).expect("read body"), "");
    drop(claim);
}

/// The last holder's PID stays in the body, and the OS can give that PID to
/// another process. Only the flock decides.
#[test]
fn unlocked_file_with_live_pid_body_is_claimed() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    std::fs::write(&lock, format!("{} some command", std::process::id())).expect("write body");
    let claim = claim_at(&lock).expect("an unlocked file must be claimed");
    let body = std::fs::read_to_string(&lock).expect("read body");
    let (pid, _) = body.split_once(' ').expect("body is `<pid> <argv>`");
    assert_eq!(pid, std::process::id().to_string());
    drop(claim);
}

#[test]
fn refusal_message_names_holder_and_stop_action() {
    let message = ClaimError::AlreadyHeld {
        holder_pid: Some(4242),
        holder_command: "rmlx serve --port 8080".to_owned(),
    }
    .to_string();
    assert!(message.contains("PID 4242"), "{message}");
    assert!(message.contains("rmlx serve --port 8080"), "{message}");
    assert!(message.contains("kill 4242"), "{message}");
    assert!(!message.contains(CLAIM_PATH), "{message}");
    let lower = message.to_lowercase();
    for word in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        assert!(
            !["rm", "unlink", "delete", "remove"].contains(&word),
            "the refusal must not tell the operator to delete a file: {message}"
        );
    }
}

#[test]
fn second_process_is_refused_and_named() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    let result = claim_at(&lock);
    let Err(ClaimError::AlreadyHeld {
        holder_pid,
        holder_command,
    }) = result
    else {
        panic!("a claim held by another process must be refused, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    assert!(
        holder_command.contains("claim_holder_child"),
        "the refusal must name the holder's command: {holder_command}"
    );
    holder.exit();
}

#[test]
fn exited_holder_is_reclaimed() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    assert!(is_held(&claim_at(&lock)));
    holder.exit();
    claim_at(&lock).expect("the claim of an exited holder must be free");
}

#[test]
fn killed_holder_is_reclaimed() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    assert!(is_held(&claim_at(&lock)));
    holder.kill();
    claim_at(&lock).expect("the claim of a killed holder must be free");
}

fn legacy_claim_in(dir: &TempDir) -> PathBuf {
    dir.path()
        .join(format!("{LEGACY_CLAIM_PREFIX}8080{LEGACY_CLAIM_SUFFIX}"))
}

#[test]
fn held_legacy_claim_is_refused_and_kept() {
    let dir = TempDir::new().expect("temp dir");
    let legacy = legacy_claim_in(&dir);
    let holder = Holder::start(&legacy);
    let result = refuse_held_legacy_claim(dir.path());
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = result else {
        panic!("a held legacy claim must be refused, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    assert!(legacy.exists(), "the probe must keep the legacy file");
    holder.exit();
    assert!(legacy.exists(), "the probe must keep the legacy file");
}

#[test]
fn free_legacy_claim_is_passed_and_kept() {
    let dir = TempDir::new().expect("temp dir");
    let legacy = legacy_claim_in(&dir);
    std::fs::write(&legacy, "4242").expect("write a legacy body");
    refuse_held_legacy_claim(dir.path()).expect("a free legacy claim must pass");
    assert_eq!(std::fs::read_to_string(&legacy).expect("read body"), "4242");
}
