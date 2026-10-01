//! A time limit for a test step that hands MLX work to another thread. The
//! test crates of `rmlx-mlx`, `rmlx-models` and `rmlx-server` include this
//! file with `#[path]`.

use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;

/// The limit for one step after the one-time MLX init of the process.
pub(crate) const LIMIT: Duration = Duration::from_secs(60);

/// The limit for a step that includes the one-time MLX init of a process. The
/// first Metal device init reads the directory of the test binary, and in a
/// large `target/debug/deps` that takes minutes.
#[allow(
    dead_code,
    reason = "only the rmlx-mlx tests time the one-time MLX init"
)]
pub(crate) const INIT_LIMIT: Duration = Duration::from_secs(600);

/// What the step sent, or `None` when its thread ended without sending (it
/// panicked).
///
/// Past `limit` the step is hung. A thread cannot be stopped, and it can hold
/// the MLX thread that every later test needs. A child process ends when this
/// process ends. So name the test and the step on stderr, and end the process.
#[allow(
    clippy::exit,
    reason = "a hung thread cannot be stopped, and it can hold the MLX thread every later test needs"
)]
pub(crate) fn within_limit<T>(done: &mpsc::Receiver<T>, limit: Duration, what: &str) -> Option<T> {
    match done.recv_timeout(limit) {
        Ok(value) => Some(value),
        Err(mpsc::RecvTimeoutError::Disconnected) => None,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let test = std::thread::current()
                .name()
                .unwrap_or("an unnamed test")
                .to_owned();
            // Not `eprintln!`: the test harness captures that, and drops what
            // it captured when the process exits.
            writeln!(
                std::io::stderr().lock(),
                "{test}: {what} did not finish within {limit:?}, so this test binary ends here."
            )
            .ok();
            std::process::exit(101);
        }
    }
}
