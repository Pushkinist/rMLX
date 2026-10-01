//! A time limit for a test step that hands MLX work to another thread. The
//! test crates of `rmlx-mlx`, `rmlx-models` and `rmlx-server` include this
//! file with `#[path]`.

use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;

/// What the step sent, or `None` when its thread ended without sending (it
/// panicked).
///
/// Past `limit` the step is hung. Nothing can stop it, and it holds the MLX
/// thread that every later test of this binary needs. So name the test and
/// the step on stderr, and end the process.
#[allow(
    clippy::exit,
    reason = "a hung step cannot be stopped, and it holds the MLX thread every later test needs"
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
                "{test}: {what} did not finish within {limit:?}. It holds the MLX thread, so \
                 this test binary ends here."
            )
            .ok();
            std::process::exit(101);
        }
    }
}
