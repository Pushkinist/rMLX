//! `rmlx claim run -- <command>` — run a command while this process holds the
//! Metal claim.
//!
//! The child's stdin is a duplicate of the claim fd. It shares the claim's
//! open file and so its flock: the claim stays held while the child runs, even
//! if this process dies first. It also keeps the terminal away from the child,
//! which runs in its own process group, where a terminal read would stop it.
//! SIGTERM, SIGINT and SIGHUP sent to this process are forwarded to the child's
//! process group. The exit status is the child's; a child killed by a signal
//! exits `128 + <signal>`, as a shell reports it.
//!
//! The command must not start `rmlx`: this process holds the claim, so a nested
//! `rmlx` GPU command is refused it.

use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd as _, OwnedFd};
use std::os::unix::process::{CommandExt as _, ExitStatusExt as _};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use anyhow::Context as _;
use rmlx_server::try_claim;
use tracing::{info, warn};

use crate::commands::parse::exit_if_held;

/// How often the wait loop checks whether the child has exited.
const POLL: Duration = Duration::from_millis(50);

/// Take the Metal claim, run `command` while holding it, and return the exit
/// code to exit with.
pub(crate) fn run_claim_run(command: &[OsString]) -> anyhow::Result<i32> {
    let claim = exit_if_held(try_claim())?;
    let lock = claim
        .as_fd()
        .try_clone_to_owned()
        .context("claim run: duplicate the claim fd")?;
    let signals = forward_signals()?;
    run_holding(lock, command, &signals)
}

/// Run `command` in its own process group with `lock` as its stdin, forwarding
/// every signal number received on `signals` to that group. This process keeps
/// no copy of `lock` once the child has started.
pub(crate) fn run_holding(
    lock: OwnedFd,
    command: &[OsString],
    signals: &Receiver<i32>,
) -> anyhow::Result<i32> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("claim run: no command given"))?;
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::from(lock))
        .process_group(0)
        .spawn()
        .with_context(|| format!("claim run: start {}", program.to_string_lossy()))?;
    let group = libc::pid_t::try_from(child.id()).context("claim run: child pid")?;
    info!(pid = child.id(), command = ?command, "claim run: child started");
    loop {
        if let Some(status) = child.try_wait().context("claim run: wait")? {
            return Ok(exit_code(status));
        }
        match signals.recv_timeout(POLL) {
            Ok(signal) => forward(group, signal),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Ok(exit_code(child.wait().context("claim run: wait")?));
            }
        }
    }
}

/// Deliver SIGTERM, SIGINT and SIGHUP to the returned channel instead of acting on
/// them. The handlers are installed before this returns, so no signal that
/// arrives after the child starts kills this process.
fn forward_signals() -> anyhow::Result<Receiver<i32>> {
    use tokio::signal::unix::{signal, SignalKind};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("claim run: signal runtime")?;
    let (mut terminate, mut interrupt, mut hangup) = {
        let _entered = runtime.enter();
        (
            signal(SignalKind::terminate()).context("claim run: SIGTERM handler")?,
            signal(SignalKind::interrupt()).context("claim run: SIGINT handler")?,
            signal(SignalKind::hangup()).context("claim run: SIGHUP handler")?,
        )
    };
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        runtime.block_on(async move {
            loop {
                let signal = tokio::select! {
                    Some(()) = terminate.recv() => libc::SIGTERM,
                    Some(()) = interrupt.recv() => libc::SIGINT,
                    Some(()) = hangup.recv() => libc::SIGHUP,
                    else => break,
                };
                if sender.send(signal).is_err() {
                    break;
                }
            }
        });
    });
    Ok(receiver)
}

#[allow(
    unsafe_code,
    reason = "libc::kill has no safe std equivalent for a process group"
)]
fn forward(group: libc::pid_t, signal: i32) {
    // SAFETY: kill takes plain integers; a negative pid names a process group.
    if unsafe { libc::kill(-group, signal) } == 0 {
        info!(
            group,
            signal, "claim run: signal forwarded to the child's group"
        );
    } else {
        warn!(
            group,
            signal,
            error = %io::Error::last_os_error(),
            "claim run: signal not forwarded"
        );
    }
}

fn exit_code(status: ExitStatus) -> i32 {
    status
        .code()
        .or_else(|| status.signal().map(|signal| 128 + signal))
        .unwrap_or(1)
}

#[cfg(test)]
#[path = "claim_run_tests.rs"]
mod tests;
