// unsafe_code: POSIX libc FFI — fcntl (clear FD_CLOEXEC on the claim fd) and
// kill (forward a signal to the child's process group)
#![allow(unsafe_code)]
//! `rmlx claim run -- <command>` — run a command while this process holds the
//! Metal claim.
//!
//! The child inherits the claim fd, so the flock stays held while the child
//! runs even if this process dies first. SIGTERM and SIGINT sent to this
//! process are forwarded to the child's process group. The exit status is the
//! child's; a child killed by a signal exits `128 + <signal>`, as a shell
//! reports it.
//!
//! The command must not start `rmlx`: this process holds the claim, so a nested
//! `rmlx` GPU command is refused it.

use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd};
use std::os::unix::process::{CommandExt as _, ExitStatusExt as _};
use std::process::{Command, ExitStatus};
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
    let signals = forward_signals()?;
    run_holding(claim.as_fd(), command, &signals)
}

/// Run `command` in its own process group with `lock` inherited, forwarding
/// every signal number received on `signals` to that group.
pub(crate) fn run_holding(
    lock: BorrowedFd<'_>,
    command: &[OsString],
    signals: &Receiver<i32>,
) -> anyhow::Result<i32> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("claim run: no command given"))?;
    inherit_across_exec(lock).context("claim run: keep the claim fd open across exec")?;
    let mut child = Command::new(program)
        .args(args)
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

/// Deliver SIGTERM and SIGINT to the returned channel instead of acting on
/// them. The handlers are installed before this returns, so no signal that
/// arrives after the child starts kills this process.
fn forward_signals() -> anyhow::Result<Receiver<i32>> {
    use tokio::signal::unix::{signal, SignalKind};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("claim run: signal runtime")?;
    let (mut terminate, mut interrupt) = {
        let _entered = runtime.enter();
        (
            signal(SignalKind::terminate()).context("claim run: SIGTERM handler")?,
            signal(SignalKind::interrupt()).context("claim run: SIGINT handler")?,
        )
    };
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        runtime.block_on(async move {
            loop {
                let signal = tokio::select! {
                    Some(()) = terminate.recv() => libc::SIGTERM,
                    Some(()) = interrupt.recv() => libc::SIGINT,
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

fn inherit_across_exec(fd: BorrowedFd<'_>) -> io::Result<()> {
    let raw = fd.as_raw_fd();
    // SAFETY: F_GETFD / F_SETFD on an open, borrowed fd read and write only
    // its descriptor flags.
    let flags = unsafe { libc::fcntl(raw, libc::F_GETFD) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: as above.
    if unsafe { libc::fcntl(raw, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

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
