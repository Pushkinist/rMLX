// unsafe_code: POSIX libc FFI — libc::flock advisory lock for single-MLX-process enforcement
#![allow(unsafe_code)]

//! Metal claim file — single MLX process per Mac enforcement.
//!
//! Apple Silicon Metal context is exclusive per process. rMLX holds a
//! flock'd POSIX advisory lock at `/tmp/rmlx.<port>.claim` for the lifetime
//! of any GPU-using subcommand. A second rMLX process attempting to claim
//! the same port receives `ClaimError::AlreadyHeld` with the holder's PID.
//!
//! CPU-only runs (`--device cpu`) must skip `try_claim` entirely — this module
//! is no-op for them.
//!
//! ## Sentinel port
//! Non-server CLI ops (info, baseline, chat) use the fixed sentinel port
//! `0xCAFE = 51966`. This represents "a single-shot GPU op in progress".
//!
//! ## Safety note
//! `flock(2)` is an advisory lock: it prevents two rMLX processes from
//! clobbering each other, but does not block Python mlx_lm.server or ollama.
//! Those are handled by the unload/stop hints printed on `ClaimError`.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

/// Sentinel port used by non-server GPU CLI ops (info, baseline, chat).
pub const SENTINEL_PORT: u16 = 0xCAFE; // 51966

/// Error returned when `try_claim` cannot acquire the lock.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ClaimError {
    /// Another rMLX process holds the lock on this port.
    #[error("another rMLX process (PID {holder_pid}) holds the Metal claim for port {port}")]
    AlreadyHeld {
        /// Port number the claim file is keyed to.
        port: u16,
        /// PID of the process that currently holds the lock.
        holder_pid: u32,
    },

    /// OS error while creating or locking the file (e.g. permission denied).
    #[error("Metal claim file I/O error for port {port}: {source}")]
    Io {
        /// Port number the claim file is keyed to.
        port: u16,
        /// Underlying I/O error from the OS.
        #[source]
        source: std::io::Error,
    },
}

/// RAII guard that holds the Metal claim for the lifetime of the GPU-using
/// subcommand. Dropping the guard removes the lock file.
#[derive(Debug)]
pub struct MetalClaim {
    path: PathBuf,
    // Keep the File open so the flock is held.
    _file: File,
}

impl Drop for MetalClaim {
    fn drop(&mut self) {
        // Best-effort: ignore errors — the file will be stale but harmless
        // (flock is released when the fd closes, and `/tmp` is ephemeral).
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Try to acquire the Metal claim for `port`.
///
/// - Creates `/tmp/rmlx.<port>.claim` with `O_CREAT | O_EXCL`.
/// - Acquires an exclusive non-blocking `flock` on the file.
/// - Writes the current PID as a decimal string into the file body.
///
/// Returns `Ok(MetalClaim)` on success. The lock is held until the returned
/// guard is dropped (i.e. until the subcommand exits).
///
/// Returns `ClaimError::AlreadyHeld` if the file already exists *and* is
/// flock-locked by another process. The holder's PID is read from the file.
pub fn try_claim(port: u16) -> Result<MetalClaim, ClaimError> {
    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));

    // --- attempt O_CREAT|O_EXCL -------------------------------------------
    // We try exclusive creation first. If that fails because the file already
    // exists, we fall through to the flock path which will tell us whether the
    // owner is still alive.

    let file = match OpenOptions::new()
        .write(true)
        .create_new(true) // O_CREAT | O_EXCL
        .open(&path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // File exists — try to open it and read the PID.
            let holder_pid = read_holder_pid(&path, port)?;
            // Try to take the flock now; if it succeeds the old owner exited
            // without cleanup — we can steal the file.
            let candidate = OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|src| ClaimError::Io { port, source: src })?;
            if flock_ex_nb(candidate.as_raw_fd()) {
                // Stale file, old process gone — we got the lock.
                candidate
            } else {
                return Err(ClaimError::AlreadyHeld { port, holder_pid });
            }
        }
        Err(e) => return Err(ClaimError::Io { port, source: e }),
    };

    // --- acquire exclusive non-blocking flock --------------------------------
    if !flock_ex_nb(file.as_raw_fd()) {
        // We just created the file but couldn't lock it — race with another
        // starter. Read PID and report.
        let holder_pid = read_holder_pid(&path, port).unwrap_or(0);
        // Remove the file we just created so we don't leave a stale entry.
        let _ = std::fs::remove_file(&path);
        return Err(ClaimError::AlreadyHeld { port, holder_pid });
    }

    // --- write our PID -------------------------------------------------------
    let pid = std::process::id();
    let mut f = file;
    write!(f, "{pid}").map_err(|src| ClaimError::Io { port, source: src })?;
    f.flush()
        .map_err(|src| ClaimError::Io { port, source: src })?;

    tracing::info!(port, pid, path = %path.display(), "Metal claim acquired");

    Ok(MetalClaim { path, _file: f })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Call `flock(fd, LOCK_EX | LOCK_NB)`. Returns `true` on success (lock
/// acquired), `false` if the lock is held by another process (EWOULDBLOCK).
///
/// # Safety
/// `fd` is a valid open file descriptor.
fn flock_ex_nb(fd: std::os::unix::io::RawFd) -> bool {
    // SAFETY: fd is valid. flock is async-signal-safe and thread-safe.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    ret == 0
}

/// Read the PID written in the claim file body. Returns 0 on any parse error
/// (non-fatal — caller uses it only for the error message).
fn read_holder_pid(path: &PathBuf, port: u16) -> Result<u32, ClaimError> {
    let contents =
        std::fs::read_to_string(path).map_err(|src| ClaimError::Io { port, source: src })?;
    Ok(contents.trim().parse::<u32>().unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "claim_tests.rs"]
mod tests;
