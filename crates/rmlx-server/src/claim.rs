// unsafe_code: POSIX libc FFI — libc::flock advisory lock for single-MLX-process enforcement
#![allow(unsafe_code)]

//! Metal claim — one MLX process per Mac.
//!
//! The Apple Silicon Metal context is exclusive per process. A GPU command
//! holds an exclusive `flock` on one machine-wide file, [`CLAIM_PATH`], from
//! its first GPU call to its exit. The flock is the only gate. The kernel
//! releases it when the holder exits or is killed, so the next process gets
//! the claim with no operator action. The file stays on disk: a process that
//! deleted it would let the next process lock a new file while the old holder
//! still runs.
//!
//! The file body is `<pid> <argv>` of the last holder. It only gives the
//! refusal message its holder. A body is never a reason to claim or to refuse.
//!
//! CPU-only runs (`--device cpu`) take no claim.
//!
//! `flock(2)` is advisory: it stops two rMLX processes, not Python
//! `mlx_lm.server` or ollama.

use std::fs::{File, OpenOptions, Permissions};
use std::io::{self, Read as _};
use std::os::unix::fs::{FileExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::os::unix::io::AsRawFd as _;
use std::path::{Path, PathBuf};

/// The one claim file. `/var/tmp` is machine-wide, and macOS `tmp_cleaner`
/// does not clean it (it cleans `/tmp`). `std::env::temp_dir()` is per user.
const CLAIM_PATH: &str = "/var/tmp/rmlx.claim";

/// Builds before the machine-wide claim hold `/tmp/rmlx.<port>.claim`. The
/// probe of these files is for one release: remove it in the release after
/// next.
const LEGACY_CLAIM_DIR: &str = "/tmp";
const LEGACY_CLAIM_PREFIX: &str = "rmlx.";
const LEGACY_CLAIM_SUFFIX: &str = ".claim";

/// `O_NONBLOCK`: the open of a FIFO planted at a claim path must not block.
const OPEN_FLAGS: libc::c_int = libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;

/// Error returned when `try_claim` cannot get the claim.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ClaimError {
    /// Another process holds the claim.
    #[error("{}", refusal_message(*.holder_pid, .holder_command))]
    AlreadyHeld {
        /// PID of the holder. `None` when the holder has not written its
        /// PID yet.
        holder_pid: Option<u32>,
        /// Command line of the holder, as it recorded it.
        holder_command: String,
    },

    /// OS error on the claim file, or the path is not a regular file.
    #[error("Metal claim I/O error at {}: {source}", .path.display())]
    Io {
        /// The file the claim operation used.
        path: PathBuf,
        /// Underlying I/O error from the OS.
        #[source]
        source: io::Error,
    },
}

/// Holds the Metal claim until it is dropped. The flock releases when the fd
/// closes; the file stays.
#[derive(Debug)]
pub struct MetalClaim {
    _file: File,
}

/// Get the machine-wide Metal claim, or refuse if another process holds it.
///
/// For one release this also refuses when a build before the machine-wide
/// claim holds a `/tmp/rmlx.<port>.claim`.
///
/// # Errors
/// [`ClaimError::AlreadyHeld`] names the holder. [`ClaimError::Io`] is an OS
/// error; the caller must not use the GPU then either.
pub fn try_claim() -> Result<MetalClaim, ClaimError> {
    refuse_held_legacy_claim(Path::new(LEGACY_CLAIM_DIR))?;
    claim_at(Path::new(CLAIM_PATH))
}

fn claim_at(path: &Path) -> Result<MetalClaim, ClaimError> {
    let io_error = |source| ClaimError::Io {
        path: path.to_path_buf(),
        source,
    };
    let (file, writable) = open_claim_file(path).map_err(io_error)?;
    if !flock_nb(&file, libc::LOCK_EX).map_err(io_error)? {
        let (holder_pid, holder_command) = read_holder(&file);
        return Err(ClaimError::AlreadyHeld {
            holder_pid,
            holder_command,
        });
    }
    if writable {
        write_holder(&file).map_err(io_error)?;
    }
    tracing::info!(pid = std::process::id(), path = %path.display(), writable, "Metal claim acquired");
    Ok(MetalClaim { _file: file })
}

/// Open the claim file for the lock. Returns the file and whether it is
/// writable. The open follows no symlink, and anything that is not a regular
/// file is refused. A file this call creates gets mode 0666, so every user can
/// open it later. When another user's file refuses a write open, a read-only
/// open still takes the flock.
fn open_claim_file(path: &Path) -> io::Result<(File, bool)> {
    let open = |write: bool, create: bool| {
        OpenOptions::new()
            .read(true)
            .write(write)
            .create_new(create)
            .mode(0o666)
            .custom_flags(OPEN_FLAGS)
            .open(path)
    };
    let (file, writable) = match open(true, true) {
        Ok(file) => {
            // The umask masks the create mode.
            file.set_permissions(Permissions::from_mode(0o666))?;
            (file, true)
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => match open(true, false) {
            Ok(file) => (file, true),
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => (open(false, false)?, false),
            Err(e) => return Err(e),
        },
        Err(e) => return Err(e),
    };
    if !file.metadata()?.file_type().is_file() {
        return Err(io::Error::other("the claim path is not a regular file"));
    }
    Ok((file, writable))
}

/// `flock(fd, op | LOCK_NB)`. `Ok(false)` when another open file holds a
/// conflicting lock.
fn flock_nb(file: &File, op: libc::c_int) -> io::Result<bool> {
    // SAFETY: the fd is open for the lifetime of `file`. flock takes no pointer.
    if unsafe { libc::flock(file.as_raw_fd(), op | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
        Ok(false)
    } else {
        Err(error)
    }
}

fn write_holder(file: &File) -> io::Result<()> {
    let argv: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let body = format!("{} {}", std::process::id(), argv.join(" "));
    file.set_len(0)?;
    file.write_all_at(body.as_bytes(), 0)
}

/// Parse a body `<pid> <argv>`, or the bare `<pid>` a legacy claim holds.
fn read_holder(mut file: &File) -> (Option<u32>, String) {
    let mut body = String::new();
    if file.read_to_string(&mut body).is_err() {
        return (None, String::new());
    }
    let body = body.trim();
    let (pid, command) = body.split_once(' ').unwrap_or((body, ""));
    (pid.parse().ok(), command.to_owned())
}

fn refusal_message(holder_pid: Option<u32>, holder_command: &str) -> String {
    let command = if holder_command.is_empty() {
        "command not recorded"
    } else {
        holder_command
    };
    match holder_pid {
        Some(pid) => format!(
            "the Metal claim is held by PID {pid} ({command}). Stop that process \
             (`kill {pid}`), then run this command again."
        ),
        None => format!(
            "the Metal claim is held by a process that has not recorded its PID yet \
             ({command}). Run this command again in a moment."
        ),
    }
}

/// Refuse when a legacy per-port claim in `dir` is held. The probe takes a
/// shared lock for a moment and changes no file.
fn refuse_held_legacy_claim(dir: &Path) -> Result<(), ClaimError> {
    let entries = std::fs::read_dir(dir).map_err(|source| ClaimError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(LEGACY_CLAIM_PREFIX) && name.ends_with(LEGACY_CLAIM_SUFFIX)) {
            continue;
        }
        let path = entry.path();
        let Ok(file) = OpenOptions::new()
            .read(true)
            .custom_flags(OPEN_FLAGS)
            .open(&path)
        else {
            continue;
        };
        if !file.metadata().is_ok_and(|meta| meta.file_type().is_file()) {
            continue;
        }
        let held = !flock_nb(&file, libc::LOCK_SH).map_err(|source| ClaimError::Io {
            path: path.clone(),
            source,
        })?;
        if held {
            let (holder_pid, _) = read_holder(&file);
            return Err(ClaimError::AlreadyHeld {
                holder_pid,
                holder_command: format!(
                    "an rmlx build older than the machine-wide claim, holding {}",
                    path.display()
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "claim_tests.rs"]
mod tests;
