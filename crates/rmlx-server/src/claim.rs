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

use std::fs::{File, OpenOptions, Permissions, TryLockError};
use std::io::{self, Read as _};
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::fs::{FileExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
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
    #[error("{}", refusal_message(*.holder_pid, .holder_command, .path))]
    AlreadyHeld {
        /// PID the holder wrote into the file body. It can be stale or
        /// forged. It is `None` when the holder wrote no body, and while a
        /// holder is starting the body can be empty or still the previous
        /// holder's: the holder truncates and rewrites it after it locks.
        holder_pid: Option<u32>,
        /// Command line of the holder, as it recorded it.
        holder_command: String,
        /// The lock file the holder has locked.
        path: PathBuf,
    },

    /// OS error on the claim file, or the path is not a regular file with
    /// one link.
    #[error("Metal claim I/O error at {}: {source}", .path.display())]
    Io {
        /// The file the claim operation used.
        path: PathBuf,
        /// Underlying I/O error from the OS.
        #[source]
        source: io::Error,
    },
}

/// Holds the Metal claim until it is dropped. The flock releases when the
/// last fd on its open file closes; the file stays. A child that inherits the
/// fd (see [`AsFd`]) holds the flock too.
#[derive(Debug)]
pub struct MetalClaim {
    file: File,
}

impl AsFd for MetalClaim {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
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
    claim_in(Path::new(LEGACY_CLAIM_DIR), Path::new(CLAIM_PATH))
}

/// Report whether a process holds the Metal claim, without taking it.
/// `Ok(())` means no process holds it. The probe writes no body and creates
/// no file. It holds a shared lock for a moment, and a GPU command that
/// starts in that moment is refused.
///
/// # Errors
/// [`ClaimError::AlreadyHeld`] names the holder. [`ClaimError::Io`] is an OS
/// error.
pub fn probe_claim() -> Result<(), ClaimError> {
    probe_in(Path::new(LEGACY_CLAIM_DIR), Path::new(CLAIM_PATH))
}

fn claim_in(legacy_dir: &Path, path: &Path) -> Result<MetalClaim, ClaimError> {
    refuse_held_legacy_claim(legacy_dir)?;
    claim_at(path)
}

fn probe_in(legacy_dir: &Path, path: &Path) -> Result<(), ClaimError> {
    refuse_held_legacy_claim(legacy_dir)?;
    let file = match open_checked(OpenOptions::new().read(true), path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(io_error(path, source)),
    };
    refuse_if_held(&file, path, None)
}

fn claim_at(path: &Path) -> Result<MetalClaim, ClaimError> {
    let (file, writable) = open_claim_file(path).map_err(|e| io_error(path, e))?;
    if !locked(file.try_lock()).map_err(|e| io_error(path, e))? {
        let (holder_pid, holder_command) = read_holder(&file);
        return Err(ClaimError::AlreadyHeld {
            holder_pid,
            holder_command,
            path: path.to_path_buf(),
        });
    }
    if writable {
        write_holder(&file).map_err(|e| io_error(path, e))?;
    }
    tracing::info!(pid = std::process::id(), path = %path.display(), writable, "Metal claim acquired");
    Ok(MetalClaim { file })
}

fn io_error(path: &Path, source: io::Error) -> ClaimError {
    ClaimError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Open the claim file for the lock. Returns the file and whether it is
/// writable. A file this call creates gets mode 0666, so every user can open
/// it later. When another user's file refuses a write open, a read-only open
/// still takes the flock.
fn open_claim_file(path: &Path) -> io::Result<(File, bool)> {
    let open = |write: bool, create: bool| {
        open_checked(
            OpenOptions::new()
                .read(true)
                .write(write)
                .create_new(create)
                .mode(0o666),
            path,
        )
    };
    match open(true, true) {
        Ok(file) => {
            // The umask masks the create mode.
            file.set_permissions(Permissions::from_mode(0o666))?;
            Ok((file, true))
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => match open(true, false) {
            Ok(file) => Ok((file, true)),
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
                Ok((open(false, false)?, false))
            }
            Err(e) => Err(e),
        },
        Err(e) => Err(e),
    }
}

/// Open `path` following no symlink, and refuse anything but a regular file
/// with one link: a hard link planted at the path would make the body write
/// land in its target.
fn open_checked(options: &mut OpenOptions, path: &Path) -> io::Result<File> {
    let file = open_no_follow(options, path)?;
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        return Err(io::Error::other("the claim path is not a regular file"));
    }
    if meta.nlink() != 1 {
        return Err(io::Error::other("the claim file has more than one link"));
    }
    Ok(file)
}

fn open_no_follow(options: &mut OpenOptions, path: &Path) -> io::Result<File> {
    options.custom_flags(OPEN_FLAGS).open(path)
}

/// `Ok(false)` when another open file holds a conflicting lock.
fn locked(result: Result<(), TryLockError>) -> io::Result<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// Refuse when another open file holds an exclusive lock on `file`. The
/// shared lock taken here is released when `file` closes.
fn refuse_if_held(file: &File, path: &Path, command: Option<&str>) -> Result<(), ClaimError> {
    if locked(file.try_lock_shared()).map_err(|e| io_error(path, e))? {
        return Ok(());
    }
    let (holder_pid, recorded_command) = read_holder(file);
    Err(ClaimError::AlreadyHeld {
        holder_pid,
        holder_command: command.map_or(recorded_command, str::to_owned),
        path: path.to_path_buf(),
    })
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

fn refusal_message(holder_pid: Option<u32>, holder_command: &str, path: &Path) -> String {
    let command = if holder_command.is_empty() {
        "command not recorded"
    } else {
        holder_command
    };
    let recorded = match holder_pid {
        Some(pid) => format!("The holder recorded PID {pid} ({command}); the record can be stale."),
        None => format!("The holder recorded no PID ({command})."),
    };
    format!(
        "the Metal claim is held. {recorded} Find the process that holds the lock with \
         `lsof {}` (as root to see another user's process), stop it with `kill <PID>`, \
         then run this command again.",
        path.display()
    )
}

/// Refuse when a legacy per-port claim in `dir` is held. The probe takes a
/// shared lock for a moment and changes no file. Old builds made only regular
/// files, so it skips an entry that vanished, is a symlink or is not a regular
/// file; any other open error refuses. A hard-linked file keeps its inode's
/// flock and is probed.
fn refuse_held_legacy_claim(dir: &Path) -> Result<(), ClaimError> {
    let entries = std::fs::read_dir(dir).map_err(|e| io_error(dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_error(dir, e))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(LEGACY_CLAIM_PREFIX) && name.ends_with(LEGACY_CLAIM_SUFFIX)) {
            continue;
        }
        let path = entry.path();
        let file = match open_no_follow(OpenOptions::new().read(true), &path) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => continue,
            Err(source) => return Err(io_error(&path, source)),
        };
        if !file
            .metadata()
            .map_err(|e| io_error(&path, e))?
            .file_type()
            .is_file()
        {
            continue;
        }
        refuse_if_held(
            &file,
            &path,
            Some("an rmlx build older than the machine-wide claim"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "claim_tests.rs"]
mod tests;
