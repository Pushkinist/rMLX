//! How a command asks for a non-zero exit code.
//!
//! A command never calls `std::process::exit`: it returns [`ExitWith`], and
//! `main` flushes the log writer before it exits with the code.

use anyhow::Result;

/// A recorded run measured, and its record did not reach the DB: the buffer
/// file is kept for `rmlx metrics record --replay-pending`. 75 is sysexits'
/// `EX_TEMPFAIL`, and no other exit path or script in the tree uses it, so a
/// caller can tell this from a crash (1) or a refusal.
pub(crate) const RECORD_PENDING: i32 = 75;

/// Exit with this code. The command has already reported why.
#[derive(Debug)]
pub(crate) struct ExitWith(pub(crate) i32);

impl std::fmt::Display for ExitWith {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "exit code {}", self.0)
    }
}

impl std::error::Error for ExitWith {}

/// The exit code for a command's outcome: its own code, or the code an
/// [`ExitWith`] carries. Any other error is returned.
pub(crate) fn exit_code(outcome: Result<i32>) -> Result<i32> {
    match outcome {
        Ok(code) => Ok(code),
        Err(e) => e.downcast::<ExitWith>().map(|exit| exit.0),
    }
}
