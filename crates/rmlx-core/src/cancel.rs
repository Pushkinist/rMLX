//! The stop of a request that is gone.
//!
//! A request has one stop flag. The flag is set when the request is dropped:
//! by a timeout, by a closed connection or by a server shutdown. Each loop
//! that can keep the request on the GPU calls [`ensure_not_cancelled`] as the
//! first statement of a pass and hands its error on with `?`.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};

/// Return [`Error::Cancelled`] when `flag` is set.
///
/// The read is one relaxed load: no allocation, no lock, no system call.
#[inline]
pub fn ensure_not_cancelled(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    Ok(())
}

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod tests;
