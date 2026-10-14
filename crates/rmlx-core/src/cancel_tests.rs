use std::sync::atomic::{AtomicBool, Ordering};

use super::ensure_not_cancelled;
use crate::error::Error;

#[test]
fn a_clear_flag_reads_as_no_stop() {
    assert!(ensure_not_cancelled(&AtomicBool::new(false)).is_ok());
}

#[test]
fn a_set_flag_reads_as_the_stop_outcome() {
    let stop = ensure_not_cancelled(&AtomicBool::new(true));
    assert!(matches!(stop, Err(Error::Cancelled)), "{stop:?}");
}

/// The read does not clear the flag: each loop of a stopped request reads the
/// same stop.
#[test]
fn a_read_leaves_the_flag_set() {
    let flag = AtomicBool::new(true);
    assert!(ensure_not_cancelled(&flag).is_err());
    assert!(flag.load(Ordering::Relaxed));
    assert!(ensure_not_cancelled(&flag).is_err());
}
