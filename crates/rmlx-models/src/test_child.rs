//! Run one `#[ignore]`d test of this test binary in a child process, for a
//! check that sets process-global state that cannot be reset, such as the GPU
//! latch of `rmlx_mlx::forbid_gpu`.
#![allow(
    clippy::unwrap_used,
    reason = "test harness: a child that cannot start or report is a test failure"
)]

use std::process::{Command, Stdio};

const CHILD_MARKER: &str = "started-by-a-parent-test";

/// The last line a child test prints. The parent requires it, so a path that
/// selects no test cannot pass.
pub(crate) const CHILD_DONE: &str = "child test done";

/// True in a child that [`run_child`] started. A child test returns at once
/// when this is false, so an ordinary `--ignored` run does nothing in it.
pub(crate) fn started_by_parent() -> bool {
    std::env::args().any(|arg| arg == CHILD_MARKER)
}

/// Run the `#[ignore]`d test at `path` (its module path in this crate) in a
/// child of this test binary. Assert that it passed and printed
/// [`CHILD_DONE`].
pub(crate) fn run_child(path: &str) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            path,
            CHILD_MARKER,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains(CHILD_DONE),
        "child did not run to its end:\n{stdout}"
    );
}
