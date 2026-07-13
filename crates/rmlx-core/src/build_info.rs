//! Test coverage for the pure logic in `build.rs`.
//!
//! `build.rs` is a standalone program Cargo compiles and runs itself; `cargo
//! test` never invokes it as a test target, so it has no way to be covered
//! directly. The one function in it whose output actually needs verifying
//! (`profile_from_out_dir` — see the reasoning in `build.rs`'s doc comment) is
//! shared here via `include!`, so the tests below run against the *exact same
//! source text* the build script uses, not a copy that can silently drift.
//!
//! Test-only: this module only exists under `cfg(test)` (see `lib.rs`).

use std::path::Path;

include!("../build_support.rs");

#[cfg(test)]
#[path = "build_info_tests.rs"]
mod tests;
