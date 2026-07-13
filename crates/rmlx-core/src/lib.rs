//! rMLX core: MLX reexport, KV-cache trait, error types.

#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        // disallowed_methods is a separate lint from unwrap_used;
        // test code (bucket-B) is already exempted for unwrap_used, extend here.
        clippy::disallowed_methods,
    )
)]

pub mod apple_gpu;
// Test-only: shares + tests the pure `profile_from_out_dir` logic that
// `build.rs` uses. See `src/build_info.rs` for why this exists.
#[cfg(test)]
mod build_info;
pub mod error;
pub mod kvcache;
pub mod mach_mem;
pub mod paths;
pub mod projects_config;
pub mod runinfo;
pub mod unified_memory;

pub use error::Error;
pub use error::OomPhase;
pub use error::Result;
