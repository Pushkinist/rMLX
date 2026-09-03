//! rMLX metrics database — see docs/METRICS_DB.md for the spec.
#![forbid(unsafe_code)]
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

pub mod bests_view;
pub mod cell;
pub mod error;
pub mod events;
pub mod export;
pub mod identity;
pub mod ingest;
pub mod legacy_ingest;
pub mod migrate;
pub mod mode;
pub mod prompts;
pub mod query;
pub mod recorder;
pub mod registry;
pub mod schema;
pub mod scope;
pub mod time_util;

/// Placeholder so the crate is not considered dead code by the compiler.
pub fn placeholder() -> &'static str {
    "rmlx-metrics"
}
