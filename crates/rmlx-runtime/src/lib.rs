//! rMLX runtime — shared decode/forward kernels.
//!
//! This crate extracts machinery duplicated across the per-architecture graphs
//! in `rmlx-models` (qwen2, qwen3, qwen3_5_moe, gemma3, gemma4, laguna). The
//! goal is **zero behavior change**: every helper here is bit-for-bit identical
//! to the per-arch copy it replaces.
//!
//! # Why a separate crate?
//!
//! Per-class optimization wants to target one shared decode
//! kernel rather than six near-duplicates. By centralising the scaffolding
//! here, future passes can speed up all six archs at once.
//!
//! # What lives here vs in `rmlx-models::layers`
//!
//! | Concern | Where |
//! |-------------------------------|-------------------------------------|
//! | `Linear`, `Embedding`, `Mlp` | `rmlx-models::layers` (existing) |
//! | `RmsNorm` (plain gamma) | `rmlx-models::layers` (existing) |
//! | Mask builders (causal/SWA) | `rmlx-models::layers` (existing) |
//! | `RmsNormShifted` (gamma + 1) | `rmlx-runtime::rmsnorm` (new) |
//! | `count_nan_in_bytes`/`max_abs_from_bytes` | `rmlx-runtime::probe` (new) |
//! | Chunked prefill outer loop | `rmlx-runtime::decode_loop` (new) |
//! | Decode profile timers | `rmlx-runtime::decode_loop` (new) |
//! | `repeat_kv` (GQA expansion) | `rmlx-runtime::attention` (new) |
//! | SDPA mask-mode dispatch | `rmlx-runtime::attention` (new) |
//!
//! Everything in `rmlx-models::layers` stays where it is. The runtime crate
//! depends on `rmlx-models` (for `KvCache`, `Embedding`, `Linear`, mask
//! builders) and re-exports the most commonly used types for callers.
//!
//! # Migration template (single-arch decode loop)
//!
//! Per-arch `generate.rs` modules can replace their hand-written prefill +
//! decode scaffold with `decode_loop::generate_greedy`, leaving only the
//! arch-specific `forward_seq_with_cache` adapter behind.

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

pub mod attention;
pub mod decode_loop;
pub mod probe;
pub mod rmsnorm;

pub use attention::repeat_kv;
pub use decode_loop::{DecodeProfile, ProbeStep, SmokeVerdict};
pub use probe::{count_nan_in_bytes, max_abs_from_bytes};
pub use rmsnorm::RmsNormShifted;
