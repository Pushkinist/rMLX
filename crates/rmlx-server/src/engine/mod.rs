//! Token-generation trait + stage-1 placeholder implementation.
//!
//! `Generator` is the single seam between the HTTP routes and the actual
//! inference engine. `NotReadyGenerator` is the placeholder (returns 503).
//! `ArchGenerator` is the real engine backed by rmlx-models::arch.
//!
//! Submodules:
//! - `types`         — Phase, NormalizedTool, SamplingParams, GenerationRequest,
//!   GpuAdmission, Admission, admit_request, GenerationToken
//! - `think`         — ThinkSplitter `<think>...</think>` state machine
//! - `generator`     — Generator trait + NotReadyGenerator placeholder
//! - `image`         — VisionBundle, build_image_prompt, run_qwen3vl_image
//! - `arch_generator` — ArchGenerator (real arch-dispatch engine)
//! - `speculative`   — SpeculativeGenerator (greedy speculative decoding)
//! - `helpers`       — shared private helpers (ITL/TTFT writers, kv_quant_label, etc.)

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_fields_in_debug,
    clippy::rc_buffer,
    clippy::too_many_lines,
    clippy::used_underscore_binding,
    trivial_casts
)]

pub(crate) mod arch_generator;
pub(crate) mod audio;
pub(crate) mod generator;
pub(crate) mod helpers;
pub(crate) mod image;
pub(crate) mod speculative;
pub(crate) mod think;
pub(crate) mod types;

#[cfg(test)]
mod tests;

// ── Public API surface (frozen — matches lib.rs pub use) ─────────────────────

pub use arch_generator::ArchGenerator;
pub use generator::{Generator, NotReadyGenerator};
pub use speculative::SpeculativeGenerator;
pub use types::{
    admit_request, normalized_to_jinja_tool, Admission, GenerationRequest, GenerationToken,
    GpuAdmission, ModelLoadConfig, NormalizedResponseFormat, NormalizedTool, NormalizedToolChoice,
    Phase, SamplingParams,
};

// ── Crate-internal re-exports ─────────────────────────────────────────────────

pub(crate) use helpers::{parse_request_kv_quant, record_ttft_and_prefill};

// ── Test-visible re-exports ───────────────────────────────────────────────────

#[cfg(test)]
pub(crate) use helpers::{
    compute_itl_stats, record_itl_percentiles, tests_support_is_reconstructible_tool_marker,
};
#[cfg(test)]
pub(crate) use parking_lot::Mutex;
#[cfg(test)]
pub(crate) use rmlx_core::Error;
#[cfg(test)]
pub(crate) use rmlx_metrics::events::EventRecorder;
#[cfg(test)]
pub(crate) use std::sync::Arc;
#[cfg(test)]
pub(crate) use std::time::Instant;
#[cfg(test)]
pub(crate) use think::ThinkSplitter;
