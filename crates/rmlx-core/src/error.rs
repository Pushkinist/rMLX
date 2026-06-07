//! Workspace-shared error type and `Result` alias.
//!
//! Every crate in the rMLX workspace surfaces failures through
//! [`Error`] and the [`Result`] type alias re-exported here. Library crates
//! return `rmlx_core::error::Result<T>`; binary entry-points wrap into
//! `anyhow::Result` at the CLI boundary.
//!
//! # Public API
//!
//! - [`Error`] — non-exhaustive enum covering I/O, config, loader, quant,
//!   model, MLX FFI, OOM, and generic failures.
//! - [`OomPhase`] — allocation phase tag inside [`Error::Oom`], used to
//!   distinguish load-time OOM (evict + retry) from mid-generation OOM
//!   (abort stream).
//! - [`Result<T>`] — `std::result::Result<T, Error>` alias.

use thiserror::Error;

/// Workspace-shared error enum covering all rMLX failure domains.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Wraps a standard [`std::io::Error`] (file not found, permission denied, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration parse or validation failure (malformed TOML, out-of-range value).
    #[error("config: {0}")]
    Config(String),

    /// Model loader failure (safetensors read error, missing weight key, shape mismatch).
    #[error("loader: {0}")]
    Loader(String),

    /// Quantisation operation failure (unsupported bit-width, group-size mismatch).
    #[error("quant: {0}")]
    Quant(String),

    /// Model-layer or architecture error (unsupported op, unexpected tensor rank).
    #[error("model: {0}")]
    Model(String),

    /// Smoke-probe generation produced incoherent output; the model snapshot is suspect.
    #[error("smoke probe failed: {0}")]
    SmokeProbe(String),

    /// MLX FFI error returned through the opaque mlx-c string channel.
    #[error("mlx: {0}")]
    Mlx(String),

    /// Out-of-memory during a known allocation phase.
    ///
    /// J3: distinct from the generic [`Error::Mlx`] catch-all so automation can
    /// tell "evict + retry" (load phases) from "stream is dead, don't retry"
    /// (mid-generation). Only constructed at call sites where the allocation
    /// phase is unambiguous by construction — the mlx-c FFI surfaces every
    /// failure through one opaque string channel, so OOM is NOT reliably
    /// distinguishable from a shape/kernel error at the status-code level.
    /// `requested_bytes` / `peak_alloc_mb` are best-effort and may be `None`.
    #[error("oom during {phase:?}: {msg}")]
    Oom {
        /// The allocation phase in which the OOM occurred.
        phase: OomPhase,
        /// Best-effort allocation size that triggered OOM (`None` if unavailable).
        requested_bytes: Option<u64>,
        /// Best-effort peak allocator usage in MiB at the time of failure (`None` if unavailable).
        peak_alloc_mb: Option<u64>,
        /// Human-readable OOM message from the MLX FFI channel.
        msg: String,
    },

    /// Catch-all for errors that do not fit a more specific variant.
    #[error("other: {0}")]
    Other(String),

    /// Architecture not supported by the requested operation or dispatch path.
    ///
    /// Raised when an architecture string passes the `KNOWN_ARCHS` guard but
    /// has no corresponding match arm — indicating a BUG where KNOWN_ARCHS was
    /// extended without adding the implementation arm.
    #[error("arch unsupported: {arch}")]
    ArchUnsupported {
        /// The architecture name that has no implementation.
        arch: String,
    },

    /// KV storage variant does not match the expected variant for the active
    /// `KvQuant` mode. Indicates a construction-time BUG in cache allocation.
    #[error("kv storage mismatch: expected {expected}, got {got}")]
    KvStorageMismatch {
        /// The storage variant that was expected based on the active `KvQuant`.
        expected: &'static str,
        /// The storage variant that was actually present.
        got: &'static str,
    },

    /// `ssd_tier::install_config` was called more than once. The SSD tier
    /// config is process-global and must be installed exactly once at startup.
    #[error("ssd tier config already installed — refusing to re-install")]
    SsdTierAlreadyInstalled,

    /// An operation that is structurally non-implementable was called. The
    /// message explains the constraint and the correct alternative.
    #[error("unimplemented: {0}")]
    Unimplemented(&'static str),

    /// Prefill requested a sequence length above the configured KV
    /// hard cap. The cap is opt-in via `RMLX_KV_MAX_SEQ_HARD_CAP` (no cap
    /// when unset). Raised before any large allocation so the caller can
    /// reject the request cleanly instead of triggering a broadcast-shape
    /// error deep inside the prefill loop.
    #[error("kv prefill request exceeds hard cap: requested={requested}, cap={cap}")]
    KvHardCapExceeded {
        /// The needed sequence length (in tokens) that triggered the guard.
        requested: i32,
        /// The configured hard cap (in tokens), from `RMLX_KV_MAX_SEQ_HARD_CAP`.
        cap: i32,
    },

    /// Prefill requested a sequence length above the per-cache virtual
    /// ceiling. The ceiling is the resolved `--max-ctx` (a virtual cap, not an
    /// eager allocation): the KV ring grows lazily up to it. Raised before any
    /// allocation past the ceiling so an over-long prompt is rejected cleanly
    /// instead of growing the ring beyond the operator-declared context bound.
    #[error(
        "kv prefill request exceeds max-ctx ceiling: requested={requested}, ceiling={ceiling}"
    )]
    KvCeilingExceeded {
        /// The needed sequence length (in tokens) that triggered the guard.
        requested: i32,
        /// The configured virtual ceiling (in tokens), from `--max-ctx`.
        ceiling: i32,
    },

    /// A `--draft-model` + `--draft-kind` combination is structurally
    /// unsupported. Raised at load (before first inference) when the draft
    /// model's detected architecture family cannot back the requested draft
    /// kind — e.g. a plain `Gemma4ForConditionalGeneration` snapshot passed
    /// with `--draft-kind mtp`, which has no MTP sidecar head and is not the
    /// dedicated Gemma4 assistant drafter. The message names the detected
    /// family and the correct alternative so the operator can fix the flags
    /// instead of seeing an unrelated loader error leak from the wrong path.
    #[error("unsupported speculative pairing: {reason}")]
    SpeculativePairing {
        /// Human-readable explanation naming the detected draft arch and the
        /// correct alternative (e.g. the required assistant snapshot).
        reason: String,
    },
}

/// The allocation phase in which an [`Error::Oom`] was raised.
///
/// Drives the HTTP status / `Retry-After` decision server-side: load-phase OOM
/// is retryable after eviction (507 + `Retry-After`); a mid-generation OOM
/// leaves the KV cache corrupt past the failure point and is NOT retryable
/// (503, no `Retry-After`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed enum — exactly three OOM phases; adding a phase requires updating HTTP status/Retry-After decision logic in both server routes"
)]
pub enum OomPhase {
    /// Allocating model weight tensors during model load.
    LoadWeights,
    /// Allocating / growing the KV cache buffers.
    LoadKvCache,
    /// A per-step allocation during token generation (decode).
    Generation,
}

/// `std::result::Result` alias with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
