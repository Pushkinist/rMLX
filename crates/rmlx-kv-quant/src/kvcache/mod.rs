//! Per-layer KV cache — append, SDPA dispatch, and prompt-cache spill.
//!
//! Owns [`KvCache`], the central data structure for autoregressive decode.
//! Each layer holds one `KvCache` instance; the model's generate loop calls
//! [`KvCache::update_and_sdpa`] (or the mixed / flash variants) on every
//! decode step and [`KvCache::update`] during chunked prefill.
//!
//! # Modes
//!
//! - **BF16 (no quantization)** — K and V are stored in `[B, H, S, D]` bf16
//!   tensors and passed directly to fused SDPA.
//! - **Quantized K/V** — K and/or V are quantized on-GPU via the MSL kernel
//!   families in `q8_msl`, `turboquant_msl`, `planarquant_msl`, etc., then
//!   dequantized at SDPA time. See [`super::storage`].
//! - **Rotating (SWA)** — sliding-window layers use the ring-buffer code path
//!   ported from mlx-lm's `RotatingKVCache`. SWA layers stay bf16 even when
//!   full-attention layers are quantized.
//! - **K8V4 flash** — the TurboFlash split-K path for `KvQuant::K8V4` when
//!   head-major persistent storage is enabled.
//!
//! # Public API
//!
//! - [`KvCache`] — per-layer cache struct.
//! - Constructors: [`KvCache::with_quant`], [`KvCache::with_quant_max_seq`],
//!   [`KvCache::with_quant_max_seq_window`].
//! - Hot-path: [`KvCache::update_and_sdpa`], [`KvCache::update_and_sdpa_mixed`],
//!   [`KvCache::update_and_sdpa_k8v4_flash`], [`KvCache::update`].
//! - Lifecycle: [`KvCache::enter_prefill`], [`KvCache::exit_prefill`],
//!   [`KvCache::reset`], [`KvCache::truncate_to`].
//! - Diagnostics: [`KvCache::approx_bytes`], [`KvCache::resident_bytes`],
//!   [`KvCache::offset`], [`KvCache::seq_len`].
//!
//! # See also
//!
//! - [`super::storage`] — quantized buffer types (`QuantK`, `QuantV`, `QuantPlanarV`, `KvStorage`).
//! - [`super::mixed_quant`] — mixed K8/V4 SDPA dispatch.
//! - `docs/KV_CACHE.md` — subsystem spec.
// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::too_many_lines
)]

mod core;
mod fused_qk_dispatch;
mod fused_qk_shadow;
mod helpers;
mod sdpa;
mod update;

// Warm-TTFT shortcut regression for PlanarK.
#[cfg(test)]
#[path = "warm_ttft_tests.rs"]
mod warm_ttft_tests;

// Warm-TTFT bf16-K cross-codec audit (shortcut vs K-only).
#[cfg(test)]
#[path = "warm_ttft_cross_codec_tests.rs"]
mod warm_ttft_cross_codec_tests;

// Cross-layer-KV producer dispatch-chain extension.
#[cfg(test)]
#[path = "returning_kv_tests.rs"]
mod returning_kv_tests;

// Fused-quant shared-KV decode share (Mixed global path).
#[cfg(test)]
#[path = "shared_kv_quant_tests.rs"]
mod shared_kv_quant_tests;

// Dynamic grow + hard cap for `update_prefill_raw`.
#[cfg(test)]
#[path = "prefill_grow_tests.rs"]
mod prefill_grow_tests;

// Thread real layer_idx into rotor3/rotor4 KV-cache construction.
#[cfg(test)]
#[path = "rotor3_layer_idx_tests.rs"]
mod rotor3_layer_idx_tests;

// resident_bytes: actual on-device byte accounting.
#[cfg(test)]
#[path = "resident_bytes_tests.rs"]
mod resident_bytes_tests;

// Windowed (SWA) ring sizing diagnostic — issue #35 falsification.
#[cfg(test)]
#[path = "windowed_ring_sizing_tests.rs"]
mod windowed_ring_sizing_tests;

pub use core::KvCache;
pub use fused_qk_dispatch::fused_qk_total_dispatch_count;
pub use sdpa::SharedKvOut;
// `FusedQkLayout` / `FusedQkShadow` are crate-internal
// only; they carry mlx-rs `Array` in their storage and would leak the
// upstream type if exposed. Downstream callers don't need them — the
// dispatch entry is `KvCache::update_and_sdpa` and the only public
// observable is `fused_qk_total_dispatch_count`. Submodules access them
// directly via `crate::kvcache::fused_qk_shadow::*`.
