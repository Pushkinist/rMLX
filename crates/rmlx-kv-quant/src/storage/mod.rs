// Promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill — which stay in `rmlx-models`) can still
// reach them across the crate boundary. Doc/visibility warnings on the
// promoted surface are silenced; the API is otherwise unchanged.
#![allow(missing_docs, missing_debug_implementations, unreachable_pub)]
//! Quantized KV buffer types: `QuantK`, `QuantV`, `QuantPlanarV`, and `KvStorage`.
//!
//! Each struct wraps the on-GPU (or paged-GPU) storage for one axis of the
//! KV cache under a specific quantization scheme. `KvStorage` is the
//! top-level enum that [`KvCache`][super::kvcache::KvCache] holds.
//!
//! # Storage types
//!
//! - [`QuantK`] — quantized K buffers (q8_0 or rot-K affine-8-bit).
//! - [`QuantV`] — quantized V buffers (TurboQuant V4 or q8_0).
//! - [`QuantPlanarV`] — quantized V buffers (PlanarQuant V4).
//! - [`KvStorage`] — enum that selects the active K/V codec and holds the
//!   matching buffer pair.
//!
//! # Paged growth
//!
//! GPU buffers are allocated in multiples of `KV_PAGE_SIZE` (256) tokens.
//! On every `append`, if the filled sequence would exceed the current
//! allocation, the buffer grows by one page (reallocate + prefix copy).
//!
//! # See also
//!
//! - [`super::q8`] — CPU-side q8_0 encode/decode.
//! - `docs/KV_CACHE.md` — subsystem spec and codec matrix.
// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for quantized scale arrays
#![allow(unsafe_code)]

mod kv_storage;
mod quant_iso_k;
mod quant_iso_k4;
mod quant_iso_v;
mod quant_iso_v4;
mod quant_k;
mod quant_k_gpu_ring;
mod quant_k_turbo3;
mod quant_k_turbo4;
mod quant_planar_k;
mod quant_planar_v;
mod quant_rotor_k3;
mod quant_rotor_k4;
mod quant_rotor_v3;
mod quant_rotor_v4;
mod quant_v;
mod seq_layout;

pub use kv_storage::{
    KvStorage, ISOV3_LAYOUT_TAG, ISOV4_LAYOUT_TAG, ISO_K_ONLY_3_LAYOUT_TAG,
    ISO_K_ONLY_4_LAYOUT_TAG, ISO_SYM_3_LAYOUT_TAG, ISO_SYM_4_LAYOUT_TAG, K8VTURBO2_TCQ_LAYOUT_TAG,
    K8VTURBO3_TCQ_LAYOUT_TAG, PLANARK4_LAYOUT_TAG, ROTORV3_LAYOUT_TAG, ROTORV4_LAYOUT_TAG,
    ROTOR_K_ASYM_3_LAYOUT_PREFIX, ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX, ROTOR_K_ASYM_4_LAYOUT_PREFIX,
    ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX, ROTOR_K_ONLY_3_LAYOUT_TAG, ROTOR_K_ONLY_3_QJL_LAYOUT_TAG,
    ROTOR_K_ONLY_4_LAYOUT_TAG, ROTOR_K_ONLY_4_QJL_LAYOUT_TAG, ROTOR_SYM_3_LAYOUT_TAG,
    ROTOR_SYM_3_QJL_LAYOUT_TAG, ROTOR_SYM_4_LAYOUT_TAG, ROTOR_SYM_4_QJL_LAYOUT_TAG,
    TURBOSYM3_LAYOUT_TAG, TURBOSYM4_LAYOUT_TAG,
};
pub use quant_iso_k::{
    iso_n_groups_for, QuantIsoK3, ISO_K3_BITS, ISO_K3_GROUP_SIZE, ISO_QUAT_BLOCK_SIZE,
};
pub use quant_iso_k4::{QuantIsoK4, ISO_K4_BITS, ISO_K4_GROUP_SIZE};
pub use quant_iso_v::{IsoBlocks, QuantIsoV3, ISO3_BITS, ISO3_GROUP_SIZE};
pub use quant_iso_v4::{QuantIsoV4, ISO4_BITS, ISO4_GROUP_SIZE};
pub use quant_k::QuantK;
pub use quant_k_gpu_ring::QuantKGpuRing;
pub use quant_k_turbo3::{QuantKTurbo3, TURBO3_K_BITS};
pub use quant_k_turbo4::QuantKTurbo4;
pub use quant_planar_k::QuantPlanarK;
pub use quant_planar_v::QuantPlanarV;
pub(crate) use quant_rotor_k3::synced_rotor_k_blocks;
pub use quant_rotor_k3::{QuantRotorK3, RotorKBlocks, ROTOR3_K_BITS, ROTOR3_K_GROUP_SIZE};
pub use quant_rotor_k4::{QuantRotorK4, ROTOR4_K_BITS, ROTOR4_K_GROUP_SIZE};
pub use quant_rotor_v3::{QuantRotorV3, RotorBlocks, ROTOR3_V_BITS, ROTOR3_V_GROUP_SIZE};
pub use quant_rotor_v4::{QuantRotorV4, ROTOR4_V_BITS, ROTOR4_V_GROUP_SIZE};
pub use quant_v::QuantV;

#[cfg(test)]
#[path = "quant_rotor_k_qjl_tests.rs"]
mod quant_rotor_k_qjl_tests;

// Long-prompt PlanarK chunked-prefill regression tests.
#[cfg(test)]
#[path = "quant_planar_k_tests.rs"]
mod quant_planar_k_tests;

// ── Paged KV growth ───────────────────────────────────────────────────────────
//
// GPU quantized buffers are allocated in multiples of PAGE_SIZE tokens instead
// of sizing to `max_seq` immediately. On every `append`, if the filled
// sequence would exceed the current allocation, we grow by another page block
// (reallocate + copy prefix). This reduces peak-prefill memory at long ctx:
// a 64K max_seq sequence that only uses 8K tokens carries only ~12.5% of the
// original buffer cost.
//
// Growth algorithm: next_capacity = ceil((prev_seq + new_seq) / PAGE_SIZE) × PAGE_SIZE,
// capped at max_seq. At 64K / 256 that is at most 256 reallocations total per
// layer per request — acceptable versus ~40% peak-memory reduction.
pub const KV_PAGE_SIZE: i32 = 256;
