//! `CacheType` enum, `CacheTypeSpec`, and the string parser.
//!
//! This module implements the naming namespace from §D1 of the
//! `--cache-type-k` / `--cache-type-v` implementation plan
//! (`docs/superpowers/plans/cache-type-flags.md`).
//!
//! Implemented: enum + parser + inline tests.
//! Not yet: Resolver, ResolverContext, ResolveError.
//!
//! ## Codec-parameter audit (kept as ground truth)
//!
//! The claims below were verified by reading the source symbols listed in the
//! "Symbol inspected" columns. No assumptions were carried forward from the
//! plan document without cross-checking the code.
//!
//! ---
//!
//! # Audit methodology
//!
//! Every claim below was verified by reading the source symbols listed in the
//! "Symbol inspected" columns. No assumptions were carried forward from the
//! plan document without cross-checking the code.
//!
//! ---
//!
//! # `KvQuant::None`
//!
//! **Codec**: unquantized; both K and V stored as `bf16` (the dtype of the
//! incoming attention tensors).
//!
//! **Implementation path**: `KvCache::update_none` →
//! `KvCache::update_decode_fp16`. Storage is a pre-allocated
//! `[B, kv_h, max_seq, head_dim]` bf16 buffer per side, filled via
//! `slice_update` at the current offset each step.
//!
//! **Symbols inspected**:
//! - `KvCache::update_none` — dispatches to `update_decode_fp16`.
//! - `KvCache::update_decode_fp16` — bf16 `slice_update` path; no quantize call.
//! - `KvCache::exit_prefill` — `KvQuant::None` arm; promotes `raw_k/v` directly
//!   to `decode_fp16_k/v` without any `quantize()` call.
//! - `KvStorage::None` — holds only `max_seq: i32`; no quantized buffers.
//!
//! ---
//!
//! # `KvQuant::K8V8`
//!
//! **K codec**: rMLX MSL 8-bit symmetric affine (`q8_0`), `group_size=128`.
//! **V codec**: same codec as K — rMLX MSL 8-bit symmetric affine (`q8_0`),
//! `group_size=128`.
//!
//! Both sides use the same `QuantK` storage struct and the same `QuantK::append`
//! / `QuantK::dequantize_choice` methods.
//!
//! **Group size source**: `crates/rmlx-models/src/kv_cache/q8.rs`,
//! line `pub(super) const Q8_GROUP_SIZE: usize = 128;`.
//!
//! **Symbols inspected**:
//! - `KvCache::update_k8v8` — constructs both `k` and `v` as `QuantK {..}`;
//!   calls `qs.append(...)` for both sides. `QuantK` is the q8_0 struct.
//! - `KvCache::exit_prefill` — `KvQuant::K8V8` arm; calls `qk.append` and
//!   `qv.append` where both locals are `QuantK`.
//! - `storage::QuantK` doc comment: "Accumulated q8_0 K cache (group_size=128)".
//! - `storage::QuantK::append` GPU path: `scales_per_step = b * kv_h * d / Q8_GROUP_SIZE`.
//!
//! **Confirmed**: plan assumption correct — K and V both `q8_g128`.
//!
//! ---
//!
//! # `KvQuant::K8V4`
//!
//! **K codec**: rMLX MSL 8-bit symmetric affine (`q8_0`), `group_size=128`.
//! **V codec**: TurboQuant 4-bit Lloyd-Max N(0,1) codebook, `group_size=32`.
//!
//! The K side uses `QuantK` (same as K8V8). The V side uses `QuantV` with
//! `bits=4`. `QuantV::append` calls `turbo_quantize_v4_gpu` on GPU and
//! `turbo_quantize_v` on CPU; the quantizer is `rmlx_kv_quant::turboquant`.
//!
//! **Group size source**:
//! - K: `q8.rs` → `Q8_GROUP_SIZE = 128`.
//! - V: `crates/rmlx-kv-quant/src/turboquant.rs`,
//!   line `pub const GROUP_SIZE: usize = 32;`.
//!   Also confirmed in `k8v4_append_msl.rs` comment (line 33–35):
//!   "K: q8_0 (group_size=128 …). V: TurboQuant 4-bit Lloyd-Max N(0,1)
//!   (group_size=32 …)."
//!
//! **Symbols inspected**:
//! - `KvCache::update_k8v4` — `k` constructed as `QuantK`; `v` constructed as
//!   `QuantV { bits: 4, .. }`.
//! - `KvCache::exit_prefill` — `KvQuant::K8V4` arm; `k` is `QuantK`, `v` is
//!   `QuantV { bits: 4, .. }`.
//! - `KvCache::alloc_flash_buffers` — K scales shaped `[.., head_dim/Q8_GROUP_SIZE]`
//!   (=128); V scales shaped `[.., head_dim/TQ4_GROUP]` where `TQ4_GROUP` is
//!   re-exported as `rmlx_kv_quant::turboquant::GROUP_SIZE = 32`.
//! - `k8v4_append_msl.rs` header comments and `alloc_k8_codes_buf` /
//!   `alloc_v4_codes_buf` helpers: K group=128, V group=32.
//!
//! **Confirmed**: plan assumption correct — K is `q8_g128`, V is `tq4` (group=32).
//!
//! ---
//!
//! # `KvQuant::Planar`
//!
//! **K codec**: rMLX MSL 8-bit symmetric affine (`q8_0`), `group_size=128`.
//! **V codec**: PlanarQuant 4-bit with per-pair Hadamard rotation, `group_size=32`.
//!
//! The K side uses `QuantK` (same as K8V8 / K8V4 K-side). The V side uses
//! `QuantPlanarV`, which calls `planar_quantize_v4_gpu` (GPU) or
//! `planar_quantize` (CPU) from `rmlx_kv_quant::planarquant`.
//!
//! **Group size source**:
//! - K: `Q8_GROUP_SIZE = 128` (same as above).
//! - V: `crates/rmlx-kv-quant/src/planarquant.rs`,
//!   inline comments "Blocks are `GROUP_SIZE = 32` element groups" and
//!   "`D` must be a multiple of `GROUP_SIZE = 32`". The `GROUP_SIZE` constant
//!   itself is not `pub` in `planarquant.rs` but is documented as 32 in multiple
//!   doc comments and enforced by the `group_size != GROUP_SIZE` guard at the
//!   top of `planar_quantize`.
//!   The plan (v4 §self-review) also explicitly corrects the earlier draft that
//!   said group=2: "PlanarQuant actual group=32 (not 2)".
//!
//! **Symbols inspected**:
//! - `KvCache::update_planar` — `k` is `QuantK`; `v` is `QuantPlanarV`.
//! - `KvCache::exit_prefill` — `KvQuant::Planar` arm; `k` is `QuantK`, `v` is
//!   `QuantPlanarV`.
//! - `storage::QuantPlanarV` doc comment: "u32 codes buffer (4 words per group of
//!   32 elements)"; `append` GPU path: `codes_words_per_step = total_per_step * 4 / GROUP_SIZE`.
//! - `storage.rs` import: `use rmlx_kv_quant::planarquant::{planar_dequantize, planar_quantize, PlanarBlocks}`.
//! - `storage.rs` `planar_quantize` call site: `planar_quantize(f32_data, GROUP_SIZE, 4, new_shape)?`
//!   (the `GROUP_SIZE` here is the turboquant re-export, confirming both
//!   turbo and planar share `GROUP_SIZE=32`).
//!
//! **Note**: PlanarQuant differs from TurboQuant in that it stores per-pair
//! Hadamard rotation coefficients alongside the codes and scales (three buffers:
//! codes, scales, rotations). Both use group=32.
//!
//! **Confirmed**: plan assumption correct — K is `q8_g128`, V is `planar4`
//! (group=32).
//!
//! ---
//!
//! # `KvQuant::Mixed { k_bits, v_bits, k_group_size, v_group_size }`
//!
//! **K codec**: MLX `mx.quantize(..., mode="affine")` at `k_bits` / `k_group_size`.
//! **V codec**: MLX `mx.quantize(..., mode="affine")` at `v_bits` / `v_group_size`.
//!
//! Both sides use the same MLX affine quantizer (`mx.quantize` / `mlx_rs::quantize`
//! in the `rmlx_mlx` crate bindings), parametrized independently by the four
//! fields. The default values wired by `KvCacheBuilder::resolve_default` for
//! Qwen3/Bonsai are `k_bits=8, v_bits=4, k_group_size=64, v_group_size=64`
//! (matching `mlx-lm-turboquant`'s `MixedQuantKVCache` defaults).
//!
//! **Implementation path**:
//! - Decode: `KvCache::update_and_sdpa_mixed` → `MixedKvState::update_and_fetch`
//!   (in `mixed_quant.rs`) → two `mx.quantize` calls (one per side).
//! - Prefill: `KvCache::enter_prefill` / `KvCache::exit_prefill` accumulate raw
//!   fp16 during prefill (via `update_prefill_raw`), then `exit_prefill`'s
//!   `KvQuant::Mixed` arm calls `state.bulk_init_from_fp16` which issues a single
//!   batched `mx.quantize` per side (direct-quantize path).
//! - The SDPA step uses `mixed_quantized_sdpa` from `mixed_quant.rs`, which calls
//!   `mx.quantized_matmul` directly on the stored 3-tuples (codes, scales, biases)
//!   without a round-trip dequantize.
//!
//! **Key distinction vs K8V4/K8V8/Planar**: Mixed uses MLX's portable affine
//! quantizer (Python-visible `mx.quantize`) with arbitrary bit-width and
//! group_size. K8V4/K8V8 K-side use the rMLX-custom MSL `q8_0` kernel
//! (`Q8_GROUP_SIZE=128`). The two 8-bit K codecs are **different** despite
//! both being "8-bit affine":
//! - `q8_g128` (K8V4/K8V8 K-side): symmetric, no bias term; scale = max(|x|)/127.
//! - `mixed_k8g64` (Mixed K-side): MLX affine with separate scale + bias terms;
//!   group_size=64 by default.
//!
//! **Symbols inspected**:
//! - `KvCache::update_and_sdpa_mixed` — destructures `KvQuant::Mixed { k_bits,
//! v_bits, k_group_size, v_group_size }` and passes all four to
//!   `mixed_quantized_sdpa`.
//! - `mixed_quant.rs` module doc: "Stores K and V as the canonical 3-tuple
//!   `(codes_u32, scales, biases)` produced by `mx.quantize(..., mode="affine")`,
//!   at independent bit widths and group sizes (default K=8 / V=4 / group=64 each)."
//! - `MixedKvState::update_and_fetch` in `mixed_quant.rs`: calls `quantize(new_k,
//! k_group_size, k_bits, device)` and `quantize(new_v, v_group_size, v_bits, device)`.
//! - `KvCache::exit_prefill` — `KvQuant::Mixed` arm calls
//!   `state.bulk_init_from_fp16(&k_full, &v_full, device)`.
//!
//! **Confirmed**: plan assumption correct — both sides use MLX `mx.quantize`
//! affine with independent (bits, group_size) parameters.
//!
//! ---
//!
//! # Summary table
//!
//! | KvQuant variant | K codec | K group | V codec | V group |
//! |------------------|-----------------------|---------|----------------------------|---------|
//! | `None` | bf16 (no quant) | — | bf16 (no quant) | — |
//! | `K8V8` | rMLX MSL q8_0 affine | 128 | rMLX MSL q8_0 affine | 128 |
//! | `K8V4` | rMLX MSL q8_0 affine | 128 | TurboQuant 4-bit Lloyd-Max | 32 |
//! | `Planar` | rMLX MSL q8_0 affine | 128 | PlanarQuant 4-bit+rotation | 32 |
//! | `Mixed{..}` | MLX affine (k_bits) | k_group | MLX affine (v_bits) | v_group |
//!
//! K8V4 K-side and K8V8 K-side use the **same** rMLX MSL codec (`q8_0`,
//! `Q8_GROUP_SIZE=128`, symmetric, no bias). Planar K-side is identical.
//! Mixed's K-side is the **portable MLX affine quantizer** — a different codec
//! even at 8 bits because it includes a bias term and defaults to group=64.

#![allow(
    clippy::elidable_lifetime_names,
    clippy::match_same_arms,
    clippy::trivially_copy_pass_by_ref
)]
use rmlx_kv_quant::KvQuant;
use thiserror::Error;

// ── ParseError ────────────────────────────────────────────────────────────────

/// Error returned by [`parse`].
#[non_exhaustive]
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    /// The tag is not recognized by this parser.
    #[error("unknown cache type '{0}' — run `rmlx info --list-cache-types` for valid tags")]
    Unknown(String),

    /// The tag is a llama.cpp-legacy block-32 codec that rMLX does not implement.
    ///
    /// The message always contains `"llama.cpp legacy"` and the substitute tag.
    #[error("{0}")]
    NotImplemented(&'static str),
}

// ── CacheType ─────────────────────────────────────────────────────────────────

/// A single-side KV cache codec type, corresponding to one tag in §D1 of the
/// `--cache-type-k` / `--cache-type-v` plan.
///
/// Variants map 1-to-1 to the canonical tag strings in §D1. Aliases are
/// handled only in [`parse`] and never stored as a separate variant.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — codec registry; adding a codec requires updating parse(), tag(), bits(), group_size(), all(), and combo_to_kv_quant()"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    /// `auto` — engine picks via `resolve_default`. Valid on both K and V sides.
    Auto,
    /// `bf16` (aliases `f16`, `none`) — unquantized; stored as bf16.
    Bf16,
    /// `q8_g128` — rMLX MSL 8-bit affine, group=128 (K8V8 / K8V4 / Planar K-side codec).
    Q8G128,
    /// `q8_g64` — MLX affine 8-bit, group=64.
    Q8G64,
    /// `q8_g32` — MLX affine 8-bit, group=32.
    Q8G32,
    /// `q6_g64` — MLX affine 6-bit, group=64. V-side only (K-side on non-MoE).
    Q6G64,
    /// `q5_g64` — MLX affine 5-bit, group=64. V-side only.
    Q5G64,
    /// `q4_g128` — MLX affine 4-bit, group=128. V-side only.
    Q4G128,
    /// `q4_g64` — MLX affine 4-bit, group=64. V-side only.
    Q4G64,
    /// `q4_g32` — MLX affine 4-bit, group=32. V-side only.
    Q4G32,
    /// `q3_g64` — MLX affine 3-bit, group=64. V-side only (exploratory).
    Q3G64,
    /// `q2_g64` — MLX affine 2-bit, group=64. **V-side only**.
    ///
    /// 2-bit is the lowest rung MLX's affine quantizer supports (16 vals/u32).
    /// On the V side it gives ~8× compression vs bf16 and stays coherent on
    /// Bonsai. Pure 2-bit K is **not** a supported combo — `combo_to_kv_quant`
    /// rejects K-side 2-bit because 2-bit K degrades attention scores into
    /// incoherent output (CLAUDE.md hard rule 6). Use the asymmetric
    /// `--ctk q8_g128 --ctv q2_g64` (or `--kv-bits 2`, K stays 8-bit) instead.
    Q2G64,
    /// `tq4` (alias `turbo4`) — TurboQuant 4-bit; V-side only; requires head_dim ∈ {128, 256}.
    Tq4,
    /// `planar4` — PlanarQuant 4-bit; V-side only; requires head_dim % 32 == 0.
    Planar4,
    /// `planar_k4` — PlanarQuant 4-bit on the **K** axis.
    ///
    /// Opposite of `Planar4` (V-side). Pairs with `bf16` V (the `KvQuant::PlanarK`
    /// resolution). Arch guard (Contract A.y): rejected on Qwen MoE.
    /// Requires `head_dim % 32 == 0`.
    PlanarK4,
    /// `rot_k` — **K-side** rotation codec: K is affine-quantized at
    /// 8-bit/group=64 in a Hadamard-rotated basis; Q is pre-rotated before SDPA
    /// so the rotations cancel (the pre-rotate-Q trick — see `rot_k.rs`).
    ///
    /// This is the *only* K-side member of the rotation family. The V-side
    /// rotation codecs (`tq4`, `planar4`) stay K-guarded; `rot_k` lifts that
    /// guard for itself only. Requires a power-of-two head_dim (Hadamard).
    /// Pair with any affine V codec (default V = `q4_g64`).
    RotK,
    /// `planar3` — PlanarQuant 3-bit V; V-side only.
    ///
    /// 3.25-bit effective V codec — same Givens-rotation + per-pair scale
    /// algorithm as `planar4` but with the 8-centroid Lloyd-Max N(0,1) codebook.
    /// Pack format: 10 vals/u32 (ForgeAttention-compatible; same u32 word count
    /// per group as 4-bit). Requires `head_dim % 32 == 0`.
    /// Pairs with K-side `q8_g128` (coerced to `KvQuant::Planar3`).
    Planar3,
    /// `iso_v_3` (alias `iso3`) — IsoQuant 3-bit V; V-side only.
    ///
    /// Quaternion SO(4) rotation + 3-bit Lloyd-Max codebook. Requires
    /// `head_dim % 4 == 0` (quaternion block alignment). Pairs with K-side
    /// `q8_g128` (coerced to `KvQuant::Iso3`). Ships the CPU codec
    /// only — SDPA falls through the dequant-then-SDPA legacy path; the MSL
    /// kernel is deferred.
    Iso3,
    /// `iso_v_4` (alias `iso4`) — IsoQuant 4-bit V; V-side only.
    ///
    /// 4.25-bit V codec: same quaternion SO(4) rotation as `Iso3` with the
    /// 16-centroid Lloyd-Max codebook and dense 8-vals-per-u32 pack. Requires
    /// `head_dim % 4 == 0`. Pairs with K-side `q8_g128` (coerced to
    /// `KvQuant::Iso4`). CPU-only (no MSL kernel — the iso3 MSL
    /// kernel is hard-coded for `bits=3`).
    Iso4,
    /// `rotor_v_3` (alias `rotor3`) — rotor3 (Cl(3,0) Clifford rotor sandwich)
    /// V; V-side only.
    ///
    /// 3-bit V codec built on Cl(3,0) multivectors (8 components per group of
    /// 3 grade-1 elements). Static per-layer rotor table; per-token codes +
    /// scales + L2 norm. Pack format: 10 vals/u32 (planar3 / iso3 convention).
    /// Pairs with K-side `q8_g128` (coerced to `KvQuant::Rotor3`). CPU-only
    /// (no MSL kernel — same precedent as iso3 / iso4).
    Rotor3,
    /// `rotor_v_4` (alias `rotor4`) — rotor4 (Cl(3,0) Clifford rotor sandwich)
    /// V; V-side only.
    ///
    /// 4-bit V codec built on Cl(3,0) multivectors — same algebra as `Rotor3`
    /// but with the 16-centroid Lloyd-Max codebook and dense 8-vals-per-u32
    /// pack. ~10.7-bit effective storage per element counting per-group scale +
    /// per-token norm; rotor table amortises across tokens. `head_dim` may be
    /// any positive integer (the last group is tail-padded when
    /// `head_dim % 3 != 0`). Pairs with K-side `q8_g128` (coerced to
    /// `KvQuant::Rotor4`). CPU-only (no MSL kernel).
    Rotor4,
    /// `k8v_turbo_3_tcq` (alias `turbo3_tcq`) — TurboQuant 3-bit with Viterbi
    /// trellis (TCQ) assignment; V-side only.
    ///
    /// 3.25-bit V codec — same Lloyd-Max N(0,1) 8-centroid codebook and same
    /// on-disk pack as plain [`Tq3`](CacheType)-equivalent (re-exported via
    /// `KvQuant::K8VTurbo3`), but the **encoder** picks centroid indices by
    /// Viterbi-optimal path search through a 4-state trellis instead of
    /// nearest-centroid. The decoder is unchanged. Pairs with K-side
    /// `q8_g128` (coerced to `KvQuant::K8VTurbo3Tcq`). Ships CPU encode
    /// plus CPU dequant on the hot path; the MSL Viterbi kernel is a
    /// future-reference hook (precedent: K8VTurbo3 / K8VTurbo2 MSL hooks).
    Turbo3Tcq,
    /// `k8v_turbo_2_tcq` (alias `turbo2_tcq`) — TurboQuant 2-bit with Viterbi
    /// trellis (TCQ) assignment; V-side only.
    ///
    /// 2.25-bit V codec — same Lloyd-Max N(0,1) 4-centroid codebook and same
    /// on-disk pack as plain `turbo2` (2-bit LSB-first, 16 values per u32), but
    /// the **encoder** picks centroid indices via Viterbi-optimal path search
    /// through a 4-state trellis. The decoder is unchanged. Pairs with K-side
    /// `q8_g128` (coerced to `KvQuant::K8VTurbo2Tcq`). Ships CPU encode
    /// plus CPU dequant on the hot path; the MSL Viterbi kernel is a
    /// future-reference hook. Maps to the `max_compression` preset in mtq.
    Turbo2Tcq,
    /// `iso_k_3` (alias `k_iso3`) — IsoQuant 3-bit K; K-side
    /// codec.
    ///
    /// Quaternion SO(4) rotation + 3-bit Lloyd-Max codebook applied to the K
    /// axis. Pairs with V=`iso_v_3` to form `KvQuant::Iso3Sym` (symmetric K+V)
    /// or with V=`bf16` to form `KvQuant::IsoKOnly3` (K-only). Requires
    /// `head_dim % 4 == 0` (quaternion block alignment).
    ///
    /// **Arch guard (Contract A.y — mandatory)**: K-side ≤4-bit on Qwen MoE
    /// is the PPL-disaster (218→8641); `combo_to_kv_quant` paths that map to
    /// `Iso3Sym` / `IsoKOnly3` are rejected on Qwen MoE via the
    /// `validate_resolved` post-decompose guard. Opt-in only.
    IsoK3,
    /// `iso_k_4` (alias `k_iso4`) — IsoQuant 4-bit K; K-side
    /// codec.
    ///
    /// Same Cl(0)+quaternion SO(4) rotation as `IsoK3` with the 16-centroid
    /// 4-bit Lloyd-Max codebook. Pairs with V=`iso_v_4` to form
    /// `KvQuant::Iso4Sym` or V=`bf16` for `KvQuant::IsoKOnly4`.
    /// Arch-guarded against Qwen MoE.
    IsoK4,
    /// `rotor_k_3` (alias `k_rotor3`) — rotor3 (Cl(3,0) Clifford rotor) 3-bit K
    /// K-side codec.
    ///
    /// Static per-(layer, head) rotor table + 3-bit Lloyd-Max codebook applied
    /// to the K axis. Carries an **optional 1-bit QJL residual sideband**
    /// (default ON, toggle via `--rotor-qjl`). Pairs with V=`rotor_v_3` to
    /// form `KvQuant::Rotor3Sym` or V=`bf16` to form `KvQuant::RotorKOnly3`.
    ///
    /// **Arch guard (Contract A.y — mandatory)**: K-side ≤4-bit on Qwen MoE
    /// is the PPL-disaster (218→8641); `combo_to_kv_quant` paths that map to
    /// `Rotor3Sym` / `RotorKOnly3` are rejected on Qwen MoE via the
    /// `validate_resolved` post-decompose guard. Opt-in only.
    RotorK3,
    /// `rotor_k_4` (alias `k_rotor4`) — rotor4 (Cl(3,0) Clifford rotor) 4-bit
    /// K; K-side codec.
    ///
    /// Same Cl(3,0) rotor sandwich as `RotorK3` with the 16-centroid 4-bit
    /// Lloyd-Max codebook. Pairs with V=`rotor_v_4` to form
    /// `KvQuant::Rotor4Sym` or V=`bf16` for `KvQuant::RotorKOnly4`.
    /// Arch-guarded against Qwen MoE.
    RotorK4,
    /// `tsym3` — symmetric WHT-3 K + turbo3 V.
    ///
    /// Both K and V use the TurboQuant 3-bit Lloyd-Max N(0,1) 8-centroid
    /// codebook. K is `QuantKTurbo3` (GPU: same MSL kernel as V-side turbo3).
    /// V is `QuantV { bits: 3 }`. Maps to `KvQuant::TurboSym3` when both
    /// sides are specified as `tsym3` (the codec is inherently symmetric —
    /// there is no `tsym3` K-only or V-only decomposition). Maps to the
    /// `speed` preset in mtq.
    ///
    /// **Arch guard (Contract A.y)**: K-side 3-bit on Qwen MoE is the
    /// PPL-disaster zone; `TurboSym3` is rejected on Qwen MoE via
    /// `validate_resolved`.
    TurboSym3,
}

impl CacheType {
    /// The canonical tag string for this variant (as printed in docs and `--list-cache-types`).
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Bf16 => "bf16",
            Self::Q8G128 => "q8_g128",
            Self::Q8G64 => "q8_g64",
            Self::Q8G32 => "q8_g32",
            Self::Q6G64 => "q6_g64",
            Self::Q5G64 => "q5_g64",
            Self::Q4G128 => "q4_g128",
            Self::Q4G64 => "q4_g64",
            Self::Q4G32 => "q4_g32",
            Self::Q3G64 => "q3_g64",
            Self::Q2G64 => "q2_g64",
            Self::Tq4 => "tq4",
            Self::Planar4 => "planar4",
            Self::PlanarK4 => "planar_k4",
            Self::RotK => "rot_k",
            Self::Planar3 => "planar3",
            Self::Iso3 => "iso_v_3",
            Self::Iso4 => "iso_v_4",
            Self::Rotor3 => "rotor_v_3",
            Self::Rotor4 => "rotor_v_4",
            Self::Turbo3Tcq => "k8v_turbo_3_tcq",
            Self::Turbo2Tcq => "k8v_turbo_2_tcq",
            Self::IsoK3 => "iso_k_3",
            Self::IsoK4 => "iso_k_4",
            Self::RotorK3 => "rotor_k_3",
            Self::RotorK4 => "rotor_k_4",
            Self::TurboSym3 => "tsym3",
        }
    }

    /// Affine bit-width, or `None` for non-affine codecs (`Auto`, `Bf16`, rotation codecs).
    ///
    /// Rotation codecs (`Tq4`, `Planar4`) deliberately return `None` even though
    /// they are nominally 4-bit: their validation matrix differs from the MLX
    /// affine codecs (no MLX bit-packing rule), so they should not be passed to
    /// the affine validators. See [`ResolveError`] doc for which validators apply.
    pub fn bits(&self) -> Option<u8> {
        match self {
            Self::Auto
            | Self::Bf16
            | Self::Tq4
            | Self::Planar4
            | Self::PlanarK4
            | Self::RotK
            | Self::Planar3
            | Self::Iso3
            | Self::Iso4
            | Self::Rotor3
            | Self::Rotor4
            | Self::Turbo3Tcq
            | Self::Turbo2Tcq
            | Self::IsoK3
            | Self::IsoK4
            | Self::RotorK3
            | Self::RotorK4
            | Self::TurboSym3 => None,
            Self::Q8G128 | Self::Q8G64 | Self::Q8G32 => Some(8),
            Self::Q6G64 => Some(6),
            Self::Q5G64 => Some(5),
            Self::Q4G128 | Self::Q4G64 | Self::Q4G32 => Some(4),
            Self::Q3G64 => Some(3),
            Self::Q2G64 => Some(2),
        }
    }

    /// Affine group size, or `None` for non-affine codecs.
    ///
    /// See [`Self::bits`] for the same caveat about rotation codecs.
    pub fn group_size(&self) -> Option<usize> {
        match self {
            Self::Auto
            | Self::Bf16
            | Self::Tq4
            | Self::Planar4
            | Self::PlanarK4
            | Self::RotK
            | Self::Planar3
            | Self::Iso3
            | Self::Iso4
            | Self::Rotor3
            | Self::Rotor4
            | Self::Turbo3Tcq
            | Self::Turbo2Tcq
            | Self::IsoK3
            | Self::IsoK4
            | Self::RotorK3
            | Self::RotorK4
            | Self::TurboSym3 => None,
            Self::Q8G128 | Self::Q4G128 => Some(128),
            Self::Q8G64 | Self::Q6G64 | Self::Q5G64 | Self::Q4G64 | Self::Q3G64 | Self::Q2G64 => {
                Some(64)
            }
            Self::Q8G32 | Self::Q4G32 => Some(32),
        }
    }

    /// True for the MLX-affine family — codecs that share the
    /// `head_dim % group_size == 0` and `head_dim % (32/bits) == 0` invariants.
    fn is_affine(&self) -> bool {
        self.bits().is_some()
    }

    /// Every canonical [`CacheType`] variant, in §D1 table order.
    ///
    /// Used by `rmlx info --list-cache-types` to render the full codec table
    /// without external file dependencies.
    pub fn all() -> &'static [CacheType] {
        &[
            CacheType::Auto,
            CacheType::Bf16,
            CacheType::Q8G128,
            CacheType::Q8G64,
            CacheType::Q8G32,
            CacheType::Q6G64,
            CacheType::Q5G64,
            CacheType::Q4G128,
            CacheType::Q4G64,
            CacheType::Q4G32,
            CacheType::Q3G64,
            CacheType::Q2G64,
            CacheType::Tq4,
            CacheType::Planar4,
            CacheType::Planar3,
            CacheType::PlanarK4,
            CacheType::RotK,
            CacheType::Iso3,
            CacheType::Iso4,
            CacheType::Rotor3,
            CacheType::Rotor4,
            CacheType::Turbo3Tcq,
            CacheType::Turbo2Tcq,
            CacheType::IsoK3,
            CacheType::IsoK4,
            CacheType::RotorK3,
            CacheType::RotorK4,
            CacheType::TurboSym3,
        ]
    }
}

// ── CacheTypeSpec ─────────────────────────────────────────────────────────────

/// Per-side KV cache type specification, holding one [`CacheType`] for K and
/// one for V.
///
/// Parsed from `--cache-type-k` / `--cache-type-v`.
/// Resolved to a concrete [`super::KvQuant`] by the resolver.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — two codec fields (K, V); adding a side requires updating resolve() and all cache-type CLI parsing"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Explicit K/V cache-type selection (from `--cache-type-k` / `--cache-type-v`).
pub struct CacheTypeSpec {
    /// Cache type for the K (key) side.
    pub k: CacheType,
    /// Cache type for the V (value) side.
    pub v: CacheType,
}

// ── parse ─────────────────────────────────────────────────────────────────────

/// Parse a `--cache-type-k` / `--cache-type-v` tag string into a [`CacheType`].
///
/// Accepts all canonical tags from §D1 plus the documented aliases.
///
/// ## Aliases
/// - `"f16"`, `"none"` → [`CacheType::Bf16`]
/// - `"turbo4"` → [`CacheType::Tq4`]
///
/// ## Reserved-but-not-implemented
/// Returns [`ParseError::NotImplemented`] for llama.cpp legacy block-32 tags
/// (`q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `iq4_nl`). The error message
/// always contains `"llama.cpp legacy"` and the closest rMLX substitute tag.
pub fn parse(s: &str) -> Result<CacheType, ParseError> {
    match s {
        // Canonical tags
        "auto" => Ok(CacheType::Auto),
        "bf16" => Ok(CacheType::Bf16),
        "q8_g128" => Ok(CacheType::Q8G128),
        "q8_g64" => Ok(CacheType::Q8G64),
        "q8_g32" => Ok(CacheType::Q8G32),
        "q6_g64" => Ok(CacheType::Q6G64),
        "q5_g64" => Ok(CacheType::Q5G64),
        "q4_g128" => Ok(CacheType::Q4G128),
        "q4_g64" => Ok(CacheType::Q4G64),
        "q4_g32" => Ok(CacheType::Q4G32),
        "q3_g64" => Ok(CacheType::Q3G64),
        "q2_g64" => Ok(CacheType::Q2G64),
        "tq4" => Ok(CacheType::Tq4),
        "planar4" => Ok(CacheType::Planar4),
        "planar3" | "planar_3" => Ok(CacheType::Planar3),
        "planar_k4" => Ok(CacheType::PlanarK4),
        "rot_k" => Ok(CacheType::RotK),
        // Both spellings accepted, mirroring Planar3.
        "iso_v_3" | "iso3" => Ok(CacheType::Iso3),
        // iso4 — same dual-spelling pattern as iso3.
        "iso_v_4" | "iso4" => Ok(CacheType::Iso4),
        // rotor3 — same dual-spelling pattern.
        "rotor_v_3" | "rotor3" => Ok(CacheType::Rotor3),
        // rotor4 — same dual-spelling pattern.
        "rotor_v_4" | "rotor4" => Ok(CacheType::Rotor4),
        // TCQ — canonical tag matches the §D1 `k8v_*` pattern;
        // alias matches the `--kv-quant k8vturbo3tcq` selector and mtq's
        // `turbo3_tcq` row.
        "k8v_turbo_3_tcq" | "turbo3_tcq" => Ok(CacheType::Turbo3Tcq),
        // 2-bit TCQ — same §D1 pattern; alias matches
        // `--kv-quant k8vturbo2tcq` selector and mtq's `turbo2_tcq` row.
        "k8v_turbo_2_tcq" | "turbo2_tcq" => Ok(CacheType::Turbo2Tcq),
        // K-side IsoQuant — dual spelling.
        "iso_k_3" | "k_iso3" => Ok(CacheType::IsoK3),
        "iso_k_4" | "k_iso4" => Ok(CacheType::IsoK4),
        // K-side rotor — dual spelling.
        "rotor_k_3" | "k_rotor3" => Ok(CacheType::RotorK3),
        "rotor_k_4" | "k_rotor4" => Ok(CacheType::RotorK4),
        // symmetric WHT-3 (K+V both turbo3).
        "tsym3" => Ok(CacheType::TurboSym3),

        // Aliases
        "f16" | "none" => Ok(CacheType::Bf16),
        "turbo4" => Ok(CacheType::Tq4),

        // Reserved — llama.cpp legacy block-32 codecs.
        // On-disk layout (fp16 per-block scale, block=32) is incompatible with
        // rMLX's affine 3-tuple (codes, scales, biases) stored at arbitrary group sizes.
        "q8_0" => Err(ParseError::NotImplemented(
            "q8_0 is a llama.cpp legacy block-32 codec (fp16 per-block scale) \
             not implemented in rMLX. \
             Use q8_g32 for block=32 fidelity, or q8_g128 (faster rMLX default).",
        )),
        "q4_0" => Err(ParseError::NotImplemented(
            "q4_0 is a llama.cpp legacy block-32 codec (fp16 per-block scale) \
             not implemented in rMLX. \
             Use q4_g32 for block=32 fidelity, or q4_g64 (rMLX default group).",
        )),
        "q4_1" => Err(ParseError::NotImplemented(
            "q4_1 is a llama.cpp legacy block-32 codec with per-block min/max \
             not implemented in rMLX. \
             Use q4_g32 as the closest rMLX equivalent.",
        )),
        "q5_0" => Err(ParseError::NotImplemented(
            "q5_0 is a llama.cpp legacy block-32 codec (fp16 per-block scale) \
             not implemented in rMLX. \
             Use q5_g64 (no g32 5-bit codec today; note the granularity difference).",
        )),
        "q5_1" => Err(ParseError::NotImplemented(
            "q5_1 is a llama.cpp legacy block-32 codec with per-block min/max \
             not implemented in rMLX. \
             Use q5_g64 as the closest rMLX equivalent.",
        )),
        "iq4_nl" => Err(ParseError::NotImplemented(
            "iq4_nl is a llama.cpp legacy non-linear 4-bit codec \
             not implemented in rMLX. \
             Use q4_g64 as the closest rMLX equivalent.",
        )),

        other => Err(ParseError::Unknown(other.to_owned())),
    }
}

// ── ResolverContext ───────────────────────────────────────────────────────────

/// Inputs the resolver needs to validate a [`CacheTypeSpec`] against §D6
/// invariants.
///
/// `head_dim` is the model's **full-attention** head_dim — see §D6.9. When the
/// model config does not declare it (and no safe fallback derives it), pass
/// `None`; the resolver refuses to operate rather than guess.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — two resolver context fields; adding a field requires updating resolve() and all resolve call sites"
)]
#[derive(Debug, Clone, Copy)]
/// Context passed to the KV-quant resolver.
pub struct ResolverContext<'a> {
    /// Architecture class string (e.g. `"Gemma4"`, `"Qwen3"`).
    pub arch_class: &'a str,
    /// Per-head dimension from the model config, if known.
    pub head_dim: Option<usize>,
}

// ── ResolveError ──────────────────────────────────────────────────────────────

/// Error returned by [`resolve`] and [`validate_resolved`].
///
/// Every variant's `Display` names the offending input AND suggests an
/// actionable alternative. No panic paths — every guard returns one of these.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResolveError {
    /// §D6.9 — model config did not declare a `head_dim` and no safe fallback derived one.
    #[error(
        "head_dim is not declared by the model config and could not be derived; \
         the resolver refuses to guess. \
         Either set `--kv-quant` (preset path) or report this model so its loader \
         can populate ModelConfig::head_dim()."
    )]
    HeadDimUnknown,

    /// §D6.3 — K-side rotation codecs (`tq4`, `planar4`) are V-side only.
    #[error(
        "K-side rotation codec '{0}' not implemented — V-side only. \
         Try '--ctk q8_g128' (the canonical K codec for K8V4/K8V8/Planar)."
    )]
    KSideRotationCodec(&'static str),

    /// — `rot_k` requires a power-of-two head_dim (Walsh-Hadamard rotation).
    #[error(
        "rot_k requires a power-of-two head_dim (Walsh-Hadamard rotation); got head_dim={0}. \
         Use '--ctk q8_g128' for this head_dim."
    )]
    RotKHeadDimNotPow2(usize),

    /// — `rot_k` is a K-side codec; it is invalid on the V side.
    #[error(
        "rot_k is a K-side rotation codec and cannot be used on V. \
         Use '--ctk rot_k' with an affine V codec (e.g. '--ctv q4_g64' or '--ctv tq4')."
    )]
    RotKVSide,

    /// §D6.5 — TurboQuant 4-bit kernel only supports head_dim ∈ {128, 256}.
    #[error(
        "tq4 requires head_dim ∈ {{128, 256}}; got head_dim={0}. \
         Try '--ctv q4_g64' or '--ctv planar4' instead."
    )]
    Tq4UnsupportedHeadDim(usize),

    /// §D6.1 — `head_dim % group_size != 0` for an affine codec.
    #[error(
        "head_dim={head_dim} not divisible by group_size={group_size} (affine codec invariant); \
         pick a codec whose group divides head_dim (e.g. q*_g32 or q*_g128 if applicable)."
    )]
    GroupSizeNotDivisible {
        /// The head dimension that failed the divisibility check.
        head_dim: usize,
        /// The group size that does not divide `head_dim`.
        group_size: usize,
    },

    /// §D6.2 — MLX bit-packing requires `head_dim % (32 / bits) == 0`.
    #[error(
        "MLX bit-packing rule violated: head_dim={head_dim} not divisible by (32 / bits) \
         where bits={bits}; pick a different head_dim or use a higher-bit codec \
         (e.g. q4_g* or q8_g*)."
    )]
    MlxBitPackingViolation {
        /// The head dimension that violates bit-packing alignment.
        head_dim: usize,
        /// The bit-width that triggered the violation.
        bits: u8,
    },

    /// §D6.4 — Qwen MoE family requires K-side bits ≥ 8 (Qwen MoE PPL disaster).
    #[error(
        "Qwen MoE family requires K-side bits >= 8 (PPL disaster on K<8); got K-bits={0}. \
         Try '--ctk q8_g128' or '--ctk q8_g64' or use '--kv-quant k8v8' preset."
    )]
    QwenMoeKBitsTooLow(u8),

    /// Contract A.y — PlanarK on Qwen MoE is PPL-disaster zone.
    ///
    /// K-side 4-bit on Qwen MoE causes catastrophic PPL collapse (218→8641 on
    /// Q4_K_M baseline; 7:1 GQA amplifies K-head error through softmax).
    /// `KvQuant::PlanarK` is rejected outright on Qwen MoE — no warn-and-proceed.
    #[error(
        "K-side 4-bit on Qwen MoE is PPL-disaster: --kv-quant planar_k (and '--ctk planar_k4') \
         are rejected for Qwen3.5/3.6 MoE. Use '--kv-quant k8v8' or '--kv-quant planar' \
         (V-side rotation; K stays 8-bit)."
    )]
    QwenMoePlanarKRejected,

    /// Contract A.y — IsoQuant K-side codecs are rejected on
    /// Qwen MoE. K-side ≤4-bit on Qwen MoE is the PPL-disaster zone (same
    /// 7:1 GQA softmax-amplification reason as `QwenMoePlanarKRejected`).
    /// Applies to variants: `Iso3Sym`, `Iso4Sym`, `IsoKOnly3`,
    /// `IsoKOnly4`. The `variant` field carries the offending KvQuant
    /// `Display` form (e.g. `"iso3_sym"`, `"k_iso4"`).
    #[error(
        "K-side ≤4-bit on Qwen MoE is PPL-disaster: --kv-quant {variant} \
         (and the matching '--ctk iso_k_*' selector) is rejected for Qwen3.5/3.6 MoE. \
         Use '--kv-quant k8v8' (K stays 8-bit) or a V-only iso variant \
         ('--kv-quant iso3' / '--kv-quant iso4')."
    )]
    QwenMoeIsoKRejected {
        /// The KvQuant `Display` form that triggered the guard.
        variant: String,
    },

    /// Contract A.y — rotor K-side codecs are rejected on Qwen
    /// MoE. K-side ≤4-bit on Qwen MoE is the PPL-disaster zone (same 7:1 GQA
    /// softmax-amplification reason as `QwenMoeIsoKRejected`). Applies to
    /// variants: `Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`,
    /// `RotorKOnly4`. The `variant` field carries the offending KvQuant
    /// `Display` form (e.g. `"rotor3_sym"`, `"k_rotor4"`).
    #[error(
        "K-side ≤4-bit on Qwen MoE is PPL-disaster: --kv-quant {variant} is rejected for \
         Qwen3.5/3.6 MoE. Use '--kv-quant k8v8' (K stays 8-bit) or a V-only rotor variant \
         ('--kv-quant rotor3' / '--kv-quant rotor4')."
    )]
    QwenMoeRotorKRejected {
        /// The KvQuant `Display` form that triggered the guard.
        variant: String,
    },

    /// Contract A.y — TurboSym3 symmetric K+V 3-bit is rejected
    /// on Qwen MoE. K-side 3-bit on Qwen MoE is the PPL-disaster zone (same
    /// 7:1 GQA softmax-amplification as the other K-side guards). The `variant`
    /// field carries the offending KvQuant `Display` form (`"tsym3"`).
    #[error(
        "K-side 3-bit on Qwen MoE is PPL-disaster: --kv-quant {variant} is rejected for \
         Qwen3.5/3.6 MoE. Use '--kv-quant k8v8' (K stays 8-bit) or '--kv-quant k8vturbo3' \
         (K=8-bit, V=turbo3)."
    )]
    QwenMoeTurboKRejected {
        /// The KvQuant `Display` form that triggered the guard.
        variant: String,
    },

    // The former `SharedKvIncompatibleWithMixed` variant was removed.
    // Gemma3 / Gemma4 cross-layer KV sharing now supports `Mixed` via
    // dequant-before-share in `KvCache::update_and_sdpa_returning_kv`,
    // so the combination is valid.
    /// No defined `KvQuant` mapping for this `(K, V)` tuple.
    ///
    /// Message names the actual K codec and explains the constraint.
    #[error("{0}")]
    UnsupportedCombo(String),
}

// ── decompose_auto ────────────────────────────────────────────────────────────

/// Decompose a concrete [`KvQuant`] into the `(k, v)` [`CacheType`] pair it
/// would have come from in the §D1 mapping.
///
/// Inverse of [`combo_to_kv_quant`] for the canonical resolutions of
/// `KvCacheBuilder::resolve_default`. Used by [`resolve`] to override only the
/// user-specified side when one side is `Auto`.
///
/// For `Mixed { k_bits, v_bits, k_group_size, v_group_size }` this maps each
/// side to the matching `Q*G*` variant. If a side's (bits, group_size) does
/// not correspond to any canonical variant, the side falls back to `Bf16` — the
/// resolver will then surface an `UnsupportedCombo` downstream rather than
/// silently picking a wrong codec.
#[allow(
    clippy::panic,
    reason = "K8VTurbo2 arm is a contract-violation guard: the variant has no decomposable CacheType pair and must never reach this path via auto-baseline selection; panic surfaces the misconfiguration at startup rather than silently producing wrong output"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "single match over the closed KvQuant enum — splitting per-variant arms into helpers would add indirection without reducing local complexity; each arm is small and self-contained"
)]
#[allow(
    clippy::too_many_lines,
    reason = "single match over the closed KvQuant enum — each arm is a 2–8 line direct mapping; splitting per-variant arms would scatter the registry across helpers without reducing per-arm complexity"
)]
pub fn decompose_auto(kq: KvQuant) -> (CacheType, CacheType) {
    match kq {
        KvQuant::None => (CacheType::Bf16, CacheType::Bf16),
        KvQuant::K8V4 => (CacheType::Q8G128, CacheType::Tq4),
        KvQuant::K8V8 => (CacheType::Q8G128, CacheType::Q8G128),
        KvQuant::Planar => (CacheType::Q8G128, CacheType::Planar4),
        KvQuant::Mixed {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
        } => (
            affine_to_cache_type(k_bits, k_group_size as usize),
            affine_to_cache_type(v_bits, v_group_size as usize),
        ),
        // RotK is never an `auto` base (opt-in only via `--ctk rot_k`),
        // so this arm is unreachable in practice. Decompose to K=rot_k and the
        // matching affine V so a defensive caller still round-trips correctly.
        KvQuant::RotK {
            v_bits,
            v_group_size,
        } => (
            CacheType::RotK,
            affine_to_cache_type(v_bits, v_group_size as usize),
        ),
        // RotKTq4V is never an auto base; decompose to (rot_k, tq4).
        KvQuant::RotKTq4V => (CacheType::RotK, CacheType::Tq4),
        // K8VTurbo3 — auto default for Gemma4 small.
        // Decomposes to (Q8G128, Tq3). No dedicated CacheType::Tq3 variant exists yet;
        // per-side --ctk/--ctv overrides that hit this arm fall back to the Mixed{v_bits:3}
        // affine path (Q8G128, Q3G64) rather than K8VTurbo3 — acceptable because the
        // --ctv override is unusual and the fallback is semantically correct (3-bit V).
        KvQuant::K8VTurbo3 => {
            tracing::debug!(
                kv_quant = "k8vturbo3",
                "decompose_auto: no Tq3 CacheType yet; V side falls back to Q3G64 (affine 3-bit)"
            );
            (CacheType::Q8G128, CacheType::Q3G64)
        }
        // TurboSym4 is never an auto base — symmetric 4-bit K is
        // the Qwen MoE PPL-218→8641 disaster; opt-in only via `--kv-quant
        // tsym4` or the `quality` preset. Decompose to (Tq4, Tq4) for
        // completeness; combo_to_kv_quant will reject a K-side Tq4 with
        // `KSideRotationCodec` if a defensive caller round-trips this.
        KvQuant::TurboSym4 => {
            tracing::warn!("unexpected TurboSym4 decompose_auto reached — TurboSym4 should never be an auto baseline");
            (CacheType::Tq4, CacheType::Tq4)
        }
        // PlanarK is never an auto base — K-side 4-bit on Qwen MoE
        // is the PPL-disaster; opt-in only via `--kv-quant planar_k`. Decompose
        // to (PlanarK4, Bf16) — the canonical pairing.
        KvQuant::PlanarK => (CacheType::PlanarK4, CacheType::Bf16),
        // K8VTurbo2 is never an auto base (opt-in via --kv-quant
        // k8vturbo2 only). There is no valid CacheType pair for K8VTurbo2 —
        // the codec is not decomposable into affine (K, V) sides. Reaching
        // this arm is a contract violation: the caller must not pass K8VTurbo2
        // as an auto-baseline, and the CLI/config layer must have filtered it.
        KvQuant::K8VTurbo2 => {
            tracing::error!(
                kv_quant = "k8vturbo2",
                "decompose_auto called with K8VTurbo2 — contract violation; K8VTurbo2 is opt-in only"
            );
            panic!(
                "decompose_auto: K8VTurbo2 has no CacheType pair; --kv-quant k8vturbo2 must be set explicitly"
            );
        }
        // Planar3 is never an auto base — opt-in only via
        // --kv-quant planar3. Decompose to (Q8G128, Planar3) for the canonical
        // K8/V3-bit pairing.
        KvQuant::Planar3 => (CacheType::Q8G128, CacheType::Planar3),
        // Iso3 is never an auto base — opt-in only via
        // --kv-quant iso3. Decompose to (Q8G128, Iso3); the K side stays at
        // 8-bit affine and the V side is the IsoQuant 3-bit rotation codec.
        // Mirrors the K8VTurbo3 / Planar3 patterns.
        KvQuant::Iso3 => {
            tracing::debug!(
                kv_quant = "iso3",
                "decompose_auto: Iso3 → (Q8G128, Iso3); never an auto baseline (opt-in only)"
            );
            (CacheType::Q8G128, CacheType::Iso3)
        }
        // Iso4 is never an auto base — opt-in only via
        // --kv-quant iso4. Decompose to (Q8G128, Iso4); same pattern as Iso3.
        KvQuant::Iso4 => {
            tracing::debug!(
                kv_quant = "iso4",
                "decompose_auto: Iso4 → (Q8G128, Iso4); never an auto baseline (opt-in only)"
            );
            (CacheType::Q8G128, CacheType::Iso4)
        }
        // Rotor3 is never an auto base — opt-in only via
        // --kv-quant rotor3. Decompose to (Q8G128, Rotor3); mirrors Iso3/Iso4.
        KvQuant::Rotor3 => {
            tracing::debug!(
                kv_quant = "rotor3",
                "decompose_auto: Rotor3 → (Q8G128, Rotor3); never an auto baseline (opt-in only)"
            );
            (CacheType::Q8G128, CacheType::Rotor3)
        }
        // Rotor4 is never an auto base — opt-in only via
        // --kv-quant rotor4. Decompose to (Q8G128, Rotor4); mirrors Rotor3.
        KvQuant::Rotor4 => {
            tracing::debug!(
                kv_quant = "rotor4",
                "decompose_auto: Rotor4 → (Q8G128, Rotor4); never an auto baseline (opt-in only)"
            );
            (CacheType::Q8G128, CacheType::Rotor4)
        }
        // K8VTurbo3Tcq is never an auto base — opt-in only via
        // --kv-quant k8vturbo3tcq. Decompose to (Q8G128, Turbo3Tcq); the K
        // side stays at 8-bit affine and the V side is the Viterbi-trellis
        // codec.
        KvQuant::K8VTurbo3Tcq => {
            tracing::debug!(
                kv_quant = "k8vturbo3tcq",
                "decompose_auto: K8VTurbo3Tcq → (Q8G128, Turbo3Tcq); never an auto baseline (opt-in only)"
            );
            (CacheType::Q8G128, CacheType::Turbo3Tcq)
        }
        // K8VTurbo2Tcq is never an auto base — opt-in only via
        // --kv-quant k8vturbo2tcq. Decompose to (Q8G128, Turbo2Tcq).
        KvQuant::K8VTurbo2Tcq => {
            tracing::debug!(
                kv_quant = "k8vturbo2tcq",
                "decompose_auto: K8VTurbo2Tcq → (Q8G128, Turbo2Tcq); never an auto baseline (opt-in only)"
            );
            (CacheType::Q8G128, CacheType::Turbo2Tcq)
        }
        // Iso3Sym — never auto. K = iso_k_3, V = iso_v_3.
        KvQuant::Iso3Sym => {
            tracing::debug!(
                kv_quant = "iso3_sym",
                "decompose_auto: Iso3Sym → (IsoK3, Iso3); never an auto baseline (opt-in only)"
            );
            (CacheType::IsoK3, CacheType::Iso3)
        }
        // Iso4Sym — never auto. K = iso_k_4, V = iso_v_4.
        KvQuant::Iso4Sym => {
            tracing::debug!(
                kv_quant = "iso4_sym",
                "decompose_auto: Iso4Sym → (IsoK4, Iso4); never an auto baseline (opt-in only)"
            );
            (CacheType::IsoK4, CacheType::Iso4)
        }
        // IsoKOnly3 — K only; V is bf16.
        KvQuant::IsoKOnly3 => {
            tracing::debug!(
                kv_quant = "k_iso3",
                "decompose_auto: IsoKOnly3 → (IsoK3, Bf16); never an auto baseline (opt-in only)"
            );
            (CacheType::IsoK3, CacheType::Bf16)
        }
        // IsoKOnly4 — K only; V is bf16.
        KvQuant::IsoKOnly4 => {
            tracing::debug!(
                kv_quant = "k_iso4",
                "decompose_auto: IsoKOnly4 → (IsoK4, Bf16); never an auto baseline (opt-in only)"
            );
            (CacheType::IsoK4, CacheType::Bf16)
        }
        // Rotor3Sym — never auto. K = rotor_k_3, V = rotor_v_3.
        KvQuant::Rotor3Sym => {
            tracing::debug!(
                kv_quant = "rotor3_sym",
                "decompose_auto: Rotor3Sym → (RotorK3, Rotor3); never an auto baseline (opt-in only)"
            );
            (CacheType::RotorK3, CacheType::Rotor3)
        }
        // Rotor4Sym — never auto. K = rotor_k_4, V = rotor_v_4.
        KvQuant::Rotor4Sym => {
            tracing::debug!(
                kv_quant = "rotor4_sym",
                "decompose_auto: Rotor4Sym → (RotorK4, Rotor4); never an auto baseline (opt-in only)"
            );
            (CacheType::RotorK4, CacheType::Rotor4)
        }
        // RotorKOnly3 — K only; V is bf16.
        KvQuant::RotorKOnly3 => {
            tracing::debug!(
                kv_quant = "k_rotor3",
                "decompose_auto: RotorKOnly3 → (RotorK3, Bf16); never an auto baseline (opt-in only)"
            );
            (CacheType::RotorK3, CacheType::Bf16)
        }
        // RotorKOnly4 — K only; V is bf16.
        KvQuant::RotorKOnly4 => {
            tracing::debug!(
                kv_quant = "k_rotor4",
                "decompose_auto: RotorKOnly4 → (RotorK4, Bf16); never an auto baseline (opt-in only)"
            );
            (CacheType::RotorK4, CacheType::Bf16)
        }
        // RotorK3Asym / RotorK4Asym — never auto. K = rotor_k_*,
        // V = affine `(v_bits, v_group_size)`. Mirrors RotorKOnly{3,4} for the
        // K side and the Mixed-style affine fallback for the V side.
        KvQuant::RotorK3Asym {
            v_bits,
            v_group_size,
        } => {
            tracing::debug!(
                kv_quant = %KvQuant::RotorK3Asym { v_bits, v_group_size },
                "decompose_auto: RotorK3Asym → (RotorK3, affine V); never an auto baseline (opt-in only)"
            );
            (
                CacheType::RotorK3,
                affine_to_cache_type(v_bits, v_group_size as usize),
            )
        }
        KvQuant::RotorK4Asym {
            v_bits,
            v_group_size,
        } => {
            tracing::debug!(
                kv_quant = %KvQuant::RotorK4Asym { v_bits, v_group_size },
                "decompose_auto: RotorK4Asym → (RotorK4, affine V); never an auto baseline (opt-in only)"
            );
            (
                CacheType::RotorK4,
                affine_to_cache_type(v_bits, v_group_size as usize),
            )
        }
        // TurboSym3 is never an auto base (opt-in via --kv-quant tsym3
        // or the `speed` preset). Decompose to (TurboSym3, TurboSym3) — the symmetric
        // K+V pairing. combo_to_kv_quant will resolve back to KvQuant::TurboSym3
        // if a defensive caller round-trips this.
        KvQuant::TurboSym3 => {
            tracing::debug!(
                kv_quant = "tsym3",
                "decompose_auto: TurboSym3 → (TurboSym3, TurboSym3); never an auto baseline (opt-in only)"
            );
            (CacheType::TurboSym3, CacheType::TurboSym3)
        }
    }
}

/// Map an (bits, group_size) tuple back to its canonical [`CacheType`] variant.
///
/// Returns `Bf16` for any tuple that has no canonical variant — the resolver
/// will reject downstream rather than silently match the wrong codec.
fn affine_to_cache_type(bits: u8, group_size: usize) -> CacheType {
    match (bits, group_size) {
        (8, 128) => CacheType::Q8G128,
        (8, 64) => CacheType::Q8G64,
        (8, 32) => CacheType::Q8G32,
        (6, 64) => CacheType::Q6G64,
        (5, 64) => CacheType::Q5G64,
        (4, 128) => CacheType::Q4G128,
        (4, 64) => CacheType::Q4G64,
        (4, 32) => CacheType::Q4G32,
        (3, 64) => CacheType::Q3G64,
        (2, 64) => CacheType::Q2G64,
        _ => CacheType::Bf16,
    }
}

// ── combo_to_kv_quant ─────────────────────────────────────────────────────────

/// Map a resolved `(K, V)` [`CacheType`] tuple to the concrete [`KvQuant`].
///
/// **Asymmetric-auto coercion**: when K is the canonical K8V4/K8V8/Planar K-side
/// (`Q8G128`) and V is a rotation codec, coerce to `KvQuant::K8V4` or
/// `KvQuant::Planar` as appropriate. This is the **only** path that "promotes"
/// a tuple — it requires K's group_size to be exactly 128. Without this guard,
/// a Bonsai-style auto decomposition (`Mixed{k_group_size=64,..}`) plus
/// `--ctv tq4` would silently swap K from g=64 to g=128.
///
/// Returns `UnsupportedCombo` for tuples that have no defined `KvQuant`
/// mapping — including rotation codec on K, rotation codec on V paired with a
/// non-canonical K, and any affine K with `bits < 3` or unfamiliar group sizes.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
#[allow(
    clippy::too_many_lines,
    reason = "every V-side codec carries its own (K, V) coercion + non-canonical-K error arm; splitting would scatter the registry across helpers"
)]
pub fn combo_to_kv_quant(k: CacheType, v: CacheType) -> Result<KvQuant, ResolveError> {
    // K-side rotation codec is never valid (defense; resolve() should reject earlier).
    match k {
        CacheType::Tq4 => return Err(ResolveError::KSideRotationCodec(CacheType::Tq4.tag())),
        CacheType::Planar4 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Planar4.tag()));
        }
        // iso3 is V-side only (mirrors Planar3/Planar4/Tq4).
        CacheType::Iso3 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Iso3.tag()));
        }
        // iso4 is V-side only (mirrors Iso3).
        CacheType::Iso4 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Iso4.tag()));
        }
        // rotor3 is V-side only (mirrors Iso3/Iso4).
        CacheType::Rotor3 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Rotor3.tag()));
        }
        // rotor4 is V-side only (mirrors rotor3).
        CacheType::Rotor4 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Rotor4.tag()));
        }
        // Turbo3Tcq is V-side only (mirrors Tq4 / Planar3 / Iso3).
        CacheType::Turbo3Tcq => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Turbo3Tcq.tag()));
        }
        // Turbo2Tcq is V-side only (mirrors Turbo3Tcq).
        CacheType::Turbo2Tcq => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Turbo2Tcq.tag()));
        }
        // `Auto` should never reach here — `resolve` decomposes before calling.
        CacheType::Auto => {
            return Err(ResolveError::UnsupportedCombo(
                "internal: combo_to_kv_quant called with K=Auto (resolver bug)".to_string(),
            ));
        }
        _ => {}
    }
    if v == CacheType::Auto {
        return Err(ResolveError::UnsupportedCombo(
            "internal: combo_to_kv_quant called with V=Auto (resolver bug)".to_string(),
        ));
    }

    // PlanarK4 is K-side only — reject on V (mirrors Tq4/Planar4
    // V-only / RotK K-only convention).
    if matches!(v, CacheType::PlanarK4) {
        return Err(ResolveError::UnsupportedCombo(
            "V='planar_k4' rejected: planar_k4 is a K-side codec (opt-in via '--ctk planar_k4'). \
             For V-side PlanarQuant use '--ctv planar4'."
                .to_string(),
        ));
    }

    // PlanarK4 K-side resolves with V=bf16 to KvQuant::PlanarK.
    if k == CacheType::PlanarK4 {
        if v == CacheType::Bf16 {
            return Ok(KvQuant::PlanarK);
        }
        return Err(ResolveError::UnsupportedCombo(format!(
            "K='planar_k4' requires V='bf16' (PlanarK pairs 4-bit rotation K with full-precision V). \
             Got V='{}'. Use '--ctk planar_k4 --ctv bf16' (or '--kv-quant planar_k').",
            v.tag()
        )));
    }

    // IsoK3 K-side resolves with V=Iso3 (sym) or V=Bf16 (k-only).
    if k == CacheType::IsoK3 {
        return match v {
            CacheType::Iso3 => Ok(KvQuant::Iso3Sym),
            CacheType::Bf16 => Ok(KvQuant::IsoKOnly3),
            _ => Err(ResolveError::UnsupportedCombo(format!(
                "K='iso_k_3' requires V='iso_v_3' (symmetric → Iso3Sym) or V='bf16' (K-only → IsoKOnly3). \
                 Got V='{}'. Use '--kv-quant iso3_sym' or '--kv-quant k_iso3'.",
                v.tag()
            ))),
        };
    }

    // IsoK4 K-side resolves with V=Iso4 (sym) or V=Bf16 (k-only).
    if k == CacheType::IsoK4 {
        return match v {
            CacheType::Iso4 => Ok(KvQuant::Iso4Sym),
            CacheType::Bf16 => Ok(KvQuant::IsoKOnly4),
            _ => Err(ResolveError::UnsupportedCombo(format!(
                "K='iso_k_4' requires V='iso_v_4' (symmetric → Iso4Sym) or V='bf16' (K-only → IsoKOnly4). \
                 Got V='{}'. Use '--kv-quant iso4_sym' or '--kv-quant k_iso4'.",
                v.tag()
            ))),
        };
    }

    // IsoK3 / IsoK4 are K-side codecs — reject on V (analog to PlanarK4 V-side rejection).
    if matches!(v, CacheType::IsoK3 | CacheType::IsoK4) {
        return Err(ResolveError::UnsupportedCombo(format!(
            "V='{}' rejected: '{}' is a K-side codec. For V-side IsoQuant use '--ctv iso_v_3' / '--ctv iso_v_4'.",
            v.tag(),
            v.tag()
        )));
    }

    // RotorK3 K-side resolves with V=Rotor3 (sym),
    // V=Bf16 (k-only), or V=affine q*_g* (RotorK3Asym).
    if k == CacheType::RotorK3 {
        return match v {
            CacheType::Rotor3 => Ok(KvQuant::Rotor3Sym),
            CacheType::Bf16 => Ok(KvQuant::RotorKOnly3),
            _ => {
                // Affine V → RotorK3Asym { v_bits, v_group_size }.
                if let (Some(vb), Some(vg)) = (v.bits(), v.group_size()) {
                    return rmlx_kv_quant::validate_rotor_k_asym_v(vb, vg as u16)
                        .map(|()| KvQuant::RotorK3Asym {
                            v_bits: vb,
                            v_group_size: vg as u16,
                        })
                        .map_err(ResolveError::UnsupportedCombo);
                }
                Err(ResolveError::UnsupportedCombo(format!(
                    "K='rotor_k_3' requires V='rotor_v_3' (symmetric → Rotor3Sym), V='bf16' (K-only → RotorKOnly3), \
                     or an affine V codec (q4_g128, q4_g64, q4_g32, q3_g64, q2_g64) → RotorK3Asym. \
                     Got V='{}'.",
                    v.tag()
                )))
            }
        };
    }

    // RotorK4 K-side resolves with V=Rotor4 (sym),
    // V=Bf16 (k-only), or V=affine q*_g* (RotorK4Asym).
    if k == CacheType::RotorK4 {
        return match v {
            CacheType::Rotor4 => Ok(KvQuant::Rotor4Sym),
            CacheType::Bf16 => Ok(KvQuant::RotorKOnly4),
            _ => {
                if let (Some(vb), Some(vg)) = (v.bits(), v.group_size()) {
                    return rmlx_kv_quant::validate_rotor_k_asym_v(vb, vg as u16)
                        .map(|()| KvQuant::RotorK4Asym {
                            v_bits: vb,
                            v_group_size: vg as u16,
                        })
                        .map_err(ResolveError::UnsupportedCombo);
                }
                Err(ResolveError::UnsupportedCombo(format!(
                    "K='rotor_k_4' requires V='rotor_v_4' (symmetric → Rotor4Sym), V='bf16' (K-only → RotorKOnly4), \
                     or an affine V codec (q4_g128, q4_g64, q4_g32, q3_g64, q2_g64) → RotorK4Asym. \
                     Got V='{}'.",
                    v.tag()
                )))
            }
        };
    }

    // TurboSym3 is a symmetric K+V codec — requires both sides.
    // K=TurboSym3 + V=TurboSym3 → KvQuant::TurboSym3.
    // TurboSym3 on only one side is an error (no asymmetric pairing defined).
    if k == CacheType::TurboSym3 {
        return match v {
            CacheType::TurboSym3 => Ok(KvQuant::TurboSym3),
            _ => Err(ResolveError::UnsupportedCombo(format!(
                "K='tsym3' requires V='tsym3' (TurboSym3 is symmetric K+V). \
                 Got V='{}'. Use '--kv-quant tsym3' or '--ctk tsym3 --ctv tsym3'.",
                v.tag()
            ))),
        };
    }
    if v == CacheType::TurboSym3 {
        return Err(ResolveError::UnsupportedCombo(format!(
            "V='tsym3' requires K='tsym3' (TurboSym3 is symmetric K+V). \
             Got K='{}'. Use '--kv-quant tsym3' or '--ctk tsym3 --ctv tsym3'.",
            k.tag()
        )));
    }

    // RotorK3 / RotorK4 are K-side codecs — reject on V.
    if matches!(v, CacheType::RotorK3 | CacheType::RotorK4) {
        return Err(ResolveError::UnsupportedCombo(format!(
            "V='{}' rejected: '{}' is a K-side codec. For V-side Clifford rotor use '--ctv rotor_v_3' / '--ctv rotor_v_4'.",
            v.tag(),
            v.tag()
        )));
    }

    // / : K-side rotation codec.
    // - V=tq4: resolves to RotKTq4V (rotated affine K + TurboFlash 4-bit V).
    // - V=affine: resolves to RotK (rotated affine K + affine V via mx.quantize).
    // - V=planar4 or V=bf16: unsupported (no rot_k pairing defined for these).
    if k == CacheType::RotK {
        // rot_k + tq4 → new RotKTq4V hybrid variant.
        if v == CacheType::Tq4 {
            return Ok(KvQuant::RotKTq4V);
        }
        let (Some(vb), Some(vg)) = (v.bits(), v.group_size()) else {
            return Err(ResolveError::UnsupportedCombo(format!(
                "K='rot_k' requires an affine V codec (q*_g*) or V='tq4'; \
                 got V='{}'. Try '--ctv q4_g64', '--ctv q8_g64', or '--ctv tq4'.",
                v.tag()
            )));
        };
        return Ok(KvQuant::RotK {
            v_bits: vb,
            v_group_size: vg as u16,
        });
    }

    match (k, v) {
        (CacheType::Bf16, CacheType::Bf16) => Ok(KvQuant::None),

        // Coerce to K8V8 only when the layout matches exactly (q8_g128 on both sides).
        (CacheType::Q8G128, CacheType::Q8G128) => Ok(KvQuant::K8V8),

        // Coerce to K8V4 only when K is exactly q8_g128 (rMLX MSL codec at group=128).
        (CacheType::Q8G128, CacheType::Tq4) => Ok(KvQuant::K8V4),

        // Coerce to Planar only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Planar4) => Ok(KvQuant::Planar),

        // Coerce to Planar3 only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Planar3) => Ok(KvQuant::Planar3),

        // Coerce to Iso3 only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Iso3) => Ok(KvQuant::Iso3),

        // Coerce to Iso4 only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Iso4) => Ok(KvQuant::Iso4),

        // Coerce to Rotor3 only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Rotor3) => Ok(KvQuant::Rotor3),

        // Coerce to Rotor4 only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Rotor4) => Ok(KvQuant::Rotor4),

        // Coerce to K8VTurbo3Tcq only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Turbo3Tcq) => Ok(KvQuant::K8VTurbo3Tcq),

        // Coerce to K8VTurbo2Tcq only when K is exactly q8_g128.
        (CacheType::Q8G128, CacheType::Turbo2Tcq) => Ok(KvQuant::K8VTurbo2Tcq),

        // Iso3 V paired with non-canonical K — reject.
        (k, CacheType::Iso3) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='iso_v_3'; Iso3 requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Iso4 V paired with non-canonical K — reject.
        (k, CacheType::Iso4) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='iso_v_4'; Iso4 requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Rotor3 V paired with non-canonical K — reject (mirrors Iso3/Iso4).
        (k, CacheType::Rotor3) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='rotor_v_3'; Rotor3 requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Rotor4 V paired with non-canonical K — reject (mirrors Rotor3).
        (k, CacheType::Rotor4) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='rotor_v_4'; Rotor4 requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Turbo3Tcq V paired with non-canonical K — reject.
        (k, CacheType::Turbo3Tcq) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='k8v_turbo_3_tcq'; K8VTurbo3Tcq requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Turbo2Tcq V paired with non-canonical K — reject.
        (k, CacheType::Turbo2Tcq) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='k8v_turbo_2_tcq'; K8VTurbo2Tcq requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Planar3 V paired with non-canonical K — reject.
        (k, CacheType::Planar3) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='planar3'; Planar3 requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // Rotation V paired with non-canonical K — do not silently promote.
        (k, CacheType::Tq4) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='tq4'; K8V4 requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),
        (k, CacheType::Planar4) => Err(ResolveError::UnsupportedCombo(format!(
            "K-side '{}' not paired with V='planar4'; Planar requires K='q8_g128' (group=128). \
             Either set '--ctk q8_g128' or pick an affine V codec.",
            k.tag()
        ))),

        // bf16 on one side and quantized on the other: unsupported (no canonical KvQuant).
        (CacheType::Bf16, v) => Err(ResolveError::UnsupportedCombo(format!(
            "K='bf16' paired with V='{}': no canonical KvQuant for bf16-K + quantized-V. \
             Use '--ctk q8_g128' (or an affine K codec) or set V='bf16' as well.",
            v.tag()
        ))),
        (k, CacheType::Bf16) => Err(ResolveError::UnsupportedCombo(format!(
            "K='{}' paired with V='bf16': no canonical KvQuant for quantized-K + bf16-V. \
             Set V='bf16' AND K='bf16', or pick a quantized V codec.",
            k.tag()
        ))),

        // pure 2-bit K is gated. 2-bit on the K side degrades attention
        // scores into incoherent output (CLAUDE.md hard rule 6 — smoke-probed
        // on Bonsai). 2-bit is V-side only; K must stay >= 3-bit.
        (CacheType::Q2G64, _) => Err(ResolveError::UnsupportedCombo(format!(
            "K='q2_g64' rejected: 2-bit K degrades attention into incoherent output. \
             2-bit is V-side only. Use '--ctk q8_g128 --ctv q2_g64' (asymmetric) \
             or '--kv-bits 2' (K stays 8-bit, V=2-bit). Paired V was '{}'.",
            v.tag()
        ))),

        // Both sides MLX affine → Mixed.
        (k, v) if k.is_affine() && v.is_affine() => {
            // Both bits/group_size are Some by is_affine() definition; bind via expect-free pattern.
            let (Some(kb), Some(kg)) = (k.bits(), k.group_size()) else {
                return Err(ResolveError::UnsupportedCombo(format!(
                    "internal: '{}' marked affine but bits/group_size missing",
                    k.tag()
                )));
            };
            let (Some(vb), Some(vg)) = (v.bits(), v.group_size()) else {
                return Err(ResolveError::UnsupportedCombo(format!(
                    "internal: '{}' marked affine but bits/group_size missing",
                    v.tag()
                )));
            };
            Ok(KvQuant::Mixed {
                k_bits: kb,
                v_bits: vb,
                k_group_size: kg as u16,
                v_group_size: vg as u16,
            })
        }

        // Catch-all: should be unreachable but defends against future enum additions.
        (k, v) => Err(ResolveError::UnsupportedCombo(format!(
            "no canonical KvQuant for K='{}' V='{}'.",
            k.tag(),
            v.tag()
        ))),
    }
}

// ── validate_resolved ─────────────────────────────────────────────────────────

// Returns `true` for any Qwen sparse-MoE architecture that requires K-bits ≥ 8.
//
// Both text-only (`Qwen3_5MoeForConditionalGeneration`) and vision-language MoE
// (`Qwen3VLMoeForConditionalGeneration`) share the same PPL-disaster sensitivity
// on low K-bit quantization (§D6.4). Centralising the check here ensures that
// future Qwen MoE variants are added in one place.
fn is_qwen_moe(arch: &str) -> bool {
    matches!(
        arch,
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3VLMoeForConditionalGeneration"
    )
}

/// Re-check post-decompose invariants on a concrete [`KvQuant`].
///
/// Enforces:
/// - §D6.4 (Qwen MoE K-bits ≥ 8) — inspects the K side of `Mixed`.
///   `K8V4`/`K8V8`/`Planar` always have K=8 so they pass; `None` passes.
///
/// This runs **after** auto-decompose so future `resolve_default` table
/// changes cannot bypass the invariant.
///
/// The former guard that rejected `Mixed` on Gemma3 / Gemma4 (cross-layer KV
/// sharing) was removed. `KvCache::update_and_sdpa_returning_kv` supports
/// `Mixed` via dequant-before-share — it surfaces the accumulated bf16 K/V
/// (prefill-raw during prefill, maintained `decode_fp16` during decode) to
/// the shared-KV consumer layers.
#[allow(
    clippy::cognitive_complexity,
    reason = "sequential Qwen MoE guard chain — each arm is a distinct error variant; refactoring would obscure the invariant order"
)]
pub fn validate_resolved(arch_class: &str, kq: &KvQuant) -> Result<(), ResolveError> {
    if is_qwen_moe(arch_class) {
        if let KvQuant::Mixed { k_bits, .. } = kq {
            if *k_bits < 8 {
                return Err(ResolveError::QwenMoeKBitsTooLow(*k_bits));
            }
        }
        // Contract A.y — PlanarK is K-side 4-bit rotation. Hard reject
        // with a dedicated error so the diagnostic surfaces the K-side disaster
        // (separate from the generic Mixed-K<8 path and from TurboSym4).
        if matches!(kq, KvQuant::PlanarK) {
            tracing::warn!(
                arch = arch_class,
                kv_quant = ?kq,
                "rejecting PlanarK (K-axis PlanarQuant 4-bit) on Qwen MoE — PPL disaster path"
            );
            return Err(ResolveError::QwenMoePlanarKRejected);
        }
        // Contract A.y — IsoQuant K-side codecs are surfaced via a
        // dedicated error so the diagnostic names the variant. Runs BEFORE the
        // generic `k_below_8bit → QwenMoeKBitsTooLow(4)` fallthrough.
        if matches!(
            kq,
            KvQuant::Iso3Sym | KvQuant::Iso4Sym | KvQuant::IsoKOnly3 | KvQuant::IsoKOnly4
        ) {
            tracing::warn!(
                arch = arch_class,
                kv_quant = ?kq,
                "rejecting iso K-side codec on Qwen MoE — PPL disaster path"
            );
            return Err(ResolveError::QwenMoeIsoKRejected {
                variant: format!("{kq}"),
            });
        }
        // Contract A.y — rotor K-side codecs use a dedicated error
        // for the same reason (variant-named diagnostic). Runs after iso K-side
        // guard (no overlap) and before the generic fallthrough.
        if matches!(
            kq,
            KvQuant::Rotor3Sym
                | KvQuant::Rotor4Sym
                | KvQuant::RotorKOnly3
                | KvQuant::RotorKOnly4
                | KvQuant::RotorK3Asym { .. }
                | KvQuant::RotorK4Asym { .. }
        ) {
            tracing::warn!(
                arch = arch_class,
                kv_quant = ?kq,
                "rejecting rotor K-side codec on Qwen MoE — PPL disaster path"
            );
            return Err(ResolveError::QwenMoeRotorKRejected {
                variant: format!("{kq}"),
            });
        }
        // Contract A.y — TurboSym3 (symmetric WHT-3 K+V) is K-side
        // 3-bit on Qwen MoE — rejected with a dedicated error so the diagnostic
        // names the variant. Runs after rotor guard and before `k_below_8bit`.
        if matches!(kq, KvQuant::TurboSym3) {
            tracing::warn!(
                arch = arch_class,
                kv_quant = ?kq,
                "rejecting TurboSym3 (symmetric K+V 3-bit) on Qwen MoE — PPL disaster path"
            );
            return Err(ResolveError::QwenMoeTurboKRejected {
                variant: format!("{kq}"),
            });
        }
        // Symmetric WHT-4 K + tq4 V (`KvQuant::TurboSym4`) is
        // the PPL-218→8641 disaster path on Qwen MoE (CLAUDE.md hard rule 6).
        // Surface as `QwenMoeKBitsTooLow(4)` so the error class is uniform with
        // the existing Mixed K<8 rejection — same exit code, same hint text.
        if kq.k_below_8bit() {
            tracing::warn!(
                arch = arch_class,
                kv_quant = ?kq,
                "rejecting low-K-bit codec on Qwen MoE — PPL disaster path"
            );
            return Err(ResolveError::QwenMoeKBitsTooLow(4));
        }
    }

    // General (arch-agnostic) Metal-vs-CPU classification. Codecs whose KV
    // encode + dequant run on the CPU on the default hot path (the iso / rotor
    // families) are honestly surfaced here with a loud structured warn so the
    // 30–60× cost is never silent. These codecs still produce correct output
    // and are not rejected — only flagged.
    if let Some(reason) = kq.cpu_hot_path_reason() {
        tracing::warn!(
            arch = arch_class,
            kv_quant = %kq,
            reason,
            "KV codec runs its encode + dequant on CPU on the default hot path — \
             expect a slow first forward and decode that slows as KV grows. \
             This is NOT a Metal kernel. \
             Pick a Metal codec (k8v4 / k8v8 / planar / rot_k_tq4v) to avoid this."
        );
    }

    Ok(())
}

// ── resolve ───────────────────────────────────────────────────────────────────

/// Resolve a user-supplied [`CacheTypeSpec`] against a [`ResolverContext`] and
/// a base `auto` [`KvQuant`] (typically from `KvCacheBuilder::resolve_default`).
///
/// Steps, in order:
/// 1. `head_dim` required (else `HeadDimUnknown`).
/// 2. K-side rotation rejected (`KSideRotationCodec`).
/// 3. Each non-`Auto` side checked for affine group-divisibility (§D6.1) and
/// 4. `tq4` on V requires `head_dim ∈ {128, 256}` (§D6.5).
/// 5. Decompose `Auto` sides via [`decompose_auto`], overriding only the user-
/// 6. Resolve `(k, v)` → `KvQuant` via [`combo_to_kv_quant`] (which holds the
/// 7. Re-validate via [`validate_resolved`] (post-decompose §D6.4 check).
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn resolve(
    spec: CacheTypeSpec,
    ctx: ResolverContext<'_>,
    auto: KvQuant,
) -> Result<KvQuant, ResolveError> {
    // (a) head_dim required.
    let head_dim = ctx.head_dim.ok_or(ResolveError::HeadDimUnknown)?;

    // (b) V-side rotation codecs (tq4, planar4) are rejected on K (V-only).
    // `rot_k` is the ONE K-side rotation codec — it lifts this guard for
    // itself only, and requires a power-of-two head_dim (Hadamard rotation).
    match spec.k {
        CacheType::Tq4 => return Err(ResolveError::KSideRotationCodec(CacheType::Tq4.tag())),
        CacheType::Planar4 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Planar4.tag()));
        }
        // Planar3 is V-side only (like Planar4).
        CacheType::Planar3 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Planar3.tag()));
        }
        // Iso3 is V-side only.
        CacheType::Iso3 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Iso3.tag()));
        }
        // Iso4 is V-side only (mirrors Iso3).
        CacheType::Iso4 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Iso4.tag()));
        }
        // Rotor3 is V-side only.
        CacheType::Rotor3 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Rotor3.tag()));
        }
        // Rotor4 is V-side only (mirrors Rotor3).
        CacheType::Rotor4 => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Rotor4.tag()));
        }
        // Turbo3Tcq is V-side only.
        CacheType::Turbo3Tcq => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Turbo3Tcq.tag()));
        }
        // Turbo2Tcq is V-side only (mirrors Turbo3Tcq).
        CacheType::Turbo2Tcq => {
            return Err(ResolveError::KSideRotationCodec(CacheType::Turbo2Tcq.tag()));
        }
        CacheType::RotK if !head_dim.is_power_of_two() => {
            return Err(ResolveError::RotKHeadDimNotPow2(head_dim));
        }
        _ => {}
    }
    // `rot_k` is K-side only — reject it on V.
    if matches!(spec.v, CacheType::RotK) {
        return Err(ResolveError::RotKVSide);
    }

    // (c) Affine invariants on each non-Auto side.
    check_affine_invariants(spec.k, head_dim)?;
    check_affine_invariants(spec.v, head_dim)?;

    // (d) Codec-specific: tq4 on V requires head_dim ∈ {128, 256}.
    if matches!(spec.v, CacheType::Tq4) && !matches!(head_dim, 128 | 256) {
        return Err(ResolveError::Tq4UnsupportedHeadDim(head_dim));
    }

    // (e) Decompose Auto, overriding only user-specified sides.
    let (auto_k, auto_v) = decompose_auto(auto);
    let k = if matches!(spec.k, CacheType::Auto) {
        auto_k
    } else {
        spec.k
    };
    let v = if matches!(spec.v, CacheType::Auto) {
        auto_v
    } else {
        spec.v
    };

    // (f) Map to KvQuant. Note the asymmetric-auto coercion guard lives in
    // combo_to_kv_quant — only (Q8G128, Tq4) coerces to K8V4, never (Q8G64, Tq4).
    let kq = combo_to_kv_quant(k, v)?;

    // (h) Re-validate the final concrete KvQuant.
    validate_resolved(ctx.arch_class, &kq)?;

    Ok(kq)
}

/// §D6.1 + §D6.2 — group-divisibility and MLX bit-packing for affine codecs.
///
/// No-op for non-affine codecs (`Auto`, `Bf16`, `Tq4`, `Planar4`).
fn check_affine_invariants(ct: CacheType, head_dim: usize) -> Result<(), ResolveError> {
    let (Some(bits), Some(group_size)) = (ct.bits(), ct.group_size()) else {
        return Ok(());
    };
    // §D6.1
    if !head_dim.is_multiple_of(group_size) {
        return Err(ResolveError::GroupSizeNotDivisible {
            head_dim,
            group_size,
        });
    }
    // §D6.2 — MLX bit-packing: head_dim % (32 / bits) == 0 for bits ∈ {2..8}.
    if (2..=8).contains(&bits) {
        let pack = 32usize / bits as usize;
        if pack > 0 && !head_dim.is_multiple_of(pack) {
            return Err(ResolveError::MlxBitPackingViolation { head_dim, bits });
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "cache_type_tests.rs"]
mod cache_type_tests;
