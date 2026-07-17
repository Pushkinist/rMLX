// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel
// input data (dims arrays) for MSL dispatch.
#![allow(unsafe_code)]

//! iso_flash_decode: fused QK over iso-quant K + online softmax + bf16-V SV
//! in a single MSL pass per (B, H, tile).
//!
//! # What this is
//!
//! A FlashAttention-style decode over the IsoQuant K store. Pre-softmax
//! scores, online softmax, and the SV product all run in one threadgroup pass
//! per (batch, head, tile). K is read directly from the
//! [`crate::storage::QuantIsoK3`] / [`crate::storage::QuantIsoK4`] packed GPU
//! ring (codes + per-group scales + per-token L2 norms) — no bf16 / f32 K is
//! ever materialised, and nothing restages through the host.
//!
//! Without this kernel the iso K-only decode path calls
//! `QuantIsoK{3,4}::dequant()` — a full-prefix **CPU** dequant plus re-upload
//! on every decode step, i.e. O(seq) host work per token with the GPU idle.
//!
//! Two Metal dispatches per call:
//! * **Pass 1** (`rmlx_iso_flash_decode_p1_b{3,4}`): per-tile threadgroups
//!   compute partial outputs + per-tile LSE state `(tile_max, tile_sum_exp)`.
//!   Each threadgroup runs `head_dim` threads over `TILE_SIZE` tokens.
//! * **Pass 2** (`rmlx_iso_flash_decode_p2`): one threadgroup per `(B, head)`
//!   merges the per-tile partials via log-sum-exp into the final output. The
//!   body is the codec-agnostic `metal/flash_decode_merge_p2.metal`, shared
//!   with `rotor_flash_decode_msl` / `planar_flash_decode_msl`.
//!
//! # Bit-width parameterisation
//!
//! `bits ∈ {3, 4}` is a template parameter carried by the **header**, not by a
//! second kernel body: [`build_iso_flash_header`] emits `IF_BITS` / `IF_MASK`
//! alongside the matching Lloyd-Max codebook, so one
//! `metal/iso_flash_decode_p1.metal` serves both variants. Selection is
//! explicit — any other `bits` is an `Err`, never a silent fallback to the
//! wrong unpack width.
//!
//! # Reusable K-decode half
//!
//! The per-lane iso decode lives in the header as the MSL function
//! `if_decode_k_lane(...)` rather than inline in the body — the same split
//! `rotor_flash_decode_msl` uses. A future quantized-V flash kernel needs the
//! identical quaternion decode against the V store's `(codes, scales, norms)`
//! triple; emitting it as a header function lets that kernel call it directly
//! instead of copying the Hamilton product. (MSL bodies in this repo are
//! statement sequences spliced inside a generated kernel signature, so a body
//! cannot define functions — the header is the only place a shared function
//! can live.)
//!
//! Unlike the fused-QK kernel's phrasing of the same math, this one is
//! **self-contained per lane**: it unpacks all four of its group's codes from
//! the group's single u32 and applies the Hamilton product in registers. The
//! fused-QK kernel instead stages the group's rotated centroids through
//! threadgroup memory and reads its neighbours' lanes after a barrier. A
//! barrier per token inside a flash inner loop would serialise the tile, and
//! the redundant work it saves is 3 extra codebook lookups — so the register
//! form is both faster here and callable from any kernel without imposing a
//! threadgroup layout on it.
//!
//! # Codec contract
//!
//! Bit-exact with [`crate::isoquant::iso_decode_fast`]:
//!
//! * `codes`  u32 `[B * S_kv * kv_h * n_groups]` — 1 u32 per group of 4
//!   quaternion components. Element `e ∈ 0..4` occupies bits
//!   `[e*BITS, e*BITS + BITS)` LSB-first (`mask = 0x7` for 3-bit, `0xF` for
//!   4-bit). Both widths fit one group in one word
//!   (`words_per_group = ceil(4 / (32 / BITS)) = 1`).
//! * `scales` f32 `[B * S_kv * kv_h * n_groups]` — one f32 per group of 4.
//! * `norms`  f32 `[B * S_kv * kv_h]` — per-token L2 norm.
//!
//! Buffers are **sequence-major** (`[B, S, kv_h, ...]`) — the canonical flat-KV
//! layout in this crate, matching the codec's chunk-append order.
//!
//! # Fixed quaternion
//!
//! The iso codec rotates every group by the same golden-ratio unit quaternion
//! ([`crate::isoquant::FIXED_QUAT`]); `iso_encode_fast` writes that constant
//! into every slot of its per-group `quaternions` array. The header therefore
//! bakes `q̄` in as a constant and the ring does not carry the quaternion
//! table at all — storing `n_tokens * n_groups * 4` copies of one constant per
//! decode step would be pure bandwidth. [`crate::isoquant_msl`] does the same.
//!
//! This is a real coupling, not an assumption: if the codec ever emits
//! per-group quaternions (the encoder's own docs float that as future work),
//! this kernel becomes silently wrong rather than merely stale. The
//! dispatcher's [`assert_fixed_quat_blocks`] rejects a store whose quaternions
//! are not `FIXED_QUAT`, so that change fails loudly here instead of decoding
//! K against the wrong rotation.
//!
//! # Single-MLX claim
//!
//! Per CLAUDE.md "Single MLX process per Mac", callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Pattern reference
//!
//! * [`crate::rotor_flash_decode_msl`] — the two-pass flash-decode shell this
//!   is a sibling of (grid / tile / online-softmax broadcast / P2 merge).
//! * [`crate::iso_fused_qk_msl`] — iso quaternion K-decode in MSL, and the
//!   codebook / fixed-quaternion header-rendering pattern.
//! * [`crate::isoquant::iso_decode_fast`] — CPU reference (Rust).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::isoquant::FIXED_QUAT;
use crate::storage::{iso_n_groups_for, ISO_QUAT_BLOCK_SIZE};
use crate::turboquant::lloyd_gaussian_codebook;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Tokens per pass-1 tile. Matches `rotor_flash_decode_msl::TILE_SIZE` so the
/// flash kernels share one tuning surface.
pub(crate) const TILE_SIZE: i32 = 64;

/// Maximum supported `head_dim`.
///
/// Sized to the largest production head_dim reached by an iso-quantized
/// growing layer: Gemma4 e2b/e4b global layers run `head_dim = 512` (their
/// `global_head_dim`), while Bonsai / Qwen3 run 128. The kernel uses
/// static-sized threadgroup arrays of this length; raising it widens SMEM and
/// lowers occupancy.
pub(crate) const ISO_FLASH_HEAD_DIM_MAX: i32 = 512;

// ── Dispatch counters ─────────────────────────────────────────────────────────

/// Incremented once per `iso_flash_decode_sdpa::<3>` P1 enqueue.
static ISO3_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Incremented once per `iso_flash_decode_sdpa::<4>` P1 enqueue.
static ISO4_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Process-lifetime count of iso3 flash-decode P1 dispatches.
///
/// The per-width breakdown of [`iso_flash_decode_dispatch_count`], for a caller
/// that needs to tell the two apart. Production code does not consult it.
pub fn iso3_flash_decode_dispatch_count() -> u64 {
    ISO3_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

/// Process-lifetime count of iso4 flash-decode P1 dispatches. See
/// [`iso3_flash_decode_dispatch_count`].
pub fn iso4_flash_decode_dispatch_count() -> u64 {
    ISO4_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

/// Combined process-lifetime count (3-bit + 4-bit).
///
/// `kvcache::iso_flash_dispatch_tests` asserts a positive delta across decode
/// steps to prove the MSL kernel actually fired, rather than the caller silently
/// falling back to the CPU dequant path this kernel exists to remove.
///
/// The counter is process-global, so a concurrent test can only ever *inflate* a
/// delta: `>=` assertions on it are race-free, `== 0` ones are not. For the
/// negative case assert on the cache's own ring instead (the flash path cannot
/// run without it). Production code does not consult this counter.
pub fn iso_flash_decode_dispatch_count() -> u64 {
    iso3_flash_decode_dispatch_count() + iso4_flash_decode_dispatch_count()
}

// ── MSL header builder ────────────────────────────────────────────────────────

/// Emit the reusable per-lane iso K-decode as an MSL function.
///
/// This is the half a quantized-V flash kernel reuses verbatim: it consumes a
/// packed iso store (codes / scales / norms) plus a token index and a head-dim
/// lane, and returns that lane's dequantised value. Keeping it a header
/// function (rather than inlining it into the pass-1 body) is what makes it
/// callable from another kernel body.
///
/// Mirrors [`crate::isoquant::iso_decode_fast`] step for step: unpack the
/// group's 4 `IF_BITS`-bit codes from its single u32, centroid lookup × the
/// per-group scale, inverse Hamilton product `q̄ * r`, × the per-token L2 norm.
///
/// Note this is a single **left** Hamilton product, not a sandwich — iso
/// encodes with `r = q * v_unit`, so the inverse is `q̄ * r` alone. (The rotor
/// codec's `R̃ * mv * R` sandwich is a different algebra; do not cross the two.)
fn render_decode_fn() -> String {
    // Bound to a local so the group size is an inline `{gs}` capture below.
    let gs = ISO_QUAT_BLOCK_SIZE;
    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Decode one head-dim lane of an iso-quantized token.\n\
         //\n\
         // `tok_idx` indexes the flat sequence-major token stream\n\
         // (`(b * kv_seq + t) * kv_h + kv_h_idx`); `lane` is the head-dim slot\n\
         // in [0, head_dim). Bit-exact with the CPU iso_decode_fast.\n\
         //\n\
         // Self-contained per lane: the group's four codes all live in one u32,\n\
         // so the Hamilton product runs in registers with no threadgroup\n\
         // staging and no barrier — callable from any kernel body.\n\
         //\n\
         // Shared surface: a quantized-V flash kernel calls this unchanged,\n\
         // passing the V store's (codes, scales, norms) instead of K's.\n\
         inline float if_decode_k_lane(\n\
         \x20   device const uint*  codes,\n\
         \x20   device const float* scales,\n\
         \x20   device const float* norms,\n\
         \x20   uint                tok_idx,\n\
         \x20   uint                n_groups,\n\
         \x20   uint                lane) {{\n\
         \x20   // Each group of 4 head-dim slots is one quaternion block.\n\
         \x20   uint group_id_in_head = lane / {gs}u;\n\
         \x20   uint lane_in_group    = lane - group_id_in_head * {gs}u;\n\
         \n\
         \x20   uint  word    = codes[tok_idx * n_groups + group_id_in_head];\n\
         \x20   float k_scale = scales[tok_idx * n_groups + group_id_in_head];\n\
         \n\
         \x20   // Unpack the group's 4 IF_BITS-bit codes -> centroid x scale.\n\
         \x20   float rw = ISO_CB[(word >> (0u * IF_BITS)) & IF_MASK] * k_scale;\n\
         \x20   float rx = ISO_CB[(word >> (1u * IF_BITS)) & IF_MASK] * k_scale;\n\
         \x20   float ry = ISO_CB[(word >> (2u * IF_BITS)) & IF_MASK] * k_scale;\n\
         \x20   float rz = ISO_CB[(word >> (3u * IF_BITS)) & IF_MASK] * k_scale;\n\
         \n\
         \x20   // Inverse rotation: r' = qbar * r, Hamilton product in the\n\
         \x20   // [w, x, y, z] convention. qbar = (ISO_QW, ISO_QCX, ISO_QCY,\n\
         \x20   // ISO_QCZ) is already conjugated by the header.\n\
         \x20   float v_for_lane;\n\
         \x20   if (lane_in_group == 0u) {{\n\
         \x20       v_for_lane = ISO_QW * rw - ISO_QCX * rx - ISO_QCY * ry - ISO_QCZ * rz;\n\
         \x20   }} else if (lane_in_group == 1u) {{\n\
         \x20       v_for_lane = ISO_QW * rx + ISO_QCX * rw + ISO_QCY * rz - ISO_QCZ * ry;\n\
         \x20   }} else if (lane_in_group == 2u) {{\n\
         \x20       v_for_lane = ISO_QW * ry - ISO_QCX * rz + ISO_QCY * rw + ISO_QCZ * rx;\n\
         \x20   }} else {{\n\
         \x20       v_for_lane = ISO_QW * rz + ISO_QCX * ry - ISO_QCY * rx + ISO_QCZ * rw;\n\
         \x20   }}\n\
         \n\
         \x20   return v_for_lane * norms[tok_idx];\n\
         }}\n",
    );
    s
}

/// Build the MSL header for the iso flash-decode pass-1 kernel.
///
/// Emits, in order: the `IF_BITS` / `IF_MASK` unpack parameters, the conjugated
/// fixed quaternion, the Lloyd-Max N(0,1) codebook for `bits`, the shared
/// `if_decode_k_lane` function, and the tile / head-dim sizing macros.
///
/// # Errors
///
/// Returns [`Error::Quant`] for any `bits` outside `{3, 4}` and for a codebook
/// whose length disagrees with `2^bits`.
pub(crate) fn build_iso_flash_header(bits: u8) -> Result<String> {
    let (mask, n_entries) = match bits {
        3 => (0x7_u32, 8_usize),
        4 => (0xF_u32, 16_usize),
        _ => {
            return Err(Error::Quant(format!(
                "iso_flash_decode: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    let cb = lloyd_gaussian_codebook(bits)?;
    if cb.len() != n_entries {
        return Err(Error::Quant(format!(
            "iso_flash_decode: codebook length {got} != 2^{bits} = {n_entries}",
            got = cb.len(),
        )));
    }

    // q̄ = (w, -x, -y, -z) — the inverse of the encode-side rotation.
    let [qw, qx, qy, qz] = FIXED_QUAT;
    let qw_bits = f32::to_bits(qw);
    let qcx_bits = f32::to_bits(-qx);
    let qcy_bits = f32::to_bits(-qy);
    let qcz_bits = f32::to_bits(-qz);

    let cb_hex: Vec<String> = cb
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Iso flash-decode header — unpack params + conjugated golden-ratio\n\
         // fixed quaternion + Lloyd-Max N(0,1) codebook + the shared per-lane\n\
         // K decode.\n\
         // BITS = {bits} (codebook = {n_entries} entries, mask = 0x{mask:X}).\n\
         // Bit-exact with crate::isoquant::FIXED_QUAT + \
         lloyd_gaussian_codebook({bits}).\n"
    );
    let _ = write!(
        s,
        "\n#define IF_BITS {bits}u\n#define IF_MASK 0x{mask:X}u\n"
    );
    let _ = write!(
        s,
        "\nconstant float ISO_QW  = as_type<float>(0x{qw_bits:08X}u);\n\
         constant float ISO_QCX = as_type<float>(0x{qcx_bits:08X}u);\n\
         constant float ISO_QCY = as_type<float>(0x{qcy_bits:08X}u);\n\
         constant float ISO_QCZ = as_type<float>(0x{qcz_bits:08X}u);\n"
    );
    let _ = write!(
        s,
        "\nconstant float ISO_CB[{n_entries}] = {{\n{cb_join}\n}};\n",
        cb_join = cb_hex.join(",\n")
    );
    s.push_str(&render_decode_fn());
    let _ = write!(
        s,
        "\n#define IF_TILE_SIZE {TILE_SIZE}u\n#define IF_HEAD_DIM_MAX {ISO_FLASH_HEAD_DIM_MAX}\n"
    );
    Ok(s)
}

/// Reject a store whose per-group quaternions are not [`FIXED_QUAT`].
///
/// The kernel bakes `q̄` into its header instead of reading the store's
/// quaternion table (see the module docs). That is correct only while the
/// encoder writes the one constant into every slot. This check is what turns a
/// future per-group-quaternion encoder from "silently decodes K against the
/// wrong rotation" into a loud error at the dispatch boundary.
///
/// `quaternions` is the CPU-side per-group table (`n * 4` f32, `[w, x, y, z]`
/// per group) carried by [`crate::storage::quant_iso_v::IsoBlocks`].
///
/// # Errors
///
/// Returns [`Error::Quant`] when the length is not a multiple of 4 or any group
/// differs from [`FIXED_QUAT`].
pub fn assert_fixed_quat_blocks(quaternions: &[f32], what: &str) -> Result<()> {
    if !quaternions.len().is_multiple_of(4) {
        return Err(Error::Quant(format!(
            "{what}: quaternion table length {} is not a multiple of 4",
            quaternions.len()
        )));
    }
    for (g, quat) in quaternions.chunks_exact(4).enumerate() {
        // Bit-equality: the encoder copies FIXED_QUAT verbatim, so anything
        // else means a different rotation, not a rounding difference.
        if quat != FIXED_QUAT.as_slice() {
            return Err(Error::Quant(format!(
                "{what}: group {g} carries quaternion {quat:?}, expected the fixed \
                 golden-ratio quaternion {FIXED_QUAT:?}. The iso flash-decode kernel bakes \
                 the fixed quaternion into its header and does not read a per-group table; \
                 a per-group-quaternion store must not decode through it."
            )));
        }
    }
    Ok(())
}

// ── MSL sources ───────────────────────────────────────────────────────────────
//
// Grid: (n_tiles * head_dim, B * n_q_heads, 1).  Threadgroup: (head_dim, 1, 1).
//
// P1 buffer layout (must match `add_input` order in `iso_flash_decode_sdpa`):
// K buffers are SEQUENCE-major (`[B, S, kv_h, ...]`); V stays head-major.
// 0. query     : f32  [B * n_q_heads * head_dim]
// 1. codes     : u32  [B * kv_seq * kv_h * n_groups]
// 2. scales    : f32  [B * kv_seq * kv_h * n_groups]
// 3. norms     : f32  [B * kv_seq * kv_h]
// 4. v_flat    : bf16 / f16 / f32 [B * kv_h * kv_seq * head_dim] (native dtype)
// 5. mask_flat : f32  [B * n_q_heads * kv_seq] or [1] dummy when no mask
// 6. scale_arr : f32  [1]
// 7. dims      : u32  [8] — {head_dim, kv_seq, n_bh, kv_h, heads_per_kv,
//                            n_tiles, has_mask, n_groups}
//
// P1 outputs:
// 0. partial_o     : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max      : f32 [n_tiles * n_bh]
// 2. tile_sum_exp  : f32 [n_tiles * n_bh]
//
// One body serves both bit widths — the unpack width arrives via the header's
// IF_BITS / IF_MASK.
const P1_SOURCE: &str = include_str!("metal/iso_flash_decode_p1.metal");

// P2 buffer layout:
// 0. partial_o    : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max     : f32 [n_tiles * n_bh]
// 2. tile_sum_exp : f32 [n_tiles * n_bh]
// 3. dims_p2      : u32 [3] — {head_dim, n_tiles, n_bh}
// Output:
// 0. dst          : f32 [n_bh * head_dim]
//
// Codec-agnostic; shared with `rotor_flash_decode_msl` / `planar_flash_decode_msl`.
const P2_SOURCE: &str = include_str!("metal/flash_decode_merge_p2.metal");

// ── Kernel singletons (one P1 per BITS variant) ───────────────────────────────

static P1_KERNEL_B3: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P1_KERNEL_B4: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

fn p1_kernel(bits: u8) -> Result<&'static MetalKernel> {
    let (cell, name) = match bits {
        3 => (&P1_KERNEL_B3, "rmlx_iso_flash_decode_p1_b3"),
        4 => (&P1_KERNEL_B4, "rmlx_iso_flash_decode_p1_b4"),
        _ => {
            return Err(Error::Quant(format!(
                "iso_flash_decode: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    cell.get_or_init(|| {
        let header = build_iso_flash_header(bits).map_err(|e| format!("{e}"))?;
        MetalKernel::new(
            name,
            &header,
            P1_SOURCE,
            &[
                "query",
                "codes",
                "scales",
                "norms",
                "v_flat",
                "mask_flat",
                "scale_arr",
                "dims",
            ],
            &["partial_o", "tile_max", "tile_sum_exp"],
        )
        .map_err(|e| format!("{e}"))
    })
    .as_ref()
    .map_err(|e| Error::Mlx(format!("iso_flash_decode P1(bits={bits}) kernel init: {e}")))
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_iso_flash_decode_p2",
                "", // No header — P2 is generic over (head_dim, n_tiles, n_bh).
                P2_SOURCE,
                &["partial_o", "tile_max", "tile_sum_exp", "dims_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("iso_flash_decode P2 kernel init: {e}")))
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the iso flash-decode kernel (fused QK over iso-quant K + online softmax
/// + bf16-V SV).
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`.
/// * `k_codes`   — packed iso codes, flat `u32 [B * kv_seq * kv_h * n_groups]`
///   (sequence-major).
/// * `k_scales`  — per-group f32 scales, same flat length as `k_codes`.
/// * `k_norms`   — per-token f32 L2 norms, flat `[B * kv_seq * kv_h]`.
/// * `v`         — bf16 / f16 / f32 V, shape `[B, kv_h, kv_seq, head_dim]`.
///   Read in its native dtype; the dispatcher does NOT astype-upcast.
/// * `additive_mask` — optional `f32 [B, n_q_heads, 1, kv_seq]`.
/// * `b`, `kv_h`, `kv_seq`, `head_dim`, `heads_per_kv` — shape metadata.
/// * `scale`     — softmax pre-scale (typically `1/sqrt(head_dim)`).
/// * `device`    — MLX device (must be GPU).
///
/// # Output
///
/// `f32` array of shape `[B, n_q_heads, 1, head_dim]`.
///
/// # Errors
///
/// * [`Error::Quant`] for `BITS` outside `{3, 4}`, shape-contract violations
///   (`head_dim` not a power of two, not a multiple of the quaternion block
///   size, above [`ISO_FLASH_HEAD_DIM_MAX`], non-positive shapes), an
///   unsupported V dtype, or grid overflow.
/// * [`Error::Mlx`] for kernel build / dispatch failures.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn iso_flash_decode_sdpa<const BITS: u8>(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: &Array,
    v: &Array,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Array> {
    if BITS != 3 && BITS != 4 {
        return Err(Error::Quant(format!(
            "iso_flash_decode: unsupported BITS={BITS} (only 3 and 4)"
        )));
    }
    if head_dim <= 0 {
        return Err(Error::Quant(format!(
            "iso_flash_decode: head_dim={head_dim} must be positive"
        )));
    }
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "iso_flash_decode: head_dim={head_dim} must be a power of two for the \
             tree reduction; caller falls back to the CPU dequant path"
        )));
    }
    if head_dim > ISO_FLASH_HEAD_DIM_MAX {
        let max = ISO_FLASH_HEAD_DIM_MAX;
        return Err(Error::Quant(format!(
            "iso_flash_decode: head_dim={head_dim} exceeds ISO_FLASH_HEAD_DIM_MAX={max}; \
             raise the static threadgroup-array sizes in metal/iso_flash_decode_p1.metal \
             to support it"
        )));
    }
    if !(head_dim as usize).is_multiple_of(ISO_QUAT_BLOCK_SIZE) {
        return Err(Error::Quant(format!(
            "iso_flash_decode: head_dim={head_dim} must be a multiple of the quaternion \
             block size {ISO_QUAT_BLOCK_SIZE}"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "iso_flash_decode: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "iso_flash_decode: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }

    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;
    let n_tiles = (kv_seq + TILE_SIZE - 1) / TILE_SIZE;
    let n_groups = iso_n_groups_for(head_dim as usize);
    let n_groups_i64 = n_groups as i64;

    // ── Flatten Q to [n_bh * head_dim] f32 ────────────────────────────────
    let q_total: i64 = i64::from(n_bh) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    // ── Flatten K buffers ─────────────────────────────────────────────────
    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * n_groups_i64;

    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[codes_total as i32], device)?;
    let norms_flat = k_norms.reshape(&[tok_count as i32], device)?;

    // ── V flat — keep native dtype (bf16 / f16 / f32) ─────────────────────
    let v_total: i64 = tok_count * i64::from(head_dim);
    let v_flat = match v.dtype() {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => v.reshape(&[v_total as i32], device)?,
        Dtype::U8 | Dtype::U32 | Dtype::I32 => {
            let dt = v.dtype();
            return Err(Error::Quant(format!(
                "iso_flash_decode: V dtype must be F32 / Bf16 / F16, got {dt:?}"
            )));
        }
    };

    // ── Mask ──────────────────────────────────────────────────────────────
    let (mask_flat, has_mask) = if let Some(m) = additive_mask {
        let flat_len: i64 = i64::from(n_bh) * i64::from(kv_seq);
        let m_f = if m.dtype() == Dtype::F32 {
            m.reshape(&[flat_len as i32], device)?
        } else {
            m.astype(Dtype::F32, device)?
                .reshape(&[flat_len as i32], device)?
        };
        (m_f, 1u32)
    } else {
        let zero_bytes = [0u8; 4];
        Array::from_bytes(&zero_bytes, &[1], Dtype::F32)
            .map(|a| (a, 0u32))
            .map_err(|e| Error::Mlx(format!("iso_flash_decode dummy mask: {e}")))?
    };

    // ── scale_arr ─────────────────────────────────────────────────────────
    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };

    // ── dims (8 u32) ──────────────────────────────────────────────────────
    // `bits` is not carried — the dispatcher selects the per-BITS kernel and
    // its header supplies IF_BITS / IF_MASK.
    let dims_arr = {
        let dims: [u32; 8] = [
            head_dim as u32,
            kv_seq as u32,
            n_bh as u32,
            kv_h as u32,
            heads_per_kv as u32,
            n_tiles as u32,
            has_mask,
            n_groups as u32,
        ];
        // SAFETY:
        // * `dims` is a stack-local `[u32; 8]` fully initialised above.
        // * `u32` has stricter alignment than `u8`, so the cast is sound.
        // * The byte length `8 * 4` equals `size_of::<[u32; 8]>()`.
        // * The borrow is bounded by the enclosing block; `Array::from_bytes`
        //   copies into mlx storage before this scope ends.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(dims.as_ptr().cast::<u8>(), 8 * 4) };
        Array::from_bytes(bytes, &[8], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("iso_flash_decode dims: {e}")))?
    };

    // Materialise inputs to flush any pending lazy ops before kernel dispatch.
    // The MSL kernels read by raw linear offset and ignore MLX lazy-transpose
    // strides, so a pending permutation must be resolved here.
    q_f32.eval()?;
    codes_flat.eval()?;
    scales_flat.eval()?;
    norms_flat.eval()?;
    v_flat.eval()?;
    if has_mask == 1 {
        mask_flat.eval()?;
    }
    scale_arr.eval()?;
    dims_arr.eval()?;

    // ── P1 dispatch ───────────────────────────────────────────────────────
    let kern_p1 = p1_kernel(BITS)?;
    let mut inv_p1 = MetalKernelInvoke::new();
    inv_p1.add_input(&q_f32)?;
    inv_p1.add_input(&codes_flat)?;
    inv_p1.add_input(&scales_flat)?;
    inv_p1.add_input(&norms_flat)?;
    inv_p1.add_input(&v_flat)?;
    inv_p1.add_input(&mask_flat)?;
    inv_p1.add_input(&scale_arr)?;
    inv_p1.add_input(&dims_arr)?;

    let partial_o_len: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(head_dim);
    let tile_meta_len: i64 = i64::from(n_tiles) * i64::from(n_bh);
    if partial_o_len > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "iso_flash_decode: partial_o length {partial_o_len} exceeds i32::MAX"
        )));
    }
    inv_p1.add_output_shape(&[partial_o_len as i32], Dtype::F32)?;
    inv_p1.add_output_shape(&[tile_meta_len as i32], Dtype::F32)?;
    inv_p1.add_output_shape(&[tile_meta_len as i32], Dtype::F32)?;

    // Grid X: n_tiles threadgroups × head_dim threads each.
    let grid_x: i64 = i64::from(n_tiles) * i64::from(head_dim);
    let grid_y: i64 = i64::from(n_bh);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "iso_flash_decode: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    inv_p1.set_grid(grid_x as i32, grid_y as i32, 1)?;
    inv_p1.set_thread_group(head_dim, 1, 1)?;

    // Counter increment at the actual P1 enqueue point — after every
    // validation gate, immediately before `.apply()`.
    if BITS == 3 {
        ISO3_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        ISO4_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    }
    tracing::trace!(
        bits = BITS,
        b,
        n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask,
        n_groups,
        n_tiles,
        "iso_flash_decode_sdpa: dispatch"
    );

    let mut p1_outs = kern_p1.apply(inv_p1, device)?;
    if p1_outs.len() < 3 {
        return Err(Error::Mlx("iso_flash_decode P1: expected 3 outputs".into()));
    }
    let partial_o = p1_outs.remove(0);
    let tile_max = p1_outs.remove(0);
    let tile_sum_exp = p1_outs.remove(0);

    // ── P2 dispatch ───────────────────────────────────────────────────────
    let dims_p2_arr = {
        let dims_p2: [u32; 3] = [head_dim as u32, n_tiles as u32, n_bh as u32];
        // SAFETY: same reasoning as the `dims` cast above — stack-local
        // `[u32; 3]`, `u32` alignment ≥ `u8`, byte length `3 * 4` equals
        // `size_of::<[u32; 3]>()`, and `Array::from_bytes` copies before the
        // borrow ends.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(dims_p2.as_ptr().cast::<u8>(), 3 * 4) };
        Array::from_bytes(bytes, &[3], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("iso_flash_decode dims_p2: {e}")))?
    };

    let kern_p2 = p2_kernel()?;
    let mut inv_p2 = MetalKernelInvoke::new();
    inv_p2.add_input(&partial_o)?;
    inv_p2.add_input(&tile_max)?;
    inv_p2.add_input(&tile_sum_exp)?;
    inv_p2.add_input(&dims_p2_arr)?;

    let dst_len: i64 = i64::from(n_bh) * i64::from(head_dim);
    inv_p2.add_output_shape(&[dst_len as i32], Dtype::F32)?;

    let p2_grid_x: i64 = i64::from(n_bh) * i64::from(head_dim);
    if p2_grid_x > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "iso_flash_decode P2: grid x {p2_grid_x} exceeds i32::MAX"
        )));
    }
    inv_p2.set_grid(p2_grid_x as i32, 1, 1)?;
    inv_p2.set_thread_group(head_dim, 1, 1)?;

    let mut p2_outs = kern_p2.apply(inv_p2, device)?;
    if p2_outs.is_empty() {
        return Err(Error::Mlx("iso_flash_decode P2: expected 1 output".into()));
    }
    let dst_flat = p2_outs.remove(0);

    // Reshape to canonical SDPA output.
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}

#[cfg(test)]
#[path = "iso_flash_decode_msl_tests.rs"]
mod iso_flash_decode_msl_tests;
