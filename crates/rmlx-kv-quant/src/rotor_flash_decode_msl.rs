// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel
// input data (dims arrays) for MSL dispatch.
#![allow(unsafe_code)]

//! rotor_flash_decode: fused QK over rotor-quant K + online softmax + bf16-V SV
//! in a single MSL pass per (B, H, tile).
//!
//! # What this is
//!
//! A FlashAttention-style decode over the RotorQuant K store. Pre-softmax
//! scores, online softmax, and the SV product all run in one threadgroup pass
//! per (batch, head, tile). K is read directly from the
//! [`crate::storage::QuantRotorK3`] / [`crate::storage::QuantRotorK4`] packed
//! GPU buffers (codes + per-group scales + per-token L2 norms + the static
//! per-(layer, head) rotor table) — no bf16 / f32 K is ever materialised, and
//! nothing restages through the host.
//!
//! Without this kernel the rotor K-only decode path calls
//! `QuantRotorK{3,4}::dequant()` — a full-prefix **CPU** dequant plus re-upload
//! on every decode step, i.e. O(seq) host work per token with the GPU idle.
//!
//! Two Metal dispatches per call:
//! * **Pass 1** (`rmlx_rotor_flash_decode_p1_b{3,4}`): per-tile threadgroups
//!   compute partial outputs + per-tile LSE state `(tile_max, tile_sum_exp)`.
//!   Each threadgroup runs `head_dim` threads over `TILE_SIZE` tokens.
//! * **Pass 2** (`rmlx_rotor_flash_decode_p2`): one threadgroup per `(B, head)`
//!   merges the per-tile partials via log-sum-exp into the final output. The
//!   body is the codec-agnostic `metal/flash_decode_merge_p2.metal`, shared
//!   with `planar_flash_decode_msl`.
//!
//! # Bit-width parameterisation
//!
//! `bits ∈ {3, 4}` is a template parameter carried by the **header**, not by a
//! second kernel body: [`build_rotor_flash_header`] emits `RF_BITS` / `RF_MASK`
//! alongside the matching Lloyd-Max codebook, so one
//! `metal/rotor_flash_decode_p1.metal` serves both variants. Selection is
//! explicit — any other `bits` is an `Err`, never a silent fallback to the
//! wrong unpack width.
//!
//! # Reduction + decode structure
//!
//! Pass 1 folds the QK dot with **simdgroup reductions** (`simd_sum`): each
//! simdgroup collapses its 32 lanes with no threadgroup barrier and no
//! idle-lane tree, and thread 0 folds the per-simdgroup partials. The rotor
//! K-decode runs **once per Cl(3,0) block** — the block leader stages all
//! `RF_GROUP_SIZE` grade-1 lanes into threadgroup memory — rather than having
//! every lane recompute the ~64-FMA inverse sandwich.
//!
//! # Reusable K-decode half
//!
//! The rotor decode lives in the header as two MSL functions:
//! `rf_decode_k_group(...)` runs the sandwich once and writes a block's grade-1
//! lanes, and `rf_decode_k_lane(...)` is the thin per-lane wrapper over it. A
//! future quantized-V flash kernel needs the identical K-side decode; emitting
//! them as header functions lets that kernel call them directly instead of
//! copying the sandwich. (MSL bodies in this repo are statement sequences
//! spliced inside a generated kernel signature, so a body cannot define
//! functions — the header is the only place a shared function can live.)
//!
//! # Codec contract
//!
//! Bit-exact with [`crate::rotorquant::rotor3_decode`] /
//! [`crate::rotorquant::rotor4_decode`]:
//!
//! * `codes`  u32 `[B * S_kv * kv_h * n_groups]` — 1 u32 per group of 8 Cl(3,0)
//!   multivector components. Element `e ∈ 0..8` occupies bits
//!   `[e*BITS, e*BITS + BITS)` LSB-first (`mask = 0x7` for 3-bit, `0xF` for
//!   4-bit).
//! * `scales` f32 `[B * S_kv * kv_h * n_groups]` — one f32 per group of 3
//!   head-dim slots.
//! * `norms`  f32 `[B * S_kv * kv_h]` — per-token L2 norm.
//! * `rotors` f32 `[n_groups * 4]` — static per-(layer, head) rotor table in
//!   compact `[s, b12, b13, b23]` form.
//!
//! Buffers are **sequence-major** (`[B, S, kv_h, ...]`) — the canonical flat-KV
//! layout in this crate, matching the codec's chunk-append order.
//!
//! # QJL sideband
//!
//! The optional 1-bit QJL residual is a per-token K-side correction that
//! back-projects through a dense `[head_dim, head_dim]` matrix. Reproducing it
//! per token inside the flash inner loop would require reading the whole
//! projection matrix per token per threadgroup — far more bandwidth than the
//! kernel saves. The dispatcher therefore fires only when QJL is off; with QJL
//! on the caller keeps the CPU dequant path. Same gate the fused-QK shadow path
//! uses.
//!
//! # Single-MLX claim
//!
//! Per CLAUDE.md "Single MLX process per Mac", callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Pattern reference
//!
//! * [`crate::planar_flash_decode_msl`] — two-pass flash-decode shell (grid /
//!   tile / online-softmax broadcast / P2 merge).
//! * [`crate::rotor_fused_qk_msl`] — rotor Cl(3,0) K-decode in MSL, and the
//!   `MUL_TABLE` / codebook header-rendering pattern.
//! * [`crate::rotorquant::rotor3_decode`] — CPU reference (Rust).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::clifford::MV_DIM;
use crate::rotorquant::{n_groups_for, ROTOR3_GROUP_SIZE};
use crate::turboquant::lloyd_gaussian_codebook;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Per-group rotor stride in the rotor table (`[s, b12, b13, b23]`).
const ROTOR_STRIDE: i64 = 4;

/// Multivector component count (Cl(3,0) basis size).
const ROTOR_MV: usize = MV_DIM;

/// Tokens per pass-1 tile. Matches `planar_flash_decode_msl::TILE_SIZE` so the
/// flash kernels share one tuning surface.
pub(crate) const TILE_SIZE: i32 = 64;

/// Maximum supported `head_dim`.
///
/// Sized to the largest production head_dim reached by a rotor-quantized
/// growing layer: Gemma4 e2b/e4b global layers run `head_dim = 512` (their
/// `global_head_dim`), while Bonsai / Qwen3 run 128. The kernel uses
/// static-sized threadgroup arrays of this length; raising it widens SMEM and
/// lowers occupancy.
pub(crate) const ROTOR_FLASH_HEAD_DIM_MAX: i32 = 512;

// ── Dispatch counters ─────────────────────────────────────────────────────────

/// Incremented once per `rotor_flash_decode_sdpa::<3>` P1 enqueue.
static ROTOR3_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Incremented once per `rotor_flash_decode_sdpa::<4>` P1 enqueue.
static ROTOR4_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Process-lifetime count of rotor3 flash-decode P1 dispatches.
///
/// Tests assert `delta > 0` to prove the MSL kernel actually fired rather than
/// the caller silently falling back to the CPU dequant path. Production code
/// does not consult this counter.
pub fn rotor3_flash_decode_dispatch_count() -> u64 {
    ROTOR3_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

/// Process-lifetime count of rotor4 flash-decode P1 dispatches.
pub fn rotor4_flash_decode_dispatch_count() -> u64 {
    ROTOR4_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

/// Combined process-lifetime count (3-bit + 4-bit).
pub fn rotor_flash_decode_dispatch_count() -> u64 {
    rotor3_flash_decode_dispatch_count() + rotor4_flash_decode_dispatch_count()
}

// ── MSL header builder ────────────────────────────────────────────────────────

/// Render the Cl(3,0) `MUL_TABLE` (8×8 `(target, sign)` pairs) into two MSL
/// `constant` lookup arrays: `MUL_T[64]` (target indices) and `MUL_S[64]`
/// (signs as f32 ±1). Indexed row-major as `[i * 8 + j]`.
///
/// Re-derived from `BASIS_BITS` with the same recipe as
/// `crate::clifford::MUL_TABLE` (that const is private; the table is tiny).
#[allow(
    clippy::indexing_slicing,
    reason = "fixed-size [ROTOR_MV] / [ROTOR_MV * ROTOR_MV] arrays indexed by bounded \
              counters in [0, ROTOR_MV); never panics by construction"
)]
fn render_mul_table() -> String {
    let basis_bits: [u8; ROTOR_MV] = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111];
    let mut bits_to_basis = [0_usize; ROTOR_MV];
    for (i, &b) in basis_bits.iter().enumerate() {
        bits_to_basis[b as usize] = i;
    }
    let mut targets = [0_u32; ROTOR_MV * ROTOR_MV];
    let mut signs = [0_i8; ROTOR_MV * ROTOR_MV];
    for i in 0..ROTOR_MV {
        let bits_i = basis_bits[i];
        for j in 0..ROTOR_MV {
            let bits_j = basis_bits[j];
            let result_bits = bits_i ^ bits_j;
            let mut sign: i8 = 1;
            for k in 0..3 {
                let bit_k = 1_u8 << k;
                if bits_j & bit_k != 0 {
                    for m in (k + 1)..3 {
                        if bits_i & (1_u8 << m) != 0 {
                            sign = -sign;
                        }
                    }
                }
            }
            targets[i * ROTOR_MV + j] = bits_to_basis[result_bits as usize] as u32;
            signs[i * ROTOR_MV + j] = sign;
        }
    }

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Cl(3,0) MUL_TABLE — target basis index + sign for e_I * e_J.\n\
         // Indexed row-major: MUL_T[i*8+j], MUL_S[i*8+j].\n\
         // Bit-exact with crate::clifford::MUL_TABLE (re-derived from BASIS_BITS).\n\
         constant uint RF_MUL_T[64] = {{\n"
    );
    for i in 0..ROTOR_MV {
        let _ = write!(s, "    ");
        for j in 0..ROTOR_MV {
            let t = targets[i * ROTOR_MV + j];
            let _ = write!(s, "{t}u");
            if !(i == ROTOR_MV - 1 && j == ROTOR_MV - 1) {
                let _ = write!(s, ", ");
            }
        }
        s.push('\n');
    }
    let _ = writeln!(s, "}};\n\nconstant float RF_MUL_S[64] = {{");
    for i in 0..ROTOR_MV {
        let _ = write!(s, "    ");
        for j in 0..ROTOR_MV {
            let v = f32::from(signs[i * ROTOR_MV + j]);
            let _ = write!(s, "{v:.1}f");
            if !(i == ROTOR_MV - 1 && j == ROTOR_MV - 1) {
                let _ = write!(s, ", ");
            }
        }
        s.push('\n');
    }
    let _ = writeln!(s, "}};");
    s
}

/// Emit the reusable rotor K-decode as two MSL functions.
///
/// `rf_decode_k_group` runs the ~64-FMA inverse sandwich **once** for a Cl(3,0)
/// block and writes all `RF_GROUP_SIZE` of its grade-1 lanes; the flash-decode
/// body calls it once per group (its block leader) instead of once per lane, so
/// the sandwich is not recomputed by every lane of the group.
///
/// `rf_decode_k_lane` is the thin per-lane wrapper a quantized-V flash kernel
/// reuses verbatim: it consumes the rotor store (codes / scales / norms /
/// rotors) plus a token index and a head-dim lane, and returns that lane's
/// dequantised K value. It delegates to `rf_decode_k_group` so the sandwich math
/// lives in exactly one place. Keeping both as header functions (rather than
/// inlining into the pass-1 body) is what makes them callable from another
/// kernel body.
///
/// Mirrors [`crate::rotorquant::rotor3_decode`] step for step: unpack 8
/// `RF_BITS`-bit codes from the group's single u32, centroid lookup × per-group
/// scale, inverse sandwich `R̃ * mv_q * R`, grade-1 extraction, × per-token L2.
fn render_decode_fn() -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Decode a whole Cl(3,0) block of a rotor-quantized token.\n\
         //\n\
         // `tok_idx` indexes the flat sequence-major token stream\n\
         // (`(b * kv_seq + t) * kv_h + kv_h_idx`); `group_id` is the block index\n\
         // in [0, n_groups). Writes the block's RF_GROUP_SIZE grade-1 lanes into\n\
         // `out`. Bit-exact with the CPU rotor{{3,4}}_decode.\n\
         //\n\
         // One decode per group: the sandwich runs once here rather than once\n\
         // per head-dim lane.\n\
         inline void rf_decode_k_group(\n\
         \x20   device const uint*  codes,\n\
         \x20   device const float* scales,\n\
         \x20   device const float* norms,\n\
         \x20   device const float* rotors,\n\
         \x20   uint                tok_idx,\n\
         \x20   uint                n_groups,\n\
         \x20   uint                group_id,\n\
         \x20   thread float*       out) {{\n\
         \x20   uint  word    = codes[tok_idx * n_groups + group_id];\n\
         \x20   float k_scale = scales[tok_idx * n_groups + group_id];\n\
         \n\
         \x20   uint  rotor_base = group_id * 4u;\n\
         \x20   float rs         = rotors[rotor_base + 0u];\n\
         \x20   float rb12       = rotors[rotor_base + 1u];\n\
         \x20   float rb13       = rotors[rotor_base + 2u];\n\
         \x20   float rb23       = rotors[rotor_base + 3u];\n\
         \n\
         \x20   // Unpack 8 RF_BITS-bit codes -> centroid x scale -> mv_q[0..8].\n\
         \x20   float mv_q[8];\n\
         \x20   for (uint e = 0u; e < 8u; ++e) {{\n\
         \x20       uint idx = (word >> (e * RF_BITS)) & RF_MASK;\n\
         \x20       mv_q[e]  = RF_CB[idx] * k_scale;\n\
         \x20   }}\n\
         \n\
         \x20   // Inverse sandwich: restored = R~ * mv_q * R.\n\
         \x20   //\n\
         \x20   // Rotor compact form r = (rs, rb12, rb13, rb23) sits at dense MV\n\
         \x20   // positions [0, 4, 5, 6]. The Clifford reverse R~ flips the three\n\
         \x20   // bivector signs. gp(a, b)[k] = sum over (i, j) of\n\
         \x20   // a[i] * b[j] * RF_MUL_S[i*8+j] where RF_MUL_T[i*8+j] == k.\n\
         \x20   //\n\
         \x20   // Step A: tmp = R~ * mv_q (R~ has 4 non-zero entries).\n\
         \x20   float rbar[8] = {{rs, 0.0f, 0.0f, 0.0f, -rb12, -rb13, -rb23, 0.0f}};\n\
         \x20   float tmp[8];\n\
         \x20   for (uint k = 0u; k < 8u; ++k) {{\n\
         \x20       tmp[k] = 0.0f;\n\
         \x20   }}\n\
         \x20   uint sparse_i[4] = {{0u, 4u, 5u, 6u}};\n\
         \x20   for (uint a = 0u; a < 4u; ++a) {{\n\
         \x20       uint  i  = sparse_i[a];\n\
         \x20       float ri = rbar[i];\n\
         \x20       for (uint j = 0u; j < 8u; ++j) {{\n\
         \x20           tmp[RF_MUL_T[i * 8u + j]] += ri * mv_q[j] * RF_MUL_S[i * 8u + j];\n\
         \x20       }}\n\
         \x20   }}\n\
         \n\
         \x20   // Step B: restored = tmp * R (R has 4 non-zero entries).\n\
         \x20   float r_dense[8] = {{rs, 0.0f, 0.0f, 0.0f, rb12, rb13, rb23, 0.0f}};\n\
         \x20   float restored[8];\n\
         \x20   for (uint k = 0u; k < 8u; ++k) {{\n\
         \x20       restored[k] = 0.0f;\n\
         \x20   }}\n\
         \x20   uint sparse_j[4] = {{0u, 4u, 5u, 6u}};\n\
         \x20   for (uint i = 0u; i < 8u; ++i) {{\n\
         \x20       float ti = tmp[i];\n\
         \x20       for (uint c = 0u; c < 4u; ++c) {{\n\
         \x20           uint j = sparse_j[c];\n\
         \x20           restored[RF_MUL_T[i * 8u + j]] += ti * r_dense[j] * RF_MUL_S[i * 8u + j];\n\
         \x20       }}\n\
         \x20   }}\n\
         \n\
         \x20   // Grade-1 lives at MV indices 1..=RF_GROUP_SIZE; rescale by L2.\n\
         \x20   float k_norm = norms[tok_idx];\n\
         \x20   for (uint e = 0u; e < RF_GROUP_SIZE; ++e) {{\n\
         \x20       out[e] = restored[e + 1u] * k_norm;\n\
         \x20   }}\n\
         }}\n\
         \n\
         // Decode one head-dim lane of a rotor-quantized token.\n\
         //\n\
         // Thin wrapper over rf_decode_k_group: resolves the lane's block, then\n\
         // returns that lane's grade-1 component. `lane` is the head-dim slot in\n\
         // [0, head_dim). Shared surface: a quantized-V flash kernel calls this\n\
         // unchanged.\n\
         inline float rf_decode_k_lane(\n\
         \x20   device const uint*  codes,\n\
         \x20   device const float* scales,\n\
         \x20   device const float* norms,\n\
         \x20   device const float* rotors,\n\
         \x20   uint                tok_idx,\n\
         \x20   uint                n_groups,\n\
         \x20   uint                lane) {{\n\
         \x20   // Each group of RF_GROUP_SIZE head-dim slots is one Cl(3,0) block.\n\
         \x20   uint group_id_in_head = lane / RF_GROUP_SIZE;\n\
         \x20   uint lane_in_group    = lane - group_id_in_head * RF_GROUP_SIZE;\n\
         \x20   float g[RF_GROUP_SIZE];\n\
         \x20   rf_decode_k_group(codes, scales, norms, rotors, tok_idx, n_groups,\n\
         \x20                     group_id_in_head, g);\n\
         \x20   return g[lane_in_group];\n\
         }}\n",
    );
    s
}

/// Build the MSL header for the rotor flash-decode pass-1 kernel.
///
/// Emits, in order: the `RF_BITS` / `RF_MASK` unpack parameters, the Lloyd-Max
/// N(0,1) codebook for `bits`, the Cl(3,0) multiplication table, the shared
/// `rf_decode_k_lane` function, and the tile / head-dim sizing macros.
///
/// # Errors
///
/// Returns [`Error::Quant`] for any `bits` outside `{3, 4}` and for a codebook
/// whose length disagrees with `2^bits`.
pub(crate) fn build_rotor_flash_header(bits: u8) -> Result<String> {
    let (mask, n_entries) = match bits {
        3 => (0x7_u32, 8_usize),
        4 => (0xF_u32, 16_usize),
        _ => {
            return Err(Error::Quant(format!(
                "rotor_flash_decode: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    let cb = lloyd_gaussian_codebook(bits)?;
    if cb.len() != n_entries {
        return Err(Error::Quant(format!(
            "rotor_flash_decode: codebook length {got} != 2^{bits} = {n_entries}",
            got = cb.len(),
        )));
    }

    let cb_hex: Vec<String> = cb
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Rotor flash-decode header — unpack params + Lloyd-Max N(0,1)\n\
         // codebook + Cl(3,0) MUL_TABLE + the shared per-lane K decode.\n\
         // BITS = {bits} (codebook = {n_entries} entries, mask = 0x{mask:X}).\n\
         // Bit-exact with crate::clifford::MUL_TABLE + \
         lloyd_gaussian_codebook({bits}).\n"
    );
    // RF_GROUP_SIZE = grade-1 slots per Cl(3,0) block (codec constant, same for
    // both bit widths).
    let gs = ROTOR3_GROUP_SIZE;
    let _ = write!(
        s,
        "\n#define RF_BITS {bits}u\n#define RF_MASK 0x{mask:X}u\n\
         #define RF_GROUP_SIZE {gs}u\n"
    );
    let _ = write!(
        s,
        "\nconstant float RF_CB[{n_entries}] = {{\n{cb_join}\n}};\n",
        cb_join = cb_hex.join(",\n")
    );
    s.push_str(&render_mul_table());
    s.push_str(&render_decode_fn());
    let _ = write!(
        s,
        "\n#define RF_TILE_SIZE {TILE_SIZE}u\n#define RF_HEAD_DIM_MAX {ROTOR_FLASH_HEAD_DIM_MAX}\n"
    );
    Ok(s)
}

// ── MSL sources ───────────────────────────────────────────────────────────────
//
// Grid: (n_tiles * head_dim, B * n_q_heads, 1).  Threadgroup: (head_dim, 1, 1).
//
// P1 buffer layout (must match `add_input` order in `rotor_flash_decode_sdpa`):
// K buffers are SEQUENCE-major (`[B, S, kv_h, ...]`); V stays head-major.
// 0. query     : f32  [B * n_q_heads * head_dim]
// 1. codes     : u32  [B * kv_seq * kv_h * n_groups]
// 2. scales    : f32  [B * kv_seq * kv_h * n_groups]
// 3. norms     : f32  [B * kv_seq * kv_h]
// 4. rotors    : f32  [n_groups * 4]
// 5. v_flat    : bf16 / f16 / f32 [B * kv_h * kv_seq * head_dim] (native dtype)
// 6. mask_flat : f32  [B * n_q_heads * kv_seq] or [1] dummy when no mask
// 7. scale_arr : f32  [1]
// 8. dims      : u32  [8] — {head_dim, kv_seq, n_bh, kv_h, heads_per_kv,
//                            n_tiles, has_mask, n_groups}
//
// P1 outputs:
// 0. partial_o     : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max      : f32 [n_tiles * n_bh]
// 2. tile_sum_exp  : f32 [n_tiles * n_bh]
//
// One body serves both bit widths — the unpack width arrives via the header's
// RF_BITS / RF_MASK.
const P1_SOURCE: &str = include_str!("metal/rotor_flash_decode_p1.metal");

// P2 buffer layout:
// 0. partial_o    : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max     : f32 [n_tiles * n_bh]
// 2. tile_sum_exp : f32 [n_tiles * n_bh]
// 3. dims_p2      : u32 [3] — {head_dim, n_tiles, n_bh}
// Output:
// 0. dst          : f32 [n_bh * head_dim]
//
// Codec-agnostic; shared with `planar_flash_decode_msl`.
const P2_SOURCE: &str = include_str!("metal/flash_decode_merge_p2.metal");

// ── Kernel singletons (one P1 per BITS variant) ───────────────────────────────

static P1_KERNEL_B3: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P1_KERNEL_B4: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

fn p1_kernel(bits: u8) -> Result<&'static MetalKernel> {
    let (cell, name) = match bits {
        3 => (&P1_KERNEL_B3, "rmlx_rotor_flash_decode_p1_b3"),
        4 => (&P1_KERNEL_B4, "rmlx_rotor_flash_decode_p1_b4"),
        _ => {
            return Err(Error::Quant(format!(
                "rotor_flash_decode: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    cell.get_or_init(|| {
        let header = build_rotor_flash_header(bits).map_err(|e| format!("{e}"))?;
        MetalKernel::new(
            name,
            &header,
            P1_SOURCE,
            &[
                "query",
                "codes",
                "scales",
                "norms",
                "rotors",
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
    .map_err(|e| {
        Error::Mlx(format!(
            "rotor_flash_decode P1(bits={bits}) kernel init: {e}"
        ))
    })
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_rotor_flash_decode_p2",
                "", // No header — P2 is generic over (head_dim, n_tiles, n_bh).
                P2_SOURCE,
                &["partial_o", "tile_max", "tile_sum_exp", "dims_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("rotor_flash_decode P2 kernel init: {e}")))
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the rotor flash-decode kernel (fused QK over rotor-quant K + online
/// softmax + bf16-V SV).
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`.
/// * `k_codes`   — packed rotor codes, flat `u32 [B * kv_seq * kv_h * n_groups]`
///   (sequence-major).
/// * `k_scales`  — per-group f32 scales, same flat length as `k_codes`.
/// * `k_norms`   — per-token f32 L2 norms, flat `[B * kv_seq * kv_h]`.
/// * `k_rotors`  — static per-(layer, head) rotor table, flat `[n_groups * 4]`
///   f32 in `[s, b12, b13, b23]` order.
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
///   (`head_dim` not a power of two, above [`ROTOR_FLASH_HEAD_DIM_MAX`],
///   non-positive shapes), an unsupported V dtype, or grid overflow.
/// * [`Error::Mlx`] for kernel build / dispatch failures.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn rotor_flash_decode_sdpa<const BITS: u8>(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: &Array,
    k_rotors: &Array,
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
            "rotor_flash_decode: unsupported BITS={BITS} (only 3 and 4)"
        )));
    }
    if head_dim <= 0 {
        return Err(Error::Quant(format!(
            "rotor_flash_decode: head_dim={head_dim} must be positive"
        )));
    }
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "rotor_flash_decode: head_dim={head_dim} must be a power of two for the \
             tree reduction; caller falls back to the CPU dequant path"
        )));
    }
    if head_dim > ROTOR_FLASH_HEAD_DIM_MAX {
        let max = ROTOR_FLASH_HEAD_DIM_MAX;
        return Err(Error::Quant(format!(
            "rotor_flash_decode: head_dim={head_dim} exceeds ROTOR_FLASH_HEAD_DIM_MAX={max}; \
             raise the static threadgroup-array sizes in metal/rotor_flash_decode_p1.metal \
             to support it"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "rotor_flash_decode: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "rotor_flash_decode: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }

    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;
    let n_tiles = (kv_seq + TILE_SIZE - 1) / TILE_SIZE;
    let n_groups = n_groups_for(head_dim as usize);
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
    let rotors_total: i64 = n_groups_i64 * ROTOR_STRIDE;

    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[codes_total as i32], device)?;
    let norms_flat = k_norms.reshape(&[tok_count as i32], device)?;
    let rotors_flat = k_rotors.reshape(&[rotors_total as i32], device)?;

    // ── V flat — keep native dtype (bf16 / f16 / f32) ─────────────────────
    let v_total: i64 = tok_count * i64::from(head_dim);
    let v_flat = match v.dtype() {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => v.reshape(&[v_total as i32], device)?,
        Dtype::U8 | Dtype::U32 | Dtype::I32 => {
            let dt = v.dtype();
            return Err(Error::Quant(format!(
                "rotor_flash_decode: V dtype must be F32 / Bf16 / F16, got {dt:?}"
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
            .map_err(|e| Error::Mlx(format!("rotor_flash_decode dummy mask: {e}")))?
    };

    // ── scale_arr ─────────────────────────────────────────────────────────
    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };

    // ── dims (8 u32) ──────────────────────────────────────────────────────
    // `bits` is not carried — the dispatcher selects the per-BITS kernel and
    // its header supplies RF_BITS / RF_MASK.
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
            .map_err(|e| Error::Mlx(format!("rotor_flash_decode dims: {e}")))?
    };

    // Materialise inputs to flush any pending lazy ops before kernel dispatch.
    // The MSL kernels read by raw linear offset and ignore MLX lazy-transpose
    // strides, so a pending permutation must be resolved here.
    q_f32.eval()?;
    codes_flat.eval()?;
    scales_flat.eval()?;
    norms_flat.eval()?;
    rotors_flat.eval()?;
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
    inv_p1.add_input(&rotors_flat)?;
    inv_p1.add_input(&v_flat)?;
    inv_p1.add_input(&mask_flat)?;
    inv_p1.add_input(&scale_arr)?;
    inv_p1.add_input(&dims_arr)?;

    let partial_o_len: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(head_dim);
    let tile_meta_len: i64 = i64::from(n_tiles) * i64::from(n_bh);
    if partial_o_len > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "rotor_flash_decode: partial_o length {partial_o_len} exceeds i32::MAX"
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
            "rotor_flash_decode: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    inv_p1.set_grid(grid_x as i32, grid_y as i32, 1)?;
    inv_p1.set_thread_group(head_dim, 1, 1)?;

    // Counter increment at the actual P1 enqueue point — after every
    // validation gate, immediately before `.apply()`.
    if BITS == 3 {
        ROTOR3_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        ROTOR4_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
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
        "rotor_flash_decode_sdpa: dispatch"
    );

    let mut p1_outs = kern_p1.apply(inv_p1, device)?;
    if p1_outs.len() < 3 {
        return Err(Error::Mlx(
            "rotor_flash_decode P1: expected 3 outputs".into(),
        ));
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
            .map_err(|e| Error::Mlx(format!("rotor_flash_decode dims_p2: {e}")))?
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
            "rotor_flash_decode P2: grid x {p2_grid_x} exceeds i32::MAX"
        )));
    }
    inv_p2.set_grid(p2_grid_x as i32, 1, 1)?;
    inv_p2.set_thread_group(head_dim, 1, 1)?;

    let mut p2_outs = kern_p2.apply(inv_p2, device)?;
    if p2_outs.is_empty() {
        return Err(Error::Mlx(
            "rotor_flash_decode P2: expected 1 output".into(),
        ));
    }
    let dst_flat = p2_outs.remove(0);

    // Reshape to canonical SDPA output.
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}

#[cfg(test)]
#[path = "rotor_flash_decode_msl_tests.rs"]
mod rotor_flash_decode_msl_tests;
