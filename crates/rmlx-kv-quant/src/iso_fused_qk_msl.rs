//! IsoQuant K-side fused-QK MSL kernel.
//!
//! # What this is
//!
//! Single generic MSL kernel that consumes the IsoQuant K codec (bits ∈ {3, 4})
//! packed codes (1 u32 per group of 4) + per-group f32 scales + per-token f32
//! L2 norms, applies the conjugate Hamilton product of the fixed golden-ratio
//! unit quaternion ([`crate::isoquant::FIXED_QUAT`]), and accumulates the
//! pre-softmax QK score `[B, n_q_heads, 1, S_kv]` — all without materialising
//! a bf16 / f32 K tensor in HBM.
//!
//! Parameterised by `const BITS: u8` at the Rust call site: two
//! const-function-item dispatchers ([`ISO3_FUSED_QK_FN`] / [`ISO4_FUSED_QK_FN`])
//! select 3-bit (8-entry codebook) or 4-bit (16-entry codebook) kernel
//! variants. Each variant has its own [`OnceLock`]-cached compiled
//! [`MetalKernel`] because the MSL source strings differ (codebook table
//! size + unpack mask / shift).
//!
//! # Codec contract
//!
//! Target [`KvQuant`](crate::quant::KvQuant): `Iso3Sym`, `IsoKOnly3` (BITS=3);
//! `Iso4Sym`, `IsoKOnly4` (BITS=4).  K storage is [`crate::storage::QuantIsoK3`]
//! / [`crate::storage::QuantIsoK4`] with `group_size = 4` (one quaternion block
//! per group).  V side is irrelevant for the fused-QK score — the caller
//! handles V separately.
//!
//! Bit-exact with [`crate::isoquant::iso_decode_fast`]:
//!
//! * `codes`  u32 `[B * kv_h * S_kv * n_groups]` — 1 u32 per group of 4
//!   (`words_per_group = ceil(4 / vals_per_word(BITS)) = 1` for both 3 and 4
//!   bits). Element `e ∈ 0..4` lives at bits `[e*BITS, e*BITS + BITS)` of the
//!   single u32 (LSB-first).
//! * `scales` f32 `[B * kv_h * S_kv * n_groups]` — one f32 per group of 4.
//! * `norms`  f32 `[B * kv_h * S_kv]` — one f32 per token (L2 norm of the
//!   original head-dim row).
//! * Codebook: `lloyd_gaussian_codebook(BITS)` — 8 entries for BITS=3,
//!   16 entries for BITS=4 (same Lloyd-Max N(0,1) tables as turbo3 / turbo4).
//!
//! # Decode (per element)
//!
//! 1. Unpack `BITS`-bit index from the per-group u32 word.
//! 2. Look up centroid in [`crate::turboquant::lloyd_gaussian_codebook`].
//! 3. Multiply by the per-group scale.
//! 4. Within the group of 4, apply the conjugate Hamilton product `r' = q̄ * r`
//!    where `q = FIXED_QUAT` (golden-ratio unit quat).  Each thread reads its
//!    group's 4 rotated centroids from threadgroup SMEM, then writes back its
//!    own lane (one of `[v_w, v_x, v_y, v_z]`).
//! 5. Multiply by the per-token L2 norm.
//!
//! This matches [`crate::isoquant::iso_decode_fast`] exactly:
//! `q_l_conj * (codebook[idx] * scale) → restored * norm`.
//!
//! # Kernel shape
//!
//! Grid `(S_kv, B * n_q_heads, 1)`; threadgroup `(head_dim, 1, 1)`.  Each
//! threadgroup computes one score `out[b, hq, 0, s_kv]`; per-thread handles
//! one element of `head_dim`.  Threadgroup SMEM holds Q (`f32[head_dim]`)
//! plus the dequantized-but-pre-inverse-rotation centroid stream
//! (`f32[head_dim]`), so the inverse Hamilton product can read all four
//! group lanes after one `threadgroup_barrier`.
//!
//! # GQA support
//!
//! Identical to the q8 / turbo3 / turbo4 paths: `n_q_heads = kv_h * heads_per_kv`;
//! thread group maps `(b, hq) -> kv_h_idx = hq / heads_per_kv` to share K.
//!
//! # A.y guard
//!
//! `Iso3Sym` / `Iso4Sym` are K-side ≤ 4-bit — the Qwen-MoE 218→8641 PPL
//! disaster applies.  The guard lives in
//! [`rmlx_models::kv_cache::cache_type::validate_resolved`] and rejects
//! `Qwen3_5MoeForConditionalGeneration + Iso{3,4}Sym` at session start —
//! the kernel does NOT re-check.
//!
//! # Reference
//!
//! * [`crate::turbo_k4_fused_qk_msl`] / [`crate::turbo_k3_fused_qk_msl`] — Exec
//!   C/D dispatcher / threadgroup layout.
//! * [`crate::isoquant_msl`] — bit-exact quaternion + codebook reference; the
//!   conjugate Hamilton product formula (left-multiply by `q̄`) is mirrored
//!   below.  Note: iso decode is a single left-Hamilton product, **not a
//!   sandwich** (`q̄ * r`, not `q̄ * r * q`).
//! * [`crate::isoquant::iso_decode_fast`] — CPU reference (Rust).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::isoquant::FIXED_QUAT;
use crate::turboquant::lloyd_gaussian_codebook;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Quaternion-block group size (4 elements per group).  Matches
/// [`crate::storage::quant_iso_k::ISO_K3_GROUP_SIZE`] and the iso codec.
const ISO_GROUP_SIZE: usize = 4;

// ── Dispatch counters ─────────────────────────────────────────────────────────

/// Incremented once per `iso_fused_qk_sdpa::<3>` invocation that reaches the
/// Metal enqueue point.
static ISO3_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Incremented once per `iso_fused_qk_sdpa::<4>` invocation that reaches the
/// Metal enqueue point.
static ISO4_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Process-lifetime count of iso3 fused-QK dispatches.
pub fn iso3_fused_qk_dispatch_count() -> u64 {
    ISO3_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
}

/// Process-lifetime count of iso4 fused-QK dispatches.
pub fn iso4_fused_qk_dispatch_count() -> u64 {
    ISO4_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
}

// ── MSL header / source builder ───────────────────────────────────────────────

#[allow(
    clippy::expect_used,
    reason = "lloyd_gaussian_codebook(3|4) is always Ok; verified by \
              `turbo_codebook_3bit_has_8_entries_monotonic` test in turboquant_tests.rs"
)]
fn build_iso_fused_qk_header(bits: u8) -> String {
    debug_assert!(bits == 3 || bits == 4, "iso fused-QK: bits must be 3 or 4");
    let cb = lloyd_gaussian_codebook(bits).expect("Lloyd-Max codebook for 3/4 bits");
    let n_entries = 1usize << u32::from(bits);
    debug_assert_eq!(
        cb.len(),
        n_entries,
        "iso fused-QK: codebook entry count must match 2^BITS"
    );

    // Fixed quaternion bits.  q̄ = (w, -x, -y, -z) for the inverse rotation.
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
        "\n// Iso fused-QK header — golden-ratio fixed quaternion + Lloyd-Max N(0,1) codebook.\n\
         // BITS = {bits} (codebook = {n_entries} entries).\n\
         // Bit-exact with crate::isoquant::FIXED_QUAT + lloyd_gaussian_codebook({bits}).\n\
         constant float ISO_QW  = as_type<float>(0x{qw_bits:08X}u);\n\
         constant float ISO_QCX = as_type<float>(0x{qcx_bits:08X}u);\n\
         constant float ISO_QCY = as_type<float>(0x{qcy_bits:08X}u);\n\
         constant float ISO_QCZ = as_type<float>(0x{qcz_bits:08X}u);\n"
    );
    let _ = write!(
        s,
        "\nconstant float ISO_CB[{n_entries}] = {{\n{cb_join}\n}};\n",
        cb_join = cb_hex.join(",\n")
    );
    s
}

/// Select the pre-rendered MSL kernel body for `bits`.
///
/// There are two independent body files, one per BITS. They differ only in the
/// unpack shift / mask (BITS=3 → shift `elem*3`, mask `0x7u`; BITS=4 → shift
/// `elem*4`, mask `0xFu`); the rest is duplicated between them and must be
/// edited in both.
///
/// # Errors
///
/// Returns [`Error::Quant`] for any `bits` outside {3, 4}, rather than
/// silently handing back a body for the wrong bit width.
fn build_iso_fused_qk_source(bits: u8) -> Result<String> {
    Ok(match bits {
        3 => include_str!("metal/iso_fused_qk_b3.metal"),
        4 => include_str!("metal/iso_fused_qk_b4.metal"),
        _ => {
            return Err(Error::Quant(format!(
                "iso fused-QK: bits must be 3 or 4, got {bits}"
            )))
        }
    }
    .to_owned())
}

// ── Kernel singletons (one per BITS variant) ─────────────────────────────────

static ISO3_QK_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ISO4_QK_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn iso_qk_kernel(bits: u8) -> Result<&'static MetalKernel> {
    let cell = match bits {
        3 => &ISO3_QK_KERNEL,
        4 => &ISO4_QK_KERNEL,
        _ => {
            return Err(Error::Quant(format!(
                "iso_fused_qk_sdpa: unsupported bits={bits} (only 3 and 4)"
            )))
        }
    };
    cell.get_or_init(|| {
        let source = build_iso_fused_qk_source(bits)?;
        let header = build_iso_fused_qk_header(bits);
        let name = match bits {
            3 => "rmlx_iso3_fused_qk",
            _ => "rmlx_iso4_fused_qk",
        };
        MetalKernel::new(
            name,
            &header,
            &source,
            &[
                "query",
                "codes",
                "scales",
                "norms",
                "mask",
                "scale_arr",
                "dims",
            ],
            &["out"],
        )
    })
    .as_ref()
    .map_err(|e| Error::Mlx(format!("iso_fused_qk(bits={bits}) kernel init: {e}")))
}

// ── Generic dispatcher ────────────────────────────────────────────────────────

/// Fused IsoQuant K-side QK kernel, parameterised by `const BITS: u8`.
///
/// Inverts the per-group SO(4) fixed-quaternion rotation in Metal registers,
/// looks up the Lloyd-Max centroid, multiplies by the per-group scale and the
/// per-token L2 norm, then accumulates the QK dot product.  No bf16 / f32 K
/// is materialised in HBM.
///
/// **A.y guard**: caller must ensure arch is NOT Qwen MoE before calling.
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`
///   (or `[B, n_q_heads, head_dim]`).  f32/f16/bf16; coerced to f32.
/// * `k_codes`   — packed iso codes,
///   flat `u32 [B * kv_h * kv_seq * (head_dim / 4)]`.
/// * `k_scales`  — per-group f32 scales,
///   flat `[B * kv_h * kv_seq * (head_dim / 4)]`.
/// * `k_norms`   — per-token f32 L2 norms,
///   flat `[B * kv_h * kv_seq]`.
/// * `additive_mask` — optional `f32 [B, n_q_heads, 1, kv_seq]`.
/// * `b`, `kv_h`, `kv_seq`, `head_dim`, `heads_per_kv` — shape metadata.
/// * `scale`     — softmax pre-scale (typ. `1/sqrt(head_dim)`).
/// * `device`    — MLX device (must be GPU).
///
/// # Output
///
/// Scores tensor `[B, n_q_heads, 1, kv_seq]` (f32) with the additive mask
/// already applied.
///
/// # Errors
///
/// * `Error::Quant` for shape contract violations (`head_dim` not in
///   `{128, 256}`, dims out of range, non-positive shapes).
/// * `Error::Mlx` for kernel build / dispatch failures.
#[allow(clippy::too_many_arguments)]
pub fn iso_fused_qk_sdpa<const BITS: u8>(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: &Array,
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
            "iso_fused_qk_sdpa: unsupported BITS={BITS} (only 3 and 4)"
        )));
    }
    if !(head_dim as usize).is_multiple_of(ISO_GROUP_SIZE) {
        return Err(Error::Quant(format!(
            "iso_fused_qk_sdpa: invariant: head_dim={head_dim} must be a multiple of \
             ISO_GROUP_SIZE={ISO_GROUP_SIZE}"
        )));
    }

    let setup = crate::fused_qk_common::build_fused_qk_setup(
        "iso_fused_qk_sdpa",
        query,
        additive_mask,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        device,
    )?;

    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    // iso: 1 u32 word per group of 4 (both BITS=3 and BITS=4 fit in one word).
    let codes_total: i64 = tok_count * i64::from(head_dim) / (ISO_GROUP_SIZE as i64);
    let scales_total: i64 = codes_total;
    let norms_total: i64 = tok_count;
    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[scales_total as i32], device)?;
    let norms_flat = k_norms.reshape(&[norms_total as i32], device)?;
    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.

    let kernel = iso_qk_kernel(BITS)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&setup.q_f32)?;
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&norms_flat)?;
    invoke.add_input(&setup.mask_flat)?;
    invoke.add_input(&setup.scale_arr)?;
    invoke.add_input(&setup.dims_arr)?;

    invoke.add_output_shape(&[setup.out_total as i32], Dtype::F32)?;

    invoke.set_grid(setup.grid_x, setup.grid_y, 1)?;
    invoke.set_thread_group(head_dim, 1, 1)?;

    if BITS == 3 {
        ISO3_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        ISO4_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    }
    tracing::trace!(
        bits = BITS,
        b,
        n_q_heads = setup.n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask = setup.has_mask,
        "iso_fused_qk_sdpa: dispatch"
    );

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "iso_fused_qk_sdpa: kernel produced no outputs".into(),
        ));
    }
    let out_flat = outputs.remove(0);

    out_flat.reshape(&[b, setup.n_q_heads, 1, kv_seq], device)
}

// ── `FusedQkFn`-compatible shims ──────────────────────────────────────────────
//
// `FusedQkFn` (defined in `rmlx_models::kv_cache::attention_dispatch`) is the
// canonical fused-QK dispatch type (13-arg, widened to carry iso/rotor
// sidebands). Iso needs an extra `k_norms` sideband; it arrives as a
// separate `Option<&Array>` arg to avoid a per-decode-step concat
// marshaling cost. Previously it was packed into `k_scales` and
// split at the shim — see git history for the concat shim form.
//
//     k_scales_combined = [scales_flat (tok_count * n_groups f32) |
//                          norms_flat  (tok_count f32) ]
//
// Both BITS=3 and BITS=4 use `n_groups = head_dim / 4`, so the split offset
// is deterministic from the shape metadata.  This keeps the
// `FusedQkFn` signature unchanged (no extra arg slot) and matches the spec
// hint "k_scales carries both scale + quaternion in interleaved layout per
// existing codec".
//
// Call-site wiring (the loader that produces the combined array from a
// `QuantIsoK{3,4}` GPU buffer) is the responsibility of a follow-up task —
// the shims below are exercised end-to-end by the dispatch-table tests in
// `rmlx-models::kv_cache::attention_dispatch_tests` (table lookup is
// non-mutating) and by direct calls to `iso_fused_qk_sdpa::<BITS>` in the
// crate-local GPU parity tests.

#[allow(clippy::too_many_arguments)]
fn iso_fused_qk_shim<const BITS: u8>(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: Option<&Array>,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Array> {
    if head_dim <= 0 || b <= 0 || kv_h <= 0 || kv_seq <= 0 {
        return Err(Error::Quant(format!(
            "iso_fused_qk_shim: non-positive shape b={b}, kv_h={kv_h}, \
             kv_seq={kv_seq}, head_dim={head_dim}"
        )));
    }
    // `k_norms` arrives as a separate Array (no per-step concat marshaling).
    // Iso ignores the rotor table sideband.
    let norms = k_norms.ok_or_else(|| {
        Error::Quant(
            "iso_fused_qk_shim: k_norms is None but iso codec requires per-token L2 norms".into(),
        )
    })?;

    iso_fused_qk_sdpa::<BITS>(
        query,
        k_codes,
        k_scales,
        norms,
        additive_mask,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        device,
    )
}

/// Table-bound shim for `Iso3Sym` / `IsoKOnly3` (BITS=3).
///
/// Signature matches `rmlx_models::kv_cache::attention_dispatch::FusedQkFn`.
/// `k_scales` and `k_norms` arrive as separate Array arguments (no per-decode
/// concat marshaling). `k_rotor_table` is unused by iso codecs.
/// `k_norms` MUST be `Some` (per-token L2).
#[allow(clippy::too_many_arguments)]
pub fn iso3_fused_qk_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: Option<&Array>,
    _k_rotor_table: Option<&Array>,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Array> {
    iso_fused_qk_shim::<3>(
        query,
        k_codes,
        k_scales,
        k_norms,
        additive_mask,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        device,
    )
}

/// Table-bound shim for `Iso4Sym` / `IsoKOnly4` (BITS=4).
///
/// Signature matches `rmlx_models::kv_cache::attention_dispatch::FusedQkFn`.
/// `k_scales` and `k_norms` arrive as separate Array arguments (no per-decode
/// concat marshaling). `k_rotor_table` is unused by iso codecs.
/// `k_norms` MUST be `Some` (per-token L2).
#[allow(clippy::too_many_arguments)]
pub fn iso4_fused_qk_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: Option<&Array>,
    _k_rotor_table: Option<&Array>,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Array> {
    iso_fused_qk_shim::<4>(
        query,
        k_codes,
        k_scales,
        k_norms,
        additive_mask,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        device,
    )
}

#[cfg(test)]
#[path = "iso_fused_qk_msl_tests.rs"]
mod iso_fused_qk_msl_tests;
