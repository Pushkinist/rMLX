//! RotorQuant K-side fused-QK MSL kernel.
//!
//! # What this is
//!
//! Single generic MSL kernel that consumes the rotor K codec (bits ∈ {3, 4})
//! packed codes (1 u32 per group of 8 multivector components) + per-group f32
//! scales + per-token f32 L2 norms + the static per-(layer, head) rotor table
//! (`[n_groups, 4]` f32), applies the inverse Cl(3,0) Clifford rotor sandwich
//! `R̃ * mv_q * R` per group, extracts the grade-1 lane, and accumulates the
//! pre-softmax QK score `[B, n_q_heads, 1, S_kv]` — all without materialising
//! a bf16 / f32 K tensor in HBM.
//!
//! Parameterised by `const BITS: u8` at the Rust call site
//! ([`rotor3_fused_qk_sdpa`] / [`rotor4_fused_qk_sdpa`] select 3-bit (8-entry
//! codebook) or 4-bit (16-entry codebook)). A **single** shared kernel body
//! ([`build_rotor_fused_qk_source`]) serves both widths — the unpack width and
//! codebook arrive in the per-BITS runtime header
//! ([`build_rotor_fused_qk_header`]) as `RF_BITS` / `RF_MASK` plus the codebook
//! table, so there is no per-BITS body to hand-keep in sync. Each width still
//! gets its own [`OnceLock`]-cached compiled [`MetalKernel`] because the header
//! (codebook table size + unpack macros) differs, so the assembled program
//! differs.
//!
//! # Codec contract
//!
//! Target [`KvQuant`](crate::quant::KvQuant): `RotorK3Asym` (BITS=3) and
//! `RotorK4Asym` (BITS=4). K storage is
//! [`crate::storage::QuantRotorK3`] / [`crate::storage::QuantRotorK4`] with
//! `group_size = 3` (one rotor block per group of 3 head-dim slots, encoded
//! into 8 Cl(3,0) multivector components). V side is irrelevant for the
//! fused-QK score — the caller handles V separately.
//!
//! The rotor `Sym` and `KOnly` variants do **not** reach this kernel. They
//! keep no bf16 K mirror (the fused-QK shadow is seeded by re-encoding one),
//! and each has a dedicated flash-decode-over-quant kernel —
//! [`crate::rotor_flash_decode_symv_msl`] and
//! [`crate::rotor_flash_decode_msl`] — that reads the packed ring directly.
//! The asym pair is the one rotor family member with no such arm, which makes
//! this kernel its only GPU decode path.
//!
//! Bit-exact with [`crate::rotorquant::rotor3_decode`] /
//! [`crate::rotorquant::rotor4_decode`]:
//!
//! * `codes`  u32 `[B * kv_h * S_kv * n_groups]` — 1 u32 per group of 8
//!   multivector components (`words_per_group = 1` for both 3 and 4 bits —
//!   8 codes × 3 bits = 24 bits ≤ 32; 8 codes × 4 bits = 32 bits = 32).
//!   Element `e ∈ 0..8` lives at bits `[e*BITS, e*BITS + BITS)` of the
//!   single u32 (LSB-first).
//! * `scales` f32 `[B * kv_h * S_kv * n_groups]` — one f32 per group of 3
//!   head-dim slots.
//! * `norms`  f32 `[B * kv_h * S_kv]` — one f32 per token (L2 norm of the
//!   original head-dim row).
//! * `rotors` f32 `[n_groups * 4]` — static per-(layer, head) rotor table in
//!   compact `[s, b12, b13, b23]` form.
//! * Codebook: [`crate::turboquant::lloyd_gaussian_codebook`] — 8 entries for
//!   BITS=3, 16 entries for BITS=4 (same Lloyd-Max N(0,1) tables as turbo3 /
//!   turbo4 / iso3 / iso4).
//!
//! # Decode (per group of 3 head-dim slots)
//!
//! 1. Read the single packed `u32` word and unpack 8 `BITS`-bit indices.
//! 2. Look up 8 centroids in the codebook; multiply each by the per-group
//!    scale to form `mv_q[0..8]`.
//! 3. Apply the **inverse** sandwich `restored = R̃ * mv_q * R`, where `R̃`
//!    is the Clifford reverse `[s, -b12, -b13, -b23]`. This uses two dense
//!    Cl(3,0) geometric products via the `MUL_TABLE` rendered into MSL.
//! 4. Multiply `restored[lane + 1]` (grade-1 component `e_{lane+1}`) by the
//!    per-token L2 norm to obtain `K[head_dim slot]`.
//!
//! Mirrors [`crate::rotorquant::rotor3_decode`] and
//! [`crate::rotorquant::rotor4_decode`] exactly: the algorithm is the same
//! per-group inverse sandwich, just lifted into per-thread Metal registers.
//!
//! # Kernel shape
//!
//! Grid `(S_kv * D, B * n_q_heads, 1)` (total threads, not threadgroup
//! count); threadgroup `(head_dim, 1, 1)`. Each threadgroup computes one
//! score `out[b, hq, 0, s_kv]`; per-thread `tid` handles one element of
//! `head_dim`. Each thread independently redoes the per-group rotor decode
//! for its own head-dim slot — the rotor sandwich is `O(64)` FMAs per group
//! and the work duplication factor (3 lanes per group) is acceptable for
//! the simplicity gain (no SMEM staging of the 8-element multivector
//! between the inverse-sandwich and the dot-product phases).
//!
//! # GQA support
//!
//! Identical to the q8 / turbo / iso paths: `n_q_heads = kv_h * heads_per_kv`;
//! thread group maps `(b, hq) -> kv_h_idx = hq / heads_per_kv` to share K.
//!
//! # QJL sideband
//!
//! The QJL residual correction lives on the **CPU dequant path** as a
//! per-token K-side residual-add (mathematically equivalent to the Python
//! `RotorQuantProd.inner_product` score-time term2 by linearity of `Q @ K`);
//! see `crate::rotorquant::apply_qjl_correction`. This kernel does not
//! reproduce it, so `try_fused_qk_dispatch` refuses to dispatch while
//! `rotor_qjl_enabled()` and the decode step falls back to the legacy bf16
//! SDPA path. Adding QJL support here means replicating the per-token
//! correction in MSL before the score is emitted — until then the gate is the
//! contract, not an oversight.
//!
//! # A.y guard
//!
//! `RotorK{3,4}Asym` are K-side ≤ 4-bit — the Qwen-MoE 218→8641 PPL disaster
//! applies. The guard lives in
//! `rmlx_models::kv_cache::cache_type::validate_resolved` and rejects every
//! rotor K-side codec on `Qwen3_5MoeForConditionalGeneration` at session
//! start — the kernel does NOT re-check.
//!
//! # Sandwich verification
//!
//! The CPU reference [`crate::rotorquant::rotor3_decode`] (line 458) calls
//! `rotor_sandwich(rotor_reverse(r), &mv_q)` — i.e. the full inverse sandwich
//! `R̃ * mv_q * R`. This **is** a true two-side sandwich (unlike the iso
//! Exec E case which reduced to a single left-multiply). Verified by reading
//! [`crate::clifford::rotor_sandwich`] source: `gp_rotor_mv(r, x)` then
//! `gp_mv_rotor(tmp, rotor_reverse(r))`. Composed with the outer
//! `rotor_reverse` argument fed at the call site, the net is
//! `R̃ * mv_q * R̃̃ = R̃ * mv_q * R`.
//!
//! # Reference
//!
//! * [`crate::turbo_k3_fused_qk_msl`] — sibling fused-QK dispatcher /
//!   threadgroup layout / counter pattern.
//! * [`crate::rotorquant::rotor3_decode`] / [`crate::rotorquant::rotor4_decode`]
//!   — CPU reference (Rust).
//! * [`crate::clifford`] — `MUL_TABLE`, `rotor_sandwich`, `rotor_reverse`.

// unsafe_code: dims buffer byte cast — slice::from_raw_parts over a fixed
// u32 array for MSL kernel input. SAFETY justified at each use site.
#![allow(unsafe_code)]

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::clifford::MV_DIM;
use crate::rotorquant::n_groups_for;
use crate::turboquant::lloyd_gaussian_codebook;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Per-group rotor stride in the rotor table (`[s, b12, b13, b23]`).
const ROTOR_STRIDE: usize = 4;

/// Multivector component count (Cl(3,0) basis size).
const ROTOR_MV: usize = MV_DIM;

// ── Dispatch counters ─────────────────────────────────────────────────────────

/// Incremented once per `rotor_fused_qk_sdpa_generic::<3>` invocation that
/// reaches the Metal enqueue point.
static ROTOR3_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Incremented once per `rotor_fused_qk_sdpa_generic::<4>` invocation that
/// reaches the Metal enqueue point.
static ROTOR4_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Process-lifetime count of rotor3 fused-QK dispatches.
pub fn rotor3_fused_qk_dispatch_count() -> u64 {
    ROTOR3_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
}

/// Process-lifetime count of rotor4 fused-QK dispatches.
pub fn rotor4_fused_qk_dispatch_count() -> u64 {
    ROTOR4_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
}

/// Combined process-lifetime count (3-bit + 4-bit). Kept for backward
/// compatibility with the Exec A skeleton API.
pub fn rotor_fused_qk_dispatch_count() -> u64 {
    rotor3_fused_qk_dispatch_count() + rotor4_fused_qk_dispatch_count()
}

// ── MSL header / source builder ───────────────────────────────────────────────

/// Render the Cl(3,0) `MUL_TABLE` (8×8 `(target, sign)` pairs) into two MSL
/// `constant` lookup arrays: `MUL_T[64]` (target indices) and `MUL_S[64]`
/// (signs as f32 ±1). Indexed row-major as `[i * 8 + j]`.
///
/// Mirrors `crate::clifford::MUL_TABLE` bit-for-bit (verified by the
/// table-driven `geometric_product_*` tests in `clifford_tests.rs`).
#[allow(
    clippy::indexing_slicing,
    reason = "fixed-size [ROTOR_MV] / [ROTOR_MV * ROTOR_MV] arrays indexed by bounded \
              counters in [0, ROTOR_MV); never panics by construction"
)]
fn render_mul_table() -> String {
    // Re-derive the table at codegen time using the same recipe as
    // `crate::clifford::MUL_TABLE` (compile-time const block) — we do not
    // import the const directly because it's private; the table is tiny.
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
            let target = bits_to_basis[result_bits as usize] as u32;
            targets[i * ROTOR_MV + j] = target;
            signs[i * ROTOR_MV + j] = sign;
        }
    }

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Cl(3,0) MUL_TABLE — target basis index + sign for e_I * e_J.\n\
         // Indexed row-major: MUL_T[i*8+j], MUL_S[i*8+j].\n\
         // Bit-exact with crate::clifford::MUL_TABLE (re-derived from BASIS_BITS).\n\
         constant uint MUL_T[64] = {{\n"
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
    let _ = writeln!(s, "}};\n\nconstant float MUL_S[64] = {{");
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

#[allow(
    clippy::expect_used,
    reason = "lloyd_gaussian_codebook(3|4) is always Ok; verified by \
              `turbo_codebook_3bit_has_8_entries_monotonic` test in turboquant_tests.rs"
)]
fn build_rotor_fused_qk_header(bits: u8) -> String {
    debug_assert!(
        bits == 3 || bits == 4,
        "rotor fused-QK: bits must be 3 or 4"
    );
    let cb = lloyd_gaussian_codebook(bits).expect("Lloyd-Max codebook for 3/4 bits");
    let n_entries = 1usize << u32::from(bits);
    debug_assert_eq!(
        cb.len(),
        n_entries,
        "rotor fused-QK: codebook entry count must match 2^BITS"
    );
    // Unpack width parameters consumed by the shared kernel body: each of the
    // 8 codes occupies `RF_BITS` bits of the packed u32, masked by `RF_MASK`.
    let mask = n_entries as u32 - 1;

    let cb_hex: Vec<String> = cb
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// Rotor fused-QK header — Cl(3,0) MUL_TABLE + Lloyd-Max N(0,1) codebook.\n\
         // BITS = {bits} (codebook = {n_entries} entries, mask = 0x{mask:X}).\n\
         // Bit-exact with crate::clifford::MUL_TABLE + lloyd_gaussian_codebook({bits}).\n"
    );
    let _ = write!(
        s,
        "\n#define RF_BITS {bits}u\n#define RF_MASK 0x{mask:X}u\n"
    );
    let _ = write!(
        s,
        "\nconstant float ROTOR_CB[{n_entries}] = {{\n{cb_join}\n}};\n",
        cb_join = cb_hex.join(",\n")
    );
    s.push_str(&render_mul_table());
    s
}

/// Return the shared MSL kernel body.
///
/// A single body serves both bit widths: the unpack width arrives from the
/// header as `RF_BITS` / `RF_MASK` (see [`build_rotor_fused_qk_header`]), so
/// there is no per-BITS body to keep in sync. The `bits` argument is still
/// validated so an unsupported width fails loudly rather than compiling a body
/// against a header that never defined its unpack macros.
///
/// # Errors
///
/// Returns [`Error::Quant`] for any `bits` outside {3, 4}.
fn build_rotor_fused_qk_source(bits: u8) -> Result<String> {
    if bits != 3 && bits != 4 {
        return Err(Error::Quant(format!(
            "rotor fused-QK: bits must be 3 or 4, got {bits}"
        )));
    }
    Ok(include_str!("metal/rotor_fused_qk.metal").to_owned())
}

// ── Kernel singletons (one per BITS variant) ─────────────────────────────────

static ROTOR3_QK_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
static ROTOR4_QK_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn rotor_qk_kernel(bits: u8) -> Result<&'static MetalKernel> {
    let cell = match bits {
        3 => &ROTOR3_QK_KERNEL,
        4 => &ROTOR4_QK_KERNEL,
        _ => {
            return Err(Error::Quant(format!(
                "rotor_fused_qk_sdpa: unsupported bits={bits} (only 3 and 4)"
            )))
        }
    };
    cell.get_or_init(|| {
        let source = build_rotor_fused_qk_source(bits)?;
        let header = build_rotor_fused_qk_header(bits);
        let name = match bits {
            3 => "rmlx_rotor3_fused_qk",
            _ => "rmlx_rotor4_fused_qk",
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
                "rotors",
                "mask",
                "scale_arr",
                "dims",
            ],
            &["out"],
        )
    })
    .as_ref()
    .map_err(|e| Error::Mlx(format!("rotor_fused_qk(bits={bits}) kernel init: {e}")))
}

// ── Generic dispatcher ────────────────────────────────────────────────────────

/// Fused RotorQuant K-side QK kernel, parameterised by `const BITS: u8`.
///
/// Inverts the per-group Cl(3,0) Clifford rotor sandwich in Metal registers,
/// looks up the Lloyd-Max centroid, multiplies by the per-group scale and the
/// per-token L2 norm, then accumulates the QK dot product. No bf16 / f32 K
/// is materialised in HBM.
///
/// **A.y guard**: caller must ensure arch is NOT Qwen MoE before calling.
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`
///   (or `[B, n_q_heads, head_dim]`). f32/f16/bf16; coerced to f32.
/// * `k_codes`   — packed rotor codes,
///   flat `u32 [B * kv_h * kv_seq * n_groups]`.
/// * `k_scales`  — per-group f32 scales,
///   flat `[B * kv_h * kv_seq * n_groups]`.
/// * `k_norms`   — per-token f32 L2 norms,
///   flat `[B * kv_h * kv_seq]`.
/// * `k_rotors`  — static per-(layer, head) rotor table,
///   flat `[n_groups * 4]` f32 in `[s, b12, b13, b23]` order.
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
// f32-out-ok: pre-softmax scores, not the attention output — the caller
// softmaxes them and restores the query dtype on the SV result
// (`KvCache::try_fused_qk_dispatch`), so nothing f32 reaches the residual
// stream. The scores do carry their width into that intervening matmul; that
// is a cost inside the attention op, not a promotion of the graph behind it.
#[allow(clippy::too_many_arguments)]
pub fn rotor_fused_qk_sdpa_generic<const BITS: u8>(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: &Array,
    k_rotors: &Array,
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
            "rotor_fused_qk_sdpa: unsupported BITS={BITS} (only 3 and 4)"
        )));
    }
    if head_dim != 128 && head_dim != 256 {
        return Err(Error::Quant(format!(
            "rotor_fused_qk_sdpa: head_dim={head_dim} not supported \
             (only 128 and 256 are wired; legacy dequant+SDPA path handles other dims)"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "rotor_fused_qk_sdpa: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "rotor_fused_qk_sdpa: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }
    let n_q_heads = kv_h * heads_per_kv;
    let head_dim_usize = head_dim as usize;
    let n_groups_usize = n_groups_for(head_dim_usize);
    let n_groups_i32 = n_groups_usize as i32;

    let q_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * i64::from(n_groups_i32);
    let scales_total: i64 = codes_total;
    let norms_total: i64 = tok_count;
    let rotors_total: i64 = i64::from(n_groups_i32) * (ROTOR_STRIDE as i64);
    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    // Bound at whatever dtype the caller holds, deliberately. Unlike the
    // flash-decode kernels, this body reads `scales` and `norms` directly
    // rather than through a header helper that declares them, so it is
    // dtype-agnostic — MSL widens either to `float` at the read. Its
    // production feed is the head-major fused-QK shadow (`FusedQkShadow`),
    // not the GPU ring, and that buffer spans the whole context: casting it
    // here would enqueue a conversion of tens of MB on every decode step, to
    // no end. That the shadow is f32 is a property of one allocation, not of
    // this kernel — the rotor encoder now produces its planes at the ring's
    // sideband dtype and `fused_qk_dispatch` casts them down to the shadow's
    // width on the way in. Both widths are correct here; what would not be is
    // this dispatcher assuming one.
    let scales_flat = k_scales.reshape(&[scales_total as i32], device)?;
    let norms_flat = k_norms.reshape(&[norms_total as i32], device)?;
    let rotors_flat = k_rotors.reshape(&[rotors_total as i32], device)?;

    let (mask_flat, has_mask) = if let Some(m) = additive_mask {
        let mask_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(kv_seq);
        let m_f = if m.dtype() == Dtype::F32 {
            m.reshape(&[mask_total as i32], device)?
        } else {
            m.astype(Dtype::F32, device)?
                .reshape(&[mask_total as i32], device)?
        };
        (m_f, 1u32)
    } else {
        let zero_bytes = [0u8; 4];
        let dummy = Array::from_bytes(&zero_bytes, &[1], Dtype::F32)
            .map_err(|e| Error::Mlx(format!("rotor_fused_qk dummy mask: {e}")))?;
        (dummy, 0u32)
    };

    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };
    let dims_vals: [u32; 6] = [
        head_dim as u32,
        kv_seq as u32,
        kv_h as u32,
        heads_per_kv as u32,
        has_mask,
        n_groups_usize as u32,
    ];
    let dims_arr = {
        // SAFETY: `dims_vals` is a stack array of 6 `u32`; reinterpreting the
        // pointer as `&[u8]` of length 24 is safe — `u32` has no alignment
        // requirement stricter than `u8`, every bit pattern is valid for
        // `u8`, and `Array::from_bytes` copies the bytes before this scope
        // ends, so no escaping pointer.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(dims_vals.as_ptr().cast::<u8>(), 6 * 4) };
        Array::from_bytes(bytes, &[6], Dtype::U32)?
    };

    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.

    let kernel = rotor_qk_kernel(BITS)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&q_f32)?;
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&norms_flat)?;
    invoke.add_input(&rotors_flat)?;
    invoke.add_input(&mask_flat)?;
    invoke.add_input(&scale_arr)?;
    invoke.add_input(&dims_arr)?;

    let out_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(kv_seq);
    invoke.add_output_shape(&[out_total as i32], Dtype::F32)?;

    let grid_x: i64 = i64::from(kv_seq) * i64::from(head_dim);
    let grid_y: i64 = i64::from(b) * i64::from(n_q_heads);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "rotor_fused_qk_sdpa: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    invoke.set_grid(grid_x as i32, grid_y as i32, 1)?;
    invoke.set_thread_group(head_dim, 1, 1)?;

    if BITS == 3 {
        ROTOR3_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        ROTOR4_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    }
    tracing::trace!(
        bits = BITS,
        b,
        n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask,
        n_groups = n_groups_usize,
        "rotor_fused_qk_sdpa: dispatch"
    );

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "rotor_fused_qk_sdpa: kernel produced no outputs".into(),
        ));
    }
    let out_flat = outputs.remove(0);

    out_flat.reshape(&[b, n_q_heads, 1, kv_seq], device)
}

// ── `FusedQkFn`-compatible shims ──────────────────────────────────────────────
//
// `FusedQkFn` (defined in `rmlx_models::kv_cache::attention_dispatch`) is the
// canonical fused-QK dispatch type (13-arg, widened to carry rotor sidebands).
// Rotor needs two extra sidebands: per-token L2 norms and
// per-(layer, head) static rotor table. They arrive as separate
// `Option<&Array>` args to avoid a per-decode-step concat marshaling cost.
// Previously `k_scales` + `k_norms` + `k_rotors` were packed into a single
// flat f32 array along the
// `[scales (tok_count * n_groups) | norms (tok_count) | rotors (n_groups * 4)]`
// layout and split at the shim — see git history for the concat form.
//
// QJL handling: the QJL sideband (when present in storage) is **not** consumed
// by this kernel — that's a separate score-time SDPA path. The shim does not
// include QJL in the combined layout; the caller (loader) is responsible for
// keeping the QJL byte stream in its own storage buffer until QJL GPU support lands.

#[allow(clippy::too_many_arguments)]
fn rotor_fused_qk_shim<const BITS: u8>(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: Option<&Array>,
    k_rotor_table: Option<&Array>,
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
            "rotor_fused_qk_shim: non-positive shape b={b}, kv_h={kv_h}, \
             kv_seq={kv_seq}, head_dim={head_dim}"
        )));
    }
    // `k_norms` (per-token L2) and `k_rotor_table` (per-layer static rotor
    // sandwich) arrive as separate Array args. Previously the caller packed
    // them into one concatenated `k_scales` Array per decode step
    // (~28 MB on Bonsai 8B at kv_seq=8192), which swamped the kernel compute
    // savings; the widened `FusedQkFn` signature carries them directly so
    // the dispatch path can pass through without a per-step concat.
    let norms = k_norms.ok_or_else(|| {
        Error::Quant(
            "rotor_fused_qk_shim: k_norms is None but rotor codec requires per-token L2 norms"
                .into(),
        )
    })?;
    let rotors = k_rotor_table.ok_or_else(|| {
        Error::Quant(
            "rotor_fused_qk_shim: k_rotor_table is None but rotor codec requires the static \
             per-(layer, head) rotor sandwich"
                .into(),
        )
    })?;
    rotor_fused_qk_sdpa_generic::<BITS>(
        query,
        k_codes,
        k_scales,
        norms,
        rotors,
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

/// Table-bound shim for `Rotor3Sym` / `RotorKOnly3` (BITS=3).
///
/// Signature matches `rmlx_models::kv_cache::attention_dispatch::FusedQkFn`.
/// `k_scales`, `k_norms`, and `k_rotor_table` arrive as separate Array
/// arguments (no per-decode concat marshaling). `k_norms` and `k_rotor_table`
/// MUST both be `Some` for rotor codecs (per-token L2 + per-(layer, head)
/// static rotor sandwich).
#[allow(clippy::too_many_arguments)]
pub fn rotor3_fused_qk_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: Option<&Array>,
    k_rotor_table: Option<&Array>,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Array> {
    rotor_fused_qk_shim::<3>(
        query,
        k_codes,
        k_scales,
        k_norms,
        k_rotor_table,
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

/// Table-bound shim for `Rotor4Sym` / `RotorKOnly4` (BITS=4).
///
/// Signature matches `rmlx_models::kv_cache::attention_dispatch::FusedQkFn`.
/// Same separate-Array layout as [`rotor3_fused_qk_sdpa`].
#[allow(clippy::too_many_arguments)]
pub fn rotor4_fused_qk_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_norms: Option<&Array>,
    k_rotor_table: Option<&Array>,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Array> {
    rotor_fused_qk_shim::<4>(
        query,
        k_codes,
        k_scales,
        k_norms,
        k_rotor_table,
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
#[path = "rotor_fused_qk_msl_tests.rs"]
mod rotor_fused_qk_msl_tests;
