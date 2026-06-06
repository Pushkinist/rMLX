//! Fused-QK dispatch table.
//!
//! # Purpose
//!
//! This module owns the static dispatch table that maps a [`KvQuant`] variant
//! to the corresponding fused-QK kernel function.
//! The table is a `&[FusedQkEntry]` of 7 entries covering all K-side codec
//! variants that have a fused-QK kernel.
//!
//! # Production dispatch
//!
//! [`FUSED_QK_TABLE`] is also reachable through the production
//! dispatch path inside [`rmlx_kv_quant::kvcache::KvCache::try_fused_qk_dispatch`]
//! that uses an **in-crate mirror** of the same table
//! (`rmlx_kv_quant::kvcache::fused_qk_dispatch::lookup_fused_qk_kernel`)
//! — the codec-layer dispatcher cannot depend on this crate
//! (`rmlx-models`) per the workspace dep-graph rule. Both tables list
//! the same 8 entries and must stay in sync; an inline-test gate in
//! `attention_dispatch_tests.rs::lookup_fused_qk_pending_kernels_match_executors`
//! verifies every variant resolves through this side.
//!
//! Wired codec coverage: q8 (K8V*, K8V4), TurboSym3,
//! TurboSym4. HOLD codecs (kernel landed, K-side GPU encoder pending):
//! Iso3Sym, Iso4Sym, IsoKOnly3, IsoKOnly4, Rotor3Sym, Rotor4Sym,
//! RotorKOnly3, RotorKOnly4. See `docs/reports/fused-qk-head-major-k.md`
//! for the storage shape, dispatch site (file:line in `rmlx-kv-quant`),
//! and HOLD ticket details.
//!
//! # Kernel function type
//!
//! The concrete kernel signature is "TBD by first kernel".
//! Executor B (q8) defines the canonical type; this file documents the expected
//! shape in the `FusedQkEntry` rustdoc and uses a placeholder `fn` type until
//! the first executor lands.
//!
//! Expected (non-binding) shape for Executor B:
//!
//! ```text
//! fn(
//!     query:    &rmlx_mlx::Array,   // [B, n_q_heads, head_dim]
//!     k_packed: &rmlx_mlx::Array,   // codec-specific packed K tensor
//!     k_scales: &rmlx_mlx::Array,   // per-group scales (or rotation aux)
//!     scale:    f32,                // 1/sqrt(head_dim) softmax pre-scale
//! ) -> rmlx_core::error::Result<rmlx_mlx::Array>
//! // Returns [B, n_q_heads, S_kv] pre-softmax scores (f32).
//! ```
//!
//! Iso / rotor variants need an extra `bits: u8` param and rotor needs
//! `k_qjl: Option<&Array>`.  The dispatch site adapts via a thin per-codec
//! shim function stored as the `kernel` slot.
//!
//! # Usage
//!
//! ```ignore
//! if let Some(_kernel_fn) = lookup_fused_qk(kv_quant) {
//!     // call kernel_fn(query, k_packed, k_scales, scale)?
//! }
//! ```
//!
//! # Table maintenance
//!
//! One row per (codec, KvQuant) pair.  The table is `&[FusedQkEntry]` so
//! entries are iterated linearly; with 7 entries the cost is negligible at
//! decode time.  Each entry documents which Executor fills the `kernel` slot.

use rmlx_core::error::Result;
use rmlx_kv_quant::iso_fused_qk_msl::{iso3_fused_qk_sdpa, iso4_fused_qk_sdpa};
use rmlx_kv_quant::q8_fused_qk_msl::q8_fused_qk_sdpa;
use rmlx_kv_quant::rotor_fused_qk_msl::{rotor3_fused_qk_sdpa, rotor4_fused_qk_sdpa};
use rmlx_kv_quant::sparse_attn::phase1_score_msl::{phase1_score, TOP_PER_TILE};
use rmlx_kv_quant::sparse_attn::phase2_sparse_attend_msl::{
    phase2_lse_merge, phase2_sparse_attend,
};
use rmlx_kv_quant::turbo_k3_fused_qk_msl::turbo_k3_fused_qk_sdpa;
use rmlx_kv_quant::turbo_k4_fused_qk_msl::turbo_k4_fused_qk_sdpa;
use rmlx_kv_quant::{sparse_attn_enabled, KvQuant};
use rmlx_loader::HeadBudgets;
use rmlx_mlx::{Array, Device, Dtype};

// ── Kernel function type ──────────────────────────────────────────────────────

/// Canonical fused-QK kernel function type.
///
/// Canonical fused-QK kernel function type, first defined by Executor B (q8).
/// Every codec dispatches through this signature; codecs that need extra
/// codec-specific data (rotor `k_qjl`, iso `bits`) pass it via a thin
/// per-codec shim closure stored at the call site, or extend this type.
///
/// # Parameter contract
///
/// * `query`     — `&Array` Q for the new token, shape
///   `[B, n_q_heads, 1, head_dim]` or `[B, n_q_heads, head_dim]`.
///   Coerced to f32 by the kernel dispatcher.
/// * `k_codes`   — `&Array` codec-specific packed K codes (q8: u32 packed i8;
///   planar / iso / rotor: u32 packed n-bit codes).
/// * `k_scales`  — `&Array` per-group K scales (f32).  Codec-specific
///   group layout — see each codec's storage docs.
/// * `additive_mask` — `Option<&Array>` causal/attention mask
///   `f32 [B, n_q_heads, 1, kv_seq]`.  Added pre-softmax in-kernel.
/// * `b, kv_h, kv_seq, head_dim, heads_per_kv` — shape metadata.
/// * `scale`     — `f32` softmax pre-scale (typ. `1/sqrt(head_dim)`).
/// * `device`    — `Device` MLX device (GPU for production dispatch).
///
/// Returns `[B, n_q_heads, 1, kv_seq]` pre-softmax scores (f32, mask-added).
#[allow(
    clippy::type_complexity,
    reason = "function pointer alias is the dispatch contract; field-by-field clarity matters more than brevity"
)]
pub type FusedQkFn = fn(
    /* query         */ &Array,
    /* k_codes       */ &Array,
    /* k_scales      */ &Array,
    /* k_norms       */ Option<&Array>,
    /* k_rotor_table */ Option<&Array>,
    /* additive_mask */ Option<&Array>,
    /* b             */ i32,
    /* kv_h          */ i32,
    /* kv_seq        */ i32,
    /* head_dim      */ i32,
    /* heads_per_kv  */ i32,
    /* scale         */ f32,
    /* device        */ Device,
) -> Result<Array>;

// ── FusedQkEntry ─────────────────────────────────────────────────────────────

/// One entry in the fused-QK dispatch table.
///
/// Maps a [`KvQuant`] variant to its compiled kernel function.  `kernel` is
/// `None` until the corresponding Executor lands the MSL implementation.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed dispatch-table entry — field set is the complete (kv_quant, kernel) \
              contract; new fields require reviewing all table-construction sites"
)]
#[derive(Debug)]
pub struct FusedQkEntry {
    /// The KvQuant variant this entry covers.
    pub kv_quant: KvQuant,
    /// The compiled kernel function, or `None` while the Executor is pending.
    ///
    /// `None` entries cause the dispatch site to fall through to the legacy
    /// dequant+SDPA path — correct behaviour until the kernel is implemented.
    pub kernel: Option<FusedQkFn>,
}

// ── Concrete kernel fn-item coerced to the `FusedQkFn` pointer type ──────────
//
// The `const` slot triggers the fn-item → fn-pointer coercion implicitly,
// so the table entries can write `Some(Q8_FUSED_QK_FN)` without the
// `trivial_casts` lint firing on `... as FusedQkFn`.
const Q8_FUSED_QK_FN: FusedQkFn = q8_fused_qk_sdpa;

// TurboSym3 K-side fused-QK kernel.
const TURBO_K3_FUSED_QK_FN: FusedQkFn = turbo_k3_fused_qk_sdpa;

// TurboSym4 K-side fused-QK kernel.
const TURBO_K4_FUSED_QK_FN: FusedQkFn = turbo_k4_fused_qk_sdpa;

// IsoQuant K-side fused-QK kernels (bits=3 and bits=4).
const ISO3_FUSED_QK_FN: FusedQkFn = iso3_fused_qk_sdpa;
const ISO4_FUSED_QK_FN: FusedQkFn = iso4_fused_qk_sdpa;

// RotorQuant K-side fused-QK kernels (bits=3 and bits=4).
const ROTOR3_FUSED_QK_FN: FusedQkFn = rotor3_fused_qk_sdpa;
const ROTOR4_FUSED_QK_FN: FusedQkFn = rotor4_fused_qk_sdpa;

// ── Static dispatch table ─────────────────────────────────────────────────────

/// Fused-QK dispatch table.
///
/// 8 entries; all `kernel` slots are `Some(...)`.
///
/// | Entry | KvQuant       | K codec         | A.y? | Executor |
/// |-------|---------------|-----------------|------|----------|
/// | 0     | K8V4          | q8_0 8-bit K    | No   | B        |
/// | 1     | K8V8          | q8_0 8-bit K    | No   | B        |
/// | 2     | TurboSym3     | turbo3 K 3-bit  | Yes  | C        |
/// | 3     | TurboSym4     | turbo4 K 4-bit  | Yes  | D        |
/// | 4     | Iso3Sym       | iso K bits=3    | Yes  | E        |
/// | 5     | Iso4Sym       | iso K bits=4    | Yes  | E        |
/// | 6     | Rotor3Sym     | rotor K bits=3  | Yes  | F        |
/// | 7     | Rotor4Sym     | rotor K bits=4  | Yes  | F        |
///
/// Note: `IsoKOnly3`, `IsoKOnly4`, `RotorKOnly3`, `RotorKOnly4` share the
/// same kernel as their `*Sym` counterpart (V side differs but K decode is
/// identical).  Executor E / F may add those entries when wiring.
pub static FUSED_QK_TABLE: &[FusedQkEntry] = &[
    // ── q8_0 K ───────────────────────────────────────────────────────────────
    // Both K8V4 and K8V8 share the same K-side codec (q8_0 affine 8-bit,
    // group_size=128); the V-side codec differs but is the SDPA caller's
    // responsibility, not the fused-QK kernel's.
    //
    // `Q8_FUSED_QK_FN` coerces the concrete fn-item to the `FusedQkFn`
    // pointer type via a `const` slot — avoids the `trivial_casts` lint
    // that fires on inline `... as FusedQkFn`.
    FusedQkEntry {
        kv_quant: KvQuant::K8V4,
        kernel: Some(Q8_FUSED_QK_FN),
    },
    FusedQkEntry {
        kv_quant: KvQuant::K8V8,
        kernel: Some(Q8_FUSED_QK_FN),
    },
    // ── turbo3 K-side ────────────────────────────────────────────────────────
    // 3-bit Lloyd-Max codebook lookup × per-group f32 scale; no WHT.
    // K-side 3-bit ⇒ A.y guard required (Qwen3.5-MoE rejected at session
    // start by `validate_resolved` in `cache_type.rs`, not here).
    FusedQkEntry {
        kv_quant: KvQuant::TurboSym3,
        kernel: Some(TURBO_K3_FUSED_QK_FN),
    },
    // ── turbo4 K-side ────────────────────────────────────────────────────────
    // 4-bit Lloyd-Max codebook lookup × per-group f32 scale; no WHT.
    // K-side 4-bit ⇒ A.y guard required (Qwen3.5-MoE rejected at session
    // start by `validate_resolved` in `cache_type.rs`, not here).
    FusedQkEntry {
        kv_quant: KvQuant::TurboSym4,
        kernel: Some(TURBO_K4_FUSED_QK_FN),
    },
    // ── iso K-side bits=3 ────────────────────────────────────────────────────
    // Per-group quaternion SO(4) rotation (inverse) × Lloyd-Max 3-bit codebook
    // lookup × per-group f32 scale × per-token f32 L2 norm. The `FusedQkFn`
    // signature was widened so the caller passes
    // `k_norms: Some(per-token L2)` as a separate Array (pre-fix-cycle the
    // caller concatenated `[scales | norms]` into `k_scales` and the shim
    // split it back; that concat cost dominated decode at long context).
    // K-side 3-bit ⇒ A.y guard required (Qwen3.5-MoE rejected at session
    // start by `validate_resolved` in `cache_type.rs`, not here).
    FusedQkEntry {
        kv_quant: KvQuant::Iso3Sym,
        kernel: Some(ISO3_FUSED_QK_FN),
    },
    // ── iso K-side bits=4 ────────────────────────────────────────────────────
    // Same shape as bits=3 with a 16-entry Lloyd-Max codebook + 4-bit unpack.
    // K-side 4-bit ⇒ A.y guard required (Qwen3.5-MoE rejected at session
    // start by `validate_resolved` in `cache_type.rs`, not here).
    FusedQkEntry {
        kv_quant: KvQuant::Iso4Sym,
        kernel: Some(ISO4_FUSED_QK_FN),
    },
    // ── rotor K-side bits=3 ──────────────────────────────────────────────────
    // Per-group inverse Cl(3,0) Clifford rotor sandwich `R̃ * mv_q * R`
    // (rendered MUL_TABLE in MSL) × Lloyd-Max 3-bit codebook lookup × per-group
    // f32 scale × per-token f32 L2 norm. The `FusedQkFn` signature was widened
    // so the caller passes `k_norms: Some(per-token L2)`
    // and `k_rotor_table: Some([n_groups * 4])` as separate Arrays (previously
    // the caller concatenated `[scales | norms | rotors]` into
    // `k_scales` and the shim split it back; that concat cost dominated
    // decode at long context — see commit f42aa0f).
    // K-side 3-bit ⇒ A.y guard required (Qwen3.5-MoE rejected at session
    // start by `validate_resolved` in `cache_type.rs`, not here).
    FusedQkEntry {
        kv_quant: KvQuant::Rotor3Sym,
        kernel: Some(ROTOR3_FUSED_QK_FN),
    },
    // ── rotor K-side bits=4 ──────────────────────────────────────────────────
    // Same shape as bits=3 with a 16-entry Lloyd-Max codebook + 4-bit unpack
    // (8 codes × 4 bits = 32 bits → 1 u32 word per group, dense pack).
    // K-side 4-bit ⇒ A.y guard required (Qwen3.5-MoE rejected at session
    // start by `validate_resolved` in `cache_type.rs`, not here).
    FusedQkEntry {
        kv_quant: KvQuant::Rotor4Sym,
        kernel: Some(ROTOR4_FUSED_QK_FN),
    },
];

// ── Dispatch lookup ───────────────────────────────────────────────────────────

/// Look up the fused-QK kernel for `kv_quant`.
///
/// Returns the first `Some` kernel in [`FUSED_QK_TABLE`] whose `kv_quant`
/// matches.  Returns `None` when:
/// - The variant has no table entry (i.e. it is not a fused-QK target), or
/// - The entry's `kernel` slot is still `None` (Executor B-F pending).
///
/// Callers must fall through to the legacy dequant+SDPA path on `None`.
pub fn lookup_fused_qk(kv_quant: KvQuant) -> Option<FusedQkFn> {
    for entry in FUSED_QK_TABLE {
        if entry.kv_quant == kv_quant {
            return entry.kernel;
        }
    }
    None
}

// ── Two-phase sparse-attention dispatch ──────────────────────────────────────

/// Per-layer shape + tensor inputs for the two-phase sparse-attention
/// dispatch.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed input pack — every field is required by the downstream kernels"
)]
#[derive(Debug)]
pub struct SparseAttnInputs<'a> {
    /// Q tensor for the new token, `[B, n_q_heads, 1, head_dim]`.
    pub query: &'a Array,
    /// PlanarQuant K codes (u32 packed).
    pub k_codes: &'a Array,
    /// PlanarQuant K per-pair scales (f32).
    pub k_scales: &'a Array,
    /// PlanarQuant K 4-bit rotation indices (u32 packed).
    pub k_rot32: &'a Array,
    /// V tensor (bf16 / f16 / f32), `[B, kv_h, kv_seq, head_dim]`.
    pub v: &'a Array,
    /// Batch size.
    pub b: i32,
    /// Number of KV heads.
    pub kv_h: i32,
    /// KV sequence length.
    pub kv_seq: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Query heads per KV head (`n_q_heads / kv_h`).
    pub heads_per_kv: i32,
    /// Layer index (selects row in `head_budgets.per_layer_per_head_budget`).
    pub layer_idx: usize,
    /// Softmax pre-scale (typically `1/sqrt(head_dim)`).
    pub scale: f32,
    /// MLX device (`Device::Gpu` for production).
    pub device: Device,
}

/// Phase 1 + CPU threshold + Phase 2 + LSE merge.
///
/// Inner dispatcher; [`sparse_attn_dispatch_if_enabled`] layers the
/// `RMLX_SPARSE_ATTN` gate + `head_budgets` presence check on top.
pub fn sparse_attn_dispatch(
    inputs: &SparseAttnInputs<'_>,
    head_budgets: &HeadBudgets,
) -> Result<Array> {
    let p1 = phase1_score(
        inputs.query,
        inputs.k_codes,
        inputs.k_scales,
        inputs.k_rot32,
        inputs.b,
        inputs.kv_h,
        inputs.kv_seq,
        inputs.head_dim,
        inputs.heads_per_kv,
        inputs.scale,
        inputs.device,
    )?;

    let n_q_heads = inputs.kv_h * inputs.heads_per_kv;
    let n_bh = inputs.b * n_q_heads;
    let tts_vec = read_tile_top_scores(&p1.tile_top_scores)?;
    let thr_vec = compute_head_threshold(
        &tts_vec,
        p1.n_tiles as usize,
        n_bh as usize,
        head_budgets,
        inputs.layer_idx,
    )?;
    let head_threshold_arr = build_threshold_array(&thr_vec, n_bh)?;

    let p2 = phase2_sparse_attend(
        inputs.query,
        inputs.k_codes,
        inputs.k_scales,
        inputs.k_rot32,
        inputs.v,
        &p1.all_scores,
        &head_threshold_arr,
        inputs.b,
        inputs.kv_h,
        inputs.kv_seq,
        inputs.head_dim,
        inputs.heads_per_kv,
        p1.n_tiles,
        inputs.scale,
        inputs.device,
    )?;

    phase2_lse_merge(
        &p2.partial_o,
        &p2.tile_lse,
        inputs.b,
        n_q_heads,
        inputs.head_dim,
        p1.n_tiles,
        inputs.device,
    )
}

/// Two-phase sparse-attention dispatch with env-var gate + budget check.
///
/// Returns `Some(Array)` when:
/// 1. [`sparse_attn_enabled`] is `true` (env-var `RMLX_SPARSE_ATTN=1`),
/// 2. `head_budgets` is `Some`, and
/// 3. [`sparse_attn_dispatch`] succeeds.
///
/// Returns `None` when either gate fails OR the inner dispatch errors.
pub fn sparse_attn_dispatch_if_enabled(
    inputs: &SparseAttnInputs<'_>,
    head_budgets: Option<&HeadBudgets>,
) -> Option<Array> {
    if !sparse_attn_enabled() {
        return None;
    }
    let budgets = head_budgets?;
    match sparse_attn_dispatch(inputs, budgets) {
        Ok(out) => Some(out),
        Err(e) => {
            tracing::warn!(error = %e, "sparse_attn inner dispatch errored — falling back to dense");
            None
        }
    }
}

// ── Bridge helpers ───────────────────────────────────────────────────────────

/// Pull `tile_top_scores` `[n_tiles, n_bh, TOP_PER_TILE]` f32 to host.
fn read_tile_top_scores(tile_top_scores: &Array) -> Result<Vec<f32>> {
    tile_top_scores.eval()?;
    let bytes = tile_top_scores.to_bytes()?;
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        // chunks_exact(4) guarantees four elements.  `unwrap_or_default`
        // keeps clippy::indexing_slicing happy without changing semantics
        // (the default `[0; 4]` is unreachable on an exact-sized chunk).
        let arr: [u8; 4] = chunk.try_into().unwrap_or_default();
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// Compute the per-`(B, H)` raw QK threshold from Phase-1 tile tops.
fn compute_head_threshold(
    tile_top_scores: &[f32],
    n_tiles: usize,
    n_bh: usize,
    head_budgets: &HeadBudgets,
    layer_idx: usize,
) -> Result<Vec<f32>> {
    let n_layers = head_budgets.per_layer_per_head_budget.len();
    let row = head_budgets
        .per_layer_per_head_budget
        .get(layer_idx)
        .ok_or_else(|| {
            rmlx_core::error::Error::Quant(format!(
                "sparse_attn: layer_idx={layer_idx} exceeds head_budgets layer rows ({n_layers})"
            ))
        })?;
    let top_per_tile = TOP_PER_TILE as usize;

    let n_q_heads = row.len();
    if n_q_heads == 0 || !n_bh.is_multiple_of(n_q_heads) {
        return Err(rmlx_core::error::Error::Quant(format!(
            "sparse_attn: n_bh={n_bh} not divisible by n_q_heads={n_q_heads}"
        )));
    }

    let mut thresholds = vec![f32::NEG_INFINITY; n_bh];
    for (bh, slot) in thresholds.iter_mut().enumerate().take(n_bh) {
        let hq = bh % n_q_heads;
        let k = (*row.get(hq).ok_or_else(|| {
            rmlx_core::error::Error::Quant(format!(
                "sparse_attn: invariant violation — hq={hq} out of bounds (n_q_heads={n_q_heads})"
            ))
        })?) as usize;
        let mut all: Vec<f32> = Vec::with_capacity(n_tiles * top_per_tile);
        for t in 0..n_tiles {
            let tile_base = (t * n_bh + bh) * top_per_tile;
            let tile_slice = tile_top_scores
                .get(tile_base..tile_base + top_per_tile)
                .ok_or_else(|| {
                    rmlx_core::error::Error::Quant(format!(
                        "sparse_attn: invariant violation — tile_top_scores OOB at tile_base={tile_base} top_per_tile={top_per_tile}"
                    ))
                })?;
            for &v in tile_slice {
                if v.is_finite() {
                    all.push(v);
                }
            }
        }
        all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(all.len()).max(1) - 1;
        *slot = all.get(take).copied().unwrap_or(f32::NEG_INFINITY);
    }
    Ok(thresholds)
}

/// Build a `[n_bh]` f32 mlx `Array` from a flat threshold vec.
fn build_threshold_array(thr: &[f32], n_bh: i32) -> Result<Array> {
    let bytes: Vec<u8> = thr.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[n_bh], Dtype::F32)
}

#[cfg(test)]
#[path = "attention_dispatch_tests.rs"]
mod attention_dispatch_tests;
