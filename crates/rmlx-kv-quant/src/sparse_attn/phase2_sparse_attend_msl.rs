// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over the
// dims params array for MSL dispatch.
#![allow(unsafe_code)]

//! Phase 2: sparse attend over surviving tokens.
//!
//! # Kernel shape
//!
//! Grid: `(n_tiles, B * n_q_heads, 1)`.  Threadgroup: `(head_dim, 1, 1)`.
//! One threadgroup per `(B, head, tile)`.  Each thread handles ONE
//! `head_dim` slot of the per-tile partial output.
//!
//! # Inputs
//!
//! * `head_threshold[n_bh]` f32 — per-`(B, H)` raw QK threshold (the
//!   K-th largest score, computed CPU/MLX-side between P1 and P2 from
//!   the Phase-1 `tile_top_scores`).
//! * `all_scores[n_bh, kv_seq]` f32 — per-position raw QK scores from
//!   Phase 1.
//! * `k_codes`, `k_scales`, `k_rot32` — packed PlanarQuant K buffers
//!   (Phase 2 re-decodes K for the survivors; this preserves the
//!   "single fused kernel" property and avoids materialising a full
//!   K tensor across the P1/P2 boundary).
//! * `v_flat` — native-dtype V (bf16/f16/f32), shape
//!   `[B, kv_h, kv_seq, head_dim]`.
//!
//! # Early exit
//!
//! Thread 0 scans `all_scores[bh, tile_start..tile_end]` for any token
//! whose score ≥ `head_threshold[bh]`.  When none survive (the common
//! case at top-1024 of 50K), the kernel writes a sentinel LSE
//! `(tile_max=-inf, tile_sum_exp=0)` plus a zero `partial_o` slice and
//! returns — no K/V decode, no softmax work.
//!
//! # Outputs
//!
//! * `partial_o[n_tiles, n_bh, head_dim]` f32
//! * `tile_lse[n_tiles, n_bh, 2]` f32 — `(tile_max, tile_sum_exp)` per
//!   `(B, H, tile)`.
//!
//! The P2 LSE-merge kernel (`planar_flash_decode_msl`'s P2 source)
//! collapses these into the final attention output across tiles.  Phase 2
//! emits the same per-tile layout so the caller can either invoke the
//! P2 kernel directly or run its own reduction.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::planarquant::{planar_rotation_codebook, N_ROTATIONS};
use crate::turboquant::{lloyd_gaussian_codebook, GROUP_SIZE};
use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use super::phase1_score_msl::{SPARSE_ATTN_HEAD_DIM_MAX, TILE_SIZE};

// ── Dispatch counter ──────────────────────────────────────────────────────────

static PHASE2_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of phase-2 dispatches.
pub fn phase2_sparse_attend_dispatch_count() -> u64 {
    PHASE2_DISPATCHES.load(Ordering::Relaxed)
}

// ── MSL header builder ────────────────────────────────────────────────────────

fn build_header() -> Result<String> {
    let cb = lloyd_gaussian_codebook(4)?;
    let n_centroids = cb.len();
    let rot_cb = planar_rotation_codebook();
    if rot_cb.len() != N_ROTATIONS {
        return Err(Error::Mlx(format!(
            "phase2_sparse_attend: rotation codebook length {got} != {expected}",
            got = rot_cb.len(),
            expected = N_ROTATIONS
        )));
    }

    let rot_entries: Vec<String> = rot_cb
        .iter()
        .map(|e| {
            let c = f32::to_bits(e[0]);
            let neg_s = f32::to_bits(e[1]);
            let s = f32::to_bits(e[2]);
            let c2 = f32::to_bits(e[3]);
            format!(
                "    {{as_type<float>(0x{c:08X}u), as_type<float>(0x{neg_s:08X}u), \
                 as_type<float>(0x{s:08X}u), as_type<float>(0x{c2:08X}u)}}"
            )
        })
        .collect();

    let cb_entries: Vec<String> = cb
        .iter()
        .map(|&v| format!("    as_type<float>(0x{:08X}u)", f32::to_bits(v)))
        .collect();

    let mut s = String::new();
    let _ = write!(
        s,
        "\n// PlanarQuant 16-entry Givens rotation codebook (bit-exact with CPU).\n\
         constant float SA2_ROT_CB[{N}][4] = {{\n{entries}\n}};\n",
        N = N_ROTATIONS,
        entries = rot_entries.join(",\n")
    );
    let _ = write!(
        s,
        "\n// Lloyd-Max N(0,1) centroid codebook (4-bit, {n_centroids} entries).\n\
         constant float SA2_CB[{n_centroids}] = {{\n{cb_entries}\n}};\n",
        n_centroids = n_centroids,
        cb_entries = cb_entries.join(",\n")
    );
    let _ = write!(s, "\n#define SA2_TILE_SIZE {TILE_SIZE}u\n");

    Ok(s)
}

// ── Kernel source ────────────────────────────────────────────────────────────
//
// Buffer layout (must match `add_input` order in `phase2_sparse_attend`):
// K buffers are SEQUENCE-major (`[B, kv_seq, kv_h, ...]`); V stays head-major.
//   0. query           : f32  [n_bh * head_dim]
//   1. k_codes         : u32  [B * kv_seq * kv_h * (head_dim/32) * 4]
//   2. k_scales        : f32  [B * kv_seq * kv_h * head_dim/2]
//   3. k_rot32         : u32  [B * kv_seq * kv_h * head_dim/16]
//   4. v_flat          : bf16/f16/f32 [B * kv_h * kv_seq * head_dim]
//   5. all_scores      : f32  [n_bh * kv_seq]  (from Phase 1)
//   6. head_threshold  : f32  [n_bh]
//   7. scale_arr      : f32  [1]                — softmax pre-scale
//   8. dims            : u32  [6]              — see Phase 1
//
// Outputs:
//   0. partial_o : f32 [n_tiles * n_bh * head_dim]
//   1. tile_lse  : f32 [n_tiles * n_bh * 2]    (tile_max, tile_sum_exp)

const P2_SOURCE: &str = include_str!("../metal/sparse_attn_phase2_attend.metal");

// ── Pass-2 LSE merge MSL source (mirror planar_flash_decode P2) ──────────────
//
// Grid: (B * n_q_heads * head_dim, 1, 1).  Threadgroup: (head_dim, 1, 1).
// One threadgroup per (B, head); each thread handles one output dim.
//
// Inputs:
//   0. partial_o : f32 [n_tiles * n_bh * head_dim]
//   1. tile_lse  : f32 [n_tiles * n_bh * 2]
//   2. dims_p2   : u32 [3] = (head_dim, n_tiles, n_bh)
//
// Output:
//   0. dst       : f32 [n_bh * head_dim]
//
// This is structurally identical to the planar_flash_decode P2 merge but reads (m, l)
// from the unified `tile_lse` buffer instead of separate `tile_max` +
// `tile_sum_exp` arrays.  Mathematically: collapse n_tiles per-tile
// (m_t, l_t, O_t) into a single global (m, l, O) via log-sum-exp.
const MERGE_SOURCE: &str = include_str!("../metal/sparse_attn_phase2_merge.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_HEADER: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static MERGE_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

/// The header is generated (codebook + tile constants), so it is memoised. The
/// body is a compile-time constant and needs no cache.
pub(crate) fn p2_header() -> Result<&'static str> {
    P2_HEADER
        .get_or_init(|| build_header().map_err(|e| e.to_string()))
        .as_deref()
        .map_err(|e| Error::Mlx(format!("phase2_sparse_attend header build: {e}")))
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    let header = p2_header()?;
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_sparse_attn_phase2_sparse_attend_v4",
                header,
                P2_SOURCE,
                &[
                    "query",
                    "k_codes",
                    "k_scales",
                    "k_rot32",
                    "v_flat",
                    "all_scores",
                    "head_threshold",
                    "scale_arr",
                    "dims",
                ],
                &["partial_o", "tile_lse"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("phase2_sparse_attend kernel init: {e}")))
}

fn merge_kernel() -> Result<&'static MetalKernel> {
    MERGE_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_sparse_attn_phase2_lse_merge",
                "",
                MERGE_SOURCE,
                &["partial_o", "tile_lse", "dims_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("phase2 LSE merge kernel init: {e}")))
}

// ── Public dispatcher ────────────────────────────────────────────────────────

/// Phase-2 outputs returned by [`phase2_sparse_attend`].
#[allow(
    clippy::exhaustive_structs,
    reason = "closed return type — partial_o + tile_lse is the complete contract"
)]
#[derive(Debug)]
pub struct Phase2Out {
    /// `[n_tiles, n_bh, head_dim]` f32.
    pub partial_o: Array,
    /// `[n_tiles, n_bh, 2]` f32 — `(tile_max, tile_sum_exp)`.
    pub tile_lse: Array,
}

/// Phase-2 sparse-attend kernel: re-decode K/V on survivors, online softmax,
/// emit per-tile partial outputs + LSE state.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn phase2_sparse_attend(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_rot32: &Array,
    v: &Array,
    all_scores: &Array,
    head_threshold: &Array,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    n_tiles: i32,
    scale: f32,
    device: Device,
) -> Result<Phase2Out> {
    if head_dim <= 0 || head_dim % (GROUP_SIZE as i32) != 0 {
        return Err(Error::Quant(format!(
            "phase2_sparse_attend: head_dim={head_dim} must be a positive multiple of \
             GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "phase2_sparse_attend: head_dim={head_dim} must be a power of two"
        )));
    }
    if head_dim > SPARSE_ATTN_HEAD_DIM_MAX {
        let max = SPARSE_ATTN_HEAD_DIM_MAX;
        return Err(Error::Quant(format!(
            "phase2_sparse_attend: head_dim={head_dim} exceeds SPARSE_ATTN_HEAD_DIM_MAX={max}"
        )));
    }
    if heads_per_kv <= 0 || b <= 0 || kv_seq <= 0 || kv_h <= 0 || n_tiles <= 0 {
        return Err(Error::Quant(format!(
            "phase2_sparse_attend: b={b}, kv_h={kv_h}, kv_seq={kv_seq}, \
             heads_per_kv={heads_per_kv}, n_tiles={n_tiles} must all be > 0"
        )));
    }

    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;

    let q_total: i64 = i64::from(n_bh) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    let codes_per_tok: i64 = (i64::from(head_dim) / GROUP_SIZE as i64) * 4;
    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * codes_per_tok;
    let scales_total: i64 = tok_count * i64::from(head_dim) / 2;
    let rot_total: i64 = tok_count * i64::from(head_dim) / 16;

    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[scales_total as i32], device)?;
    let rot_flat = k_rot32.reshape(&[rot_total as i32], device)?;

    let v_total: i64 = tok_count * i64::from(head_dim);
    let v_flat = match v.dtype() {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => v.reshape(&[v_total as i32], device)?,
        Dtype::U8 | Dtype::U32 | Dtype::I32 => {
            let dt = v.dtype();
            return Err(Error::Quant(format!(
                "phase2_sparse_attend: V dtype must be F32 / Bf16 / F16, got {dt:?}"
            )));
        }
    };

    let all_scores_flat =
        all_scores.reshape(&[(i64::from(n_bh) * i64::from(kv_seq)) as i32], device)?;
    let head_threshold_flat = head_threshold.reshape(&[n_bh], device)?;

    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };

    let dims_arr = {
        let dims: [u32; 6] = [
            head_dim as u32,
            kv_seq as u32,
            n_bh as u32,
            kv_h as u32,
            heads_per_kv as u32,
            n_tiles as u32,
        ];
        // SAFETY: stack-local fully-initialised `[u32; 6]`; u32 alignment ≥ u8;
        // byte length 6*4 = size_of; borrow scoped; from_bytes copies.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(dims.as_ptr().cast::<u8>(), 6 * 4) };
        Array::from_bytes(bytes, &[6], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("phase2_sparse_attend dims: {e}")))?
    };

    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.
    //
    // `head_threshold` is built host-side between Phase 1 and Phase 2, but the
    // host sync that implies already happened in the caller, which reads
    // Phase-1's `tile_top_scores` back to CPU to compute it. Re-evaluating the
    // reshaped result here adds nothing.

    let kern = p2_kernel()?;
    let mut inv = MetalKernelInvoke::new();
    inv.add_input(&q_f32)?;
    inv.add_input(&codes_flat)?;
    inv.add_input(&scales_flat)?;
    inv.add_input(&rot_flat)?;
    inv.add_input(&v_flat)?;
    inv.add_input(&all_scores_flat)?;
    inv.add_input(&head_threshold_flat)?;
    inv.add_input(&scale_arr)?;
    inv.add_input(&dims_arr)?;

    let partial_total: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(head_dim);
    let lse_total: i64 = i64::from(n_tiles) * i64::from(n_bh) * 2;
    if partial_total > i64::from(i32::MAX) || lse_total > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "phase2_sparse_attend: output length exceeds i32::MAX (partial={partial_total}, lse={lse_total})"
        )));
    }
    inv.add_output_shape(&[partial_total as i32], Dtype::F32)?;
    inv.add_output_shape(&[lse_total as i32], Dtype::F32)?;

    // set_grid takes total threads; mlx-c divides by threadgroup size to get
    // threadgroup count: n_tiles_threadgroups = grid_x / head_dim = n_tiles.
    let grid_x: i64 = i64::from(n_tiles) * i64::from(head_dim);
    let grid_y: i64 = i64::from(n_bh);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "phase2_sparse_attend: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    inv.set_grid(grid_x as i32, grid_y as i32, 1)?;
    inv.set_thread_group(head_dim, 1, 1)?;

    PHASE2_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut outs = kern.apply(inv, device)?;
    if outs.len() < 2 {
        return Err(Error::Mlx(
            "phase2_sparse_attend: expected 2 outputs (partial_o, tile_lse)".into(),
        ));
    }
    let partial_o = outs.remove(0);
    let tile_lse = outs.remove(0);

    let partial_o = partial_o.reshape(&[n_tiles, n_bh, head_dim], device)?;
    let tile_lse = tile_lse.reshape(&[n_tiles, n_bh, 2], device)?;

    Ok(Phase2Out {
        partial_o,
        tile_lse,
    })
}

/// Merge Phase-2 per-tile partials + LSE state into the final attention output.
///
/// Mirror of `planar_flash_decode_msl`'s P2 kernel — collapses
/// `(n_tiles, n_bh, head_dim)` × `(n_tiles, n_bh, 2)` into
/// `(B, n_q_heads, 1, head_dim)` via log-sum-exp.
pub fn phase2_lse_merge(
    partial_o: &Array,
    tile_lse: &Array,
    b: i32,
    n_q_heads: i32,
    head_dim: i32,
    n_tiles: i32,
    device: Device,
) -> Result<Array> {
    let n_bh = b * n_q_heads;

    let partial_total: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(head_dim);
    let lse_total: i64 = i64::from(n_tiles) * i64::from(n_bh) * 2;

    let partial_flat = partial_o.reshape(&[partial_total as i32], device)?;
    let lse_flat = tile_lse.reshape(&[lse_total as i32], device)?;

    let dims_p2_arr = {
        let dims_p2: [u32; 3] = [head_dim as u32, n_tiles as u32, n_bh as u32];
        // SAFETY: stack-local `[u32; 3]`; alignment ≥ u8; length 3*4; scoped borrow.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(dims_p2.as_ptr().cast::<u8>(), 3 * 4) };
        Array::from_bytes(bytes, &[3], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("phase2_lse_merge dims_p2: {e}")))?
    };

    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.

    let kern = merge_kernel()?;
    let mut inv = MetalKernelInvoke::new();
    inv.add_input(&partial_flat)?;
    inv.add_input(&lse_flat)?;
    inv.add_input(&dims_p2_arr)?;

    let dst_len: i64 = i64::from(n_bh) * i64::from(head_dim);
    inv.add_output_shape(&[dst_len as i32], Dtype::F32)?;

    let grid_x: i64 = i64::from(n_bh) * i64::from(head_dim);
    if grid_x > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "phase2_lse_merge: grid x {grid_x} exceeds i32::MAX"
        )));
    }
    inv.set_grid(grid_x as i32, 1, 1)?;
    inv.set_thread_group(head_dim, 1, 1)?;

    let mut outs = kern.apply(inv, device)?;
    if outs.is_empty() {
        return Err(Error::Mlx("phase2_lse_merge: expected 1 output".into()));
    }
    let dst_flat = outs.remove(0);
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}
