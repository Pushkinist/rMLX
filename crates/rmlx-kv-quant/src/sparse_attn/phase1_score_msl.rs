// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over the
// dims params array for MSL dispatch.  Same pattern as
// `planar_flash_decode_msl`.
#![allow(unsafe_code)]

//! Phase 1: PlanarQuant-K QK scoring + per-tile top-4.
//!
//! # Kernel shape
//!
//! Grid: `(n_tiles, B * n_q_heads, 1)`.  Threadgroup: `(head_dim, 1, 1)`.
//! One threadgroup per `(B, head, tile)`.  Each of the `head_dim` threads
//! decodes its `K[dim]` element (bit-exact with `planar_fused_qk_msl`),
//! participates in a tree reduction for each per-token QK score, and
//! thread 0 maintains a per-tile top-4 heap.
//!
//! # Outputs
//!
//! * `tile_top_scores[n_tiles, n_bh, 4]` f32 — top-4 raw QK scores per
//!   `(B, H, tile)`.
//! * `tile_top_indices[n_tiles, n_bh, 4]` u32 — corresponding kv-seq
//!   positions (the global `t` index in `0..kv_seq`).
//! * `all_scores[n_bh, kv_seq]` f32 — per-position raw QK scores
//!   (kept for Phase 2 to apply the per-head threshold).
//!
//! # K decode
//!
//! Bit-exact with `metal/planar_flash_decode_p1.metal` /
//! `metal/planar_fused_qk_b4.metal` (4-bit Lloyd-Max codebook + per-pair
//! Givens rotation + per-pair f32 scale; 32 elements per "group", 16 pairs
//! per group).
//!
//! # Tile-size + head_dim ceiling
//!
//! Tile size is 64 tokens (matches the planar-flash precedent).  Max supported
//! `head_dim` is 256 (Gemma4-26B class) — the threadgroup arrays are
//! statically sized to that limit.
//!
//! # Dispatch counter
//!
//! [`phase1_score_dispatch_count`] returns the process-lifetime count of
//! `phase1_score` calls that reached the Metal enqueue point.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::planarquant::{planar_rotation_codebook, N_ROTATIONS};
use crate::turboquant::{lloyd_gaussian_codebook, GROUP_SIZE};
use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Constants (mirror planar_flash_decode) ────────────────────────────────────

/// Tokens per phase-1 tile.  Matches `planar_flash_decode_msl::TILE_SIZE`.
pub const TILE_SIZE: i32 = 64;

/// Maximum supported `head_dim`.  Mirrors `PLANAR_FLASH_HEAD_DIM_MAX`.
pub const SPARSE_ATTN_HEAD_DIM_MAX: i32 = 256;

/// Number of top-scores tracked per tile.  Fixed at 4.
pub const TOP_PER_TILE: i32 = 4;

// ── Dispatch counter ──────────────────────────────────────────────────────────

/// Incremented exactly once per `phase1_score` invocation that reaches
/// the Metal enqueue point.  Used by parity tests.
static PHASE1_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of phase-1 dispatches.
pub fn phase1_score_dispatch_count() -> u64 {
    PHASE1_DISPATCHES.load(Ordering::Relaxed)
}

// ── MSL header builder ────────────────────────────────────────────────────────

fn build_header() -> Result<String> {
    let cb = lloyd_gaussian_codebook(4)?;
    let n_centroids = cb.len();
    let rot_cb = planar_rotation_codebook();
    if rot_cb.len() != N_ROTATIONS {
        return Err(Error::Mlx(format!(
            "phase1_score: rotation codebook length {got} != {expected}",
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
         constant float SA_ROT_CB[{N}][4] = {{\n{entries}\n}};\n",
        N = N_ROTATIONS,
        entries = rot_entries.join(",\n")
    );
    let _ = write!(
        s,
        "\n// Lloyd-Max N(0,1) centroid codebook (4-bit, {n_centroids} entries).\n\
         constant float SA_CB[{n_centroids}] = {{\n{cb_entries}\n}};\n",
        n_centroids = n_centroids,
        cb_entries = cb_entries.join(",\n")
    );
    let _ = write!(s, "\n#define SA_TILE_SIZE {TILE_SIZE}u\n");
    let _ = writeln!(s, "#define SA_TOP_PER_TILE {TOP_PER_TILE}u");

    Ok(s)
}

// ── Pass-1 kernel source ─────────────────────────────────────────────────────
//
// Buffer layout (must match `add_input` order in `phase1_score`):
//   0. query            : f32  [B * n_q_heads * head_dim]
//   1. k_codes          : u32  [B * kv_h * kv_seq * (head_dim/32) * 4]
//   2. k_scales         : f32  [B * kv_h * kv_seq * head_dim/2]
//   3. k_rot32          : u32  [B * kv_h * kv_seq * head_dim/16]
//   4. scale_arr        : f32  [1]                — softmax pre-scale
//   5. dims             : u32  [6]                — see below
//
// `dims` layout:
//   dims[0] = head_dim          (D, also threadgroup size)
//   dims[1] = kv_seq            (S_kv)
//   dims[2] = B * n_q_heads     (n_bh, grid Y bound)
//   dims[3] = kv_h
//   dims[4] = heads_per_kv      (n_q_heads / kv_h)
//   dims[5] = n_tiles           (ceil_div(kv_seq, TILE_SIZE))
//
// Outputs:
//   0. tile_top_scores  : f32  [n_tiles * n_bh * TOP_PER_TILE]
//   1. tile_top_indices : u32  [n_tiles * n_bh * TOP_PER_TILE]
//   2. all_scores       : f32  [n_bh * kv_seq]

const P1_SOURCE: &str = include_str!("../metal/sparse_attn_phase1_score.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static P1_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P1_HEADER: OnceLock<std::result::Result<String, String>> = OnceLock::new();

/// The header is generated (codebook + tile constants), so it is memoised. The
/// body is a compile-time constant and needs no cache.
pub(crate) fn p1_header() -> Result<&'static str> {
    P1_HEADER
        .get_or_init(|| build_header().map_err(|e| e.to_string()))
        .as_deref()
        .map_err(|e| Error::Mlx(format!("phase1_score header build: {e}")))
}

fn p1_kernel() -> Result<&'static MetalKernel> {
    let header = p1_header()?;
    P1_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_sparse_attn_phase1_score_v4",
                header,
                P1_SOURCE,
                &[
                    "query",
                    "k_codes",
                    "k_scales",
                    "k_rot32",
                    "scale_arr",
                    "dims",
                ],
                &["tile_top_scores", "tile_top_indices", "all_scores"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("phase1_score kernel init: {e}")))
}

// ── Public dispatcher ────────────────────────────────────────────────────────

/// Phase-1 outputs returned by [`phase1_score`].
#[allow(
    clippy::exhaustive_structs,
    reason = "closed return type — three GPU buffers + n_tiles is the complete contract"
)]
#[derive(Debug)]
pub struct Phase1Out {
    /// `[n_tiles, n_bh, TOP_PER_TILE]` f32 — top-4 raw QK scores per
    /// `(B, H, tile)`, descending (`[..., 0]` is the largest).
    pub tile_top_scores: Array,
    /// `[n_tiles, n_bh, TOP_PER_TILE]` u32 — corresponding kv-seq positions.
    pub tile_top_indices: Array,
    /// `[n_bh, kv_seq]` f32 — per-position raw QK scores (Phase 2 input).
    pub all_scores: Array,
    /// Number of tiles `= ceil(kv_seq / TILE_SIZE)`.
    pub n_tiles: i32,
}

/// Run the Phase-1 score kernel: PlanarQuant K decode + QK + per-tile top-4.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn phase1_score(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_rot32: &Array,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<Phase1Out> {
    if head_dim <= 0 || head_dim % (GROUP_SIZE as i32) != 0 {
        return Err(Error::Quant(format!(
            "phase1_score: head_dim={head_dim} must be a positive multiple of \
             GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "phase1_score: head_dim={head_dim} must be a power of two"
        )));
    }
    if head_dim > SPARSE_ATTN_HEAD_DIM_MAX {
        let max = SPARSE_ATTN_HEAD_DIM_MAX;
        return Err(Error::Quant(format!(
            "phase1_score: head_dim={head_dim} exceeds SPARSE_ATTN_HEAD_DIM_MAX={max}"
        )));
    }
    if heads_per_kv <= 0 || b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "phase1_score: b={b}, kv_h={kv_h}, kv_seq={kv_seq}, heads_per_kv={heads_per_kv} \
             must all be > 0"
        )));
    }

    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;
    let n_tiles = (kv_seq + TILE_SIZE - 1) / TILE_SIZE;

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
        // SAFETY: `dims` is stack-local fully-initialised `[u32; 6]`; u32
        // alignment ≥ u8; byte length 6*4 matches size_of; borrow scoped to
        // block (Array::from_bytes copies before scope ends).
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(dims.as_ptr().cast::<u8>(), 6 * 4) };
        Array::from_bytes(bytes, &[6], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("phase1_score dims: {e}")))?
    };

    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.

    let kern = p1_kernel()?;
    let mut inv = MetalKernelInvoke::new();
    inv.add_input(&q_f32)?;
    inv.add_input(&codes_flat)?;
    inv.add_input(&scales_flat)?;
    inv.add_input(&rot_flat)?;
    inv.add_input(&scale_arr)?;
    inv.add_input(&dims_arr)?;

    let top_total: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(TOP_PER_TILE);
    let all_total: i64 = i64::from(n_bh) * i64::from(kv_seq);
    if top_total > i64::from(i32::MAX) || all_total > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "phase1_score: output length exceeds i32::MAX (top={top_total}, all={all_total})"
        )));
    }
    inv.add_output_shape(&[top_total as i32], Dtype::F32)?;
    inv.add_output_shape(&[top_total as i32], Dtype::U32)?;
    inv.add_output_shape(&[all_total as i32], Dtype::F32)?;

    // set_grid takes total threads; mlx-c divides by threadgroup size to get
    // threadgroup count: n_tiles_threadgroups = grid_x / head_dim = n_tiles.
    let grid_x: i64 = i64::from(n_tiles) * i64::from(head_dim);
    let grid_y: i64 = i64::from(n_bh);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "phase1_score: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    inv.set_grid(grid_x as i32, grid_y as i32, 1)?;
    inv.set_thread_group(head_dim, 1, 1)?;

    PHASE1_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut outs = kern.apply(inv, device)?;
    if outs.len() < 3 {
        return Err(Error::Mlx(
            "phase1_score: expected 3 outputs (tile_top_scores, tile_top_indices, all_scores)"
                .into(),
        ));
    }
    let tile_top_scores = outs.remove(0);
    let tile_top_indices = outs.remove(0);
    let all_scores = outs.remove(0);

    let tile_top_scores = tile_top_scores.reshape(&[n_tiles, n_bh, TOP_PER_TILE], device)?;
    let tile_top_indices = tile_top_indices.reshape(&[n_tiles, n_bh, TOP_PER_TILE], device)?;
    let all_scores = all_scores.reshape(&[n_bh, kv_seq], device)?;

    Ok(Phase1Out {
        tile_top_scores,
        tile_top_indices,
        all_scores,
        n_tiles,
    })
}
