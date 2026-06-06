// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel
// input data (params arrays) for MSL dispatch.
#![allow(unsafe_code)]

//! planar_flash_decode: fused QK + online softmax + SV in a single MSL pass
//! per (B, H, tile).
//!
//! # What this is
//!
//! This module extends the fused-QK kernel to a full FlashAttention-style decode:
//! pre-softmax scores, online softmax, and the SV product are all computed in
//! a single threadgroup pass per (batch, head, tile).  K is read directly from
//! the [`crate::storage::QuantPlanarK`] packed buffers (codes + per-pair
//! scales + 4-bit rotation indices), bit-exact with the fused-QK kernel.  V is passed in its
//! native dtype (bf16 / f16 / f32) — the kernel auto-typed pointer relies on
//! MSL's implicit promotion to `float` at read time, which halves V bandwidth
//! vs. an f32 astype upcast.
//!
//! Two Metal dispatches per call:
//! * **Pass 1** (`rmlx_planar_flash_decode_p1`): Per-tile threadgroups compute
//!   partial outputs + per-tile LSE state `(tile_max, tile_sum_exp)`.  Each
//!   threadgroup runs `head_dim` threads and processes `TILE_SIZE=64` tokens.
//! * **Pass 2** (`rmlx_planar_flash_decode_p2`): One threadgroup per
//!   `(B, head)` merges the per-tile partials via log-sum-exp into the final
//!   output `[B, n_q_heads, head_dim]`.
//!
//! Pass 2 must be a Metal kernel — the rMLX no-Python invariant forbids the
//! reduction the multi-turboquant Python harness uses for the merge.
//!
//! # Pattern reference
//!
//! `multi-turboquant/multi_turboquant/kernels/metal/fused_attention.py`
//! `PLANAR_FLASH_DECODE_KERNEL` (lines 151-272 of that file) is the reference
//! pass-1 layout: per-token unpack of packed K + V, threadgroup-shared online
//! softmax broadcast via four scalar slots `s_max / s_sum / s_corr / s_expsc`,
//! per-thread V accumulator in registers.  This flash-decode kernel differs
//! from the mtq reference in two ways:
//! * **rMLX PlanarQuant uses a 16-entry Givens rotation codebook + per-pair
//!   scale** (see `planar_fused_qk_msl`); the mtq reference assumes a single
//!   fixed 45° Hadamard rotation and one scale per token.  The K decode path
//!   here is verbatim with `planar_fused_qk_msl::build_qk_kernel_source`.
//! * **V is bf16 / f16 / f32 plain**, not planar-packed.  The dispatcher
//!   passes V to the kernel in its native dtype; mlx-c auto-types the
//!   `device const * v_flat` parameter and the kernel relies on MSL's
//!   implicit promotion to `float` at the per-thread read site.  This halves
//!   V bandwidth on the bf16 / f16 hot paths and matches the native-dtype V
//!   read pattern used by `turbo_flash_msl`.
//!
//! # Single-MLX claim
//!
//! Per CLAUDE.md "Single MLX process per Mac", callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//! This kernel is no exception.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::planarquant::{planar_rotation_codebook, N_ROTATIONS};
use crate::turboquant::{lloyd_gaussian_codebook, GROUP_SIZE};
use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Env-var gate (mirrors `turbo_flash_msl::turbo_flash_enabled`) ─────────────

/// Returns true when the planar_flash_decode kernel is enabled via env.
///
/// Default OFF until the NIAH gate (Bonsai + Qwen3.6 with `--kv-quant planar_k`)
/// has been run in the phase 4-5 follow-up executor.  The CLI flag
/// `--planar-flash-decode {on|off|auto}` in `rmlx-cli::commands::serve` is the
/// production switch; this env-var path mirrors the `RMLX_TURBO_FLASH` /
/// `OnceLock` pattern so unit tests and benches can opt in without going
/// through the CLI.
///
/// Precedence: env-var `RMLX_PLANAR_FLASH_DECODE=1` ⇒ ON; otherwise OFF.
pub fn planar_flash_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("RMLX_PLANAR_FLASH_DECODE").as_deref(),
            Ok("1")
        )
    })
}

// ── Dispatch counter (mirrors `TURBO_FLASH_DISPATCHES`) ───────────────────────

/// Incremented exactly once per `planar_flash_decode_sdpa` invocation that
/// reaches the P1 enqueue point.  Used by the NIAH harness in the phase 4-5
/// follow-up to prove the MSL kernel actually fired (vs. silently falling back
/// to the fused-QK + legacy softmax + legacy SV chain).
static PLANAR_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of planar_flash_decode P1 dispatches.
///
/// NIAH harness asserts `delta > 0` on ON cells and `delta == 0` on OFF
/// cells.  Production code does not consult this counter.
pub fn planar_flash_decode_dispatch_count() -> u64 {
    PLANAR_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

// ── MSL constants ─────────────────────────────────────────────────────────────

/// Tokens per pass-1 tile.  Matches `turbo_flash_msl::BLOCK_SIZE` so the two
/// flash kernels share a tuning surface; the multi-turboquant reference uses
/// 256 but on rMLX K8V4 the 64-token tile gave the best decode TPS at 32K
/// context (64-token tile gave the best decode TPS at 32K context).
pub(crate) const TILE_SIZE: i32 = 64;

/// Maximum supported `head_dim` for the planar_flash_decode kernels.  Sized
/// to the largest production head_dim (Gemma4-26B class: 256).  The kernel
/// uses static-sized threadgroup arrays of this length; raising it widens
/// SMEM and lowers occupancy.  See the threadgroup-memory ceiling comment in
/// `build_p1_kernel_source`.
pub(crate) const PLANAR_FLASH_HEAD_DIM_MAX: i32 = 256;

// ── MSL header builder ────────────────────────────────────────────────────────

/// Build the MSL header (rotation codebook + Lloyd-Max centroid codebook) for
/// the planar K decode.  Bit-exact with `planar_fused_qk_msl::build_qk_header`
/// — same Givens rotation table, same Lloyd-Max codebook, same constants.
///
/// `bits` is fixed at 4 (the only PlanarK storage variant
/// [`crate::storage::QuantPlanarK`] exposes today).  A 3-bit K kernel can be
/// added when a real caller materialises; until then the bits parameter is
/// retained on the public dispatcher signature for forward-compat only and
/// rejected at validation.
fn build_flash_header(bits: u8) -> Result<String> {
    let cb = lloyd_gaussian_codebook(bits)?;
    let n_centroids = cb.len();
    let rot_cb = planar_rotation_codebook();
    if rot_cb.len() != N_ROTATIONS {
        return Err(Error::Mlx(format!(
            "planar_flash_decode: rotation codebook length {got} != {expected}",
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
         constant float PF_ROT_CB[{N}][4] = {{\n{entries}\n}};\n",
        N = N_ROTATIONS,
        entries = rot_entries.join(",\n")
    );
    let _ = write!(
        s,
        "\n// Lloyd-Max N(0,1) centroid codebook ({bits}-bit, {n_centroids} entries).\n\
         constant float PF_CB[{n_centroids}] = {{\n{cb_entries}\n}};\n",
        bits = bits,
        n_centroids = n_centroids,
        cb_entries = cb_entries.join(",\n")
    );

    let _ = write!(s, "\n#define PF_TILE_SIZE {TILE_SIZE}u\n");

    Ok(s)
}

// ── Pass 1 MSL source ────────────────────────────────────────────────────────
//
// Grid: (n_tiles * head_dim, B * n_q_heads, 1).  Threadgroup: (head_dim, 1, 1).
// One threadgroup per (B, head, tile).  Each of the `head_dim` threads in the
// threadgroup is responsible for ONE dim of the head: it decodes its K[dim]
// from the packed buffers (bit-exact with fused-QK), participates in the tree
// reduction that produces the QK score, then reads its V[dim] in V's native
// dtype (bf16 / f16 / f32 — auto-typed pointer, implicit promotion to float)
// and accumulates the softmax-weighted contribution in a per-thread register.
//
// Buffer layout (must match `add_input` order in `planar_flash_decode_sdpa`):
// 0. query     : f32  [B * n_q_heads * head_dim]
// 1. k_codes   : u32  [B * kv_h * kv_seq * (head_dim / 32) * 4]
// 2. k_scales  : f32  [B * kv_h * kv_seq * (head_dim / 2)]
// 3. k_rot32   : u32  [B * kv_h * kv_seq * (head_dim / 16)]
// 4. v_flat    : bf16 / f16 / f32 [B * kv_h * kv_seq * head_dim] (native dtype)
// 5. mask_flat : f32  [B * n_q_heads * kv_seq] or [1] dummy when no mask
// 6. scale_arr : f32  [1]
// 7. dims      : u32  [7] — see below
//
// `dims` layout:
//   dims[0] = head_dim                (D, also threadgroup size)
//   dims[1] = kv_seq                  (S_kv)
//   dims[2] = B * n_q_heads           (n_bh, grid Y bound)
//   dims[3] = kv_h
//   dims[4] = heads_per_kv            (n_q_heads / kv_h)
//   dims[5] = n_tiles                 (ceil_div(kv_seq, TILE_SIZE))
//   dims[6] = has_mask                (0 or 1)
//
// Outputs (P1):
// 0. partial_o     : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max      : f32 [n_tiles * n_bh]
// 2. tile_sum_exp  : f32 [n_tiles * n_bh]

#[allow(
    clippy::uninlined_format_args,
    reason = "named template args keep each MSL placeholder traceable to its Rust source value"
)]
fn build_p1_kernel_source() -> String {
    // 4-bit code unpack: 8 elements per u32, 4 bits each.  The 3-bit variant
    // was wired speculatively in the initial port; YAGNI per CLAUDE.md
    // Simplicity rule 4 (inline beats premature factoring) — drop the dead
    // code path and re-add it the day a 3-bit K codec needs it.
    let vals_per_word: u32 = 8;
    let decode_idx_expr = "(k_codes[code_word_abs] >> ((elem_in_group & 7u) * 4u)) & 0xFu";
    let words_per_group: u32 = (GROUP_SIZE / 8) as u32; // 4 u32 words per 32-elem group

    format!(
        r"
    // ── Thread / threadgroup coordinates ──────────────────────────────────
    uint head_dim     = dims[0];
    uint kv_seq       = dims[1];
    uint n_bh         = dims[2];
    uint kv_h         = dims[3];
    uint heads_per_kv = dims[4];
    uint n_tiles      = dims[5];
    uint has_mask     = dims[6];

    uint tile_idx = threadgroup_position_in_grid.x;
    uint bh       = threadgroup_position_in_grid.y;
    uint tid      = thread_position_in_threadgroup.x;

    if (bh >= n_bh)        return;
    if (tile_idx >= n_tiles) return;

    uint n_q_heads = kv_h * heads_per_kv;
    uint b         = bh / n_q_heads;
    uint hq        = bh % n_q_heads;
    uint kv_h_idx  = hq / heads_per_kv;

    uint tile_start = tile_idx * PF_TILE_SIZE;
    uint tile_end   = tile_start + PF_TILE_SIZE;
    if (tile_end > kv_seq) tile_end = kv_seq;

    // Per-token K layout (matches fused-QK storage contract bit-exact).
    uint codes_words_per_tok    = (head_dim / 32u) * 4u;
    uint scales_pairs_per_tok   = head_dim / 2u;
    uint rot_words_per_tok      = head_dim / 16u;

    // Group bookkeeping (32 elements per group; 16 pairs per group).
    uint group_id_in_head = tid / 32u;
    uint elem_in_group    = tid % 32u;
    uint pair_in_group    = elem_in_group / 2u;
    uint elem_in_pair     = elem_in_group & 1u;

    // ── Load Q once into threadgroup memory ──────────────────────────────
    // SMEM ceiling: head_dim in 128 or 256 (dispatcher-enforced via
    // PLANAR_FLASH_HEAD_DIM_MAX); 256 floats * 4 B = 1 KiB per buffer keeps
    // three threadgroup arrays at 3 KiB total, well within the Apple GPU
    // 32 KiB threadgroup-memory budget and tight enough not to gate
    // occupancy on the head_dim=128 hot path.
    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];

    // ── Online softmax broadcast slots ───────────────────────────────────
    threadgroup float s_max[1];
    threadgroup float s_sum[1];
    threadgroup float s_corr[1];
    threadgroup float s_expsc[1];

    if (tid == 0u) {{
        s_max[0] = -INFINITY;
        s_sum[0] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread V accumulator (registers — never spills to threadgroup mem).
    float acc_v = 0.0f;

    // Shared scratch for K decode (pre- and post-Givens) and the dot
    // reduction.  Sized to the head_dim ceiling — see q_shared comment above.
    threadgroup float k_pre_rot[256];
    threadgroup float dot_shared[256];

    for (uint t = tile_start; t < tile_end; t++) {{
        // ── Decode K[tid] for this (b, kv_h_idx, t) ──────────────────────
        uint kv_tok          = (b * kv_h + kv_h_idx) * kv_seq + t;
        uint codes_tok_off   = kv_tok * codes_words_per_tok;
        uint scales_tok_off  = kv_tok * scales_pairs_per_tok;
        uint rot_tok_off     = kv_tok * rot_words_per_tok;

        // Code word + nibble.
        uint code_word_in_group = elem_in_group / {vals_per_word_u}u;
        uint code_word_abs      = codes_tok_off + group_id_in_head * {words_per_group_u}u + code_word_in_group;
        uint cb_idx             = {decode_idx_expr};

        // Per-pair scale.
        uint pair_global = group_id_in_head * 16u + pair_in_group;
        float scale_pair = k_scales[scales_tok_off + pair_global];

        // Per-pair Givens rotation index (4-bit, 8 pairs per u32, 2 words per group).
        uint rot_word_in_group = pair_in_group / 8u;
        uint rot_word_abs      = rot_tok_off + group_id_in_head * 2u + rot_word_in_group;
        uint rot_shift         = (pair_in_group & 7u) * 4u;
        uint rot_idx           = (k_rot32[rot_word_abs] >> rot_shift) & 0xFu;

        // Pre-rotation: centroid x per-pair scale.
        k_pre_rot[tid] = PF_CB[cb_idx] * scale_pair;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply R^T to (y_a, y_b) — matches planar_fused_qk exactly.
        uint pair_base = tid - elem_in_pair;
        float ya = k_pre_rot[pair_base];
        float yb = k_pre_rot[pair_base + 1u];

        float c     = PF_ROT_CB[rot_idx][0];
        float neg_s = PF_ROT_CB[rot_idx][1];
        float sv    = PF_ROT_CB[rot_idx][2];
        float c2    = PF_ROT_CB[rot_idx][3];

        float k_val = (elem_in_pair == 0u)
            ? (c * ya + sv * yb)
            : (neg_s * ya + c2 * yb);

        // ── QK dot product + tree reduction ─────────────────────────────
        dot_shared[tid] = q_shared[tid] * k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // REQUIRES head_dim to be a power of two (dispatcher rejects non-pow-2).
        for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {{
            if (tid < stride) {{
                dot_shared[tid] += dot_shared[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // ── Thread 0: online softmax update + broadcast ──────────────────
        if (tid == 0u) {{
            float raw = dot_shared[0] * scale_arr[0];
            // Mask is per (b, q_head, t) — add inside thread 0.
            float mask_val = 0.0f;
            if (has_mask != 0u) {{
                mask_val = mask_flat[(b * n_q_heads + hq) * kv_seq + t];
            }}
            float score = raw + mask_val;

            float old_max = s_max[0];
            float new_max = (score > old_max) ? score : old_max;
            float corr    = exp(old_max - new_max);
            float es      = exp(score - new_max);

            s_max[0]   = new_max;
            s_sum[0]   = s_sum[0] * corr + es;
            s_corr[0]  = corr;
            s_expsc[0] = es;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── V read + softmax-weighted accumulation ──────────────────────
        // V layout: [B, kv_h, kv_seq, head_dim], flat.
        float corr = s_corr[0];
        float es   = s_expsc[0];

        uint v_off = ((b * kv_h + kv_h_idx) * kv_seq + t) * head_dim + tid;
        float v_val = v_flat[v_off];

        acc_v = acc_v * corr + es * v_val;
    }}

    // ── Write per-tile partials ──────────────────────────────────────────
    uint out_base = (tile_idx * n_bh + bh) * head_dim;
    partial_o[out_base + tid] = acc_v;

    if (tid == 0u) {{
        uint meta = tile_idx * n_bh + bh;
        tile_max[meta]      = s_max[0];
        tile_sum_exp[meta]  = s_sum[0];
    }}
",
        vals_per_word_u = vals_per_word,
        words_per_group_u = words_per_group,
        decode_idx_expr = decode_idx_expr,
    )
}

// ── Pass 2 MSL source ────────────────────────────────────────────────────────
//
// Grid: (B * n_q_heads * head_dim, 1, 1).  Threadgroup: (head_dim, 1, 1).
// One threadgroup per (B, head); each thread handles one output dim.
//
// Buffer layout:
// 0. partial_o    : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max     : f32 [n_tiles * n_bh]
// 2. tile_sum_exp : f32 [n_tiles * n_bh]
// 3. dims_p2      : u32 [3] — {head_dim, n_tiles, n_bh}
//
// Output:
// 0. dst          : f32 [n_bh * head_dim]
const P2_SOURCE: &str = r"
    uint head_dim = dims_p2[0];
    uint n_tiles  = dims_p2[1];
    uint n_bh     = dims_p2[2];

    uint tid = thread_position_in_threadgroup.x;
    uint bh  = threadgroup_position_in_grid.x;
    if (bh >= n_bh) return;

    // ── Find global max across tiles (single-thread scan, broadcast) ─────
    threadgroup float g_max_buf[1];
    threadgroup float g_sum_buf[1];

    if (tid == 0u) {
        float gmax = -INFINITY;
        for (uint t = 0u; t < n_tiles; t++) {
            float tmax = tile_max[t * n_bh + bh];
            if (tmax > gmax) gmax = tmax;
        }
        g_max_buf[0] = gmax;

        // Sum the corrected per-tile masses for the LSE denominator.
        float gsum = 0.0f;
        for (uint t = 0u; t < n_tiles; t++) {
            float tmax = tile_max[t * n_bh + bh];
            float tsum = tile_sum_exp[t * n_bh + bh];
            gsum += exp(tmax - gmax) * tsum;
        }
        g_sum_buf[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = g_max_buf[0];
    float global_sum = g_sum_buf[0];
    float inv_sum    = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // ── Merge partial outputs for this dim ───────────────────────────────
    if (tid < head_dim) {
        float accum = 0.0f;
        for (uint t = 0u; t < n_tiles; t++) {
            float tmax = tile_max[t * n_bh + bh];
            float corr = exp(tmax - global_max);
            float pv   = partial_o[(t * n_bh + bh) * head_dim + tid];
            accum += corr * pv;
        }
        dst[bh * head_dim + tid] = accum * inv_sum;
    }
";

// ── Kernel singletons ─────────────────────────────────────────────────────────

static P1_KERNEL_V4: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

static P1_HEADER_V4: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static P1_SOURCE_V4: OnceLock<String> = OnceLock::new();

fn p1_header_v4() -> Result<&'static str> {
    P1_HEADER_V4
        .get_or_init(|| build_flash_header(4).map_err(|e| e.to_string()))
        .as_deref()
        .map_err(|e| Error::Mlx(format!("planar_flash_decode header build: {e}")))
}

fn p1_source_v4() -> &'static str {
    P1_SOURCE_V4.get_or_init(build_p1_kernel_source)
}

fn p1_kernel_v4() -> Result<&'static MetalKernel> {
    let header = p1_header_v4()?;
    let source = p1_source_v4();
    P1_KERNEL_V4
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_planar_flash_decode_p1_v4",
                header,
                source,
                &[
                    "query",
                    "k_codes",
                    "k_scales",
                    "k_rot32",
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
        .map_err(|e| Error::Mlx(format!("planar_flash_decode P1 kernel init: {e}")))
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_planar_flash_decode_p2",
                "", // No header — P2 is generic over (head_dim, n_tiles, n_bh).
                P2_SOURCE,
                &["partial_o", "tile_max", "tile_sum_exp", "dims_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("planar_flash_decode P2 kernel init: {e}")))
}

// ── Public dispatcher ────────────────────────────────────────────────────────

/// Run the planar_flash_decode kernel (fused QK + online softmax + SV).
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`.
/// * `k_codes`   — packed K codes from a [`crate::storage::QuantPlanarK`]
///   buffer (flat `u32` of length `B * kv_h * kv_seq * (head_dim/32) * 4`).
/// * `k_scales`  — per-pair scales (`f32`, flat
///   `B * kv_h * kv_seq * head_dim/2`).
/// * `k_rot32`   — 4-bit rotation indices (`u32`, flat
///   `B * kv_h * kv_seq * head_dim/16`).
/// * `v`         — bf16 / f16 / f32 V, shape `[B, kv_h, kv_seq, head_dim]`.
///   Read in its native dtype via implicit promotion to `float` inside MSL;
///   the dispatcher does NOT astype-upcast.  Other dtypes are rejected.
/// * `additive_mask` — optional `f32 [B, n_q_heads, 1, kv_seq]`.
/// * `b`, `kv_h`, `kv_seq`, `head_dim`, `heads_per_kv` — shape metadata.
/// * `bits`      — K code bit-width.  Must be `4` (only PlanarK 4-bit storage
///   is wired today); retained on the signature for forward-compat.
/// * `scale`     — softmax pre-scale (typically `1/sqrt(head_dim)`).
/// * `device`    — MLX device (`Device::Gpu`).
///
/// # Output
///
/// `f32` array of shape `[B, n_q_heads, 1, head_dim]`.  Caller may cast to
/// match the legacy SDPA output dtype.
///
/// # Errors
///
/// Returns `Error::Quant` for shape contract violations
/// (head_dim not a power of two, not divisible by GROUP_SIZE, etc.).
/// Returns `Error::Mlx` for kernel build / dispatch failures.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn planar_flash_decode_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    k_rot32: &Array,
    v: &Array,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    bits: u8,
    scale: f32,
    device: Device,
) -> Result<Array> {
    if bits != 4 {
        return Err(Error::Quant(format!(
            "planar_flash_decode: bits must be 4 (only PlanarK 4-bit storage is wired today), \
             got {bits}"
        )));
    }
    if head_dim <= 0 || head_dim % (GROUP_SIZE as i32) != 0 {
        return Err(Error::Quant(format!(
            "planar_flash_decode: head_dim={head_dim} must be a positive multiple of \
             GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "planar_flash_decode: head_dim={head_dim} must be a power of two for the \
             tree reduction; caller falls back to the planar fused-QK chain"
        )));
    }
    if head_dim > PLANAR_FLASH_HEAD_DIM_MAX {
        let max = PLANAR_FLASH_HEAD_DIM_MAX;
        return Err(Error::Quant(format!(
            "planar_flash_decode: head_dim={head_dim} exceeds PLANAR_FLASH_HEAD_DIM_MAX={max}; \
             raise the static threadgroup-array sizes in build_p1_kernel_source to support it"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "planar_flash_decode: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "planar_flash_decode: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }

    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;
    let n_tiles = (kv_seq + TILE_SIZE - 1) / TILE_SIZE;

    // ── Flatten Q to [n_bh * head_dim] f32 ────────────────────────────────
    let q_total: i64 = i64::from(n_bh) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    // ── Flatten K buffers ─────────────────────────────────────────────────
    let codes_per_tok: i64 = (i64::from(head_dim) / GROUP_SIZE as i64) * 4;
    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * codes_per_tok;
    let scales_total: i64 = tok_count * i64::from(head_dim) / 2;
    let rot_total: i64 = tok_count * i64::from(head_dim) / 16;

    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[scales_total as i32], device)?;
    let rot_flat = k_rot32.reshape(&[rot_total as i32], device)?;

    // ── V flat — keep native dtype (bf16 / f16 / f32). ────────────────────
    // The kernel param is auto-typed by mlx-c from the array dtype and the
    // MSL body uses `float v_val = v_flat[v_off];` to rely on implicit
    // promotion to float.  This halves V bandwidth vs. an f32 astype upcast
    // on the bf16/f16 hot paths (halves V bandwidth vs. an f32 astype upcast).
    let v_total: i64 = tok_count * i64::from(head_dim);
    let v_flat = match v.dtype() {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => v.reshape(&[v_total as i32], device)?,
        Dtype::U8 | Dtype::U32 | Dtype::I32 => {
            let dt = v.dtype();
            return Err(Error::Quant(format!(
                "planar_flash_decode: V dtype must be F32 / Bf16 / F16, got {dt:?}"
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
            .map_err(|e| Error::Mlx(format!("planar_flash_decode dummy mask: {e}")))?
    };

    // ── scale_arr ─────────────────────────────────────────────────────────
    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };

    // ── dims (7 u32) ──────────────────────────────────────────────────────
    // Layout mirrors the kernel buffer-layout comment above.  `bits` is no
    // longer carried — the dispatcher validates `bits == 4` up-front and the
    // kernel statics only have a 4-bit variant.
    let dims_arr = {
        let dims: [u32; 7] = [
            head_dim as u32,
            kv_seq as u32,
            n_bh as u32,
            kv_h as u32,
            heads_per_kv as u32,
            n_tiles as u32,
            has_mask,
        ];
        // SAFETY:
        // * `dims` is a stack-local `[u32; 7]` fully initialised above.
        // * `u32` has stricter alignment than `u8`, so the cast is sound.
        // * The byte length `7 * 4` equals `size_of::<[u32; 7]>()`.
        // * The borrow is bounded by the enclosing block; `Array::from_bytes`
        //   copies into mlx storage before this scope ends.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(dims.as_ptr().cast::<u8>(), 7 * 4) };
        Array::from_bytes(bytes, &[7], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("planar_flash_decode dims: {e}")))?
    };

    // Materialise inputs to flush any pending lazy ops before kernel dispatch
    // (guards against the planar_quantize atomic-OR race discovered during
    // fused-QK development).
    q_f32.eval()?;
    codes_flat.eval()?;
    scales_flat.eval()?;
    rot_flat.eval()?;
    v_flat.eval()?;
    if has_mask == 1 {
        mask_flat.eval()?;
    }
    scale_arr.eval()?;
    dims_arr.eval()?;

    // ── P1 dispatch ───────────────────────────────────────────────────────
    let kern_p1 = p1_kernel_v4()?;
    let mut inv_p1 = MetalKernelInvoke::new();
    inv_p1.add_input(&q_f32)?;
    inv_p1.add_input(&codes_flat)?;
    inv_p1.add_input(&scales_flat)?;
    inv_p1.add_input(&rot_flat)?;
    inv_p1.add_input(&v_flat)?;
    inv_p1.add_input(&mask_flat)?;
    inv_p1.add_input(&scale_arr)?;
    inv_p1.add_input(&dims_arr)?;

    let partial_o_len: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(head_dim);
    let tile_meta_len: i64 = i64::from(n_tiles) * i64::from(n_bh);
    if partial_o_len > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "planar_flash_decode: partial_o length {partial_o_len} exceeds i32::MAX"
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
            "planar_flash_decode: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    inv_p1.set_grid(grid_x as i32, grid_y as i32, 1)?;
    inv_p1.set_thread_group(head_dim, 1, 1)?;

    // Counter increment at the actual P1 enqueue point — after all validation
    // gates, immediately before the kernel `.apply()` call.
    PLANAR_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut p1_outs = kern_p1.apply(inv_p1, device)?;
    if p1_outs.len() < 3 {
        return Err(Error::Mlx(
            "planar_flash_decode P1: expected 3 outputs".into(),
        ));
    }
    let partial_o = p1_outs.remove(0);
    let tile_max = p1_outs.remove(0);
    let tile_sum_exp = p1_outs.remove(0);

    // ── P2 dispatch ───────────────────────────────────────────────────────
    let dims_p2_arr = {
        let dims_p2: [u32; 3] = [head_dim as u32, n_tiles as u32, n_bh as u32];
        // SAFETY:
        // * `dims_p2` is a stack-local `[u32; 3]` fully initialised above.
        // * `u32` alignment ≥ `u8`; the byte-cast is sound.
        // * Byte length `3 * 4` equals `size_of::<[u32; 3]>()`.
        // * Borrow is scoped; `Array::from_bytes` copies into mlx storage.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(dims_p2.as_ptr().cast::<u8>(), 3 * 4) };
        Array::from_bytes(bytes, &[3], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("planar_flash_decode dims_p2: {e}")))?
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
            "planar_flash_decode P2: grid x {p2_grid_x} exceeds i32::MAX"
        )));
    }
    inv_p2.set_grid(p2_grid_x as i32, 1, 1)?;
    inv_p2.set_thread_group(head_dim, 1, 1)?;

    let mut p2_outs = kern_p2.apply(inv_p2, device)?;
    if p2_outs.is_empty() {
        return Err(Error::Mlx(
            "planar_flash_decode P2: expected 1 output".into(),
        ));
    }
    let dst_flat = p2_outs.remove(0);

    // Reshape to canonical SDPA output.
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}

#[cfg(test)]
#[path = "planar_flash_decode_msl_tests.rs"]
mod tests;
