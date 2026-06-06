//! TurboSym3 K-side fused-QK MSL kernel.
//!
//! # What this is
//!
//! Single MSL kernel that consumes the TurboQuant 3-bit packed K codes
//! (3 u32 per 32-element group + one f32 scale per group; codebook =
//! 8-entry Lloyd-Max N(0,1)) **directly** and computes pre-softmax QK
//! scores `[B, n_q_heads, 1, S_kv]` without materialising a bf16 / f32 K
//! tensor in HBM.  Mirrors [`crate::q8_fused_qk_msl`] (Executor B) shape,
//! threadgroup layout, mask handling and Q reshape; only the K-decode
//! body differs (3-bit unpack + codebook lookup vs q8 affine).
//!
//! # Codec contract
//!
//! Target [`KvQuant`](crate::quant::KvQuant): `TurboSym3` — K stored as
//! TurboQuant 3-bit (group_size=32, Lloyd-Max optimal N(0,1) codebook,
//! 8 centroids).  V side is also turbo3 but is the SDPA caller's
//! responsibility — this kernel produces only the pre-softmax QK score.
//!
//! Bit-exact with [`crate::k8vturbo3_append_msl::turbo_dequantize_v3_gpu`]:
//!
//! * `codes`  u32 `[B * kv_h * S_kv * (D * 3 / 32)]` — 3 u32 words per
//!   32-element group, LSB-first across a 96-bit per-group stream.
//! * `scales` f32 `[B * kv_h * S_kv * (D / 32)]` — one f32 per group of 32.
//! * Codebook: 8-entry Lloyd-Max N(0,1) (`CB3` in `k8vturbo3_append_msl`).
//! * **No WHT / Hadamard correction**: the turbo3 K codec is plain codebook
//!   lookup × scale; the centroids are the dequantized value directly.
//!
//! # Kernel shape
//!
//! Grid `(S_kv * D, B * n_q_heads, 1)`; threadgroup `(D, 1, 1)`.  Each
//! threadgroup computes one score `out[b, hq, 0, s_kv]`; per-thread handles
//! one element of `head_dim`.
//!
//! # GQA support
//!
//! Identical to the q8 path: `n_q_heads = kv_h * heads_per_kv`; thread
//! group maps `(b, hq) -> kv_h_idx = hq / heads_per_kv` to share K.
//!
//! # A.y guard
//!
//! `TurboSym3` is K-side 3-bit, the Qwen-MoE 218->8641 PPL disaster applies.
//! The guard lives in [`rmlx_models::kv_cache::cache_type::validate_resolved`]
//! and rejects `Qwen3_5MoeForConditionalGeneration + TurboSym3` at session
//! start — the kernel does not re-check.
//!
//! # Reference
//!
//! * [`crate::q8_fused_qk_msl`] — Exec B dispatcher / threadgroup layout.
//! * [`crate::k8vturbo3_append_msl`] — bit-exact 3-bit unpack idiom (same
//!   32-bit window arithmetic, reframed per thread instead of per-output).

// unsafe_code: dims buffer byte cast — slice::from_raw_parts over a fixed
// u32 array for MSL kernel input. SAFETY justified at each use site.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::turboquant::GROUP_SIZE;

// ── Dispatch counter ─────────────────────────────────────────────────────────

/// Incremented once per [`turbo_k3_fused_qk_sdpa`] invocation that reaches
/// the Metal enqueue point.  Used by NIAH harness.
static TURBO_K3_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of turbo_k3_fused_qk dispatches.
pub fn turbo_k3_fused_qk_dispatch_count() -> u64 {
    TURBO_K3_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
}

// ── MSL kernel source ────────────────────────────────────────────────────────
//
// `dims` (uint vector, 5 elements):
//   dims[0] = head_dim       (D)            — also equals threadgroup size
//   dims[1] = kv_seq         (S_kv)         — number of K positions
//   dims[2] = kv_h           (number of K/V heads)
//   dims[3] = heads_per_kv   (n_q_heads / kv_h)
//   dims[4] = has_mask       (0 or 1)
//
// Inputs:
//   query     : float[B * n_q_heads * D]
//   codes     : uint [B * kv_h * S_kv * (D * 3 / 32)]   — turbo3 packed
//   scales    : float[B * kv_h * S_kv * (D / 32)]       — per-group f32 scale
//   mask      : float[B * n_q_heads * S_kv]             — optional additive
//   scale_arr : float[1]                                — softmax pre-scale
//
// Output:
//   out : float[B * n_q_heads * S_kv]                   — pre-softmax scores
//
// Constraints (validated by the Rust dispatcher):
//   - head_dim ∈ {128, 256} (kernel uses fixed shared-mem of 256 floats)
//   - head_dim % GROUP_SIZE (32) == 0
//   - head_dim is a power of two (tree-reduction stride loop)

/// MSL header — embeds the 8 Lloyd-Max N(0,1) 3-bit centroids as bit-exact
/// `as_type<float>(0x...)` constants.  Bit patterns mirror
/// `crate::k8vturbo3_append_msl::V3_KERNEL_HEADER` exactly (same codec).
const TURBO_K3_FUSED_QK_HEADER: &str = r"
constant float CB3[8] = {
    as_type<float>(0xC009B977u),  // -2.1519449
    as_type<float>(0xBFAC0532u),  // -1.3439085
    as_type<float>(0xBF418987u),  // -0.7560048
    as_type<float>(0xBE7AF9EBu),  // -0.2450940
    as_type<float>(0x3E7AF9EBu),  //  0.2450940
    as_type<float>(0x3F418987u),  //  0.7560048
    as_type<float>(0x3FAC0532u),  //  1.3439085
    as_type<float>(0x4009B977u)   //  2.1519449
};
";

const TURBO_K3_FUSED_QK_SOURCE: &str = r"
    uint s_kv         = threadgroup_position_in_grid.x;
    uint bh           = threadgroup_position_in_grid.y;
    uint tid          = thread_position_in_threadgroup.x;
    uint head_dim     = dims[0];
    uint kv_seq       = dims[1];
    uint kv_h         = dims[2];
    uint heads_per_kv = dims[3];
    uint has_mask     = dims[4];

    uint n_q_heads = kv_h * heads_per_kv;
    uint b         = bh / n_q_heads;
    uint hq        = bh % n_q_heads;
    uint kv_h_idx  = hq / heads_per_kv;

    // Per-token layout: D elements per K row =>
    //   groups_per_tok = D / 32
    //   words_per_tok  = D * 3 / 32     (== groups_per_tok * 3)
    //   scales_per_tok = D / 32         (== groups_per_tok)
    uint kv_tok            = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
    uint groups_per_tok    = head_dim / 32u;
    uint words_per_tok     = groups_per_tok * 3u;
    uint scales_per_tok    = groups_per_tok;
    uint codes_tok_off     = kv_tok * words_per_tok;
    uint scales_tok_off    = kv_tok * scales_per_tok;

    // tid handles one head_dim element. Which group and which lane in group?
    uint group_id_in_head = tid / 32u;
    uint elem_in_group    = tid % 32u;

    // Each thread loads one Q element into shared memory.  256-slot SMEM
    // covers head_dim in {128, 256}; trailing slots are unused for 128.
    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3-bit unpack: locate this thread's index inside the 96-bit group
    // stream.  bits [elem_in_group*3, elem_in_group*3+3) of the concatenated
    // LE stream of (codes[group*3], codes[group*3+1], codes[group*3+2]).
    uint group_codes_base = codes_tok_off + group_id_in_head * 3u;
    uint bit_off  = elem_in_group * 3u;
    uint word0_id = bit_off / 32u;          // 0, 1 or 2
    uint shift0   = bit_off - word0_id * 32u;
    ulong window  = (ulong)codes[group_codes_base + word0_id];
    if (word0_id + 1u < 3u) {
        window |= ((ulong)codes[group_codes_base + word0_id + 1u]) << 32;
    }
    uint idx = (uint)((window >> shift0) & 0x7ul);

    // Codebook lookup x per-group scale.  No WHT / Hadamard — turbo3 K
    // is plain codebook * scale (matches V3_DEQUANTIZE_SOURCE).
    float k_scale = scales[scales_tok_off + group_id_in_head];
    float k_val   = CB3[idx] * k_scale;

    // QK partial product + threadgroup tree reduction (in-place).
    threadgroup float dot_shared[256];
    dot_shared[tid] = q_shared[tid] * k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            dot_shared[tid] += dot_shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float result = dot_shared[0] * scale_arr[0];
        if (has_mask != 0u) {
            result += mask[bh * kv_seq + s_kv];
        }
        out[bh * kv_seq + s_kv] = result;
    }
";

// ── Kernel singleton ─────────────────────────────────────────────────────────

static QK_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn qk_kernel() -> Result<&'static MetalKernel> {
    QK_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_turbo_k3_fused_qk",
                TURBO_K3_FUSED_QK_HEADER,
                TURBO_K3_FUSED_QK_SOURCE,
                &["query", "codes", "scales", "mask", "scale_arr", "dims"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("turbo_k3_fused_qk kernel init: {e}")))
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the fused TurboSym3 K-side QK kernel.
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`
///   (or `[B, n_q_heads, head_dim]`).  f32/f16/bf16; coerced to f32.
/// * `k_codes`   — packed turbo3 codes from a `QuantKTurbo3` GPU buffer,
///   flat `u32 [B * kv_h * kv_seq * (head_dim * 3 / 32)]`.
/// * `k_scales`  — per-group f32 scales,
///   flat `[B * kv_h * kv_seq * (head_dim / 32)]`.
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
pub fn turbo_k3_fused_qk_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    // Turbo codec ignores norms / rotor sidebands;
    // the widened `FusedQkFn` signature carries them as `Option`.
    _k_norms: Option<&Array>,
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
    if head_dim != 128 && head_dim != 256 {
        return Err(Error::Quant(format!(
            "turbo_k3_fused_qk_sdpa: head_dim={head_dim} not supported \
             (only 128 and 256 are wired; legacy dequant+SDPA path handles other dims)"
        )));
    }
    if !(head_dim as usize).is_multiple_of(GROUP_SIZE) {
        return Err(Error::Quant(format!(
            "turbo_k3_fused_qk_sdpa: invariant: head_dim={head_dim} must be a multiple of \
             GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "turbo_k3_fused_qk_sdpa: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "turbo_k3_fused_qk_sdpa: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }
    let n_q_heads = kv_h * heads_per_kv;

    let q_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * i64::from(head_dim) * 3 / (GROUP_SIZE as i64);
    let scales_total: i64 = tok_count * i64::from(head_dim) / (GROUP_SIZE as i64);
    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[scales_total as i32], device)?;

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
            .map_err(|e| Error::Mlx(format!("turbo_k3_fused_qk dummy mask: {e}")))?;
        (dummy, 0u32)
    };

    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };
    let dims_vals: [u32; 5] = [
        head_dim as u32,
        kv_seq as u32,
        kv_h as u32,
        heads_per_kv as u32,
        has_mask,
    ];
    let dims_arr = {
        // SAFETY: `dims_vals` is a stack array of 5 `u32`; reinterpreting the
        // pointer as `&[u8]` of length 20 is safe — `u32` has no alignment
        // requirement stricter than `u8`, every bit pattern is valid for
        // `u8`, and `Array::from_bytes` copies the bytes before this scope
        // ends, so no escaping pointer.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(dims_vals.as_ptr().cast::<u8>(), 5 * 4) };
        Array::from_bytes(bytes, &[5], Dtype::U32)?
    };

    q_f32.eval()?;
    codes_flat.eval()?;
    scales_flat.eval()?;
    mask_flat.eval()?;
    scale_arr.eval()?;
    dims_arr.eval()?;

    let kernel = qk_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&q_f32)?;
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&mask_flat)?;
    invoke.add_input(&scale_arr)?;
    invoke.add_input(&dims_arr)?;

    let out_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(kv_seq);
    invoke.add_output_shape(&[out_total as i32], Dtype::F32)?;

    let grid_x: i64 = i64::from(kv_seq) * i64::from(head_dim);
    let grid_y: i64 = i64::from(b) * i64::from(n_q_heads);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "turbo_k3_fused_qk_sdpa: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    invoke.set_grid(grid_x as i32, grid_y as i32, 1)?;
    invoke.set_thread_group(head_dim, 1, 1)?;

    TURBO_K3_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    tracing::trace!(
        b,
        n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask,
        "turbo_k3_fused_qk_sdpa: dispatch"
    );

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "turbo_k3_fused_qk_sdpa: kernel produced no outputs".into(),
        ));
    }
    let out_flat = outputs.remove(0);

    out_flat.reshape(&[b, n_q_heads, 1, kv_seq], device)
}

#[cfg(test)]
#[path = "turbo_k3_fused_qk_msl_tests.rs"]
mod turbo_k3_fused_qk_msl_tests;
