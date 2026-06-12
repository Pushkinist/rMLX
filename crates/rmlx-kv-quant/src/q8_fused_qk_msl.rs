//! Fused-QK MSL kernel for q8_0-packed K (K8V4 / K8V8).
//!
//! # What this is
//!
//! Single MSL kernel that consumes q8_0-packed K (i8 codes + per-group f32
//! scales) **directly** and computes pre-softmax QK scores
//! `[B, n_q_heads, 1, S_kv]` without an intermediate dequantized K tensor in
//! HBM.  The legacy path materialises a full bf16 / f32 K via
//! [`q8_dequantize_gpu`](crate::q8_msl::q8_dequantize_gpu) and then calls
//! `scaled_dot_product_attention`; the dequant write+read of K is the
//! dominant decode-step bandwidth cost on memory-bound models.
//!
//! # Codec contract
//!
//! Target [`KvQuant`](crate::quant::KvQuant) variants: `K8V4`, `K8V8` —
//! both have q8_0 8-bit affine K-side (V-side differs but is the SDPA
//! caller's responsibility).  Bit-exact with
//! [`q8_dequantize_gpu`](crate::q8_msl::q8_dequantize_gpu):
//!
//! * `codes` u32 `[B * kv_h * S_kv * (D / 4)]` — every 4 consecutive bytes
//!   in native LE order pack 4 i8 codes (byte 0 = element 0 in group,
//!   byte 3 = element 3 in group).
//! * `scales` f32 `[B * kv_h * S_kv * (D / 128)]` — one scale per group of
//!   128 elements.
//! * No codebook (q8 is dense affine); no rotation aux (q8 has no rotor).
//!
//! # Kernel shape
//!
//! Grid `(S_kv, B * n_q_heads, 1)`; threadgroup `(head_dim, 1, 1)`.
//! Each threadgroup computes one score `out[b, hq, 0, s_kv]`.  Per-thread
//! handles one element of `head_dim`: loads its Q element, decodes its K
//! element (raw byte → signed i8 → scale-multiply), accumulates the
//! per-thread product, and participates in a tree reduction.
//!
//! # Single Q-step contract (decode-only)
//!
//! `S_q == 1`.  The caller passes the per-(b, hq) Q vector for the new
//! token; the kernel scores it against every K position `s_kv ∈ 0..S_kv`.
//! Matches the dominant cost at decode time (single token, long K context).
//!
//! # GQA support
//!
//! `n_q_heads = kv_h * heads_per_kv`; the threadgroup maps
//! `(b, hq) → kv_h_idx = hq / heads_per_kv` to share K across the GQA group.
//!
//! # Numerical contract
//!
//! Decoded K element is computed in `float` (f32) registers — matches the
//! [`q8_dequantize_gpu`](crate::q8_msl::q8_dequantize_gpu) path bit-for-bit.
//! The dot product accumulates in `float`; the optional additive mask is
//! added in `float`; the `scale * sum` write is `float`.  Q is loaded as
//! `float` even when the caller passes f16/bf16 — converting at thread load
//! is cheaper than threadgroup memory pressure.
//!
//! # A.y guard
//!
//! Not required — q8 is K-side 8-bit, so the Qwen-MoE 4-bit A.y guard
//! does not apply.  Both K8V4 and K8V8 are accepted on every architecture.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::q8::Q8_GROUP_SIZE;

// ── Dispatch counter ─────────────────────────────────────────────────────────

/// Incremented once per [`q8_fused_qk_sdpa`] invocation that reaches the
/// Metal enqueue point.  Used by NIAH harness to prove the kernel fired.
static Q8_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of q8_fused_qk dispatches.
pub fn q8_fused_qk_dispatch_count() -> u64 {
    Q8_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
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
//   query     : float[B * n_q_heads * D]           (Q for the new token)
//   codes     : uint [B * kv_h * S_kv * (D / 4)]   — q8 codes packed 4/u32 (LE i8)
//   scales    : float[B * kv_h * S_kv * (D / 128)] — one f32 per 128-element group
//   mask      : float[B * n_q_heads * S_kv]        — optional additive mask, dummy 1-elem if none
//   scale_arr : float[1]                           — softmax pre-scale (1/sqrt(D) typ.)
//
// Output:
//   out : float[B * n_q_heads * S_kv]              — pre-softmax scores (mask-added)
//
// Grid: (S_kv * D, B * n_q_heads, 1)  →  threadgroups (S_kv, B * n_q_heads, 1)
// Threadgroup: (D, 1, 1).
//
// Constraints (validated by the Rust dispatcher):
//   - head_dim ∈ {128, 256}  (kernel uses fixed register sizes)
//   - head_dim % Q8_GROUP_SIZE (128) == 0  (one or two groups per head)
//   - head_dim is a power of two  (tree-reduction stride loop)
const Q8_FUSED_QK_SOURCE: &str = r"
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

    uint kv_tok                = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
    uint codes_words_per_tok   = head_dim / 4u;
    uint scales_groups_per_tok = head_dim / 128u;
    uint codes_tok_off         = kv_tok * codes_words_per_tok;
    uint scales_tok_off        = kv_tok * scales_groups_per_tok;

    uint group_id_in_head = tid / 128u;
    uint elem_in_group    = tid % 128u;
    uint word_in_group    = elem_in_group / 4u;
    uint byte_in_word     = elem_in_group & 3u;

    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint code_word_abs = codes_tok_off + group_id_in_head * 32u + word_in_group;
    uint scale_abs     = scales_tok_off + group_id_in_head;

    uint word     = codes[code_word_abs];
    uint raw_byte = (word >> (byte_in_word * 8u)) & 0xFFu;
    int code      = (int)raw_byte;
    if (code & 0x80) { code -= 256; }

    float k_scale = scales[scale_abs];
    float k_val   = k_scale * (float)code;

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
                "rmlx_q8_fused_qk",
                "",
                Q8_FUSED_QK_SOURCE,
                &["query", "codes", "scales", "mask", "scale_arr", "dims"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("q8_fused_qk kernel init: {e}")))
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the fused q8_0 K-side QK kernel.
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`
///   (or `[B, n_q_heads, head_dim]`).  f32/f16/bf16; coerced to f32 internally.
/// * `k_codes`   — packed q8 codes from a [`QuantK`](crate::storage::QuantK)
///   GPU buffer, flat `u32 [B * kv_h * kv_seq * (head_dim / 4)]` (LE i8 packing).
/// * `k_scales`  — per-group scales (f32), flat
///   `[B * kv_h * kv_seq * (head_dim / 128)]`.
/// * `additive_mask` — optional `f32 [B, n_q_heads, 1, kv_seq]`.  Added to
///   the raw QK score inside the kernel; `None` → kernel skips the add.
/// * `b`, `kv_h`, `kv_seq`, `head_dim`, `heads_per_kv` — shape metadata.
/// * `scale`     — softmax pre-scale (typ. `1/sqrt(head_dim)`).
/// * `device`    — MLX device (must be GPU).
///
/// # Output
///
/// Scores tensor `[B, n_q_heads, 1, kv_seq]` (f32) with the additive mask
/// already applied.  The caller is responsible for the softmax + SV path.
///
/// # Errors
///
/// * `Error::Quant` for shape contract violations (`head_dim` not in
///   `{128, 256}`, dims out of range, non-positive shapes).
/// * `Error::Mlx` for kernel build / dispatch failures.
#[allow(clippy::too_many_arguments)]
pub fn q8_fused_qk_sdpa(
    query: &Array,
    k_codes: &Array,
    k_scales: &Array,
    // q8 codec ignores per-token norms / per-layer rotor table sidebands;
    // the widened `FusedQkFn` signature carries them as `Option` so iso /
    // rotor shims can read them without the per-step concat marshaling cost.
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
    let setup = crate::fused_qk_common::build_fused_qk_setup(
        "q8_fused_qk_sdpa",
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

    if !(head_dim as usize).is_multiple_of(Q8_GROUP_SIZE) {
        return Err(Error::Quant(format!(
            "q8_fused_qk_sdpa: invariant: head_dim={head_dim} must be a multiple of \
             Q8_GROUP_SIZE={Q8_GROUP_SIZE}"
        )));
    }

    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * i64::from(head_dim) / 4;
    let scales_total: i64 = tok_count * i64::from(head_dim) / (Q8_GROUP_SIZE as i64);
    let codes_flat = k_codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = k_scales.reshape(&[scales_total as i32], device)?;
    codes_flat.eval()?;
    scales_flat.eval()?;

    let kernel = qk_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&setup.q_f32)?;
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&setup.mask_flat)?;
    invoke.add_input(&setup.scale_arr)?;
    invoke.add_input(&setup.dims_arr)?;

    invoke.add_output_shape(&[setup.out_total as i32], Dtype::F32)?;

    invoke.set_grid(setup.grid_x, setup.grid_y, 1)?;
    invoke.set_thread_group(head_dim, 1, 1)?;

    Q8_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    tracing::trace!(
        b,
        n_q_heads = setup.n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask = setup.has_mask,
        "q8_fused_qk_sdpa: dispatch"
    );

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "q8_fused_qk_sdpa: kernel produced no outputs".into(),
        ));
    }
    let out_flat = outputs.remove(0);

    out_flat.reshape(&[b, setup.n_q_heads, 1, kv_seq], device)
}

#[cfg(test)]
#[path = "q8_fused_qk_msl_tests.rs"]
mod q8_fused_qk_msl_tests;
