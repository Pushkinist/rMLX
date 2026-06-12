//! TurboSym4 K-side fused-QK MSL kernel.
//!
//! # What this is
//!
//! Single MSL kernel that consumes the TurboQuant 4-bit packed K codes
//! (4 u32 per 32-element group + one f32 scale per group; codebook =
//! 16-entry Lloyd-Max N(0,1)) **directly** and computes pre-softmax QK
//! scores `[B, n_q_heads, 1, S_kv]` without materialising a bf16 / f32 K
//! tensor in HBM.  Mirrors [`crate::turbo_k3_fused_qk_msl`] (Executor C)
//! shape, threadgroup layout, mask handling and Q reshape; only the
//! K-decode body differs (4-bit nibble unpack + 16-entry codebook vs the
//! 3-bit 8-entry path).
//!
//! # Codec contract
//!
//! Target [`KvQuant`](crate::quant::KvQuant): `TurboSym4` — K stored as
//! TurboQuant 4-bit (group_size=32, Lloyd-Max optimal N(0,1) codebook,
//! 16 centroids).  V side is also turbo4 but is the SDPA caller's
//! responsibility — this kernel produces only the pre-softmax QK score.
//!
//! Bit-exact with [`crate::turboquant_msl::turbo_dequantize_v4_gpu`]:
//!
//! * `codes`  u32 `[B * kv_h * S_kv * (D / 8)]` — 4 u32 words per
//!   32-element group, 8 nibbles per u32 (LSB-first: element `e` in group
//!   lives at word `e/8`, bits `[(e%8)*4, (e%8)*4 + 4)`).
//! * `scales` f32 `[B * kv_h * S_kv * (D / 32)]` — one f32 per group of 32.
//! * Codebook: 16-entry Lloyd-Max N(0,1) (`CB` in `turboquant_msl`).
//! * **No WHT / Hadamard correction**: the turbo4 K codec is plain codebook
//!   lookup × scale (verified by reading `storage/quant_k_turbo4.rs` and
//!   the sibling MSL `DEQUANTIZE_SOURCE` — both call straight into
//!   `CB[idx] * scale` with no rotation step).
//!
//! # Kernel shape
//!
//! Grid `(S_kv * D, B * n_q_heads, 1)`; threadgroup `(D, 1, 1)`.  Each
//! threadgroup computes one score `out[b, hq, 0, s_kv]`; per-thread handles
//! one element of `head_dim`.
//!
//! # GQA support
//!
//! Identical to the q8 / turbo3 paths: `n_q_heads = kv_h * heads_per_kv`;
//! thread group maps `(b, hq) -> kv_h_idx = hq / heads_per_kv` to share K.
//!
//! # A.y guard
//!
//! `TurboSym4` is K-side 4-bit, the Qwen-MoE 218->8641 PPL disaster applies.
//! The guard lives in [`rmlx_models::kv_cache::cache_type::validate_resolved`]
//! and rejects `Qwen3_5MoeForConditionalGeneration + TurboSym4` at session
//! start — the kernel does not re-check.
//!
//! # Reference
//!
//! * [`crate::turbo_k3_fused_qk_msl`] — Exec C dispatcher / threadgroup layout.
//! * [`crate::turboquant_msl`] — bit-exact 4-bit unpack idiom (same byte/
//!   nibble window arithmetic, reframed per thread instead of per-word).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::turboquant::GROUP_SIZE;

// ── Dispatch counter ─────────────────────────────────────────────────────────

/// Incremented once per `turbo_k4_fused_qk_sdpa` invocation that reaches
/// the Metal enqueue point.  Used by NIAH harness.
static TURBO_K4_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of turbo_k4_fused_qk dispatches.
pub fn turbo_k4_fused_qk_dispatch_count() -> u64 {
    TURBO_K4_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
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
//   codes     : uint [B * kv_h * S_kv * (D / 8)]        — turbo4 packed
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

/// MSL header — embeds the 16 Lloyd-Max N(0,1) 4-bit centroids as bit-exact
/// `as_type<float>(0x...)` constants.  Bit patterns mirror
/// `crate::turboquant_msl::KERNEL_HEADER` exactly (same codec).
const TURBO_K4_FUSED_QK_HEADER: &str = r"
constant float CB4[16] = {
    as_type<float>(0xC02DEE42u),  // -2.7176671
    as_type<float>(0xC003563Bu),  // -2.0521381
    as_type<float>(0xBFCCE718u),  // -1.6008024
    as_type<float>(0xBF9EB6FAu),  // -1.2399590
    as_type<float>(0xBF6DA172u),  // -0.9282447
    as_type<float>(0xBF255816u),  // -0.6458753
    as_type<float>(0xBEC329CBu),  // -0.3811782
    as_type<float>(0xBE011273u),  // -0.1260469
    as_type<float>(0x3E011273u),  //  0.1260469
    as_type<float>(0x3EC329CBu),  //  0.3811782
    as_type<float>(0x3F255816u),  //  0.6458753
    as_type<float>(0x3F6DA172u),  //  0.9282447
    as_type<float>(0x3F9EB6FAu),  //  1.2399590
    as_type<float>(0x3FCCE718u),  //  1.6008024
    as_type<float>(0x4003563Bu),  //  2.0521381
    as_type<float>(0x402DEE42u)   //  2.7176671
};
";

const TURBO_K4_FUSED_QK_SOURCE: &str = r"
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
    //   words_per_tok  = D / 8      (4 u32 per group of 32, 8 nibbles per u32)
    //   scales_per_tok = D / 32     (== groups_per_tok)
    uint kv_tok            = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
    uint groups_per_tok    = head_dim / 32u;
    uint words_per_tok     = head_dim / 8u;
    uint scales_per_tok    = groups_per_tok;
    uint codes_tok_off     = kv_tok * words_per_tok;
    uint scales_tok_off    = kv_tok * scales_per_tok;

    // tid handles one head_dim element. Which group and which lane in group?
    uint group_id_in_head = tid / 32u;
    uint elem_in_group    = tid - group_id_in_head * 32u;

    // Each thread loads one Q element into shared memory.  256-slot SMEM
    // covers head_dim in {128, 256}; trailing slots are unused for 128.
    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4-bit unpack: locate this thread's nibble within the group's 4-word
    // (128-bit) stream.  Element `e` in group lives at word `e/8`, nibble
    // `e%8` (bits [(e%8)*4 .. (e%8)*4 + 4)).  Group base in codes: 4 words.
    uint group_codes_base = codes_tok_off + group_id_in_head * 4u;
    uint word_off         = elem_in_group / 8u;            // 0..3
    uint nibble_in_word   = elem_in_group - word_off * 8u; // 0..7
    uint word             = codes[group_codes_base + word_off];
    uint idx              = (word >> (nibble_in_word * 4u)) & 0xFu;

    // Codebook lookup x per-group scale.  No WHT / Hadamard — turbo4 K
    // is plain codebook * scale (matches DEQUANTIZE_SOURCE in
    // turboquant_msl.rs: out = CB[idx] * scale).
    float k_scale = scales[scales_tok_off + group_id_in_head];
    float k_val   = CB4[idx] * k_scale;

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
                "rmlx_turbo_k4_fused_qk",
                TURBO_K4_FUSED_QK_HEADER,
                TURBO_K4_FUSED_QK_SOURCE,
                &["query", "codes", "scales", "mask", "scale_arr", "dims"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("turbo_k4_fused_qk kernel init: {e}")))
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the fused TurboSym4 K-side QK kernel.
///
/// # Inputs
///
/// * `query`     — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`
///   (or `[B, n_q_heads, head_dim]`).  f32/f16/bf16; coerced to f32.
/// * `k_codes`   — packed turbo4 codes from a `QuantKTurbo4` GPU buffer,
///   flat `u32 [B * kv_h * kv_seq * (head_dim / 8)]`.
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
pub fn turbo_k4_fused_qk_sdpa(
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
    let setup = crate::fused_qk_common::build_fused_qk_setup(
        "turbo_k4_fused_qk_sdpa",
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

    if !(head_dim as usize).is_multiple_of(GROUP_SIZE) {
        return Err(Error::Quant(format!(
            "turbo_k4_fused_qk_sdpa: invariant: head_dim={head_dim} must be a multiple of \
             GROUP_SIZE={GROUP_SIZE}"
        )));
    }

    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    // turbo4: 4 u32 words per 32-element group => D/8 words per token.
    let codes_total: i64 = tok_count * i64::from(head_dim) / 8;
    let scales_total: i64 = tok_count * i64::from(head_dim) / (GROUP_SIZE as i64);
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

    TURBO_K4_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    tracing::trace!(
        b,
        n_q_heads = setup.n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask = setup.has_mask,
        "turbo_k4_fused_qk_sdpa: dispatch"
    );

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "turbo_k4_fused_qk_sdpa: kernel produced no outputs".into(),
        ));
    }
    let out_flat = outputs.remove(0);

    out_flat.reshape(&[b, setup.n_q_heads, 1, kv_seq], device)
}

#[cfg(test)]
#[path = "turbo_k4_fused_qk_msl_tests.rs"]
mod turbo_k4_fused_qk_msl_tests;
