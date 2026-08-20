// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! TurboFlash p1/p2 split-K FlashAttention kernel.
//!
//! # Apple10 (M5+) hazard — historical, cleared 2026-06
//!
//! TheTom's original TurboFlash kernel was default-OFF on Apple10 (M5+) due
//! to corruption producing garbage output (commit `67f076f2e` in
//! llama-cpp-turboquant). `67f076f2e` is a 4-line `return true`→
//! `return false` default-flip, NOT a kernel fix — no upstream TurboFlash
//! corruption fix exists.
//!
//! On rMLX, the initial B1 validation reproduced a more severe failure on
//! M5 Max: an `EXC_BAD_ACCESS (SIGSEGV) KERN_INVALID_ADDRESS at 0x0` the
//! instant the K8V4 flash path dispatched at 32k context on
//! Qwen3.6-35B-A3B-8bit (`head_dim = 256`). Null `Buffer::raw_ptr()` in
//! `Array::to_bytes` reading back the kernel output. The `--turbo-flash`
//! `Auto` arm therefore landed with a `family ≥ 10 → OFF` clause.
//!
//! **Re-validation (2026-06)**: the hazard was re-driven on M5 Max via
//! `crates/rmlx-kv-quant/tests/apple10_head_dim_256.rs` — a synthetic K8V4
//! cache at `head_dim = 256` driven through the public
//! `KvCache::update_and_sdpa` chain under a `turbo_flash: true` policy, smoke
//! + 16-step decode stress + a `turbo_flash: false` control. Result:
//!
//! * smoke (1 dispatch, kv_seq=65): no SIGSEGV, cosine min 0.997 vs bf16.
//! * stress (16 dispatches, kv_seq up to 80): no SIGSEGV, cosine min 0.997.
//! * control (kernel off): dispatch dormant (delta=0).
//!
//! The 0.997 SDPA cosine vs the K8V4 fused-QK 0.999998 floor is the
//! **codec floor**, not a kernel issue. A CPU baseline test
//! (`tests/apple10_cpu_baseline.rs`) confirmed the V turbo-4 codec
//! encode→decode round-trip cosine alone is 0.997 (~identical at
//! head_dim ∈ {128, 256}). The K8V4 fused-QK 0.999998 measures Q·K^T
//! (K-dominated); the full SDPA (softmax @ V) shows V's turbo-4 codec
//! floor. Same numerics at both head dims.
//!
//! That attribution used to rest on the CPU round-trip alone — i.e. on the
//! codec's error being *large enough* to explain the gap, never on the
//! kernel's own error being *small*. [`turbo_flash_reference_sdpa`] closes
//! that: it unpacks the same packed buffers and runs an ordinary SDPA, so the
//! codec cancels between the arms and only the kernel's arithmetic is left.
//! Measured <=1.7 bf16 ULP at both dispatching geometries (cosine 0.9999956 at
//! Bonsai-8B's, 0.9999962 at Qwen3.6-35B-A3B's). Any comparison against a
//! `--turbo-flash off` run is a
//! comparison against a bf16 attention — that arm does not run the codec at
//! all — and cannot answer this question in either direction.
//!
//! The documented hazard does not reproduce against the current kernel
//! surface. See `docs/reports/apple10-head-dim-256-revalidation.md` for the
//! verbatim numbers and the kernel changes that almost certainly closed the
//! original failure mode.
//!
//! That clearance is crash/fidelity only, and was never a throughput one. On
//! throughput the kernel loses: `--turbo-flash auto` resolves **OFF on every
//! host**, because at `kv_seq > 4096` it decodes 2.0-4.25x slower than the
//! generic K8V4 path it replaces (see
//! `rmlx_cli::commands::serve::TurboFlashMode` for the measured cells).
//! Enabling it is an explicit opt-in.
//!
//! Those cells were measured while this dispatcher returned its f32 kernel
//! output uncast, which promoted the whole decode graph — residual stream,
//! norms, weight GEMV, sampler — to f32 for as long as the gate was on. Part
//! of the recorded ratio was that promotion rather than the kernel, so read the
//! range as an upper bound until the cells are re-measured on a quiet host. The
//! direction is unchanged: the ON arm is still the slower one.
//!
//! The smoke-probe trip-wire below is retained as armour against any future
//! drift in the kernel that might revive the `!!!!!!`-style garbage-token
//! signature — it is cheap (≥4 consecutive identical token IDs check) and
//! gives a soft fallback if it ever fires.
//!
//! # What this is
//!
//! A 2-pass split-K FlashAttention kernel for rMLX's native KV storage format:
//! - K: q8_0 affine 8-bit, group_size=128, f32 scale, i8 codes packed as u32.
//! - V: TurboQuant 4-bit Lloyd-Max, group_size=32, f32 scale, 4-bit in u32.
//!
//! Architecture (adapted from TheTom `ggml-metal.metal:8843-9168`):
//!
//! **Pass 1** (`rmlx_turbo_flash_p1`): Each threadgroup processes a block of
//! `BLOCK_SIZE=64` KV tokens for one query head.
//! - Loads Q into registers (one lane per head_dim/32 slice).
//! - Dequants K (rMLX q8_0 format: i8 codes × f32 scale, group_size=128).
//! - Computes Q·K scores with attention mask.
//! - Online softmax within the block (running max + exp sum).
//! - Dequants V (rMLX turbo4 format: 4-bit Lloyd-Max × f32 scale, group_size=32).
//! - Accumulates softmax-weighted V in registers.
//! - Writes per-block partial {output[dim], block_max, block_sum} to DRAM.
//!
//! **Pass 2** (`rmlx_turbo_flash_p2`): One threadgroup per query head.
//! - Scans per-block {max, sum} to find global max.
//! - Rescales each block's output by `exp(block_max - global_max)`.
//! - Normalises by global sum and writes final output.
//! - NO inverse WHT: rMLX turbo4 V does NOT use WHT rotation.
//!
//! # Key differences from TheTom's kernel
//!
//! TheTom kernel uses `block_q8_0` (f16 scale, 32-element groups) and
//! `block_turbo3_0` (3-bit + sign array + WHT rotation). rMLX uses:
//! - K: `f32 scale` per 128 elements, i8 codes packed as 4 per u32.
//! - V: `f32 scale` per 32 elements, 4-bit Lloyd-Max codes (8 per u32).
//! - No WHT rotation on V — Lloyd-Max is a scalar codebook, not rotated.
//!
//! # Activation condition
//!
//! Only active for:
//! - `DispatchPolicy::turbo_flash` set on the cache's policy.
//! - Decode step (q_seq = 1).
//! - K format: q8_0 (k_bits = 8 in KvQuant::K8V4).
//! - V format: turbo4 (KvQuant::K8V4 — the only mode with turbo V).
//! - `kv_seq > DispatchPolicy::turbo_flash_min_kv_seq` (default 4096: split-K
//!   wins when K-seq is long; below this threshold the existing
//!   `mixed_quantized_sdpa` is faster due to launch overhead).
//!
//! # Reference
//!
//! TheTom `ggml-metal.metal:8843` (p1), `:9034` (p2). N69 §1.
//! Commit `67f076f2e` disables it on Apple10. N73 §3.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Smoke-probe state ─────────────────────────────────────────────────────────

/// Set to true if the smoke-probe detected corruption and forced fallback.
static FORCED_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Returns true if the smoke-probe triggered corruption fallback.
pub fn turbo_flash_corrupted() -> bool {
    FORCED_FALLBACK.load(Ordering::Relaxed)
}

// ── Dispatch counter ──────────────────────────────────────────────────────────
//
// Incremented exactly once per `turbo_flash_sdpa` invocation that reaches the
// P1 kernel-enqueue point. Used by the NIAH harness to *prove* the MSL kernel
// actually fired (vs. silently falling back to `mixed_quantized_sdpa`).
//
// Validation gates (head_dim, t_stride) live above the increment site, so a
// kernel-dispatch failure due to a bad gate increments only after those checks
// pass — i.e. the counter reflects real MSL P1 enqueues, not pre-check calls.
static TURBO_FLASH_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of TurboFlash P1 kernel dispatches.
///
/// NIAH harness uses this to assert the kernel actually fired (delta > 0 on
/// ON cells, delta == 0 on OFF cells). Production code does NOT consult this
/// counter.
pub fn turbo_flash_dispatch_count() -> u64 {
    TURBO_FLASH_DISPATCHES.load(Ordering::Relaxed)
}

/// Inspect the first `n_tokens` of the generated output for corruption.
///
/// Corruption signature: ≥4 consecutive identical token IDs (the `!!!!!!`
/// pattern observed on M5 Max with TheTom's kernel at commit `67f076f2e`).
///
/// If corruption is detected, sets `FORCED_FALLBACK` and returns `true`.
/// Once set, `turbo_flash_should_run` returns false for the process lifetime.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn smoke_probe_check(token_ids: &[u32]) -> bool {
    if token_ids.len() < 4 {
        return false;
    }
    // Check for ≥4 consecutive identical tokens.
    let mut run = 1usize;
    for i in 1..token_ids.len() {
        if token_ids[i] == token_ids[i - 1] {
            run += 1;
            if run >= 4 {
                tracing::warn!(
                    "TurboFlash: corruption detected — {} consecutive identical tokens \
                     (token_id={}). Apple10/M5 corruption (TheTom `67f076f2e`). \
                     Falling back to mixed_quantized_sdpa for remainder of process. \
                     Pass --turbo-flash off to suppress this probe.",
                    run,
                    token_ids[i]
                );
                FORCED_FALLBACK.store(true, Ordering::Relaxed);
                return true;
            }
        } else {
            run = 1;
        }
    }
    false
}

/// Returns true if TurboFlash should run for this decode step.
///
/// Conditions:
/// 1. `policy.turbo_flash` is set.
/// 2. Smoke-probe has not forced fallback.
/// 3. q_seq == 1 (decode step only).
/// 4. `kv_seq > policy.turbo_flash_min_kv_seq` (default 4096). This is a perf
///    gate, not a correctness gate — proof runs lower it to 0 so dispatch
///    fires on short prompts. Production keeps the default.
pub fn turbo_flash_should_run(policy: &DispatchPolicy, q_seq: i32, kv_seq: i32) -> bool {
    policy.turbo_flash
        && !turbo_flash_corrupted()
        && q_seq == 1
        && kv_seq > policy.turbo_flash_min_kv_seq
}

// ── MSL constants ─────────────────────────────────────────────────────────────

/// Block size (tokens per P1 threadgroup). Matches TheTom's optimal B=64.
const BLOCK_SIZE: i32 = 64;

/// Threads per threadgroup in P1: 32 (one SIMD group).
const TG_SIZE: i32 = 32;

// ── Lloyd-Max 4-bit codebook (same as turboquant_msl.rs) ─────────────────────
//
// Embedded as hex f32 bit patterns for bit-exact MSL literals.
// Source: turboquant.rs::CODEBOOK_4BIT (Lloyd-Max N(0,1)).
const KERNEL_HEADER: &str = include_str!("metal/turbo_flash_header.metal");

// ── Pass 1 MSL source ─────────────────────────────────────────────────────────
//
// Grid: (n_bh × n_blocks, 1, 1) where n_bh = B × n_q_heads, n_blocks = ceil(T_active / BLOCK_SIZE).
// Threadgroup: (TG_SIZE, 1, 1) = (32, 1, 1) — exactly one SIMD group.
//
// Buffer layout (must match `add_input` order in `turbo_flash_p1`):
// 0. q_flat: f32 [B × n_q_heads × head_dim] — scaled queries (q_seq=1)
// 1. k_codes: u32 [B × n_kv_heads × T_stride × (head_dim/4)] — i8 codes, 4/u32
// 2. k_scales: f32 [B × n_kv_heads × T_stride × (head_dim/128)] — q8_0 scales
// 3. v_codes: u32 [B × n_kv_heads × T_stride × (head_dim/8)] — turbo4 codes, 8/u32
// 4. v_scales: f32 [B × n_kv_heads × T_stride × (head_dim/32)] — turbo4 scales
// 5. mask_flat: f32 [B × n_q_heads × T_active] or empty if no mask
// 6. params_p1: u32 [11] — {B, n_q_heads, n_kv_heads, n_repeats, T_active, head_dim, n_blocks, has_mask, q8_words_per_tok, tq4_words_per_tok, T_stride}
//
// `T_active` is the count of valid tokens (iteration bound + mask length).
// `T_stride` is the per-head row stride in K/V code/scale buffers — equal to
// the persistent buffer's `max_seq` dimension for the head-major P2.A.1 layout
// (slots `[T_active..T_stride)` are zero-init and never read). Pre-P2.A.1 the
// caller passed `T_stride == T_active` (chunk-major layout requantised fresh
// per dispatch); both modes share the same kernel.
//
// Output:
// partial_out: f32 [B × n_q_heads × n_blocks × head_dim]
// partial_ms: f32 [B × n_q_heads × n_blocks × 2] — {block_max, block_sum}
const P1_SOURCE: &str = include_str!("metal/turbo_flash_p1.metal");

// ── Pass 2 MSL source ─────────────────────────────────────────────────────────
//
// Grid: (B × n_q_heads, 1, 1).
// Threadgroup: (max(head_dim, 32), 1, 1) — covers all head_dim dims in one TG.
//
// Buffer layout (must match `add_input` order in `turbo_flash_p2`):
// 0. partial_out: f32 [B × n_q_heads × n_blocks × head_dim]
// 1. partial_ms: f32 [B × n_q_heads × n_blocks × 2]
// 2. params_p2: u32 [4] — {n_blocks, head_dim, B_times_n_q_heads, _pad}
//
// Output:
// dst: f32 [B × n_q_heads × head_dim]
const P2_SOURCE: &str = include_str!("metal/turbo_flash_p2.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static P1_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

fn p1_kernel() -> Result<&'static MetalKernel> {
    P1_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_turbo_flash_p1",
                KERNEL_HEADER,
                P1_SOURCE,
                &[
                    "q_flat",
                    "k_codes",
                    "k_scales",
                    "v_codes",
                    "v_scales",
                    "mask_flat",
                    "params_p1",
                ],
                &["partial_out", "partial_ms"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("turbo_flash_p1 kernel init: {e}")))
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_turbo_flash_p2",
                KERNEL_HEADER,
                P2_SOURCE,
                &["partial_out", "partial_ms", "params_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("turbo_flash_p2 kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Shape preconditions shared by the kernel and its reference arm.
///
/// Both must refuse the same inputs, or a reference that silently accepted a
/// shape the kernel rejects would compare a computed answer against nothing.
fn validate_flash_shapes(caller: &str, head_dim: i32, t_active: i32, t_stride: i32) -> Result<()> {
    if head_dim % 32 != 0 {
        return Err(Error::Quant(format!(
            "{caller}: head_dim={head_dim} not divisible by 32"
        )));
    }
    if head_dim != 128 && head_dim != 256 {
        return Err(Error::Quant(format!(
            "{caller}: head_dim={head_dim} not supported \
             (only 128 and 256 are wired). Fallback to mixed_quantized_sdpa."
        )));
    }
    if t_stride < t_active {
        return Err(Error::Quant(format!(
            "{caller}: t_stride={t_stride} < t_active={t_active}"
        )));
    }
    Ok(())
}

/// Dequantize-then-SDPA over the **same** packed buffers [`turbo_flash_sdpa`]
/// reads — the kernel's executable specification.
///
/// This exists because the obvious comparison arm is not a reference. Turning
/// the kernel off does not turn the codec off: on `K8V4` the generic decode
/// path reads the bf16 mirror and never touches the 4-bit V store at all, so a
/// gate-off run is a bf16 attention, and **any** correct tq4-V kernel must
/// differ from it by the codec's own quantization error. Measuring the kernel
/// against that arm charges it for the codec and leaves its own error untested.
///
/// The arm below takes the identical `flash_k_codes` / `flash_k_scales` /
/// `flash_v_codes` / `flash_v_scales` buffers, unpacks them with the same two
/// codecs the kernel unpacks inline (q8_0 for K at `Q8_GROUP_SIZE`, TurboQuant
/// 4-bit Lloyd-Max for V at its own group), and runs an ordinary SDPA. Every
/// quantization error is therefore common to both arms and cancels; what
/// remains is the kernel's own arithmetic — its block tiling, its online
/// softmax and its two-pass rescale.
///
/// Correctness only. It materialises a full bf16 K and V of the whole window
/// per call, which is exactly the cost the kernel exists to avoid; nothing on
/// a decode path may call it.
///
/// Arguments, dtypes and layout are [`turbo_flash_sdpa`]'s, unchanged —
/// including `queries` being **pre-scaled**, so no scale is applied here.
#[allow(clippy::too_many_arguments)]
pub fn turbo_flash_reference_sdpa(
    queries: &Array,
    k_codes_flat: &Array,
    k_scales_flat: &Array,
    v_codes_flat: &Array,
    v_scales_flat: &Array,
    additive_mask: Option<&Array>,
    b: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    t_active: i32,
    t_stride: i32,
    head_dim: i32,
    device: Device,
) -> Result<Array> {
    validate_flash_shapes("turbo_flash_reference_sdpa", head_dim, t_active, t_stride)?;
    if n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 {
        return Err(Error::Quant(format!(
            "turbo_flash_reference_sdpa: n_q_heads={n_q_heads} is not a multiple \
             of n_kv_heads={n_kv_heads}"
        )));
    }

    // Unpack the whole provisioned window, then keep the active prefix. The
    // kernel iterates `t < t_active` over a buffer of row stride `t_stride`, so
    // slicing after the dequant is what reproduces its view of the same bytes.
    let full_shape = [b, n_kv_heads, t_stride, head_dim];
    let out_dtype = queries.dtype();
    let k_full = crate::q8_msl::q8_dequantize_gpu(
        k_codes_flat,
        k_scales_flat,
        &full_shape,
        out_dtype,
        device,
    )?;
    let v_full = crate::turboquant_msl::turbo_dequantize_v4_gpu(
        v_codes_flat,
        v_scales_flat,
        &full_shape,
        out_dtype,
        device,
    )?;

    let start = [0i32; 4];
    let stop = [b, n_kv_heads, t_active, head_dim];
    let strides = [1i32; 4];
    let k = k_full.slice(&start, &stop, &strides, device)?;
    let v = v_full.slice(&start, &stop, &strides, device)?;

    // `queries` arrives pre-scaled, same as the kernel's contract, so the
    // SDPA scale is 1.0 and the only difference between the arms is arithmetic.
    let mask_mode = if additive_mask.is_some() { "array" } else { "" };
    rmlx_mlx::scaled_dot_product_attention(queries, &k, &v, 1.0, mask_mode, additive_mask, device)
}

/// 2-pass split-K FlashAttention for rMLX's native q8_0 K + turbo4 V format.
///
/// Replaces `mixed_quantized_sdpa` when `DispatchPolicy::turbo_flash` is set
/// and `kv_seq > DispatchPolicy::turbo_flash_min_kv_seq` for the K8V4 decode
/// path.
///
/// # Arguments
///
/// - `queries`: f32 `[B, n_q_heads, 1, head_dim]` — scaled queries (q_seq=1).
/// - `k_codes_flat`: u32 flat — rMLX q8_0 K codes (`i8` packed 4/u32).
/// - `k_scales_flat`: f32 flat — rMLX q8_0 K scales (one per Q8_GROUP=128 elems).
/// - `v_codes_flat`: u32 flat — rMLX turbo4 V codes (4-bit, 8/u32).
/// - `v_scales_flat`: f32 flat — rMLX turbo4 V scales (one per TQ4_GROUP=32 elems).
/// - `additive_mask`: optional causal mask, f32 `[B, n_q_heads, 1, T_active]`.
/// - `b`, `n_q_heads`, `n_kv_heads`, `head_dim`: dimensions.
/// - `t_active`: number of valid KV tokens (iteration bound + mask length).
/// - `t_stride`: per-head row stride of the K/V buffers in tokens. For the
///   chunk-major fresh-quantise path use `t_stride = t_active`. For the
///   Head-major persistent buffer use `t_stride = max_seq` —
///   slots `[t_active..t_stride)` are zero-init and never read.
/// - `device`: MLX device.
///
/// # Returns
///
/// Array of shape `[B, n_q_heads, 1, head_dim]` in **`queries`' dtype**. The
/// two kernels accumulate in f32 (online softmax needs it), but the result is
/// cast back before it is handed to the caller: an f32 attention output enters
/// the residual stream and MLX then promotes the whole downstream graph —
/// norm, weight GEMV, every elementwise op — to f32 for the rest of the layer.
///
/// # Errors
///
/// Returns `Error::Mlx` if kernel dispatch fails.
/// Returns `Error::Quant` if head_dim is not in `{128, 256}` or not divisible
/// by 32. The kernel's register arrays (`q_vals`, `o_state`, `v_decoded`) are
/// sized for the larger of the two supported dims (head_dim=256, dims_per_lane=8).
#[allow(clippy::too_many_arguments)]
pub fn turbo_flash_sdpa(
    queries: &Array,
    k_codes_flat: &Array,
    k_scales_flat: &Array,
    v_codes_flat: &Array,
    v_scales_flat: &Array,
    additive_mask: Option<&Array>,
    b: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    t_active: i32,
    t_stride: i32,
    head_dim: i32,
    device: Device,
) -> Result<Array> {
    validate_flash_shapes("turbo_flash_sdpa", head_dim, t_active, t_stride)?;

    let n_repeats = n_q_heads / n_kv_heads;
    let n_blocks = (t_active + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ── Flatten queries to 1D f32 ─────────────────────────────────────────────
    let q_flat = {
        let q = queries.reshape(&[b * n_q_heads * head_dim], device)?;
        if q.dtype() == Dtype::F32 {
            q
        } else {
            q.astype(Dtype::F32, device)?
        }
    };

    // ── Mask flat (or dummy 1-element zero) ───────────────────────────────────
    let (mask_flat, has_mask) = if let Some(m) = additive_mask {
        let flat_len = b * n_q_heads * t_active;
        let m_f = if m.dtype() == Dtype::F32 {
            m.reshape(&[flat_len], device)?
        } else {
            m.astype(Dtype::F32, device)?.reshape(&[flat_len], device)?
        };
        (m_f, 1u32)
    } else {
        // Provide a dummy zero scalar — the kernel checks has_mask before reading.
        let zero_bytes = [0u8; 4];
        let dummy = Array::from_bytes(&zero_bytes, &[1], Dtype::F32)
            .map_err(|e| Error::Mlx(format!("turbo_flash dummy mask: {e}")))?;
        (dummy, 0u32)
    };

    // ── P1 params: see kernel layout above (11 u32 entries). ─────────────────
    let q8_words_per_tok = head_dim / 4; // i8 codes packed 4/u32
    let tq4_words_per_tok = head_dim / 8; // 4-bit codes packed 8/u32
    let params_p1: [u32; 11] = [
        b as u32,
        n_q_heads as u32,
        n_kv_heads as u32,
        n_repeats as u32,
        t_active as u32,
        head_dim as u32,
        n_blocks as u32,
        has_mask,
        q8_words_per_tok as u32,
        tq4_words_per_tok as u32,
        t_stride as u32,
    ];
    let params_p1_arr = {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(params_p1.as_ptr().cast::<u8>(), 11 * 4) };
        Array::from_bytes(bytes, &[11], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("turbo_flash p1 params: {e}")))?
    };

    // ── Dispatch P1 ───────────────────────────────────────────────────────────
    let kern_p1 = p1_kernel()?;
    let mut inv_p1 = MetalKernelInvoke::new();
    inv_p1.add_input(&q_flat)?;
    inv_p1.add_input(k_codes_flat)?;
    inv_p1.add_input(k_scales_flat)?;
    inv_p1.add_input(v_codes_flat)?;
    inv_p1.add_input(v_scales_flat)?;
    inv_p1.add_input(&mask_flat)?;
    inv_p1.add_input(&params_p1_arr)?;

    // partial_out: f32 [B × n_q_heads × n_blocks × head_dim]
    let partial_out_len = b * n_q_heads * n_blocks * head_dim;
    inv_p1.add_output_shape(&[partial_out_len], Dtype::F32)?;
    // partial_ms: f32 [B × n_q_heads × n_blocks × 2]
    let partial_ms_len = b * n_q_heads * n_blocks * 2;
    inv_p1.add_output_shape(&[partial_ms_len], Dtype::F32)?;

    // Grid is total threads (mlx-c `mlx_fast_metal_kernel` uses dispatchThreads
    // semantics — see https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
    // We want exactly `B*n_q_heads * n_blocks` threadgroups, each with TG_SIZE=32
    // threads in the x-axis. So x-grid must be `(B*n_q_heads) * TG_SIZE`.
    //
    // Earlier the grid was set to `(B*n_q_heads, n_blocks, 1)` which produced
    // a single threadgroup of 16 threads (B*n_q_heads=16 for Qwen35B): only
    // head 0 was computed, simd_sum read 16 inactive lanes, output garbage.
    inv_p1.set_grid(b * n_q_heads * TG_SIZE, n_blocks, 1)?;
    inv_p1.set_thread_group(TG_SIZE, 1, 1)?;

    // Counter increment sits at the actual P1 enqueue point — after all
    // validation gates above, immediately before the kernel `.apply()` call
    // that submits work to the Metal command queue.
    // Reads via `turbo_flash_dispatch_count()` (NIAH harness only).
    TURBO_FLASH_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut p1_outs = kern_p1.apply(inv_p1, device)?;
    if p1_outs.len() < 2 {
        return Err(Error::Mlx("turbo_flash_p1: expected 2 outputs".to_owned()));
    }
    let partial_out = p1_outs.remove(0);
    let partial_ms = p1_outs.remove(0);

    // ── P2 params: [n_blocks, head_dim, B×n_q_heads, _pad] ──────────────────
    let params_p2: [u32; 4] = [
        n_blocks as u32,
        head_dim as u32,
        (b * n_q_heads) as u32,
        0u32,
    ];
    let params_p2_arr = {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(params_p2.as_ptr().cast::<u8>(), 4 * 4) };
        Array::from_bytes(bytes, &[4], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("turbo_flash p2 params: {e}")))?
    };

    // ── Dispatch P2 ───────────────────────────────────────────────────────────
    let kern_p2 = p2_kernel()?;
    let mut inv_p2 = MetalKernelInvoke::new();
    inv_p2.add_input(&partial_out)?;
    inv_p2.add_input(&partial_ms)?;
    inv_p2.add_input(&params_p2_arr)?;

    // dst: f32 [B × n_q_heads × head_dim]
    let dst_len = b * n_q_heads * head_dim;
    inv_p2.add_output_shape(&[dst_len], Dtype::F32)?;

    // Grid is total threads (dispatchThreads). We want `B*n_q_heads`
    // threadgroups, each with `tg_size = max(head_dim, 32)` threads.
    let tg_size = head_dim.max(32);
    inv_p2.set_grid(b * n_q_heads * tg_size, 1, 1)?;
    inv_p2.set_thread_group(tg_size, 1, 1)?;

    let mut p2_outs = kern_p2.apply(inv_p2, device)?;
    if p2_outs.is_empty() {
        return Err(Error::Mlx("turbo_flash_p2: expected 1 output".to_owned()));
    }
    let dst_flat = p2_outs.remove(0);

    // Reshape to [B, n_q_heads, 1, head_dim], and restore the query dtype the
    // caller handed in — the kernels declare f32 outputs for the accumulation,
    // but f32 out of an attention op promotes the residual stream.
    let dst = dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)?;
    dst.astype(queries.dtype(), device)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "turbo_flash_msl_tests.rs"]
mod tests;
