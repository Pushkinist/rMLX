//! Fused QK MSL kernel for PlanarQuant-packed K.
//!
//! # What this is
//!
//! Single MSL kernel that consumes PlanarQuant-packed K (codes / per-pair scales
//! / 4-bit rotation indices) **directly** and computes pre-softmax QK scores
//! `[B, n_q_heads, S, S_kv]` without an intermediate dequantized K tensor in
//! HBM.  The legacy path materialises a full bf16 K via
//! [`planar_dequantize_v4_gpu`](crate::planarquant_msl::planar_dequantize_v4_gpu)
//! and then calls `scaled_dot_product_attention`; the dequant write+read of K
//! is the dominant decode-step bandwidth cost on memory-bound models.
//!
//! The flash-decode kernel (`planar_flash_decode_msl`) extends this contract to
//! fuse the V path as well.  Further generalisation to other codecs (rotor, iso)
//! is deferred.  See `docs/KV_QUANT.md` §"Fused-QK kernels".
//!
//! # Pattern reference
//!
//! `multi-turboquant` ships a `PLANAR_FUSED_QK_KERNEL` in
//! `multi_turboquant/kernels/metal/fused_attention.py`.  That kernel uses a
//! single fixed Hadamard rotation (45°) and one scale per token.  rMLX's
//! PlanarQuant codec uses a **16-entry Givens codebook** (per-pair rotation
//! index in `rot32`) and a **per-pair scale** (one f32 per 2 channels).  The
//! kernel below implements the rMLX codec contract — bit-exact with
//! [`planar_dequantize_v4_gpu`].
//!
//! # Kernel shape
//!
//! Grid `(S_kv, B * n_q_heads, 1)`; threadgroup `(head_dim, 1, 1)`.
//! Each threadgroup computes one score `out[b, hq, s_q_zero, s_kv]`.
//!
//! * Thread `tid` (in `0..head_dim`) loads its element of Q from threadgroup
//!   memory, decodes its element of K (centroid lookup + per-pair Givens
//!   rotation + scale → register), forms the per-thread product, and
//!   participates in a tree-reduction across the threadgroup.
//! * Thread 0 writes the reduced dot product (scaled by `scale[0]`).
//!
//! # Single Q-step contract (decode-only)
//!
//! The kernel is decode-only: `S_q == 1`.  The caller passes the per-(b, hq)
//! Q vector for the new token; the kernel scores it against every K position
//! `s_kv ∈ 0..S_kv`.  This matches the dominant cost at decode time (single
//! token, long K context); multi-Q-step prefill is handled by the flash-decode
//! kernel.
//!
//! # GQA support
//!
//! Q has `n_q_heads`, K has `kv_h` heads where `n_q_heads = kv_h *
//! heads_per_kv`.  The threadgroup maps `(b, hq) → kv_h_idx = hq /
//! heads_per_kv` to share K across the GQA group.  `heads_per_kv` is a kernel
//! arg passed via the `dims` uint vector.
//!
//! # Numerical contract
//!
//! Decoded K element is computed in `float` (f32) registers — matches the CPU
//! [`planar_dequantize`](crate::planarquant::planar_dequantize) path bit-for-bit.
//! The dot product accumulates in `float`; the `scale * sum` write is `float`
//! (output dtype f32, caller casts if needed).  Q is loaded as `float` even
//! when the caller passes f16 — converting at thread load is cheaper than
//! threadgroup memory pressure.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::planarquant::{planar_rotation_codebook, N_ROTATIONS};
use crate::turboquant::{lloyd_gaussian_codebook, GROUP_SIZE};
use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Dispatch counter ──────────────────────────────────────────────────────────

/// Incremented exactly once per [`planar_fused_qk`] invocation that reaches
/// the Metal enqueue point.
///
/// Used by dormancy tests to assert that the PlanarK fused-QK kernel did NOT
/// fire on a warm-seeded PlanarK decode step (the warm-TTFT gate shorts to
/// bf16-K SDPA before reaching this kernel).
///
/// # Atomic coherence note
///
/// The counter is loaded with `Relaxed` ordering.  Use before/after deltas
/// in single-threaded test contexts — no concurrent dispatch can race the
/// read in that setting.
static PLANAR_FUSED_QK_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of [`planar_fused_qk`] invocations
/// that reached the Metal enqueue point.
#[must_use]
pub fn planar_fused_qk_dispatch_count() -> u64 {
    PLANAR_FUSED_QK_DISPATCHES.load(Ordering::Relaxed)
}

// ── MSL header builder ────────────────────────────────────────────────────────

/// Build the MSL header (rotation codebook + Lloyd-Max centroid codebook) for
/// the chosen `bits`.  Bit-exact with `planarquant_msl::build_msl_header*`.
///
/// `bits` is in `{3, 4}` — the only Planar bit-widths shipped by the V codec.
/// The K-side storage ([`crate::storage::QuantPlanarK`]) is hard-coded to
/// 4-bit at present; the 3-bit kernel exists for V-axis reuse (parity tests)
/// and for the future K-axis 3-bit codec.
fn build_qk_header(bits: u8) -> Result<String> {
    let cb = lloyd_gaussian_codebook(bits)?;
    let n_centroids = cb.len();
    let rot_cb = planar_rotation_codebook();
    if rot_cb.len() != N_ROTATIONS {
        return Err(Error::Mlx(format!(
            "planar_fused_qk: rotation codebook length {got} != {expected}",
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
         constant float QK_ROT_CB[{N}][4] = {{\n{entries}\n}};\n",
        N = N_ROTATIONS,
        entries = rot_entries.join(",\n")
    );
    let _ = write!(
        s,
        "\n// Lloyd-Max N(0,1) centroid codebook ({bits}-bit, {n_centroids} entries).\n\
         constant float QK_CB[{n_centroids}] = {{\n{cb_entries}\n}};\n",
        bits = bits,
        n_centroids = n_centroids,
        cb_entries = cb_entries.join(",\n")
    );
    Ok(s)
}

// ── MSL kernel source (parametrised by bits via a different header) ──────────
//
// `dims` (uint vector, 4 elements):
//   dims[0] = head_dim       (D)            — also equals threadgroup size
//   dims[1] = kv_seq         (S_kv)         — number of K positions
//   dims[2] = kv_h           (number of K/V heads)
//   dims[3] = heads_per_kv   (n_q_heads / kv_h)
//
// Inputs (K buffers are SEQUENCE-major: `[B, S_kv, kv_h, D]` element order):
//   query  : float[B * n_q_heads * D]  (Q for the new token)
//   codes  : uint [B * S_kv * kv_h * (D/8)]   — packed 4-bit codes
//   scales : float[B * S_kv * kv_h * (D/2)]   — per-pair scale (f32)
//   rot32  : uint [B * S_kv * kv_h * (D/16)]  — 4-bit rotation index per pair
//   scale_arr : float[1]   — softmax pre-scale (1/sqrt(d))
//
// Output:
//   out : float[B * n_q_heads * S_kv]   — pre-softmax scores
//
// Grid: (S_kv, B * n_q_heads, 1).  Threadgroup: (D, 1, 1).

// Two independent per-bits kernel bodies, one file each.
//
// They differ only in the mask + shift arithmetic for `idx` and in
// `vals_per_word` (8 for 4-bit, 10 for 3-bit) with its code-word indexing —
// roughly six lines. The remaining ~97% is duplicated between the two files:
// a fix to the shared arithmetic must be applied to BOTH.
const QK_SOURCE_V4: &str = include_str!("metal/planar_fused_qk_b4.metal");
const QK_SOURCE_V3: &str = include_str!("metal/planar_fused_qk_b3.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static QK_KERNEL_V4: OnceLock<Result<MetalKernel>> = OnceLock::new();
static QK_KERNEL_V3: OnceLock<Result<MetalKernel>> = OnceLock::new();
// Headers are generated (rotation codebook), so they are memoised. The bodies
// are compile-time constants and need no cache.
static QK_HEADER_V4: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static QK_HEADER_V3: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn header_for(bits: u8) -> Result<&'static str> {
    // Surface header-build errors instead of
    // caching an empty `String` on failure.  The previous implementation
    // memoised `Ok("")` after the first failure, producing cryptic MSL
    // compile errors on every subsequent decode step.
    let cell = match bits {
        4 => &QK_HEADER_V4,
        3 => &QK_HEADER_V3,
        _ => {
            return Err(Error::Mlx(format!(
                "planar_fused_qk: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    cell.get_or_init(|| build_qk_header(bits).map_err(|e| e.to_string()))
        .as_deref()
        .map_err(|e| Error::Mlx(format!("planar_fused_qk header build failed: {e}")))
}

fn source_for(bits: u8) -> Result<&'static str> {
    match bits {
        4 => Ok(QK_SOURCE_V4),
        3 => Ok(QK_SOURCE_V3),
        _ => Err(Error::Mlx(format!(
            "planar_fused_qk: bits must be 3 or 4, got {bits}"
        ))),
    }
}

fn qk_kernel(bits: u8) -> Result<&'static MetalKernel> {
    let cell = match bits {
        4 => &QK_KERNEL_V4,
        3 => &QK_KERNEL_V3,
        _ => {
            return Err(Error::Mlx(format!(
                "planar_fused_qk: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    let name = if bits == 4 {
        "rmlx_planar_fused_qk_v4"
    } else {
        "rmlx_planar_fused_qk_v3"
    };
    let header = header_for(bits)?;
    let source = source_for(bits)?;
    cell.get_or_init(|| {
        MetalKernel::new(
            name,
            header,
            source,
            &["query", "codes", "scales", "rot32", "scale_arr", "dims"],
            &["out"],
        )
    })
    .as_ref()
    .map_err(|e| Error::Mlx(format!("planar_fused_qk kernel init: {e}")))
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the fused PlanarQuant QK kernel.
///
/// # Inputs
///
/// * `query`  — Q for the new token, shape `[B, n_q_heads, 1, head_dim]` (or
///   `[B, n_q_heads, head_dim]`).
/// * `codes`  — packed K codes from a `QuantPlanarK` / `QuantPlanarV` buffer,
///   flat `u32` of length `B * kv_h * kv_seq * (head_dim / vals_per_word)`.
/// * `scales` — per-pair scales (f32), flat length `B * kv_h * kv_seq * (head_dim / 2)`.
/// * `rot32`  — 4-bit rotation indices, flat `u32` of length
///   `B * kv_h * kv_seq * (head_dim / 16)`.
/// * `b`, `kv_h`, `kv_seq`, `head_dim`, `heads_per_kv` — shape metadata.
/// * `bits`   — code bit-width (3 or 4).
/// * `scale`  — softmax pre-scale (1/sqrt(head_dim) typically).
///
/// # Output
///
/// Scores tensor `[B, n_q_heads, 1, kv_seq]` (f32).  The caller is responsible
/// for adding any additive mask and running the softmax + SV path.
///
/// # Errors
///
/// Returns `Error::Mlx` for kernel build / dispatch failure, or `Error::Quant`
/// for shape contract violations.
// f32-out-ok: pre-softmax scores, not the attention output — the caller
// softmaxes them and restores the query dtype on the SV result
// (`KvCache::try_fused_qk_dispatch`), so nothing f32 reaches the residual
// stream. The scores do carry their width into that intervening matmul; that
// is a cost inside the attention op, not a promotion of the graph behind it.
#[allow(clippy::too_many_arguments)]
pub fn planar_fused_qk(
    query: &Array,
    codes: &Array,
    scales: &Array,
    rot32: &Array,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    bits: u8,
    scale: f32,
    device: Device,
) -> Result<Array> {
    if !matches!(bits, 3 | 4) {
        return Err(Error::Quant(format!(
            "planar_fused_qk: bits must be 3 or 4, got {bits}"
        )));
    }
    if head_dim <= 0 || head_dim % (GROUP_SIZE as i32) != 0 {
        return Err(Error::Quant(format!(
            "planar_fused_qk: head_dim={head_dim} must be a positive multiple of \
             GROUP_SIZE={GROUP_SIZE}"
        )));
    }
    // The MSL tree reduction
    // `for (stride = head_dim >> 1; stride > 0; stride >>= 1)` produces
    // wrong sums for non-power-of-two `head_dim`.  Reject here so the SDPA
    // dispatcher falls through to the legacy dequant+SDPA path instead of
    // silently returning incorrect scores.  Supported decode head dims
    // (64, 128, 256) are all powers of two; uncommon dims (80, 96) hit the
    // legacy path.
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "planar_fused_qk: head_dim={head_dim} must be a power of two for the \
             fused-QK kernel's tree reduction (legacy dequant+SDPA path handles \
             non-pow-2 dims)"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "planar_fused_qk: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    // Reject degenerate shapes before computing grid arithmetic.  Empty grids
    // surface as opaque mlx-c failures; explicit error keeps the message at the
    // codec boundary.
    if b <= 0 || kv_seq <= 0 {
        return Err(Error::Quant(format!(
            "planar_fused_qk: b={b} and kv_seq={kv_seq} must be > 0"
        )));
    }
    let n_q_heads = kv_h * heads_per_kv;

    // Flatten Q to [B * n_q_heads * head_dim].  i64 arithmetic to guard
    // against overflow on large head counts × seq lengths.
    let q_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    // Flatten K buffers — caller already passes them flat per the storage
    // contract, but reshape defensively to make the kernel signature concrete.
    // Codes per token = (head_dim / 32) * 4 for both 3-bit (10 vals/u32 with
    // 2 wasted bits per word) and 4-bit (8 vals/u32, exact).  Codec contract
    // documented in `planarquant_msl.rs::QUANTIZE_SOURCE` / `_V3`.
    let codes_per_tok: i64 = (i64::from(head_dim) / GROUP_SIZE as i64) * 4;
    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * codes_per_tok;
    let scales_total: i64 = tok_count * i64::from(head_dim) / 2;
    let rot_total: i64 = tok_count * i64::from(head_dim) / 16;
    let codes_flat = codes.reshape(&[codes_total as i32], device)?;
    let scales_flat = scales.reshape(&[scales_total as i32], device)?;
    let rot32_flat = rot32.reshape(&[rot_total as i32], device)?;

    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };
    let dims_bytes: Vec<u8> = [
        head_dim as u32,
        kv_seq as u32,
        kv_h as u32,
        heads_per_kv as u32,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let dims_arr = Array::from_bytes(&dims_bytes, &[4], Dtype::U32)?;

    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.
    //
    // The former justification here — a barrier against pending
    // `planar_quantize` atomic-OR accumulation — described a hazard that is
    // handled where it arises: `planar_quantize_v4_gpu` zero-initialises its
    // atomically-ORed outputs (`set_init_value(0.0)`).

    let kernel = qk_kernel(bits)?;
    PLANAR_FUSED_QK_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    // Per-dispatch trace, matching every sibling KV kernel. The in-process
    // counter above has no caller outside tests, so this event is the only
    // way a shipped binary can answer "did this kernel run".
    tracing::trace!(
        bits,
        b,
        n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        "planar_fused_qk: dispatch"
    );
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&q_f32)?;
    invoke.add_input(&codes_flat)?;
    invoke.add_input(&scales_flat)?;
    invoke.add_input(&rot32_flat)?;
    invoke.add_input(&scale_arr)?;
    invoke.add_input(&dims_arr)?;

    let out_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(kv_seq);
    invoke.add_output_shape(&[out_total as i32], Dtype::F32)?;

    // mlx-c `set_grid` takes the TOTAL thread count per axis, not
    // threadgroup count.  Total threads = threadgroups × threadgroup_size.
    // Grid X: kv_seq threadgroups × head_dim threads each = kv_seq * head_dim.
    //
    // Widen grid arithmetic to i64 and verify each axis fits in i32 before
    // dispatch.  At extreme contexts (e.g. kv_seq=131072 × head_dim=256) the
    // X-axis product would silently wrap on i32 multiplication; explicit
    // overflow check fails loud.
    let grid_x: i64 = i64::from(kv_seq) * i64::from(head_dim);
    let grid_y: i64 = i64::from(b) * i64::from(n_q_heads);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "planar_fused_qk: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    invoke.set_grid(grid_x as i32, grid_y as i32, 1)?;
    invoke.set_thread_group(head_dim, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if outputs.is_empty() {
        return Err(Error::Mlx(
            "planar_fused_qk: kernel produced no outputs".into(),
        ));
    }
    let out_flat = outputs.remove(0);

    // Reshape to the SDPA-canonical [B, n_q_heads, 1, kv_seq].
    out_flat.reshape(&[b, n_q_heads, 1, kv_seq], device)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "planar_fused_qk_msl_tests.rs"]
mod tests;
