//! Shared scaffold for the flash-decode MSL dispatchers.
//!
//! # The `norms` device-address-space floor
//!
//! Scoped to the symmetric (quant-K + quant-V) dispatchers —
//! [`crate::iso_flash_decode_symv_msl`] and
//! [`crate::rotor_flash_decode_symv_msl`].
//!
//! Both bind a per-token `norms` array as a kernel input. On the MLX build this
//! floor was measured against, the custom-kernel builder **bound** a small input
//! array's outer-kernel parameter in the **`constant`** address space instead of
//! `device` (an internal size heuristic), while both codecs' shared
//! per-lane/per-group decode helpers (`if_decode_k_lane` in
//! [`crate::iso_flash_decode_msl`]; `rf_decode_k_group` — the symv P1 kernel
//! body calls this directly rather than the `rf_decode_k_lane` wrapper, since
//! the Cl(3,0) sandwich needs the group result, not a single lane — in
//! [`crate::rotor_flash_decode_msl`]) declare their `norms` parameter in the
//! `device` address space. That mismatch failed the MSL compile at first
//! dispatch. The threshold was **measured** at 8 (a ring-only decode at
//! `kv_h == 1` aborted for `kv_seq` 2–7 and succeeded from 8 on, for every
//! `head_dim`); [`NORMS_DEVICE_MIN`] was set above it for margin. Only `norms`
//! (one scalar per token) came near it at low `kv_seq` — `codes` / `scales`
//! carry `n_groups` more elements per token, and rotor's `rotors` table is
//! sized off `n_groups` alone, so both stayed above the threshold at any
//! `kv_seq >= 1`.
//!
//! **The abort no longer reproduces on the pinned MLX.** Re-measured by
//! disabling [`pad_norms_to_device_floor`] and sweeping a `kv_h == 1`
//! `iso3_sym` ring-only decode over `kv_seq` 2..=24 at `head_dim = 512`: every
//! step dispatched the fused symv kernel and returned successfully, at both the
//! `f32` sideband this store used to carry and the `bf16` one it carries now.
//! So the floor is currently inert rather than load-bearing, and halving the
//! plane's element width did **not** raise it — the heuristic that produced the
//! original abort is not the byte size of this buffer on this MLX build. The
//! constant stays because the failure mode it guards is a property of MLX's
//! signature builder, not of this crate, and an MLX bump could bring it back;
//! what does not stay is a claim that it is currently doing something.
//!
//! [`pad_norms_to_device_floor`] zero-pads `norms` up to the floor before
//! dispatch instead of falling back to a CPU dequant path: each kernel's
//! per-tile loop bound is the real `kv_seq` carried in its `dims` buffer, not
//! the `norms` buffer length, so the padding is allocated but never read.
//! This keeps both fused kernels on the GPU for every `kv_seq >= 1` with no
//! CPU dequant fallback (hard rule 10) — in particular for a normal short
//! chat prompt against a single-KV-head model (Gemma4 global layers are
//! `kv_h == 1`), where a 2-token prompt reaches `kv_seq == 2` on the first
//! decode step, well below the compile floor.
//!
//! # Why a kernel dispatcher must not `eval()` its inputs
//!
//! This is the canonical statement of the rule; the MSL dispatchers point here
//! rather than repeating it, so the argument cannot drift between copies. The
//! CI gate `make check-no-kernel-input-eval` enforces it.
//!
//! These kernels read their buffers by raw linear offset, so they need
//! row-contiguous inputs. Historically every dispatcher forced
//! `Array::eval()` on each input immediately before dispatch, on the reasoning
//! that a pending lazy transpose would otherwise be read with the wrong
//! strides. **Both halves of that reasoning are wrong:**
//!
//! * **Ordering.** `rmlx_mlx::metal_kernel::MetalKernel::apply` enqueues an
//!   MLX `fast::CustomKernel` **graph node**; it does not dispatch.
//!   MLX runs that node's `eval_gpu` only once every input edge is itself
//!   materialised, and the `ensure_row_contiguous` copy — requested by
//!   `MetalKernel::new` — is applied inside that same `eval_gpu`. The kernel
//!   therefore cannot observe an uncomputed or strided buffer. A caller-side
//!   `eval()` buys no ordering the graph does not already provide; it only
//!   moves *when* the host blocks.
//! * **Layout.** `Array::eval()` materialises but does **not** relayout. MLX's
//!   `Transpose` is a strided view over a shared buffer, so an evaluated
//!   transpose is still non-row-contiguous — evaluating it would not have
//!   fixed the stride problem the comment claimed to be guarding. The layout
//!   guarantee comes from `reshape` plus `ensure_row_contiguous`, and always
//!   did.
//!
//! The cost of the eval was not small: it blocks the calling thread on the GPU
//! once per attention layer per decode step, so the forward pass advances one
//! layer at a time with nothing queued ahead. Removing it is a fixed
//! per-step saving (it collapses the per-step intercept, not the per-KV-token
//! slope) worth multiples of the decode rate at short context.
//!
//! An eval that is genuinely load-bearing — a host readback before
//! `to_bytes()`, say — stays, and says so at the call site with an
//! `// eval-ok: <reason>` marker.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{pad, Array, Device, Dtype};

/// Flatten a bf16 / f16 / f32 V mirror for a flash-decode kernel, and report
/// the stride the kernel must index its sequence axis with.
///
/// `v` is `[b, kv_h, v_seq, head_dim]` with `v_seq >= kv_seq`: the caller hands
/// over the **whole** mirror allocation, not a `..kv_seq` slice of it, and
/// `kv_seq` stays the attended length. Only the stride changes.
///
/// That distinction is the point of this helper. The mirror is head-major, so
/// cutting a `..kv_seq` prefix out of it leaves a gap between heads: the view is
/// row-contiguous only when `b * kv_h == 1`, and flattening it anywhere else
/// materialises the whole prefix — once per layer per decode step, with no
/// `contiguous()` call at the site to make it visible. Flattening the
/// allocation itself copies nothing at any `kv_h`.
///
/// # Errors
///
/// [`Error::Quant`] when `v` is not rank 4, when its `b` / `kv_h` / `head_dim`
/// axes disagree with the passed shape metadata, when its sequence axis is
/// shorter than `kv_seq` (the kernel would read past the buffer), when the flat
/// element count overflows `i32`, or for a non-float dtype. Forwards `reshape`
/// errors otherwise.
pub(crate) fn flatten_v_mirror(
    v: &Array,
    kernel: &str,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    device: Device,
) -> Result<(Array, i32)> {
    let shape = v.shape();
    let [v_b, v_kv_h, v_seq, v_head_dim] = shape[..] else {
        return Err(Error::Quant(format!(
            "{kernel}: V rank != 4, got {shape:?}"
        )));
    };
    if v_b != b || v_kv_h != kv_h || v_head_dim != head_dim {
        return Err(Error::Quant(format!(
            "{kernel}: V shape {shape:?} disagrees with b={b}, kv_h={kv_h}, head_dim={head_dim}"
        )));
    }
    if v_seq < kv_seq {
        return Err(Error::Quant(format!(
            "{kernel}: V sequence extent {v_seq} is shorter than the attended kv_seq={kv_seq}"
        )));
    }
    match v.dtype() {
        Dtype::F32 | Dtype::Bf16 | Dtype::F16 => {}
        other @ (Dtype::U8 | Dtype::U32 | Dtype::I32) => {
            return Err(Error::Quant(format!(
                "{kernel}: V dtype must be F32 / Bf16 / F16, got {other:?}"
            )))
        }
    }
    let total: i64 = i64::from(b) * i64::from(kv_h) * i64::from(v_seq) * i64::from(head_dim);
    let total = i32::try_from(total).map_err(|_| {
        Error::Quant(format!(
            "{kernel}: V element count {total} overflows i32 at v_seq={v_seq}"
        ))
    })?;
    Ok((v.reshape(&[total], device)?, v_seq))
}

/// Minimum per-token `norms` element count (`b * kv_h * kv_seq`) a symmetric
/// flash-decode kernel (iso or rotor) is dispatched with. See the module doc
/// for why this floor exists and how it is measured.
pub(crate) const NORMS_DEVICE_MIN: i64 = 16;

/// Zero-pad flat `norms` (`[tok_count]`) up to [`NORMS_DEVICE_MIN`] elements
/// when `tok_count` is below it; returns `norms` unchanged otherwise.
///
/// # Errors
///
/// Returns [`Error::Mlx`] if the pad amount overflows `i32`, and forwards
/// [`pad`] errors.
pub(crate) fn pad_norms_to_device_floor(
    norms: Array,
    tok_count: i64,
    device: Device,
) -> Result<Array> {
    if tok_count >= NORMS_DEVICE_MIN {
        return Ok(norms);
    }
    let extra = i32::try_from(NORMS_DEVICE_MIN - tok_count)
        .map_err(|_| Error::Mlx("pad_norms_to_device_floor: pad amount overflowed i32".into()))?;
    pad(&norms, &[0], &[0], &[extra], device)
}

#[cfg(test)]
#[path = "flash_decode_common_tests.rs"]
mod flash_decode_common_tests;
