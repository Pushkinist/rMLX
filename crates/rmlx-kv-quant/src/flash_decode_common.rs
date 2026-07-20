//! Shared scaffold for the symmetric (quant-K + quant-V) flash-decode MSL
//! dispatchers — [`crate::iso_flash_decode_symv_msl`] and
//! [`crate::rotor_flash_decode_symv_msl`].
//!
//! Both bind a per-token `norms` array as a kernel input. MLX's custom-kernel
//! builder binds a small input array's outer-kernel parameter in the
//! **`constant`** address space instead of `device` (an internal size
//! heuristic), but both codecs' shared per-lane/per-group decode helpers
//! (`if_decode_k_lane` in [`crate::iso_flash_decode_msl`]; `rf_decode_k_group`
//! — the symv P1 kernel body calls this directly rather than the
//! `rf_decode_k_lane` wrapper, since the Cl(3,0) sandwich needs the group
//! result, not a single lane — in [`crate::rotor_flash_decode_msl`]) declare
//! their `norms` parameter `device const float*` — an address-space mismatch
//! that fails the MSL compile at first dispatch. The threshold was
//! **measured** at 8 (a
//! ring-only decode at `kv_h == 1` aborts for `kv_seq` 2–7 and succeeds from 8
//! on, for every `head_dim`); [`NORMS_DEVICE_MIN`] is set above it for
//! margin. Only `norms` (one f32 per token) crosses it at low `kv_seq` —
//! `codes` / `scales` carry `n_groups` more elements per token, and rotor's
//! `rotors` table is sized off `n_groups` alone, so both stay above the
//! threshold at any `kv_seq >= 1`.
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

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{pad, Array, Device};

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
