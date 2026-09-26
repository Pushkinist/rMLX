//! Free helper functions, test-only probes, and unit tests for `KvCache`.
#![allow(clippy::too_many_lines)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use super::KvCache;

// ── Test-only probes ──────────────────────────────────────────────────────────
//
// The `probe_*_dequant` pair is called from the SSD round-trip tests in
// `rmlx-kv-ssd` across the crate boundary. A cross-crate `#[cfg(test)]` gate is
// not possible (each crate compiles its cfg(test) independently), so they stay
// `pub`. Both read one slot of `KvStorage::view`, so no panic path is
// reachable from production callers.
//
// Both return `Option<Result<..>>`, and the two layers mean different things:
// `None` is "this variant has no CPU-dequantizable buffer on that axis",
// `Some(Err(..))` is "the store exists and its dequant refused". Collapsing the
// second into the first reports a missing buffer for the blocks-vs-`shape[2]`
// coverage failure, which is the one thing these probes are used to detect.

impl KvCache {
    /// Dequant the K side of the cache to flat f32 (CPU paths only).
    ///
    /// Returns `None` for storage variants that have no q8 K buffer
    /// (`Paged`, `Mixed`, `None`), and `Some(Err(..))` when the K
    /// store exists but its dequant refused — see the module note above for why
    /// those stay distinct. Used by the hydrate round-trip tests to compare a
    /// reconstructed cache's K against the pre-spill K within the fp tolerance.
    pub fn probe_k_dequant(&self, device: Device) -> Option<Result<Vec<f32>>> {
        let [k, _, _] = self.storage.view().slots;
        k?.dequant_f32(device)
    }
}

impl KvCache {
    /// Dequant the V side of the cache to flat f32 (CPU paths only).
    ///
    /// Companion of [`Self::probe_k_dequant`], and `pub` for the same reason
    /// that one is: it centralises the per-variant V dispatch so the SSD
    /// round-trip tests in `rmlx-kv-ssd` — and any future codec added to them —
    /// do not each re-derive which field holds V. (A caller *could* match on
    /// `KvCache::storage()` locally, which is already `pub`; that is not the
    /// justification. The justification is one dispatch, not four.) The K probe
    /// alone cannot see the V-side codecs — `QuantV`, `QuantPlanarV` and the
    /// iso / rotor V stores — which is where the block-accumulating payload
    /// lives for most quants.
    ///
    /// The two failure modes are kept apart on purpose:
    ///
    /// * `None` — this variant has no CPU-dequantizable V store at all. The
    ///   K-only families (`PlanarK`, `IsoKOnly*`, `RotorKOnly*`) keep V as bf16
    ///   on the parent cache; `None` / `Mixed` / `Paged` hold no per-axis quant
    ///   store; and an un-initialised axis is `v: None`.
    /// * `Some(Err(..))` — the store exists and its dequant refused, which is
    ///   what the blocks-vs-`shape[2]` coverage check returns. Collapsing that
    ///   into `None` would report "no V buffer" for the one failure mode the
    ///   truncation work actually introduces.
    pub fn probe_v_dequant(&self, device: Device) -> Option<Result<Vec<f32>>> {
        let [_, v, _] = self.storage.view().slots;
        v?.dequant_f32(device)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(super) fn arrays_to_f32(k: &Array, v: &Array, device: Device) -> Result<(Vec<f32>, Vec<f32>)> {
    let k_f32 = array_to_f32_vec(k, device)?;
    let v_f32 = array_to_f32_vec(v, device)?;
    Ok((k_f32, v_f32))
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(super) fn array_to_f32_vec(a: &Array, device: Device) -> Result<Vec<f32>> {
    let a_f32 = if a.dtype() == Dtype::F32 {
        a.try_clone()?
    } else {
        a.astype(Dtype::F32, device)?
    };
    a_f32.eval()?;
    let bytes = a_f32.to_bytes()?;
    let n = bytes.len() / 4;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    out.extend(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
    );
    Ok(out)
}

/// Cut the first `seq_len` sequence positions out of a rank-4 bf16 V mirror
/// `[b, kv_h, v_seq, head_dim]`.
///
/// For consumers that need an exactly-sized tensor. The flash-decode kernels do
/// not: the mirror is head-major, so this view is row-contiguous only when
/// `b * kv_h == 1` and flattening it anywhere else copies the whole prefix.
/// Those dispatchers take the mirror whole and stride over it instead — see
/// `crate::flash_decode_common::flatten_v_mirror`.
///
/// # Errors
///
/// [`Error::Quant`] for a non-rank-4 `v` or an out-of-range `seq_len` — the
/// same shape-contract kind `flatten_v_mirror` raises for the same faults.
pub(super) fn slice_v_prefix(v: &Array, seq_len: i32, device: Device) -> Result<Array> {
    let shape = v.shape();
    let [b, kv_h, v_seq, head_dim] = shape[..] else {
        return Err(Error::Quant(format!("V mirror rank != 4, got {shape:?}")));
    };
    if seq_len < 0 || seq_len > v_seq {
        return Err(Error::Quant(format!(
            "V mirror prefix {seq_len} out of range for sequence extent {v_seq}"
        )));
    }
    v.slice(
        &[0_i32; 4],
        &[b, kv_h, seq_len, head_dim],
        &[1_i32; 4],
        device,
    )
}

pub(super) fn f32_vec_to_array(data: &[f32], shape: &[i32]) -> Result<Array> {
    // SAFETY: f32 and u8 have compatible alignment; we copy out of the slice immediately.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32)
}

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod tests;
