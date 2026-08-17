//! Free helper functions, test-only probes, and unit tests for `KvCache`.
#![allow(clippy::too_many_lines)]

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use super::KvCache;
use crate::storage::KvStorage;

// ── Test-only probes ──────────────────────────────────────────────────────────
//
// `probe_k_dequant` is called from `rmlx-models` `hydrate_tests.rs` across
// the crate boundary. A cross-crate `#[cfg(test)]` gate is not possible
// (each crate compiles its cfg(test) independently), so the function stays
// `pub`. The wildcard arm now returns `None` instead of panicking so no
// panic path is reachable from production callers.

impl KvCache {
    /// Dequant the K side of the cache to flat f32 (CPU paths only).
    ///
    /// Returns `None` for storage variants that have no q8 K buffer
    /// (`Paged`, `Mixed`, `RotKTq4V`, `None`). Used by the hydrate round-trip
    /// test to compare a reconstructed cache's K against the pre-spill K
    /// within the fp tolerance.
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn probe_k_dequant(&self, device: Device) -> Option<Vec<f32>> {
        match &self.storage {
            KvStorage::K8V4 { k, .. }
            | KvStorage::K8V8 { k, .. }
            | KvStorage::Planar { k, .. }
            | KvStorage::K8VTurbo3 { k, .. }
            | KvStorage::K8VTurbo3Tcq { k, .. }
            | KvStorage::K8VTurbo2Tcq { k, .. }
            | KvStorage::K8VTurbo2 { k, .. }
            | KvStorage::IsoV3 { k, .. }
            | KvStorage::IsoV4 { k, .. }
            | KvStorage::RotorV3 { k, .. }
            | KvStorage::RotorV4 { k, .. } => {
                let (flat, _) = k.as_ref()?.dequantize_choice(device, Dtype::F32).ok()?;
                Some(flat)
            }
            // TurboSym3 — K is `QuantKTurbo3`, independent type from
            // `QuantK` and `QuantKTurbo4`. Same dequantize_choice API, dispatched
            // separately so the type checker stays happy.
            KvStorage::TurboSym3 { k, .. } => {
                let (flat, _) = k.as_ref()?.dequantize_choice(device, Dtype::F32).ok()?;
                Some(flat)
            }
            // TurboSym4 — K is a `QuantKTurbo4`, independent type
            // from `QuantK`. Same dequantize_choice signature, dispatched
            // separately so the type checker stays happy.
            KvStorage::TurboSym4 { k, .. } => {
                let (flat, _) = k.as_ref()?.dequantize_choice(device, Dtype::F32).ok()?;
                Some(flat)
            }
            // PlanarK — K is a `QuantPlanarK`. Same API as QuantPlanarV.
            KvStorage::PlanarK { k, .. } => {
                let (flat, _) = k.as_ref()?.dequantize_choice(device, Dtype::F32).ok()?;
                Some(flat)
            }
            // Iso symmetric / K-only — K is QuantIsoK3 / QuantIsoK4
            // (CPU-only). Dequant via the codec's `dequant()` method.
            KvStorage::IsoSym3 { k, .. } | KvStorage::IsoKOnly3 { k, .. } => {
                Some(k.as_ref()?.dequant().ok()?)
            }
            KvStorage::IsoSym4 { k, .. } | KvStorage::IsoKOnly4 { k, .. } => {
                Some(k.as_ref()?.dequant().ok()?)
            }
            // Rotor symmetric / K-only — K is QuantRotorK3 / QuantRotorK4.
            // RotorKAsym3 / RotorKAsym4 — same K codec types.
            KvStorage::RotorSym3 { k, .. }
            | KvStorage::RotorKOnly3 { k, .. }
            | KvStorage::RotorKAsym3 { k, .. } => Some(k.as_ref()?.dequant().ok()?),
            KvStorage::RotorSym4 { k, .. }
            | KvStorage::RotorKOnly4 { k, .. }
            | KvStorage::RotorKAsym4 { k, .. } => Some(k.as_ref()?.dequant().ok()?),
            KvStorage::None { .. }
            | KvStorage::Mixed { .. }
            | KvStorage::Paged { .. }
            | KvStorage::RotKTq4V { .. } => None,
        }
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
        match &self.storage {
            // V is a `QuantV` (TurboQuant, any bit width).
            KvStorage::K8V4 { v, .. }
            | KvStorage::K8VTurbo3 { v, .. }
            | KvStorage::K8VTurbo3Tcq { v, .. }
            | KvStorage::K8VTurbo2 { v, .. }
            | KvStorage::K8VTurbo2Tcq { v, .. }
            | KvStorage::TurboSym3 { v, .. }
            | KvStorage::TurboSym4 { v, .. }
            | KvStorage::RotKTq4V { v, .. }
            | KvStorage::RotorKAsym3 { v, .. }
            | KvStorage::RotorKAsym4 { v, .. } => {
                let v = v.as_ref()?;
                Some(
                    v.dequantize_choice(device, Dtype::F32)
                        .map(|(flat, _)| flat),
                )
            }
            // V is a second `QuantK` (affine q8_0 on both axes).
            KvStorage::K8V8 { v, .. } => {
                let v = v.as_ref()?;
                Some(
                    v.dequantize_choice(device, Dtype::F32)
                        .map(|(flat, _)| flat),
                )
            }
            // V is a `QuantPlanarV`.
            KvStorage::Planar { v, .. } => {
                let v = v.as_ref()?;
                Some(
                    v.dequantize_choice(device, Dtype::F32)
                        .map(|(flat, _)| flat),
                )
            }
            // Iso / rotor V stores expose `dequant()` rather than the
            // device-choosing pair.
            KvStorage::IsoV3 { v, .. } | KvStorage::IsoSym3 { v, .. } => {
                Some(v.as_ref()?.dequant())
            }
            KvStorage::IsoV4 { v, .. } | KvStorage::IsoSym4 { v, .. } => {
                Some(v.as_ref()?.dequant())
            }
            KvStorage::RotorV3 { v, .. } | KvStorage::RotorSym3 { v, .. } => {
                Some(v.as_ref()?.dequant())
            }
            KvStorage::RotorV4 { v, .. } | KvStorage::RotorSym4 { v, .. } => {
                Some(v.as_ref()?.dequant())
            }
            // V is bf16 on the parent cache, or there is no per-axis store.
            KvStorage::PlanarK { .. }
            | KvStorage::IsoKOnly3 { .. }
            | KvStorage::IsoKOnly4 { .. }
            | KvStorage::RotorKOnly3 { .. }
            | KvStorage::RotorKOnly4 { .. }
            | KvStorage::None { .. }
            | KvStorage::Mixed { .. }
            | KvStorage::Paged { .. } => None,
        }
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

pub(super) fn f32_vec_to_array(data: &[f32], shape: &[i32]) -> Result<Array> {
    // SAFETY: f32 and u8 have compatible alignment; we copy out of the slice immediately.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32)
}

/// Returns a static string tag for the storage variant — used in typed error
/// messages for dispatch mismatches so callers see the actual variant name
/// rather than panicking with `unreachable!`.
pub(super) fn storage_variant_name(s: &KvStorage) -> &'static str {
    match s {
        KvStorage::K8V4 { .. } => "K8V4",
        KvStorage::K8V8 { .. } => "K8V8",
        KvStorage::Planar { .. } => "Planar",
        KvStorage::None { .. } => "None",
        KvStorage::Mixed { .. } => "Mixed",
        KvStorage::Paged { .. } => "Paged",
        KvStorage::RotKTq4V { .. } => "RotKTq4V",
        KvStorage::K8VTurbo3 { .. } => "K8VTurbo3",
        KvStorage::TurboSym3 { .. } => "TurboSym3",
        KvStorage::TurboSym4 { .. } => "TurboSym4",
        KvStorage::PlanarK { .. } => "PlanarK",
        KvStorage::K8VTurbo2 { .. } => "K8VTurbo2",
        KvStorage::IsoV3 { .. } => "IsoV3",
        KvStorage::IsoV4 { .. } => "IsoV4",
        KvStorage::RotorV3 { .. } => "RotorV3",
        KvStorage::RotorV4 { .. } => "RotorV4",
        KvStorage::K8VTurbo3Tcq { .. } => "K8VTurbo3Tcq",
        KvStorage::K8VTurbo2Tcq { .. } => "K8VTurbo2Tcq",
        KvStorage::IsoSym3 { .. } => "IsoSym3",
        KvStorage::IsoSym4 { .. } => "IsoSym4",
        KvStorage::IsoKOnly3 { .. } => "IsoKOnly3",
        KvStorage::IsoKOnly4 { .. } => "IsoKOnly4",
        KvStorage::RotorSym3 { .. } => "RotorSym3",
        KvStorage::RotorSym4 { .. } => "RotorSym4",
        KvStorage::RotorKOnly3 { .. } => "RotorKOnly3",
        KvStorage::RotorKOnly4 { .. } => "RotorKOnly4",
        KvStorage::RotorKAsym3 { .. } => "RotorKAsym3",
        KvStorage::RotorKAsym4 { .. } => "RotorKAsym4",
    }
}

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod tests;
