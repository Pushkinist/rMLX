//! The mutating sites that go through `KvStorage::view_mut` leave the same
//! storage as the per-variant matches they replace.
//!
//! The `old_*` fns below are verbatim copies of `KvStorage::reset`,
//! `KvStorage::truncate_to` and `KvStorage::clear_payload` as they stood
//! before the view. For every codec in `ALL_KV_QUANTS`, on an empty and on a
//! filled storage, and for a paged storage whose three slots hold 1, 2 and 3
//! pages, the old copy and the view-based fn run on two copies of one storage.
//! After each operation the two copies must have the same store digest and
//! the same K and V probe results, error text included. The truncation
//! targets are negative, zero, inside the first block, at a block edge, the
//! full fill and past the fill, where the unclamped stores refuse the next
//! read. A view arm that drops a slot leaves that slot uncut, and the digest
//! changes.
//!
//! The digest does not see a store's GPU bookkeeping, and that is where a
//! store's own `reset` and its `truncate_to(0)` differ. So both copies get a
//! sentinel GPU capacity before each operation, and the state after it
//! includes that capacity: a `reset` sets it to 0, a `truncate_to` keeps it.
//! The paged slots show the same difference in the page ids they allocate
//! next.
#![allow(
    clippy::too_many_lines,
    clippy::match_same_arms,
    clippy::cognitive_complexity,
    reason = "verbatim copies of the per-variant matches the view replaced"
)]

use super::core::KvCache;
use super::deep_clone_digest_tests::fill;
use super::store_bytes_tests::{store_digest, CHUNK_SEQ, TEST_MAX_SEQ};
use crate::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};
use crate::storage::{
    KvStorage, QuantIsoK, QuantIsoV, QuantK, QuantKTurbo, QuantPlanarK, QuantPlanarV, QuantRotorK,
    QuantRotorV, QuantV,
};
use crate::test_utils::env_lock;
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_core::error::Result;
use rmlx_mlx::Device;

fn old_reset(storage: &mut KvStorage) {
    match storage {
        KvStorage::K8V4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        KvStorage::K8V8 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        KvStorage::Planar { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // None: no quant state — the bf16 buffers live on KvCache and are
        // dropped/reset by KvCache::reset directly.
        KvStorage::None { .. } => {}
        KvStorage::Mixed { state, .. } => state.reset(),
        KvStorage::Paged {
            k, v_k8, v_planar, ..
        } => {
            if let Some(pk) = k.as_mut() {
                pk.reset();
            }
            if let Some(pv) = v_k8.as_mut() {
                pv.reset();
            }
            if let Some(pv) = v_planar.as_mut() {
                pv.reset();
            }
        }
        // K8VTurbo3 resets like K8V4.
        KvStorage::K8VTurbo3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // TurboSym3 — symmetric reset (K3 + V3 shape-zeroing).
        KvStorage::TurboSym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // TurboSym4 — symmetric reset (same shape-zeroing).
        KvStorage::TurboSym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // PlanarK — K only; V (bf16) lives on parent KvCache.
        KvStorage::PlanarK { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
        }
        // K8VTurbo2 resets like K8V4.
        KvStorage::K8VTurbo2 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // IsoV3 — K is q8_0; V holds CPU IsoBlocks (reset clears them so
        // the next request starts fresh).
        KvStorage::IsoV3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        // IsoV4 — same shape semantics as IsoV3.
        KvStorage::IsoV4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        // RotorV3 — K shape zeroes; V codec resets blocks but KEEPS the
        // static rotor table.
        KvStorage::RotorV3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        // RotorV4 — same semantics as RotorV3.
        KvStorage::RotorV4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        // K8VTurbo3Tcq resets like K8VTurbo3 / K8V4.
        KvStorage::K8VTurbo3Tcq { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // K8VTurbo2Tcq resets like K8VTurbo2 / K8VTurbo3Tcq.
        KvStorage::K8VTurbo2Tcq { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(0);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        // IsoSym3/IsoSym4 reset both K + V iso buffers.
        KvStorage::IsoSym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        KvStorage::IsoSym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        // IsoKOnly3/4 — K iso buffer only; V bf16 lives on parent.
        KvStorage::IsoKOnly3 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
        }
        KvStorage::IsoKOnly4 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
        }
        // RotorSym3 / RotorSym4 — reset both K + V rotor buffers (each has
        // its own per-token blocks; rotor table + QJL matrix are layer-static
        // and kept).
        KvStorage::RotorSym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        KvStorage::RotorSym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
            if let Some(vs) = v.as_mut() {
                vs.reset();
            }
        }
        // RotorKOnly3/4 — K only; V bf16 on parent.
        KvStorage::RotorKOnly3 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
        }
        KvStorage::RotorKOnly4 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
        }
        // RotorKAsym3 / RotorKAsym4 — K rotor reset; V affine shape-zero
        // (same as K8V4 V-side).
        KvStorage::RotorKAsym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
        KvStorage::RotorKAsym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.reset();
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(0);
            }
        }
    }
}

fn old_truncate_to(storage: &mut KvStorage, n: i32) {
    // Clamp the negative case once, here, so no arm can compute from a
    // negative `n` before delegating.
    //
    // The upper clamp is NOT uniform, and the divergence is worth naming
    // rather than papering over. The turbo / planar / affine stores clamp
    // `n` down to their own `shape[2]` (`storage::clamp_truncate_target`);
    // the rotor / iso stores deliberately do not, because they abort loudly
    // on an over-long target instead. So for `n > shape[2]` the mixed arms
    // leave the two axes of one codec at different lengths: `IsoV3`,
    // `IsoV4`, `RotorV3`, `RotorV4` (affine K clamps, codec V does not) and
    // `RotorKAsym3` / `RotorKAsym4` (rotor K does not, affine V does). That
    // matters on spill, where the layer geometry is derived from the K shape
    // while the V payload is written raw — the reconciliation guard on the
    // unclamped side is what surfaces it.
    //
    // `Mixed` is a third reading and belongs in the same list: it has no
    // `shape[2]` to clamp, because `state.offset` IS its coverage. It keeps
    // its fill on an over-long target and reports one through an error
    // event (`MixedKvState::truncate_to`) — loud like the rotor / iso
    // stores, but at the truncate rather than at the next read, since
    // nothing downstream of it would notice.
    let n = n.max(0);
    match storage {
        KvStorage::K8V4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        KvStorage::K8V8 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        KvStorage::Planar { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // None: bf16 buffers are sliced lazily on next read; nothing to
        // truncate here. KvCache::truncate_to drops the buffers itself.
        KvStorage::None { .. } => {}
        // Mixed: the store is a capacity buffer with `state.offset` as its
        // fill marker, so rolling that marker back IS the truncation — see
        // `MixedKvState::truncate_to`. Resetting instead dropped the kept
        // prefix too, and `KvCache::truncate_to` then set `offset = n`,
        // leaving a cache that reports `n` positions and holds none.
        KvStorage::Mixed { state, .. } => state.truncate_to(n),
        KvStorage::Paged {
            k, v_k8, v_planar, ..
        } => {
            if let Some(pk) = k.as_mut() {
                pk.truncate_to(n);
            }
            if let Some(pv) = v_k8.as_mut() {
                pv.truncate_to(n);
            }
            if let Some(pv) = v_planar.as_mut() {
                pv.truncate_to(n);
            }
        }
        // K8VTurbo3 truncates like K8V4.
        KvStorage::K8VTurbo3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // TurboSym3 — symmetric truncate (K3 + V3 shape).
        KvStorage::TurboSym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // TurboSym4 — symmetric truncate (same shape semantics).
        KvStorage::TurboSym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // PlanarK — truncate K only; V (bf16) sliced lazily.
        KvStorage::PlanarK { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
        }
        // K8VTurbo2 truncates like K8V4.
        KvStorage::K8VTurbo2 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // IsoV3 — K shape truncates; V codec is per-token so dropping
        // trailing blocks is delegated to QuantIsoV3.
        KvStorage::IsoV3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // IsoV4 — same shape semantics as IsoV3.
        KvStorage::IsoV4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // RotorV3 — K shape truncates; V codec drops trailing blocks
        // (rotor table kept).
        KvStorage::RotorV3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // RotorV4 — same semantics as RotorV3.
        KvStorage::RotorV4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // K8VTurbo3Tcq truncates like K8VTurbo3 / K8V4.
        KvStorage::K8VTurbo3Tcq { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // K8VTurbo2Tcq truncates like K8VTurbo2 / K8VTurbo3Tcq.
        KvStorage::K8VTurbo2Tcq { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // IsoSym3 / IsoSym4 — both axes are per-token block codecs;
        // delegate to each side's truncate_to (same as IsoV3).
        KvStorage::IsoSym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        KvStorage::IsoSym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // IsoKOnly3 / IsoKOnly4 — K only; V bf16 sliced lazily on parent.
        KvStorage::IsoKOnly3 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
        }
        KvStorage::IsoKOnly4 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
        }
        // RotorSym3 / RotorSym4 — both axes are per-token block codecs;
        // delegate to each side's truncate_to.
        KvStorage::RotorSym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        KvStorage::RotorSym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        // RotorKOnly3 / RotorKOnly4 — K only; V bf16 sliced lazily on parent.
        KvStorage::RotorKOnly3 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
        }
        KvStorage::RotorKOnly4 { k, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
        }
        // RotorKAsym3 / RotorKAsym4 — K rotor truncate; V affine
        // shape-truncate (same as K8V4 V-side).
        KvStorage::RotorKAsym3 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
        KvStorage::RotorKAsym4 { k, v, .. } => {
            if let Some(ks) = k.as_mut() {
                ks.truncate_to(n);
            }
            if let Some(vs) = v.as_mut() {
                vs.truncate_to(n);
            }
        }
    }
}

fn old_clear_payload(storage: &mut KvStorage) {
    match storage {
        KvStorage::None { .. } => {}
        KvStorage::K8V4 { k, v, .. }
        | KvStorage::K8VTurbo3 { k, v, .. }
        | KvStorage::K8VTurbo3Tcq { k, v, .. }
        | KvStorage::K8VTurbo2 { k, v, .. }
        | KvStorage::K8VTurbo2Tcq { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::K8V8 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::Planar { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::PlanarK { k, .. } => *k = None,
        KvStorage::TurboSym3 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::TurboSym4 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::IsoV3 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::IsoV4 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::IsoSym3 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::IsoSym4 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::IsoKOnly3 { k, .. } => *k = None,
        KvStorage::IsoKOnly4 { k, .. } => *k = None,
        KvStorage::RotorV3 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::RotorV4 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::RotorSym3 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::RotorSym4 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::RotorKOnly3 { k, .. } => *k = None,
        KvStorage::RotorKOnly4 { k, .. } => *k = None,
        KvStorage::RotorKAsym3 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        KvStorage::RotorKAsym4 { k, v, .. } => {
            *k = None;
            *v = None;
        }
        // Payload is not an `Option`; each owns a `reset`.
        KvStorage::Mixed { state, .. } => state.reset(),
        KvStorage::Paged {
            k, v_k8, v_planar, ..
        } => {
            *k = None;
            *v_k8 = None;
            *v_planar = None;
        }
    }
}

/// Positions [`fill`] writes: one chunk and two single tokens.
const FILLED: i32 = CHUNK_SEQ + 2;

/// Negative, zero, inside the chunk block, the chunk block's edge, the edge
/// between the two decode blocks, the full fill and past it.
const TARGETS: [i32; 7] = [
    -3,
    0,
    CHUNK_SEQ / 2,
    CHUNK_SEQ,
    CHUNK_SEQ + 1,
    FILLED,
    FILLED + 9,
];

#[derive(Clone, Copy, Debug)]
enum Op {
    Reset,
    Truncate(i32),
    Clear,
}

fn run_old(storage: &mut KvStorage, op: Op) {
    match op {
        Op::Reset => old_reset(storage),
        Op::Truncate(n) => old_truncate_to(storage, n),
        Op::Clear => old_clear_payload(storage),
    }
}

fn run_new(storage: &mut KvStorage, op: Op) {
    match op {
        Op::Reset => storage.reset(),
        Op::Truncate(n) => storage.truncate_to(n),
        Op::Clear => storage.clear_payload(),
    }
}

fn ops() -> Vec<Op> {
    let mut ops = vec![Op::Reset, Op::Clear];
    ops.extend(TARGETS.into_iter().map(Op::Truncate));
    ops
}

type ProbeBits = Option<std::result::Result<Vec<u32>, String>>;

fn probe_bits(probe: Option<Result<Vec<f32>>>) -> ProbeBits {
    probe.map(|result| {
        result
            .map(|flat| flat.iter().map(|x| x.to_bits()).collect())
            .map_err(|err| err.to_string())
    })
}

/// A GPU capacity no append writes, so a store that still holds it after an
/// operation kept its GPU buffers.
const SENTINEL_CAPACITY: i32 = 7;

/// The GPU bookkeeping of a store that its `reset` clears and its
/// `truncate_to` keeps.
trait GpuBookkeeping {
    fn set_sentinel(&mut self);
    fn read(&self) -> Vec<i32>;
}

macro_rules! flat_gpu_bookkeeping {
    ($($store:ty),* $(,)?) => {$(
        impl GpuBookkeeping for $store {
            fn set_sentinel(&mut self) {
                self.gpu_capacity = SENTINEL_CAPACITY;
            }
            fn read(&self) -> Vec<i32> {
                vec![self.gpu_capacity]
            }
        }
    )*};
}

flat_gpu_bookkeeping!(QuantK, QuantV, QuantPlanarK, QuantPlanarV);

impl<const BITS: u8> GpuBookkeeping for QuantKTurbo<BITS> {
    fn set_sentinel(&mut self) {
        self.gpu_capacity = SENTINEL_CAPACITY;
    }
    fn read(&self) -> Vec<i32> {
        vec![self.gpu_capacity]
    }
}

impl<const BITS: u8> GpuBookkeeping for QuantIsoK<BITS> {
    fn set_sentinel(&mut self) {
        self.gpu.capacity = SENTINEL_CAPACITY;
    }
    fn read(&self) -> Vec<i32> {
        vec![self.gpu.capacity]
    }
}

impl<const BITS: u8> GpuBookkeeping for QuantIsoV<BITS> {
    fn set_sentinel(&mut self) {
        self.gpu.capacity = SENTINEL_CAPACITY;
        self.gpu_capacity = SENTINEL_CAPACITY;
    }
    fn read(&self) -> Vec<i32> {
        vec![self.gpu.capacity, self.gpu_capacity, self.gpu_offset]
    }
}

impl<const BITS: u8> GpuBookkeeping for QuantRotorK<BITS> {
    fn set_sentinel(&mut self) {
        self.gpu.capacity = SENTINEL_CAPACITY;
    }
    fn read(&self) -> Vec<i32> {
        vec![self.gpu.capacity]
    }
}

impl<const BITS: u8> GpuBookkeeping for QuantRotorV<BITS> {
    fn set_sentinel(&mut self) {
        self.gpu.capacity = SENTINEL_CAPACITY;
    }
    fn read(&self) -> Vec<i32> {
        vec![self.gpu.capacity]
    }
}

/// Call `f` on every filled store slot of `storage`, in K, V order.
fn for_each_store(storage: &mut KvStorage, f: &mut dyn FnMut(Option<&mut dyn GpuBookkeeping>)) {
    fn visit<T: GpuBookkeeping>(
        slot: &mut Option<T>,
        f: &mut dyn FnMut(Option<&mut dyn GpuBookkeeping>),
    ) {
        match slot {
            Some(store) => f(Some(store)),
            None => f(None),
        }
    }
    match storage {
        KvStorage::K8V4 { k, v }
        | KvStorage::K8VTurbo3 { k, v }
        | KvStorage::K8VTurbo3Tcq { k, v }
        | KvStorage::K8VTurbo2 { k, v }
        | KvStorage::K8VTurbo2Tcq { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::K8V8 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::Planar { k, v, bits: _ } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::PlanarK { k } => visit(k, f),
        KvStorage::TurboSym3 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::TurboSym4 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::IsoV3 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::IsoV4 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::IsoSym3 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::IsoSym4 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::IsoKOnly3 { k } => visit(k, f),
        KvStorage::IsoKOnly4 { k } => visit(k, f),
        KvStorage::RotorV3 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::RotorV4 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::RotorSym3 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::RotorSym4 { k, v } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::RotorKOnly3 { k } => visit(k, f),
        KvStorage::RotorKOnly4 { k } => visit(k, f),
        KvStorage::RotorKAsym3 {
            k,
            v,
            v_bits: _,
            v_group_size: _,
        } => {
            visit(k, f);
            visit(v, f);
        }
        KvStorage::RotorKAsym4 {
            k,
            v,
            v_bits: _,
            v_group_size: _,
        } => {
            visit(k, f);
            visit(v, f);
        }
        // `Mixed` keeps no GPU bookkeeping beside its payload, and `None` has
        // no store. The paged slots have their own test below.
        KvStorage::None {} | KvStorage::Mixed { .. } | KvStorage::Paged { .. } => {}
    }
}

fn set_sentinels(storage: &mut KvStorage) {
    for_each_store(storage, &mut |store| {
        if let Some(store) = store {
            store.set_sentinel();
        }
    });
}

fn gpu_bookkeeping(storage: &mut KvStorage) -> Vec<Option<Vec<i32>>> {
    let mut out = Vec::new();
    for_each_store(storage, &mut |store| out.push(store.map(|s| s.read())));
    out
}

type State = (u64, ProbeBits, ProbeBits, Vec<Option<Vec<i32>>>);

/// The store digest, the CPU dequant of the K and V slots, and the GPU
/// bookkeeping of every store.
fn state(storage: &mut KvStorage) -> State {
    let [k, v, _] = storage.view().slots;
    let k = probe_bits(k.and_then(|slot| slot.dequant_f32(Device::Cpu)));
    let v = probe_bits(v.and_then(|slot| slot.dequant_f32(Device::Cpu)));
    (store_digest(storage), k, v, gpu_bookkeeping(storage))
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: a clone of a store this test just filled must succeed, and the panic names the codec"
)]
fn view_mut_sites_leave_the_storage_of_the_old_matches_for_every_codec() {
    // The rotor K stores read the QJL switch when they are built.
    let _guard = env_lock();
    let mut refused = 0_usize;
    for &quant in ALL_KV_QUANTS {
        for filled in [false, true] {
            let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
            if filled {
                fill(&mut cache, quant);
            }
            for op in ops() {
                let mut old = cache
                    .storage
                    .try_deep_clone()
                    .expect("clone for the old copy");
                let mut new = cache.storage.try_deep_clone().expect("clone for the view");
                set_sentinels(&mut old);
                set_sentinels(&mut new);
                run_old(&mut old, op);
                run_new(&mut new, op);
                let (old_state, new_state) = (state(&mut old), state(&mut new));
                if matches!(old_state.1, Some(Err(_))) || matches!(old_state.2, Some(Err(_))) {
                    refused += 1;
                }
                assert_eq!(
                    new_state, old_state,
                    "{quant} filled={filled} {op:?}: the view-based fn left another storage"
                );
            }
        }
    }
    assert!(
        refused > 0,
        "no case reached a refused read, so the error text is compared nowhere"
    );
}

const PAGE_TOKENS: i32 = 16;
const PAGE_SLOTS: usize = 4;

/// A paged storage whose three slots hold 1, 2 and 3 pages, each with its last
/// page partly filled.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a CPU page allocation must succeed, and the panic names the slot"
)]
fn paged_filled() -> KvStorage {
    let tokens = |pages: i32| pages * PAGE_TOKENS - 3;
    let mut k = PagedKStorage::new(TEST_MAX_SEQ, PAGE_TOKENS, PAGE_SLOTS);
    let id = k.codes.alloc(Device::Cpu).expect("allocate a K page");
    k.scales
        .alloc(Device::Cpu)
        .expect("allocate a K scale page");
    k.block_table.push(id);
    k.total_tokens = tokens(1);
    k.shape = vec![1, 1, tokens(1), 64];
    let mut v_k8 = PagedVStorage::new(TEST_MAX_SEQ, PAGE_TOKENS, PAGE_SLOTS, 8);
    for _ in 0..2 {
        let id = v_k8.codes.alloc(Device::Cpu).expect("allocate a V page");
        v_k8.scales
            .alloc(Device::Cpu)
            .expect("allocate a V scale page");
        v_k8.block_table.push(id);
    }
    v_k8.total_tokens = tokens(2);
    v_k8.shape = vec![1, 1, tokens(2), 64];
    let mut v_planar = PagedPlanarVStorage::new(TEST_MAX_SEQ, PAGE_TOKENS, PAGE_SLOTS);
    for _ in 0..3 {
        let id = v_planar
            .codes
            .alloc(Device::Cpu)
            .expect("allocate a planar V page");
        v_planar
            .scales
            .alloc(Device::Cpu)
            .expect("allocate a planar V scale page");
        v_planar
            .rotations
            .alloc(Device::Cpu)
            .expect("allocate a planar V rotation page");
        v_planar.block_table.push(id);
    }
    v_planar.total_tokens = tokens(3);
    v_planar.shape = vec![1, 1, tokens(3), 64];
    KvStorage::Paged {
        quant: KvQuant::K8V4,
        k: Some(k),
        v_k8: Some(Box::new(v_k8)),
        v_planar: Some(Box::new(v_planar)),
    }
}

type PagedSlot = Option<(Vec<usize>, i32, Vec<i32>, u64)>;

/// The block table, fill, shape and page bytes of each paged slot. The paged
/// storage has no store-digest cell.
fn paged_state(storage: &KvStorage) -> [PagedSlot; 3] {
    let KvStorage::Paged {
        k, v_k8, v_planar, ..
    } = storage
    else {
        panic!("the paged fixture is not paged");
    };
    [
        k.as_ref().map(|s| {
            (
                s.block_table.clone(),
                s.total_tokens,
                s.shape.clone(),
                s.resident_bytes(),
            )
        }),
        v_k8.as_ref().map(|s| {
            (
                s.block_table.clone(),
                s.total_tokens,
                s.shape.clone(),
                s.resident_bytes(),
            )
        }),
        v_planar.as_ref().map(|s| {
            (
                s.block_table.clone(),
                s.total_tokens,
                s.shape.clone(),
                s.resident_bytes(),
            )
        }),
    ]
}

fn paged_empty() -> KvStorage {
    KvStorage::Paged {
        quant: KvQuant::K8V4,
        k: None,
        v_k8: None,
        v_planar: None,
    }
}

/// The ids of the next two pages each paged slot allocates. `reset` returns
/// every page to the free list and restarts the fresh ids at 0; `truncate_to`
/// returns only the pages past the cut.
#[allow(
    clippy::expect_used,
    reason = "test probe: a CPU page allocation must succeed, and the panic names the slot"
)]
fn next_page_ids(storage: &mut KvStorage) -> [Option<[usize; 2]>; 3] {
    let KvStorage::Paged {
        k, v_k8, v_planar, ..
    } = storage
    else {
        panic!("the paged fixture is not paged");
    };
    macro_rules! two {
        ($slot:expr) => {
            $slot.as_mut().map(|s| {
                [
                    s.codes.alloc(Device::Cpu).expect("allocate a page"),
                    s.codes.alloc(Device::Cpu).expect("allocate a page"),
                ]
            })
        };
    }
    [two!(k), two!(v_k8), two!(v_planar)]
}

#[test]
fn view_mut_sites_leave_the_paged_storage_of_the_old_matches() {
    let page_targets = [-1, 0, 5, PAGE_TOKENS, PAGE_TOKENS + 7, 2 * PAGE_TOKENS];
    let mut paged_ops = vec![Op::Reset, Op::Clear];
    paged_ops.extend(page_targets.into_iter().map(Op::Truncate));
    let builds: [(&str, fn() -> KvStorage); 2] =
        [("empty", paged_empty), ("with pages", paged_filled)];
    for (label, build) in builds {
        for &op in &paged_ops {
            let mut old = build();
            let mut new = build();
            run_old(&mut old, op);
            run_new(&mut new, op);
            assert_eq!(
                paged_state(&new),
                paged_state(&old),
                "paged {label} {op:?}: the view-based fn left another storage"
            );
            assert_eq!(
                next_page_ids(&mut new),
                next_page_ids(&mut old),
                "paged {label} {op:?}: the view-based fn left another page free list"
            );
        }
    }
    let mut cut = paged_filled();
    cut.truncate_to(PAGE_TOKENS);
    let [k, v_k8, v_planar] = paged_state(&cut);
    let pages = |slot: PagedSlot| slot.map(|(table, ..)| table.len());
    assert_eq!(
        [pages(k), pages(v_k8), pages(v_planar)],
        [Some(1), Some(1), Some(1)],
        "a cut to one page keeps one page in every slot"
    );
}
