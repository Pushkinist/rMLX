//! A flash-decode step must stride over the bf16 V mirror, not copy its prefix.
//!
//! The mirror is head-major `[b, kv_h, max_seq, head_dim]`. Cutting its valid
//! `..kv_seq` prefix out gives a view that is row-contiguous only when
//! `b * kv_h == 1`; flattening that view for a kernel materialises the whole
//! prefix, once per layer per decode step, with no `contiguous()` call at the
//! site to make it visible. So the dispatchers take the mirror whole and carry
//! its sequence stride in `dims` instead.
//!
//! # Why the oracle is bytes and not tokens per second
//!
//! A regression here is silent: it costs throughput and changes no output bit.
//! Decode rate is the wrong instrument — this host's rate drifts by more than
//! the effect — but the copy's size is exact arithmetic,
//! `kv_h * kv_seq * head_dim * sizeof(bf16)` bytes, and the Metal allocator
//! reports it.
//!
//! # Which allocator reading
//!
//! MLX frees the temporaries of a command buffer in that buffer's completion
//! handler, and an evaluation can return before Metal calls it. So the live
//! count right after an evaluation holds the copy or does not, by timing, and
//! no test here reads it there. Every reading follows
//! `rmlx_mlx::synchronize_gpu`, which waits for the handlers. From that settled
//! state the copy is gone from the live count, so the instrument for the copy
//! is the peak over one dispatch ([`headroom_from_settled`]), and the settled
//! live count is the instrument for what the cache keeps.
//!
//! # What each test covers, and what it does not
//!
//! * [`every_flash_decode_dispatcher_strides_over_the_whole_v_mirror`] — all
//!   three codecs, at `kv_h > 1` and at `kv_h == 1`. Bytes, not allocation: it
//!   is what pins each kernel's `dims` slot, and it fails if any one of them
//!   reads the stride from the wrong place.
//! * [`the_slice_the_dispatchers_no_longer_take_costs_a_measurable_prefix_copy`]
//!   — the allocation oracle's power measurement. It pays a prefix copy on
//!   purpose so the peak is shown to be able to see one.
//! * [`iso_decode_does_not_copy_the_v_mirror`] and its rotor sibling — the
//!   production seam, through `update_and_sdpa`. A copy of one plane of the
//!   packed K prefix per step is below their step bound; a whole K prefix and
//!   a V prefix are not.
//!
//! **Planar has no production-seam probe.** Its fused arm is warm-TTFT
//! quiescent: after `exit_prefill` the bf16 K seed is live and the dispatcher
//! bypasses it by design (`warm_ttft_tests`), and its flash arm is additionally
//! off by default (`DispatchPolicy::planar_flash_decode`). A probe would have
//! to construct a configuration production does not reach. Its kernel indexing
//! is covered by the equivalence test above; what is untested is whether the
//! saving is realised in a served request, and on the default policy it is not.
//!
//! # Two shapes, for opposite reasons
//!
//! `kv_h > 1` is the only shape the copy exists at, so the allocation bounds
//! run there. `kv_h == 1` is the shape where the `(b * kv_h + kv_h_idx)` term
//! vanishes and the stride cancels out of the address arithmetic entirely — a
//! stride bug and a correct stride are indistinguishable there, which is why it
//! is an equivalence arm proving the change is inert and not an allocation arm.
//! It is also the production shape of Gemma4's global layers.

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::flash_decode_common::VMirror;
use crate::iso_flash_decode_msl::{iso_flash_decode_dispatch_count, iso_flash_decode_sdpa};
use crate::isoquant::iso_encode_fast;
use crate::kvcache::helpers::slice_v_prefix;
use crate::planar_flash_decode_msl::planar_flash_decode_sdpa;
use crate::planarquant_msl::planar_quantize_v4_gpu;
use crate::quant::KvQuant;
use crate::rotor_flash_decode_msl::{rotor_flash_decode_dispatch_count, rotor_flash_decode_sdpa};
use crate::rotorquant::{n_groups_for, rotor3_encode};
use crate::storage::{
    iso_n_groups_for, KvStorage, QuantRotorK3, ISO_QUAT_BLOCK_SIZE, KV_PAGE_SIZE,
};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{mlx_active_memory_bytes, synchronize_gpu, Array, Device, Dtype, PeakBracket};

/// One direct-dispatch fixture shape.
#[derive(Debug, Clone, Copy)]
struct Shape {
    kv_h: i32,
    heads_per_kv: i32,
    /// Attended length.
    kv_seq: i32,
    /// The mirror allocation's own sequence extent, `> kv_seq` so the
    /// `..kv_seq` cut is a strict, non-contiguous sub-view.
    mirror_seq: i32,
}

const B: i32 = 1;
const HEAD_DIM: i32 = 128;

/// The shape the copy exists at (`b * kv_h > 1`), and the one it does not.
const MULTI_HEAD: Shape = Shape {
    kv_h: 8,
    heads_per_kv: 4,
    kv_seq: 1024,
    mirror_seq: 2048,
};
const SINGLE_HEAD: Shape = Shape {
    kv_h: 1,
    heads_per_kv: 4,
    kv_seq: 1024,
    mirror_seq: 2048,
};

impl Shape {
    const fn n_q_heads(self) -> i32 {
        self.kv_h * self.heads_per_kv
    }

    /// Bytes one `..kv_seq` prefix copy of this shape's bf16 mirror costs.
    const fn prefix_copy_bytes(self, tokens: i32) -> u64 {
        (self.kv_h as u64) * (tokens as u64) * (HEAD_DIM as u64) * 2
    }
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn u32_array(data: &[u32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::U32).expect("u32_array")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn evaluated(a: Array) -> Array {
    a.eval().expect("fixture eval");
    a
}

/// The bf16 V mirror and the `..kv_seq` cut of it the dispatchers used to take.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn v_mirror_and_cut(shape: Shape) -> (Array, Array) {
    let n = (B * shape.kv_h * shape.mirror_seq * HEAD_DIM) as usize;
    let mirror = f32_array(
        &lcg_data(n, 0x5EED),
        &[B, shape.kv_h, shape.mirror_seq, HEAD_DIM],
    )
    .astype(Dtype::Bf16, Device::Gpu)
    .expect("V mirror astype bf16");
    let mirror = evaluated(mirror);
    let cut =
        evaluated(slice_v_prefix(&mirror, shape.kv_seq, Device::Gpu).expect("V mirror prefix"));
    (mirror, cut)
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn decode_query(shape: Shape, seed: u64) -> Array {
    f32_array(
        &lcg_data((B * shape.n_q_heads() * HEAD_DIM) as usize, seed),
        &[B, shape.n_q_heads(), 1, HEAD_DIM],
    )
}

/// Sequence-major K: token `(s, h)` at row `s * kv_h + h`, the layout every
/// packed K store writes.
fn k_seq_major(shape: Shape) -> Vec<f32> {
    lcg_data((shape.kv_seq * shape.kv_h * HEAD_DIM) as usize, 0xB0B)
}

// ── Per-codec direct dispatch, parameterised only by the V argument ───────────
//
// Each codec's packed K is built once, up front, and evaluated: an allocation
// measurement around a dispatch must not also contain the fixture's own buffers.

/// iso3-packed K: `(codes, scales, norms)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn iso3_packed(shape: Shape) -> (Array, Array, Array) {
    let (codes, scales, _quat, norms) = iso_encode_fast(
        &k_seq_major(shape),
        HEAD_DIM as usize,
        ISO_QUAT_BLOCK_SIZE,
        3,
    )
    .expect("iso_encode_fast");
    (
        evaluated(u32_array(&codes, &[codes.len() as i32])),
        evaluated(f32_array(&scales, &[scales.len() as i32])),
        evaluated(f32_array(&norms, &[norms.len() as i32])),
    )
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn iso3_dispatch(shape: Shape, packed: &(Array, Array, Array), v: &Array) -> Vec<u8> {
    let (codes, scales, norms) = packed;
    let out = iso_flash_decode_sdpa::<3>(
        &decode_query(shape, 0xA1),
        codes,
        scales,
        norms,
        VMirror::new(v, shape.kv_seq),
        None,
        B,
        shape.kv_h,
        shape.kv_seq,
        HEAD_DIM,
        shape.heads_per_kv,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Device::Gpu,
    )
    .expect("iso_flash_decode_sdpa");
    out.eval().expect("iso out eval");
    out.to_bytes().expect("iso out bytes")
}

/// rotor3-packed K: `(codes, scales, norms, rotors)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor3_packed(shape: Shape) -> (Array, Array, Array, Array) {
    let rotors = make_rotor_table(0, 0, n_groups_for(HEAD_DIM as usize));
    let (codes, scales, norms) =
        rotor3_encode(&k_seq_major(shape), &rotors, HEAD_DIM as usize).expect("rotor3_encode");
    (
        evaluated(u32_array(&codes, &[codes.len() as i32])),
        evaluated(f32_array(&scales, &[scales.len() as i32])),
        evaluated(f32_array(&norms, &[norms.len() as i32])),
        evaluated(f32_array(&rotors, &[rotors.len() as i32])),
    )
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor3_dispatch(shape: Shape, packed: &(Array, Array, Array, Array), v: &Array) -> Vec<u8> {
    let (codes, scales, norms, rotors) = packed;
    let out = rotor_flash_decode_sdpa::<3>(
        &decode_query(shape, 0xA2),
        codes,
        scales,
        norms,
        rotors,
        VMirror::new(v, shape.kv_seq),
        None,
        B,
        shape.kv_h,
        shape.kv_seq,
        HEAD_DIM,
        shape.heads_per_kv,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Device::Gpu,
    )
    .expect("rotor_flash_decode_sdpa");
    out.eval().expect("rotor out eval");
    out.to_bytes().expect("rotor out bytes")
}

/// planar4-packed K: `(codes, scales, rot32)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn planar4_packed(shape: Shape) -> (Array, Array, Array) {
    let k = f32_array(
        &k_seq_major(shape),
        &[B, shape.kv_seq, shape.kv_h, HEAD_DIM],
    );
    let (codes, scales, rot32) =
        planar_quantize_v4_gpu(&k, Device::Gpu).expect("planar_quantize_v4_gpu");
    (evaluated(codes), evaluated(scales), evaluated(rot32))
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn planar4_dispatch(shape: Shape, packed: &(Array, Array, Array), v: &Array) -> Vec<u8> {
    let (codes, scales, rot32) = packed;
    let out = planar_flash_decode_sdpa(
        &decode_query(shape, 0xA3),
        codes,
        scales,
        rot32,
        VMirror::new(v, shape.kv_seq),
        None,
        B,
        shape.kv_h,
        shape.kv_seq,
        HEAD_DIM,
        shape.heads_per_kv,
        4,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Device::Gpu,
    )
    .expect("planar_flash_decode_sdpa");
    out.eval().expect("planar out eval");
    out.to_bytes().expect("planar out bytes")
}

// ── The stride reaches every dispatcher's kernel, and changes no bit ──────────

/// Each dispatcher must return the same bytes whether it is handed the whole
/// mirror or the `..kv_seq` cut of it, at both a multi-head and a single-head
/// shape.
///
/// The two arms differ only in V's sequence extent, so the stride the kernel
/// indexes V with differs between them. A dispatcher that dropped the stride,
/// or a kernel body that read it from the wrong `dims` slot, reads V at the
/// wrong offset in the whole-mirror arm and the bytes diverge — which is the
/// per-codec wiring this covers and the allocation bounds below do not.
#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn every_flash_decode_dispatcher_strides_over_the_whole_v_mirror() {
    if skip_if_no_gpu_env() {
        return;
    }
    for shape in [MULTI_HEAD, SINGLE_HEAD] {
        let (mirror, cut) = v_mirror_and_cut(shape);
        assert_ne!(
            mirror.shape()[2],
            cut.shape()[2],
            "the two arms must differ in V's sequence extent, or the comparison is trivial"
        );
        let iso = iso3_packed(shape);
        let rotor = rotor3_packed(shape);
        let planar = planar4_packed(shape);
        #[allow(
            clippy::type_complexity,
            reason = "one local table of per-codec dispatch closures; naming the type buys nothing"
        )]
        let codecs: [(&str, Box<dyn Fn(&Array) -> Vec<u8>>); 3] = [
            ("iso3", Box::new(|v| iso3_dispatch(shape, &iso, v))),
            ("rotor3", Box::new(|v| rotor3_dispatch(shape, &rotor, v))),
            ("planar4", Box::new(|v| planar4_dispatch(shape, &planar, v))),
        ];
        for (codec, dispatch) in codecs {
            assert_eq!(
                dispatch(&mirror),
                dispatch(&cut),
                "{codec} at kv_h={}: striding over the whole V mirror must be bit-identical \
                 to attending its cut prefix",
                shape.kv_h
            );
        }
    }
}

// ── The copy is real, is visible, and the dispatchers no longer pay it ────────

/// Bytes `region` needs on top of the settled live count.
///
/// An evaluation returns before Metal calls the completion handlers that free
/// its temporaries, so the live count right after one has two states and which
/// one a reading sees is a race. `synchronize_gpu` waits for the handlers, so
/// the bracket opens from the one state that does not depend on that timing,
/// and the peak inside it is then a property of `region` alone.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn headroom_from_settled(region: impl FnOnce()) -> u64 {
    synchronize_gpu().expect("synchronize before the bracket opens");
    let bracket = PeakBracket::open();
    region();
    let reading = bracket.close();
    assert!(
        reading.measurable(),
        "the peak mark was not reset, so nothing below is scoped to the region ({reading:?})"
    );
    reading.headroom_bytes()
}

/// Measure the bytes one iso dispatch needs when V arrives whole against when
/// it arrives as the `..kv_seq` cut, and hold both halves of the claim: the cut
/// costs one full prefix copy, and the whole mirror costs none of it.
///
/// The first half is the gate's power measurement. Without it, the bounds of
/// the production-seam tests below could pass against a fixture that never
/// allocated a copy in either arm.
///
/// A peak bracket from a settled state, not the live count after the dispatch:
/// MLX frees the copy in a completion handler, so the live count holds it or
/// does not, by timing. The peak holds it in both cases.
#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn the_slice_the_dispatchers_no_longer_take_costs_a_measurable_prefix_copy() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = MULTI_HEAD;
    let (mirror, cut) = v_mirror_and_cut(shape);
    let iso = iso3_packed(shape);

    // The first dispatch of a kernel compiles it; keep that out of the brackets.
    let _ = iso3_dispatch(shape, &iso, &mirror);
    let headroom = |v: &Array| {
        headroom_from_settled(|| {
            let _ = iso3_dispatch(shape, &iso, v);
        })
    };

    let whole_before = headroom(&mirror);
    let sliced = headroom(&cut);
    let whole_after = headroom(&mirror);
    eprintln!(
        "iso3: readings whole_before={whole_before} sliced={sliced} whole_after={whole_after}"
    );
    assert_eq!(
        whole_before, whole_after,
        "the whole-mirror arm must need the same bytes either side of the cut arm, \
         or this is measuring drift rather than the copy"
    );

    let copy = shape.prefix_copy_bytes(shape.kv_seq);
    assert!(
        (1..copy).contains(&whole_before),
        "the strided arm must allocate, and must need less than one prefix copy \
         ({copy} B); measured {whole_before} B"
    );
    let extra = sliced.saturating_sub(whole_before);
    assert!(
        (copy..2 * copy).contains(&extra),
        "the cut arm must need exactly one prefix copy ({copy} B) more than the strided \
         arm; measured {extra} B (whole={whole_before}, sliced={sliced}). A smaller delta \
         means this fixture no longer reproduces the copy, and the bounds below are blind."
    );
}

// ── The production decode loop does not pay it either ────────────────────────

const MAX_SEQ: i32 = 4096;
const PREFILL: i32 = 512;
const SETTLE_STEPS: u64 = 256;
const MEASURED_STEPS: u64 = 1024;

/// MLX's Metal allocator rounds a buffer larger than one VM page up to whole
/// VM pages (`mlx/backend/metal/allocator.cpp`), and the page is 16 KiB on
/// Apple Silicon. So the allocator's count for a buffer is less than one page
/// above the buffer's own size.
const ALLOCATOR_PAGE: u64 = 16 * 1024;

/// The buffers a packed K ring holds: codes, scales, norms.
const RING_PLANES: u64 = 3;

/// Bytes the packed K store of `quant` holds for `positions` positions of
/// [`MULTI_HEAD`], from the codec's own byte model: its per-layer estimate
/// less the bf16 V buffer a K-only codec keeps, which is half of what `None`
/// holds.
fn packed_k_bytes(quant: KvQuant, positions: i32) -> u64 {
    assert!(
        !quant.feeds_bf16_k_at_decode(false),
        "{quant}: the estimate also holds a bf16 K seed, so this is not the packed store alone"
    );
    let per_layer = |q: KvQuant| {
        q.estimated_resident_bytes_per_layer(
            positions as u64,
            HEAD_DIM as u64,
            MULTI_HEAD.kv_h as u64,
            false,
        )
    };
    per_layer(quant) - per_layer(KvQuant::None) / 2
}

/// The capacity the ring policy gives `positions`: whole `KV_PAGE_SIZE` pages.
/// Stated here on purpose, and not read from the ring, so that a ring that
/// grows by more than the policy fails.
fn paged(positions: i32) -> i32 {
    (positions + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE * KV_PAGE_SIZE
}

/// Drive `update_and_sdpa` on `cache` and hold what each decode step needed
/// and what stayed allocated.
///
/// Every reading is taken from a settled state (see [`headroom_from_settled`]),
/// so none depends on when Metal calls a completion handler.
///
/// **Each step.** The step's headroom, less the new ring of a step that regrows
/// the store, is what the step allocated and freed again. It must be less than
/// one packed K prefix of the attended length. A re-materialised V prefix is at
/// least `kv_h * kv_seq * head_dim * 2` bytes of exactly that (the mirror of
/// this fixture is f32, twice that), and the packed K prefix is smaller than
/// the bf16 V prefix, so the bound fails a V prefix copy and a K prefix copy
/// both. Only a peak sees either: the copy is freed
/// when the step is done.
///
/// **The whole run.** After every step the store holds its layer-static tables
/// plus whole pages of positions, by the codec's own byte model. Over the run
/// the settled live count grew by what the store grew by, to within the
/// allocator's rounding of the ring's planes. Every buffer sized at `max_seq` is allocated before the run
/// opens and cancels out.
///
/// `dispatch_count` is the codec's own kernel counter: a run that did not
/// advance it once per step never reached the kernel, and its readings would
/// describe a CPU-dequant fallback instead.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn assert_decode_allocates_no_prefix_copy(
    quant: KvQuant,
    mut cache: KvCache,
    dispatch_count: fn() -> u64,
) {
    let device = Device::Gpu;
    let shape = MULTI_HEAD;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let static_bytes = cache.storage.resident_bytes();

    let pf = (PREFILL * shape.kv_h * HEAD_DIM) as usize;
    let k = f32_array(&lcg_data(pf, 1), &[B, shape.kv_h, PREFILL, HEAD_DIM]);
    let v = f32_array(&lcg_data(pf, 2), &[B, shape.kv_h, PREFILL, HEAD_DIM]);
    let q = f32_array(
        &lcg_data((PREFILL * shape.n_q_heads() * HEAD_DIM) as usize, 3),
        &[B, shape.n_q_heads(), PREFILL, HEAD_DIM],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    let step = |cache: &mut KvCache, seed: u64| {
        let one = (shape.kv_h * HEAD_DIM) as usize;
        let k1 = f32_array(&lcg_data(one, 10 + seed), &[B, shape.kv_h, 1, HEAD_DIM]);
        let v1 = f32_array(&lcg_data(one, 20 + seed), &[B, shape.kv_h, 1, HEAD_DIM]);
        let q1 = f32_array(
            &lcg_data((shape.n_q_heads() * HEAD_DIM) as usize, 30 + seed),
            &[B, shape.n_q_heads(), 1, HEAD_DIM],
        );
        let out = cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa");
        out.eval().expect("decode out eval");
    };
    let settled_live = || {
        synchronize_gpu().expect("synchronize before a live reading");
        mlx_active_memory_bytes()
            .expect("no Metal allocator reading — the bounds below would be vacuous")
    };
    let store_holds_whole_pages = |cache: &KvCache| {
        let (held, positions) = (cache.storage.resident_bytes(), cache.offset());
        let modelled = static_bytes + packed_k_bytes(quant, paged(positions));
        assert_eq!(
            held,
            modelled,
            "{quant}: the store holds {held} B at {positions} positions; its byte model gives \
             {modelled} B for {} positions in whole pages plus {static_bytes} B of static tables",
            paged(positions)
        );
        held
    };

    for s in 0..SETTLE_STEPS {
        step(&mut cache, s);
    }
    let live_before = settled_live();
    let store_before = store_holds_whole_pages(&cache);
    let offset_before = cache.offset();
    let dispatches_before = dispatch_count();

    let mut most_per_position = 0;
    for s in SETTLE_STEPS..SETTLE_STEPS + MEASURED_STEPS {
        let store_at_open = cache.storage.resident_bytes();
        let headroom = headroom_from_settled(|| step(&mut cache, s));
        let kv_seq = cache.offset();
        // A step that regrows the ring allocates the whole new ring while the
        // old one is still live.
        let store_now = store_holds_whole_pages(&cache);
        let new_ring = if store_now == store_at_open {
            0
        } else {
            store_now - static_bytes
        };
        let transient = headroom.saturating_sub(new_ring);
        let k_prefix = packed_k_bytes(quant, kv_seq);
        assert!(
            transient < k_prefix,
            "{quant}: the decode step at kv_seq={kv_seq} allocated and freed {transient} B. \
             One packed K prefix is {k_prefix} B and a bf16 V prefix, the narrowest a V \
             mirror gets, is {} B: the step materialised a prefix (headroom {headroom} B)",
            shape.prefix_copy_bytes(kv_seq)
        );
        most_per_position = most_per_position.max(transient / kv_seq as u64);
    }
    let live_after = settled_live();
    let store_after = store_holds_whole_pages(&cache);
    let grown = live_after.saturating_sub(live_before);
    eprintln!(
        "{quant}: readings grown={grown} store_grown={} most_transient_per_position=\
         {most_per_position}",
        store_after - store_before
    );

    assert_eq!(
        cache.offset() - offset_before,
        MEASURED_STEPS as i32,
        "{quant}: the cache did not advance one position per decode step"
    );
    assert!(
        dispatch_count() - dispatches_before >= MEASURED_STEPS,
        "{quant}: the measured steps did not each reach the flash-decode kernel — \
         the readings describe a CPU-dequant fallback, not the dispatch this gate is about"
    );
    assert!(
        most_per_position > 0,
        "{quant}: no decode step allocated anything; the step bound passed by measuring nothing"
    );
    assert!(
        store_after > store_before,
        "{quant}: the store did not grow over {MEASURED_STEPS} steps; the bound below would \
         pass by measuring nothing"
    );
    assert!(
        grown.abs_diff(store_after - store_before) < RING_PLANES * ALLOCATOR_PAGE,
        "{quant}: the settled live count grew by {grown} B over {MEASURED_STEPS} steps and the \
         store by {} B. They must agree to within the allocator's rounding of the ring's \
         {RING_PLANES} planes: the decode loop keeps a buffer that the store does not account for",
        store_after - store_before
    );
}

#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn iso_decode_does_not_copy_the_v_mirror() {
    if skip_if_no_gpu_env() {
        return;
    }
    let cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly3, MAX_SEQ);
    assert_decode_allocates_no_prefix_copy(
        KvQuant::IsoKOnly3,
        cache,
        iso_flash_decode_dispatch_count,
    );
}

#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn rotor_decode_does_not_copy_the_v_mirror() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Seeding the rotor table pins the store's QJL decision off without touching
    // the process-global toggle: both append paths only set `qjl_s_matrix` in
    // their `rotors.is_empty()` lazy-init branch. A QJL-carrying store keeps the
    // CPU dequant path and would never reach the kernel.
    let rotors = make_rotor_table(0, 0, n_groups_for(HEAD_DIM as usize));
    let storage = KvStorage::RotorKOnly3 {
        k: Some(QuantRotorK3::from_cpu_blocks(
            rotors,
            None,
            Vec::new(),
            vec![B, MULTI_HEAD.kv_h, 0, HEAD_DIM],
            0,
        )),
    };
    let cache = KvCache::from_storage(
        storage,
        MAX_SEQ,
        KvQuant::RotorKOnly3,
        0,
        0,
        DispatchPolicy::default(),
        false,
    );
    assert_decode_allocates_no_prefix_copy(
        KvQuant::RotorKOnly3,
        cache,
        rotor_flash_decode_dispatch_count,
    );
}

// ── The mirror's valid length is checked in every profile ─────────────────────

/// A mirror whose valid length disagrees with the attended `kv_seq` must be a
/// hard error, in every build profile.
///
/// This is the invariant the pre-stride code enforced implicitly: it flattened
/// V to `b * kv_h * kv_seq * head_dim` elements and MLX's `reshape` rejects an
/// element-count mismatch. Striding over the allocation decouples the two, so
/// the check is now explicit — and it has to be an `Err`, not a `debug_assert`,
/// because the profile the engine ships under compiles assertions out and
/// executes no GPU test. The failure it prevents is a decode step attending the
/// mirror's tail (zeros, or a previous longer sequence's V) and returning
/// plausible-but-wrong output with no error at all.
///
/// `Device::Cpu` on purpose: the check runs before any Metal dispatch, so this
/// is not a GPU test and must not be `#[ignore]`d — `#[ignore]` is what would
/// keep it out of the profile it exists to cover.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: every expect is on a fixture built immediately above"
)]
fn a_v_mirror_shorter_than_the_attended_prefix_is_rejected() {
    let shape = MULTI_HEAD;
    let kv_seq = 8_i32;
    let n_groups = iso_n_groups_for(HEAD_DIM as usize) as i32;

    // Correctly sized K planes: the V check sits behind the K flattens, so
    // undersized dummies would fail earlier and this would pass for the wrong
    // reason.
    let tok = B * kv_seq * shape.kv_h;
    let code_words = crate::storage::iso_row_words(HEAD_DIM as usize, 3) as i32;
    let codes = u32_array(
        &vec![0_u32; (tok * code_words) as usize],
        &[tok * code_words],
    );
    let scales = f32_array(&vec![0.0_f32; (tok * n_groups) as usize], &[tok * n_groups]);
    let norms = f32_array(&vec![0.0_f32; tok as usize], &[tok]);
    let q = decode_query(shape, 0xD1);
    let mirror = f32_array(
        &vec![0.0_f32; (B * shape.kv_h * 16 * HEAD_DIM) as usize],
        &[B, shape.kv_h, 16, HEAD_DIM],
    );

    let err = iso_flash_decode_sdpa::<3>(
        &q,
        &codes,
        &scales,
        &norms,
        VMirror::new(&mirror, kv_seq - 1),
        None,
        B,
        shape.kv_h,
        kv_seq,
        HEAD_DIM,
        shape.heads_per_kv,
        1.0,
        Device::Cpu,
    )
    .expect_err("a mirror shorter than the attended prefix must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("out of step"),
        "expected the store/mirror desync error, got: {msg}"
    );
}

/// `slice_v_prefix` must cut the right rows at a head-major shape whose
/// `head_dim` is not a power of two.
///
/// The flash dispatchers reject non-power-of-two `head_dim` outright (their
/// tree reduction needs one), so that shape cannot be an arm of the equivalence
/// test above. It does reach `slice_v_prefix`: planar's non-flash fused-QK
/// fallback calls it, and that arm is exactly where a non-power-of-two
/// `head_dim` lands.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: every expect is on a fixture built immediately above"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test: chunks_exact(4) guarantees the 4-byte length"
)]
fn slice_v_prefix_cuts_the_valid_rows_at_a_non_pow2_head_dim() {
    let (kv_h, head_dim, mirror_seq, valid) = (3_i32, 96_i32, 8_i32, 5_i32);
    let n = (B * kv_h * mirror_seq * head_dim) as usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mirror = f32_array(&data, &[B, kv_h, mirror_seq, head_dim]);

    let cut = slice_v_prefix(&mirror, valid, Device::Cpu).expect("slice_v_prefix");
    assert_eq!(cut.shape(), vec![B, kv_h, valid, head_dim]);
    let got: Vec<f32> = cut
        .to_bytes()
        .expect("cut bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Head-major: head h's valid rows start at h * mirror_seq * head_dim, which
    // is what makes the cut non-contiguous once kv_h > 1.
    let want: Vec<f32> = (0..kv_h)
        .flat_map(|h| {
            let base = (h * mirror_seq * head_dim) as usize;
            (0..(valid * head_dim) as usize).map(move |i| (base + i) as f32)
        })
        .collect();
    assert_eq!(
        got, want,
        "the cut must hold each head's first {valid} rows"
    );
}
