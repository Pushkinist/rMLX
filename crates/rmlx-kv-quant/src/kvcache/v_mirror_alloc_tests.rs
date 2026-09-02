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
//! reports it. [`PeakBracket`] scopes that reading to one region.
//!
//! # Two things every test here holds
//!
//! * **`kv_h > 1`.** At `kv_h == 1` the prefix slice is contiguous and the copy
//!   does not exist, so a fixture at that shape would pass against the defect.
//! * **The kernel actually ran.** Each allocation bound is paired with a
//!   dispatch-count delta or an output comparison, so a bound cannot pass by
//!   measuring a path that never reached the kernel.
//!
//! [`the_slice_the_dispatchers_no_longer_take_costs_a_measurable_prefix_copy`]
//! is what gives the bounds their power: it pays the copy on purpose, in the
//! same process at the same shape, so the oracle is shown to see one rather
//! than assumed to.

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::iso_flash_decode_msl::{iso_flash_decode_dispatch_count, iso_flash_decode_sdpa};
use crate::isoquant::iso_encode_fast;
use crate::kvcache::helpers::slice_v_prefix;
use crate::planar_flash_decode_msl::planar_flash_decode_sdpa;
use crate::planarquant_msl::planar_quantize_v4_gpu;
use crate::quant::KvQuant;
use crate::rotor_flash_decode_msl::{rotor_flash_decode_dispatch_count, rotor_flash_decode_sdpa};
use crate::rotorquant::{n_groups_for, rotor3_encode};
use crate::storage::{KvStorage, QuantRotorK3, ISO_QUAT_BLOCK_SIZE};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{mlx_active_memory_bytes, Array, Device, Dtype, PeakBracket};

/// `b * kv_h > 1` — the shape the copy exists at.
const KV_H: i32 = 8;
const HEAD_DIM: i32 = 128;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;
const B: i32 = 1;

/// Attended length in the direct-dispatch tests, and the mirror's own extent.
/// They differ so the `..kv_seq` cut is a strict, non-contiguous sub-view.
const KV_SEQ: i32 = 1024;
const MIRROR_SEQ: i32 = 2048;

/// Bytes one `..kv_seq` prefix copy of a `[1, KV_H, _, HEAD_DIM]` bf16 mirror
/// costs — the size of the defect, straight from the shape.
const fn prefix_copy_bytes(kv_seq: i32) -> u64 {
    (KV_H as u64) * (kv_seq as u64) * (HEAD_DIM as u64) * 2
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

/// The bf16 V mirror and the `..KV_SEQ` cut of it the dispatchers used to take.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn v_mirror_and_cut() -> (Array, Array) {
    let n = (B * KV_H * MIRROR_SEQ * HEAD_DIM) as usize;
    let mirror = f32_array(&lcg_data(n, 0x5EED), &[B, KV_H, MIRROR_SEQ, HEAD_DIM])
        .astype(Dtype::Bf16, Device::Gpu)
        .expect("V mirror astype bf16");
    mirror.eval().expect("V mirror eval");
    let cut = slice_v_prefix(&mirror, KV_SEQ, Device::Gpu).expect("V mirror prefix");
    (mirror, cut)
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn decode_query(seed: u64) -> Array {
    f32_array(
        &lcg_data((B * N_Q_HEADS * HEAD_DIM) as usize, seed),
        &[B, N_Q_HEADS, 1, HEAD_DIM],
    )
}

/// Sequence-major K for the direct-dispatch tests: token `(s, h)` at row
/// `s * kv_h + h`, which is the layout every packed K store writes.
fn k_seq_major() -> Vec<f32> {
    lcg_data((KV_SEQ * KV_H * HEAD_DIM) as usize, 0xB0B)
}

// ── Per-codec direct dispatch, parameterised only by the V argument ───────────
//
// Each codec's packed K is built once, up front, and evaluated: an allocation
// bracket around a dispatch must not also contain the fixture's own buffers.

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn evaluated(a: Array) -> Array {
    a.eval().expect("fixture eval");
    a
}

/// iso3-packed K: `(codes, scales, norms)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn iso3_packed() -> (Array, Array, Array) {
    let (codes, scales, _quat, norms) =
        iso_encode_fast(&k_seq_major(), HEAD_DIM as usize, ISO_QUAT_BLOCK_SIZE, 3)
            .expect("iso_encode_fast");
    (
        evaluated(u32_array(&codes, &[codes.len() as i32])),
        evaluated(f32_array(&scales, &[scales.len() as i32])),
        evaluated(f32_array(&norms, &[norms.len() as i32])),
    )
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn iso3_dispatch(packed: &(Array, Array, Array), v: &Array) -> Vec<u8> {
    let (codes, scales, norms) = packed;
    let out = iso_flash_decode_sdpa::<3>(
        &decode_query(0xA1),
        codes,
        scales,
        norms,
        v,
        None,
        B,
        KV_H,
        KV_SEQ,
        HEAD_DIM,
        HEADS_PER_KV,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Device::Gpu,
    )
    .expect("iso_flash_decode_sdpa");
    out.eval().expect("iso out eval");
    out.to_bytes().expect("iso out bytes")
}

/// rotor3-packed K: `(codes, scales, norms, rotors)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor3_packed() -> (Array, Array, Array, Array) {
    let rotors = make_rotor_table(0, 0, n_groups_for(HEAD_DIM as usize));
    let (codes, scales, norms) =
        rotor3_encode(&k_seq_major(), &rotors, HEAD_DIM as usize).expect("rotor3_encode");
    (
        evaluated(u32_array(&codes, &[codes.len() as i32])),
        evaluated(f32_array(&scales, &[scales.len() as i32])),
        evaluated(f32_array(&norms, &[norms.len() as i32])),
        evaluated(f32_array(&rotors, &[rotors.len() as i32])),
    )
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor3_dispatch(packed: &(Array, Array, Array, Array), v: &Array) -> Vec<u8> {
    let (codes, scales, norms, rotors) = packed;
    let out = rotor_flash_decode_sdpa::<3>(
        &decode_query(0xA2),
        codes,
        scales,
        norms,
        rotors,
        v,
        None,
        B,
        KV_H,
        KV_SEQ,
        HEAD_DIM,
        HEADS_PER_KV,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Device::Gpu,
    )
    .expect("rotor_flash_decode_sdpa");
    out.eval().expect("rotor out eval");
    out.to_bytes().expect("rotor out bytes")
}

/// planar4-packed K: `(codes, scales, rot32)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn planar4_packed() -> (Array, Array, Array) {
    let k = f32_array(&k_seq_major(), &[B, KV_SEQ, KV_H, HEAD_DIM]);
    let (codes, scales, rot32) =
        planar_quantize_v4_gpu(&k, Device::Gpu).expect("planar_quantize_v4_gpu");
    (evaluated(codes), evaluated(scales), evaluated(rot32))
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn planar4_dispatch(packed: &(Array, Array, Array), v: &Array) -> Vec<u8> {
    let (codes, scales, rot32) = packed;
    let out = planar_flash_decode_sdpa(
        &decode_query(0xA3),
        codes,
        scales,
        rot32,
        v,
        None,
        B,
        KV_H,
        KV_SEQ,
        HEAD_DIM,
        HEADS_PER_KV,
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
/// mirror or the `..kv_seq` cut of it.
///
/// The two arms differ only in V's sequence extent, so the stride the kernel
/// indexes V with differs between them. A dispatcher that dropped the stride,
/// or a kernel body that read it from the wrong `dims` slot, would read V at
/// the wrong offset in the whole-mirror arm and the bytes would diverge —
/// which is the per-codec wiring this covers and the allocation bounds below
/// do not.
#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn every_flash_decode_dispatcher_strides_over_the_whole_v_mirror() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (mirror, cut) = v_mirror_and_cut();
    assert_ne!(
        mirror.shape()[2],
        cut.shape()[2],
        "the two arms must differ in V's sequence extent, or the comparison is trivial"
    );
    let iso = iso3_packed();
    let rotor = rotor3_packed();
    let planar = planar4_packed();
    #[allow(
        clippy::type_complexity,
        reason = "one local table of per-codec dispatch closures; naming the type buys nothing"
    )]
    let codecs: [(&str, Box<dyn Fn(&Array) -> Vec<u8>>); 3] = [
        ("iso3", Box::new(|v| iso3_dispatch(&iso, v))),
        ("rotor3", Box::new(|v| rotor3_dispatch(&rotor, v))),
        ("planar4", Box::new(|v| planar4_dispatch(&planar, v))),
    ];
    for (codec, dispatch) in codecs {
        assert_eq!(
            dispatch(&mirror),
            dispatch(&cut),
            "{codec}: striding over the whole V mirror must be bit-identical to \
             attending its cut prefix"
        );
    }
}

// ── The copy is real, is visible, and the dispatchers no longer pay it ────────

/// Measure the resident bytes an iso dispatch holds when V arrives whole
/// against when it arrives as the `..kv_seq` cut, and hold both halves of the
/// claim: the cut costs one full prefix copy, and the whole mirror costs none
/// of it.
///
/// The first half is the gate's power measurement. Without it, the second half
/// could pass against a fixture that never allocated a copy in either arm.
///
/// Resident bytes rather than a peak bracket, because Metal releases a
/// completed dispatch's buffers on the next dispatch: at steady state the copy
/// is already live when a bracket opens and is replaced in place, so the peak
/// never rises above the open reading and every delta derived from it is zero.
/// What the copy does move is the settled live count, exactly and repeatably.
#[test]
#[ignore = "GPU Metal context — run explicitly"]
#[allow(
    clippy::expect_used,
    reason = "test: every expect is on a fixture built immediately above"
)]
fn the_slice_the_dispatchers_no_longer_take_costs_a_measurable_prefix_copy() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (mirror, cut) = v_mirror_and_cut();
    let iso = iso3_packed();

    // Repeat until the allocator settles: the first dispatch of a kernel
    // compiles it, and each dispatch's buffers are released on the next one.
    let settled_live = |v: &Array| -> u64 {
        for _ in 0..4 {
            let _ = iso3_dispatch(&iso, v);
        }
        let live = mlx_active_memory_bytes()
            .expect("no Metal allocator reading — the measurement below is vacuous");
        assert!(live > 0, "the allocator reports nothing resident at all");
        live
    };

    let whole_before = settled_live(&mirror);
    let sliced = settled_live(&cut);
    let whole_after = settled_live(&mirror);
    assert_eq!(
        whole_before, whole_after,
        "the whole-mirror arm must settle to the same resident bytes either side \
         of the cut arm, or this is measuring drift rather than the copy"
    );

    let copy = prefix_copy_bytes(KV_SEQ);
    let extra = sliced.saturating_sub(whole_before);
    assert!(
        extra >= copy,
        "the cut arm must hold one prefix copy ({copy} B) more than the strided \
         arm; measured {extra} B (whole={whole_before}, sliced={sliced}). A smaller \
         delta means this fixture no longer reproduces the copy, and the bounds \
         below are blind."
    );
}

// ── The production decode step does not pay it either ─────────────────────────

const MAX_SEQ: i32 = 1024;
const PREFILL: i32 = 1016;
const WARMUP_STEPS: u64 = 2;

/// What one bracketed decode step observed.
#[derive(Debug)]
struct StepProbe {
    headroom: u64,
    kv_seq: i32,
}

/// Prefill a cache, warm it, then bracket exactly one steady-state decode step.
///
/// `dispatch_count` is the codec's own kernel counter; a step that did not
/// advance it never reached the kernel, and its allocation reading would
/// describe a CPU-dequant fallback instead.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn bracket_one_decode_step(
    codec: &str,
    mut cache: KvCache,
    dispatch_count: fn() -> u64,
) -> StepProbe {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    let pf = (PREFILL * KV_H * HEAD_DIM) as usize;
    let k = f32_array(&lcg_data(pf, 1), &[B, KV_H, PREFILL, HEAD_DIM]);
    let v = f32_array(&lcg_data(pf, 2), &[B, KV_H, PREFILL, HEAD_DIM]);
    let q = f32_array(
        &lcg_data((PREFILL * N_Q_HEADS * HEAD_DIM) as usize, 3),
        &[B, N_Q_HEADS, PREFILL, HEAD_DIM],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    let step = |cache: &mut KvCache, seed: u64| {
        let one = (KV_H * HEAD_DIM) as usize;
        let k1 = f32_array(&lcg_data(one, 10 + seed), &[B, KV_H, 1, HEAD_DIM]);
        let v1 = f32_array(&lcg_data(one, 20 + seed), &[B, KV_H, 1, HEAD_DIM]);
        let q1 = f32_array(
            &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 30 + seed),
            &[B, N_Q_HEADS, 1, HEAD_DIM],
        );
        let out = cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa");
        out.eval().expect("decode out eval");
    };
    for s in 0..WARMUP_STEPS {
        step(&mut cache, s);
    }

    let before = dispatch_count();
    let bracket = PeakBracket::open();
    step(&mut cache, 99);
    let reading = bracket.close();

    assert!(
        dispatch_count() > before,
        "{codec}: the bracketed step never reached the flash-decode kernel — the \
         reading describes a CPU-dequant fallback, not the dispatch this gate is about"
    );
    assert!(
        reading.measurable(),
        "{codec}: peak mark could not be zeroed — the reading is not scoped to the step"
    );
    assert!(
        reading.observed_allocation(),
        "{codec}: the bracketed decode step allocated nothing: {reading:?}"
    );
    StepProbe {
        headroom: reading.headroom_bytes(),
        kv_seq: cache.offset(),
    }
}

/// One decode step must allocate far less than one V-mirror prefix copy.
fn assert_step_pays_no_prefix_copy(codec: &str, probe: &StepProbe) {
    let copy = prefix_copy_bytes(probe.kv_seq);
    assert!(
        probe.headroom < copy / 2,
        "{codec}: one decode step allocated {} B, on the order of the {copy} B it \
         costs to flatten a non-contiguous `..kv_seq` cut of the bf16 V mirror at \
         kv_h={KV_H} — the mirror is being copied per layer per step again ({probe:?})",
        probe.headroom
    );
}

#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn iso_decode_step_does_not_copy_the_v_mirror() {
    if skip_if_no_gpu_env() {
        return;
    }
    let cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly3, MAX_SEQ);
    let probe = bracket_one_decode_step("iso3", cache, iso_flash_decode_dispatch_count);
    assert_step_pays_no_prefix_copy("iso3", &probe);
}

#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn rotor_decode_step_does_not_copy_the_v_mirror() {
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
            vec![B, KV_H, 0, HEAD_DIM],
            0,
        )),
        max_seq: MAX_SEQ,
    };
    let cache = KvCache::from_storage(
        storage,
        KvQuant::RotorKOnly3,
        0,
        0,
        DispatchPolicy::default(),
        false,
    );
    let probe = bracket_one_decode_step("rotor3", cache, rotor_flash_decode_dispatch_count);
    assert_step_pays_no_prefix_copy("rotor3", &probe);
}
