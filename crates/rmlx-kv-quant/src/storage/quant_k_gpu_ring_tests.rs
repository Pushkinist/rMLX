//! Unit tests for the packed K GPU ring ([`QuantKGpuRing`]).
//!
//! The ring is the thing that makes a packed K store (rotor / iso) readable by
//! a Metal kernel without a host round-trip, so its prefix bookkeeping is
//! correctness-critical: a mis-seeded or mis-strided ring is silently wrong
//! attention, not a slow path.

use super::*;
use crate::test_utils::skip_if_no_gpu_env;

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn u32_arr(vals: &[u32]) -> Array {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[vals.len() as i32], Dtype::U32).expect("u32_arr")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_arr(vals: &[f32]) -> Array {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[vals.len() as i32], Dtype::F32).expect("f32_arr")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
#[allow(
    clippy::unwrap_used,
    reason = "test helper: chunks_exact(4) guarantees length"
)]
fn read_u32(a: &Array) -> Vec<u32> {
    a.eval().expect("eval");
    a.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Read a sideband plane back, at the width the ring stores it.
///
/// Every value asserted against this in the tests below is exactly
/// representable at that width, so the comparisons stay exact equalities and do
/// not need a tolerance that would also swallow a mis-strided read.
use crate::test_utils::read_sideband_plane as read_f32;

/// n_groups 2, so one sequence position with kv_h=2 is 4 code words and 2
/// norms. Small enough to assert element-by-element.
///
/// The ring takes `n_groups` directly — how a codec derives it from `head_dim`
/// (rotor `ceil(D/3)`, iso `D/4`) is the codec's rule, not the ring's, so no
/// head_dim appears here.
const N_GROUPS: i32 = 2;
const KV_H: i32 = 2;
const CODES_PER_STEP: usize = 4; // kv_h * n_groups = 2 * 2
const NORMS_PER_STEP: usize = 2; // kv_h

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn new_ring_is_unallocated_and_views_none() {
    if skip_if_no_gpu_env() {
        return;
    }
    let ring = QuantKGpuRing::default();
    assert!(!ring.is_allocated());
    assert_eq!(ring.byte_size(), 0);
    let view = ring.packed_view(4, Device::Gpu).expect("packed_view");
    assert!(
        view.is_none(),
        "an unallocated ring must report None so the caller falls back to CPU dequant"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn append_then_view_round_trips_payload() {
    if skip_if_no_gpu_env() {
        return;
    }
    let mut ring = QuantKGpuRing::default();
    let codes: Vec<u32> = (0..CODES_PER_STEP as u32).collect();
    let scales: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let norms: Vec<f32> = vec![7.0, 8.0];

    ring.append_encoded(
        &u32_arr(&codes),
        &f32_arr(&scales),
        &f32_arr(&norms),
        KV_H,
        N_GROUPS,
        0,
        1,
        64,
        Device::Gpu,
    )
    .expect("append_encoded");

    let (c, s, n) = ring
        .packed_view(1, Device::Gpu)
        .expect("packed_view")
        .expect("ring is live");
    assert_eq!(read_u32(&c), codes);
    assert_eq!(read_f32(&s), scales);
    assert_eq!(read_f32(&n), norms);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn sequential_appends_land_at_increasing_offsets() {
    if skip_if_no_gpu_env() {
        return;
    }
    let mut ring = QuantKGpuRing::default();
    // Three single-position appends; each step's payload is tagged by value so
    // a mis-strided write shows up as a wrong element, not just a wrong length.
    for step in 0..3_u32 {
        let codes: Vec<u32> = (0..CODES_PER_STEP as u32).map(|i| step * 100 + i).collect();
        let scales: Vec<f32> = (0..CODES_PER_STEP)
            .map(|i| step as f32 + i as f32)
            .collect();
        let norms: Vec<f32> = (0..NORMS_PER_STEP)
            .map(|i| step as f32 * 10.0 + i as f32)
            .collect();
        ring.append_encoded(
            &u32_arr(&codes),
            &f32_arr(&scales),
            &f32_arr(&norms),
            KV_H,
            N_GROUPS,
            step as i32,
            1,
            64,
            Device::Gpu,
        )
        .expect("append_encoded");
    }

    let (c, _s, n) = ring
        .packed_view(3, Device::Gpu)
        .expect("packed_view")
        .expect("ring is live");
    let got_codes = read_u32(&c);
    assert_eq!(got_codes.len(), 3 * CODES_PER_STEP);
    assert_eq!(got_codes[0..4], [0, 1, 2, 3], "step 0 codes");
    assert_eq!(got_codes[4..8], [100, 101, 102, 103], "step 1 codes");
    assert_eq!(got_codes[8..12], [200, 201, 202, 203], "step 2 codes");
    let got_norms = read_f32(&n);
    assert_eq!(got_norms, vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn seed_from_cpu_then_append_preserves_prefix() {
    if skip_if_no_gpu_env() {
        return;
    }
    // This is the live prefill→decode transition: prefill bulk-quantizes K on
    // the CPU, so the first decode step must seed the ring from the
    // accumulated blocks. Appending at prev_seq without seeding would leave
    // [0, prev_seq) as zeros.
    let filled = 2_i32;
    let cpu_codes: Vec<u32> = (0..(filled as usize * CODES_PER_STEP) as u32).collect();
    let cpu_scales: Vec<f32> = (0..filled as usize * CODES_PER_STEP)
        .map(|i| i as f32 * 0.25)
        .collect();
    let cpu_norms: Vec<f32> = (0..filled as usize * NORMS_PER_STEP)
        .map(|i| i as f32 + 0.5)
        .collect();

    let mut ring = QuantKGpuRing::default();
    ring.seed_from_cpu(
        &cpu_codes,
        &cpu_scales,
        &cpu_norms,
        KV_H,
        N_GROUPS,
        filled,
        64,
        Device::Gpu,
    )
    .expect("seed_from_cpu");
    assert!(ring.is_allocated());

    let next_codes: Vec<u32> = vec![900, 901, 902, 903];
    ring.append_encoded(
        &u32_arr(&next_codes),
        &f32_arr(&[9.0, 9.1, 9.2, 9.3]),
        &f32_arr(&[99.0, 99.1]),
        KV_H,
        N_GROUPS,
        filled,
        1,
        64,
        Device::Gpu,
    )
    .expect("append_encoded after seed");

    let (c, _s, n) = ring
        .packed_view(filled + 1, Device::Gpu)
        .expect("packed_view")
        .expect("ring is live");
    let got = read_u32(&c);
    assert_eq!(
        got[0..cpu_codes.len()],
        cpu_codes[..],
        "seeded CPU prefix must survive the append"
    );
    assert_eq!(got[cpu_codes.len()..], next_codes[..], "appended chunk");
    let got_n = read_f32(&n);
    assert_eq!(got_n[0..cpu_norms.len()], cpu_norms[..], "seeded norms");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn seed_is_a_no_op_once_allocated() {
    if skip_if_no_gpu_env() {
        return;
    }
    let mut ring = QuantKGpuRing::default();
    ring.append_encoded(
        &u32_arr(&[1, 2, 3, 4]),
        &f32_arr(&[1.0, 2.0, 3.0, 4.0]),
        &f32_arr(&[5.0, 6.0]),
        KV_H,
        N_GROUPS,
        0,
        1,
        64,
        Device::Gpu,
    )
    .expect("append_encoded");

    // A second seed must not clobber the live ring.
    ring.seed_from_cpu(
        &[0; CODES_PER_STEP],
        &[0.0; CODES_PER_STEP],
        &[0.0; NORMS_PER_STEP],
        KV_H,
        N_GROUPS,
        1,
        64,
        Device::Gpu,
    )
    .expect("seed_from_cpu on a live ring is a no-op");

    let (c, _s, _n) = ring
        .packed_view(1, Device::Gpu)
        .expect("packed_view")
        .expect("ring is live");
    assert_eq!(
        read_u32(&c),
        vec![1, 2, 3, 4],
        "live ring must not be reset"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn growth_across_a_page_boundary_preserves_prefix() {
    if skip_if_no_gpu_env() {
        return;
    }
    // KV_PAGE_SIZE positions fit in the first page; position KV_PAGE_SIZE
    // forces a realloc + prefix copy.
    let mut ring = QuantKGpuRing::default();
    let first = KV_PAGE_SIZE;
    let cpu_codes: Vec<u32> = (0..(first as usize * CODES_PER_STEP) as u32).collect();
    let cpu_scales: Vec<f32> = vec![1.0; first as usize * CODES_PER_STEP];
    let cpu_norms: Vec<f32> = vec![2.0; first as usize * NORMS_PER_STEP];
    ring.seed_from_cpu(
        &cpu_codes,
        &cpu_scales,
        &cpu_norms,
        KV_H,
        N_GROUPS,
        first,
        4096,
        Device::Gpu,
    )
    .expect("seed_from_cpu");
    let cap_before = ring.capacity;

    ring.append_encoded(
        &u32_arr(&[7, 7, 7, 7]),
        &f32_arr(&[1.0, 1.0, 1.0, 1.0]),
        &f32_arr(&[3.0, 3.0]),
        KV_H,
        N_GROUPS,
        first,
        1,
        4096,
        Device::Gpu,
    )
    .expect("append across the page boundary");

    assert!(
        ring.capacity > cap_before,
        "ring must grow past the first page (was {cap_before}, now {})",
        ring.capacity
    );
    let (c, _s, _n) = ring
        .packed_view(first + 1, Device::Gpu)
        .expect("packed_view")
        .expect("ring is live");
    let got = read_u32(&c);
    assert_eq!(
        got[0..cpu_codes.len()],
        cpu_codes[..],
        "prefix must survive the grow-realloc"
    );
    assert_eq!(got[cpu_codes.len()..], [7, 7, 7, 7], "post-grow append");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn clear_drops_the_ring() {
    if skip_if_no_gpu_env() {
        return;
    }
    let mut ring = QuantKGpuRing::default();
    ring.append_encoded(
        &u32_arr(&[1, 2, 3, 4]),
        &f32_arr(&[1.0, 2.0, 3.0, 4.0]),
        &f32_arr(&[5.0, 6.0]),
        KV_H,
        N_GROUPS,
        0,
        1,
        64,
        Device::Gpu,
    )
    .expect("append_encoded");
    assert!(ring.is_allocated());
    ring.clear();
    assert!(!ring.is_allocated());
    assert_eq!(ring.capacity, 0);
    assert!(ring
        .packed_view(1, Device::Gpu)
        .expect("packed_view")
        .is_none());
}

// ── Shape guards ─────────────────────────────────────────────────────────────
//
// These assert that a malformed append / seed is rejected. None of them needs a
// Metal context, so they pass `Device::Cpu` and stay un-ignored — the cheap
// always-on half of the suite.
//
// The device argument is **load-bearing in `chunk_length_mismatch_errors`**,
// not incidental: `prev_seq = 0` on an unallocated ring clears every
// pre-allocation check, so the ring really does allocate (`alloc` -> `zeros`)
// and only then does `write_range` catch the length mismatch. Passing
// `Device::Gpu` there would make that a live Metal allocation and drag the test
// behind `--ignored`. The other three are rejected before the ring allocates at
// all: `prev_seq > 0` on an unallocated ring, `prev_seq + new_seq > max_seq`,
// and the `seed_from_cpu` stride check all return first.

#[test]
fn append_on_unallocated_ring_with_existing_prefix_errors() {
    // The zeroed-prefix footgun: allocating here and writing only
    // [prev_seq, needed) would leave [0, prev_seq) as zeros — silently wrong
    // attention. Must be rejected rather than "helpfully" allocated.
    let mut ring = QuantKGpuRing::default();
    let err = ring.append_encoded(
        &u32_arr(&[1, 2, 3, 4]),
        &f32_arr(&[1.0, 2.0, 3.0, 4.0]),
        &f32_arr(&[5.0, 6.0]),
        KV_H,
        N_GROUPS,
        5, // prev_seq > 0 on an unallocated ring
        1,
        64,
        Device::Cpu,
    );
    assert!(
        err.is_err(),
        "appending at prev_seq>0 on an unallocated ring must error — \
         seed_from_cpu() has to upload the prefix first"
    );
    assert!(
        !ring.is_allocated(),
        "the rejected append must not allocate"
    );
}

#[test]
fn append_beyond_max_seq_errors() {
    let mut ring = QuantKGpuRing::default();
    let err = ring.append_encoded(
        &u32_arr(&[1, 2, 3, 4]),
        &f32_arr(&[1.0, 2.0, 3.0, 4.0]),
        &f32_arr(&[5.0, 6.0]),
        KV_H,
        N_GROUPS,
        8,
        1,
        8, // max_seq — prev_seq + new_seq = 9 > 8
        Device::Cpu,
    );
    assert!(
        err.is_err(),
        "overflowing max_seq must error, not write out of bounds"
    );
}

#[test]
fn chunk_length_mismatch_errors() {
    let mut ring = QuantKGpuRing::default();
    // Two positions' worth of codes declared as new_seq = 1.
    let err = ring.append_encoded(
        &u32_arr(&[1, 2, 3, 4, 5, 6, 7, 8]),
        &f32_arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        &f32_arr(&[5.0, 6.0]),
        KV_H,
        N_GROUPS,
        0,
        1,
        64,
        Device::Cpu,
    );
    assert!(
        err.is_err(),
        "a chunk whose length disagrees with new_seq must error"
    );
}

#[test]
fn seed_length_mismatch_errors() {
    let mut ring = QuantKGpuRing::default();
    let err = ring.seed_from_cpu(
        &[1, 2, 3], // not filled_seq * kv_h * n_groups
        &[1.0, 2.0, 3.0],
        &[1.0, 2.0],
        KV_H,
        N_GROUPS,
        1,
        64,
        Device::Cpu,
    );
    assert!(
        err.is_err(),
        "a CPU prefix that disagrees with the derived stride must error"
    );
}

#[test]
fn page_round_covers_needed_and_caps_at_max() {
    assert_eq!(page_round(1, 4096), KV_PAGE_SIZE);
    assert_eq!(page_round(KV_PAGE_SIZE, 4096), KV_PAGE_SIZE);
    assert_eq!(page_round(KV_PAGE_SIZE + 1, 4096), KV_PAGE_SIZE * 2);
    // Capped at max_seq, but never below what the caller needs.
    assert_eq!(page_round(300, 300), 300);
}

// ── The watermark is a prefix, enforced in both directions ───────────────────

/// Appending past `filled` must be refused, not written and then advertised.
///
/// `filled` is a contiguous prefix. Writing `[prev_seq, needed)` on a ring that
/// only holds `filled` positions leaves `[filled, prev_seq)` as the
/// allocation's zeros, and the `self.filled = needed` at the end of
/// `append_encoded` would then cover that gap — after which `packed_view`
/// cannot tell it from written data and the flash-decode kernel attends a
/// zeroed K/V hole with no error. Enforcing the watermark on reads alone leaves
/// exactly the state this branch exists to remove reachable one append later.
///
/// Runs on `Device::Cpu`: the ring is device-parameterised, and the guard is
/// checked before any allocation, so nothing here selects a Metal stream.
///
/// Mutation check: delete the `prev_seq > self.filled` refusal from
/// `append_encoded` and the `expect_err` below returns `Ok` instead — the ring
/// then reports `filled == 5` while positions 2 and 3 hold zeros, and the
/// `packed_view` assertion that follows stops being reachable at all.
#[test]
fn appending_past_the_fill_watermark_is_refused() {
    let codes: Vec<u32> = (0..CODES_PER_STEP as u32 * 2).collect();
    let scales: Vec<f32> = vec![0.5; CODES_PER_STEP * 2];
    let norms: Vec<f32> = vec![7.0; NORMS_PER_STEP * 2];

    let mut ring = QuantKGpuRing::default();
    ring.append_encoded(
        &u32_arr(&codes),
        &f32_arr(&scales),
        &f32_arr(&norms),
        KV_H,
        N_GROUPS,
        0,
        2,
        64,
        Device::Cpu,
    )
    .expect("seed the first two positions");
    assert_eq!(ring.filled, 2, "two positions written");

    // A stale caller: its blocks reached position 4 while the ring stopped at 2.
    let one_codes: Vec<u32> = (0..CODES_PER_STEP as u32).collect();
    let one_scales: Vec<f32> = vec![1.5; CODES_PER_STEP];
    let one_norms: Vec<f32> = vec![9.0; NORMS_PER_STEP];
    let err = ring
        .append_encoded(
            &u32_arr(&one_codes),
            &f32_arr(&one_scales),
            &f32_arr(&one_norms),
            KV_H,
            N_GROUPS,
            4,
            1,
            64,
            Device::Cpu,
        )
        .expect_err("appending at prev_seq=4 onto a ring filled to 2 must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("prev_seq=4") && msg.contains("filled=2"),
        "the refusal must name both positions, got: {msg}"
    );
    assert_eq!(
        ring.filled, 2,
        "a rejected append commits nothing — the watermark must not move"
    );

    // And the read side still refuses the same gap, so neither direction can
    // hand back the allocation's zeros.
    let err = ring
        .packed_view(4, Device::Cpu)
        .expect_err("reading past filled must be refused too");
    assert!(
        err.to_string().contains("exceeds filled="),
        "expected the read-side watermark guard, got: {err}"
    );

    // The in-step append still works: the guard keys on the gap, not on being
    // strict about equality.
    ring.append_encoded(
        &u32_arr(&one_codes),
        &f32_arr(&one_scales),
        &f32_arr(&one_norms),
        KV_H,
        N_GROUPS,
        2,
        1,
        64,
        Device::Cpu,
    )
    .expect("an append that starts exactly at `filled` is the healthy case");
    assert_eq!(ring.filled, 3, "the healthy append advances the watermark");
}

// ── sideband narrowing ───────────────────────────────────────────────────────

/// Widen a plane back to `f32` for comparison, whatever width it is stored at.
#[allow(clippy::expect_used, reason = "test oracle: invariants documented")]
#[allow(
    clippy::unwrap_used,
    reason = "test oracle: chunks_exact(4) guarantees the slice length"
)]
fn widen_to_f32(a: &Array) -> Vec<f32> {
    let w = a.astype(Dtype::F32, Device::Cpu).expect("widen");
    w.eval().expect("eval");
    w.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The four `f32` bit patterns that decide the rounding rule.
///
/// A tie is an `f32` whose low 16 bits are exactly `0x8000` — halfway between
/// two bf16 values — and round-to-nearest-**even** then keeps whichever
/// neighbour has an even low bit in the surviving half. That is the one input a
/// half-up rounder and an RNE rounder disagree on, and a random `f32` lands on
/// one with probability ~2^-16, so the cases are stated by bit pattern rather
/// than swept for.
const TIE_TRUNCATION_EVEN: u32 = 0x3F80_8000; // 1.00390625 -> 1.0
const TIE_TRUNCATION_ODD: u32 = 0x3F81_8000; // 1.01171875 -> 1.015625
const BELOW_TIE: u32 = 0x3F80_4000; // low 16 bits < 0x8000 -> rounds down
const ABOVE_TIE: u32 = 0x3F80_C000; // low 16 bits > 0x8000 -> rounds up

/// The values the two narrowing paths are compared over.
///
/// Both signs of each tie, because RNE ties-to-even is magnitude-symmetric and
/// a rounder that broke ties away from zero would agree on the positives alone.
/// The remaining entries are the magnitudes a scale or an L2 norm takes, the
/// exponent extremes, and the value whose narrowing overflows to infinity.
fn rounding_cases() -> Vec<f32> {
    let tie_even = f32::from_bits(TIE_TRUNCATION_EVEN);
    let tie_odd = f32::from_bits(TIE_TRUNCATION_ODD);
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        tie_even,
        -tie_even,
        tie_odd,
        -tie_odd,
        f32::from_bits(BELOW_TIE),
        f32::from_bits(ABOVE_TIE),
        core::f32::consts::PI,
        f32::from_bits(0x40C9_1234),
        f32::from_bits(0x3800_1111),
        f32::from_bits(0x449A_4321),
        f32::from_bits(0xBB00_9999),
        1e-30,
        1e30,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]
}

/// [`bf16_round`] agrees with MLX's own narrowing, tie for tie.
///
/// Three places narrow a sideband value: the CPU encoders call `bf16_round`
/// before quantizing against a scale, the upload path narrows through MLX
/// `astype`, and the MSL encoders through `bfloat(x)`. They must agree in the
/// last mantissa bit or a ring seeded from CPU blocks decodes differently from
/// the tail appended into it after — a divergence with no error, on a prefix
/// boundary, which is this crate's documented silent-corruption shape.
///
/// MLX is the oracle here rather than a second hand-rolled rounder: it is the
/// path the shipped upload takes, so an agreement asserted against anything
/// else would not be the agreement that matters.
///
/// `Device::Cpu` on purpose — this is arithmetic, not a Metal dispatch, and an
/// `#[ignore]` here would mean it never ran under any gate.
#[test]
#[allow(clippy::expect_used, reason = "test: invariants documented")]
fn sideband_rounding_matches_mlx() {
    let cases = rounding_cases();
    let src = Array::from_f32_slice(&cases, &[cases.len() as i32]).expect("build the f32 fixture");
    let via_mlx = widen_to_f32(
        &src.astype(KV_SIDEBAND_DTYPE, Device::Cpu)
            .expect("MLX narrowing"),
    );

    assert_eq!(via_mlx.len(), cases.len());
    for (i, (&x, &mlx)) in cases.iter().zip(via_mlx.iter()).enumerate() {
        let ours = bf16_round(x);
        assert_eq!(
            ours.to_bits(),
            mlx.to_bits(),
            "case {i} ({x:e}): bf16_round gave {ours:e} (0x{:08x}), MLX gave {mlx:e} \
             (0x{:08x}) — the CPU encoders and the upload path would store different \
             values for the same input",
            ours.to_bits(),
            mlx.to_bits()
        );
    }

    // The two tie outcomes, stated on their own so the rule is readable without
    // re-deriving it from the sweep: the tie whose truncation is EVEN rounds
    // down, the one whose truncation is ODD rounds up, at both signs.
    let even_down = f32::from_bits(TIE_TRUNCATION_EVEN);
    let odd_up = f32::from_bits(TIE_TRUNCATION_ODD);
    assert_eq!(bf16_round(even_down).to_bits(), 0x3F80_0000);
    assert_eq!(bf16_round(-even_down).to_bits(), 0xBF80_0000);
    assert_eq!(bf16_round(odd_up).to_bits(), 0x3F82_0000);
    assert_eq!(bf16_round(-odd_up).to_bits(), 0xBF82_0000);

    // NaN is compared by class, not by bits: narrowing a NaN keeps a NaN but
    // need not keep the payload, and `bf16_round` deliberately returns the
    // input untouched rather than shifting a payload that could land on an
    // infinity.
    assert!(bf16_round(f32::NAN).is_nan());
    let nan_via_mlx = widen_to_f32(
        &Array::from_f32_slice(&[f32::NAN], &[1])
            .expect("nan fixture")
            .astype(KV_SIDEBAND_DTYPE, Device::Cpu)
            .expect("MLX narrowing"),
    );
    assert!(nan_via_mlx[0].is_nan(), "MLX must keep a NaN a NaN");
}

/// A ring's stored rate is the arithmetic [`ring_bits_per_value`] states, read
/// off a real allocation.
///
/// The formula is the single producer every rate figure in the tree is printed
/// and asserted from, so it needs one place that checks it against bytes rather
/// than against itself.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --ignored --test-threads=1"]
fn ring_bits_per_value_is_what_the_allocation_holds() {
    if skip_if_no_gpu_env() {
        return;
    }
    // iso geometry at head_dim 128: one quaternion group per 4 slots.
    let head_dim: u64 = 128;
    let n_groups: u64 = head_dim / 4;
    let seq: u64 = 8;
    let kv_h: u64 = 1;
    let values = seq * kv_h * head_dim;

    let codes: Vec<u32> = (0..(seq * kv_h * n_groups) as usize)
        .map(|i| i as u32)
        .collect();
    let scales: Vec<f32> = vec![1.0; codes.len()];
    let norms: Vec<f32> = vec![1.0; (seq * kv_h) as usize];

    let mut ring = QuantKGpuRing::default();
    ring.seed_from_cpu(
        &codes,
        &scales,
        &norms,
        kv_h as i32,
        n_groups as i32,
        seq as i32,
        seq as i32,
        Device::Gpu,
    )
    .expect("seed the ring");

    let measured = (ring.byte_size() * 8) as f64 / values as f64;
    let stated = ring_bits_per_value(head_dim, n_groups);
    assert!(
        (measured - stated).abs() < 1e-9,
        "the ring holds {measured} bits per value, ring_bits_per_value states {stated}"
    );
}
