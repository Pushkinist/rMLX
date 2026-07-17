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

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
#[allow(
    clippy::unwrap_used,
    reason = "test helper: chunks_exact(4) guarantees length"
)]
fn read_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    a.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k_gpu_ring -- --include-ignored --test-threads=1"]
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
// Every guard below is rejected by a scalar shape check that runs before the
// ring allocates or writes, so these pass `Device::Cpu` and stay un-ignored —
// they are the cheap always-on half of the suite. Handing them `Device::Gpu`
// would claim a Metal context they never reach and force them behind
// `--ignored` with the tests that do.

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
