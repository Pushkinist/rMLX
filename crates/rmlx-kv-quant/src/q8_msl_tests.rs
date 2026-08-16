use super::*;
use rmlx_mlx::{Array, Device, Dtype, PeakBracket};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32);
            frac * 2.0 - 1.0
        })
        .collect()
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test q8_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn q8_msl_roundtrip_within_tolerance() {
    let shape = [1i32, 4, 128, 128];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales) = q8_quantize_gpu(&arr, Device::Gpu).expect("GPU quantize");
    let recon =
        q8_dequantize_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu).expect("GPU dequant");

    let recon_vec = to_f32_vec(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.01,
        "GPU q8 roundtrip max abs error {max_err:.6} exceeds 0.01"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test q8_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn q8_msl_matches_cpu_within_eps() {
    let shape = [1i32, 4, 128, 128];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xCAFE_BABE_u64);

    let mut cpu_recon = vec![0.0_f32; n];
    for g in 0..(n / 128) {
        let start = g * 128;
        let group = &data[start..start + 128];
        let abs_max = group
            .iter()
            .cloned()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (i, &v) in group.iter().enumerate() {
            let c = (v * inv).round().clamp(-128.0, 127.0) as i32;
            cpu_recon[start + i] = scale * (c as f32);
        }
    }

    let arr = make_f32_array(&data, &shape);
    let (codes, scales) = q8_quantize_gpu(&arr, Device::Gpu).expect("GPU quantize");
    let recon =
        q8_dequantize_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu).expect("GPU dequant");
    let gpu_recon = to_f32_vec(&recon);

    let max_diff = cpu_recon
        .iter()
        .zip(gpu_recon.iter())
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1.0e-4,
        "CPU vs GPU max abs diff {max_diff:.6} exceeds 1e-4"
    );
}

/// Allocation gate for the q8 MSL round trip.
///
/// Numerics tests cannot see a change that leaves every output bit identical
/// while allocating an extra scratch buffer per dispatch. This one can: it
/// brackets the Metal allocator across quantize + dequantize and bounds the
/// bytes the region needed on top of what was already live.
///
/// The bound is a multiple of the *input* size, never an absolute byte count —
/// MLX pools its buffers, so an absolute figure would encode whatever ran
/// before this test rather than what this region did. The outputs are
/// materialised inside the bracket because MLX is lazy: without the `eval`
/// the allocation would happen after `close()` and the gate would measure
/// nothing.
///
/// Budget: codes are 1/4 of the input (i8 vs f32) and scales 1/128 of it, and
/// the dequantized reconstruction is a full input-sized f32 buffer. That is
/// ~1.26x live at the peak. 4x leaves room for one full-size scratch buffer
/// on top and still fails a per-element temporary or a doubled reconstruction.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test q8_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn q8_msl_roundtrip_allocation_stays_within_budget() {
    let shape = [1i32, 4, 128, 128];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let input_bytes = (n * 4) as u64;
    let data = lcg_data(n, 0x5EED_1234_u64);
    let arr = make_f32_array(&data, &shape);
    // Materialise the input before the bracket opens: its bytes are part of
    // "already live", not part of what the round trip costs.
    arr.eval().expect("eval input");

    let bracket = PeakBracket::open();
    let (codes, scales) = q8_quantize_gpu(&arr, Device::Gpu).expect("GPU quantize");
    let recon =
        q8_dequantize_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu).expect("GPU dequant");
    recon.eval().expect("eval recon");
    let reading = bracket.close();

    // Anti-vacuous: an upper bound is free to hold against a region that never
    // allocated. Prove the bracket actually saw the round trip first.
    assert!(
        reading.observed_allocation(),
        "peak bracket recorded no allocation across a q8 round trip — the bracket \
         is not measuring the region ({reading:?})"
    );

    // Every byte the region peaked at is still live at close: the round trip
    // allocates its codes, scales and reconstruction and holds all three.
    //
    // What this can see: a scratch buffer LARGER than the round trip's own
    // steady state, allocated and released inside the region. That is the
    // regression worth catching, and no numerics test can see it — the output
    // bits do not change.
    //
    // What it cannot see: a transient smaller than the steady-state peak. It
    // hides under the peak the surviving buffers reach anyway, and a
    // peak-based measure has no way to distinguish it. `headroom_bytes` above
    // does not see it either. Neither bound is a general no-scratch proof.
    assert_eq!(
        reading.transient_bytes(),
        0,
        "q8 round trip allocated {} bytes it then released before the bracket closed — \
         a transient larger than the round trip's own working set appeared in a path \
         that had none ({reading:?})",
        reading.transient_bytes(),
    );

    let budget = 4 * input_bytes;
    assert!(
        reading.headroom_bytes() <= budget,
        "q8 round trip needed {} bytes over the {} already live, budget is {budget} \
         ({:.2}x the {input_bytes}-byte input) — {reading:?}",
        reading.headroom_bytes(),
        reading.live_at_open_bytes,
        reading.headroom_bytes() as f64 / input_bytes as f64,
    );
}
