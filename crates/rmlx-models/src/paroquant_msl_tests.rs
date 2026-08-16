use super::*;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
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
    let a_f32 = a.astype(Dtype::F32, Device::Cpu).expect("astype f32");
    a_f32.eval().expect("array materialise");
    a_f32
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Kernel registration: MLX accepts the source and hands back a handle.
///
/// This does NOT prove the MSL compiles — MLX compiles lazily on first
/// `apply()`, so a syntax error surfaces there, not here. The two round-trip
/// tests below are what force the compile (and one instantiation each of the
/// `ROWS_PER_TILE` template int); `make check-metal-compiles` is what catches a
/// syntax error without a GPU.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test paroquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn paro_kernel_registration() {
    paro_rotate_kernel().expect("paro rotate kernel should register");
}

/// The compile probe's `MAX_KROT` / `MAX_GROUP_SIZE` must equal the Rust consts
/// the dispatch passes as template ints.
///
/// Those manifest values are hand-written, so they are a second source of the
/// same numbers — the drift the `probes/*.hdr.metal` snapshots are pinned
/// against, one field over. Production numerics cannot go wrong (the dispatch
/// reads the consts directly), but the probe silently stops representing the
/// shape production requests: raising `MAX_GROUP_SIZE` in Rust would leave the
/// gate compiling a threadgroup allocation nothing ever asks for, and the real
/// one would first be seen on a GPU dispatch. Equality here turns that into a
/// hard failure.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[test]
fn probe_manifest_defines_match_rust_consts() {
    const MANIFEST: &str = include_str!("metal/probes/kernels.manifest");
    const BODY: &str = "paroquant_rotate.metal";

    let line = MANIFEST
        .lines()
        .find(|l| !l.trim_start().starts_with('#') && l.trim_start().starts_with(BODY))
        .expect("manifest must carry a paroquant_rotate.metal line");
    let defines = line
        .split('|')
        .nth(3)
        .expect("paroquant_rotate.metal line must carry a defines field");

    let lookup = |name: &str| -> Option<usize> {
        defines.split(',').find_map(|d| {
            let (k, v) = d.split_once('=')?;
            (k.trim() == name).then(|| v.trim().parse::<usize>().ok())?
        })
    };

    assert_eq!(
        lookup("MAX_KROT"),
        Some(MAX_KROT),
        "stale probe: metal/probes/kernels.manifest sets MAX_KROT to something other than \
         paroquant_msl.rs's MAX_KROT={MAX_KROT}"
    );
    assert_eq!(
        lookup("MAX_GROUP_SIZE"),
        Some(MAX_GROUP_SIZE),
        "stale probe: metal/probes/kernels.manifest sets MAX_GROUP_SIZE to something other \
         than paroquant_msl.rs's MAX_GROUP_SIZE={MAX_GROUP_SIZE}"
    );
}

/// Round-trip identity at `batch = 1`, which selects `ROWS_PER_TILE = 1` — the
/// decode-step template instantiation.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test paroquant_msl -- --ignored --test-threads=1"]
fn paro_rotate_identity_roundtrip_rpt1() {
    paro_rotate_identity_roundtrip(1);
}

/// Round-trip identity at `batch = 2`, which selects `ROWS_PER_TILE = 4` — the
/// prefill / batch template instantiation.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test paroquant_msl -- --ignored --test-threads=1"]
fn paro_rotate_identity_roundtrip_rpt4() {
    paro_rotate_identity_roundtrip(2);
}

/// Round-trip identity test: zero rotation angles (cos=1.0, sin=0.0) and
/// channel_scales=1.0 must produce output equal to input.
///
/// hidden=4, group_size=4, krot=2.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn paro_rotate_identity_roundtrip(batch: usize) {
    let hidden: usize = 4;
    let group_size: usize = 4;
    let krot: usize = 2;
    let half_hidden = hidden / 2;

    let x_data: Vec<f32> = (0..batch * hidden).map(|i| i as f32 + 1.0).collect();
    let x = make_f32_array(&x_data, &[batch as i32, hidden as i32]);

    // packed_pairs: [krot=2, half_hidden=2] I32.
    // Use pairs (i=0,j=1) and (i=2,j=3) — with angle=0 these are identity.
    let mut pairs_data = vec![0i32; krot * half_hidden];
    for k in 0..krot {
        for t in 0..half_hidden {
            let i_local = (2 * t) as u32;
            let j_local = (2 * t + 1) as u32;
            pairs_data[k * half_hidden + t] = (i_local | (j_local << 16)) as i32;
        }
    }
    let pairs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(pairs_data.as_ptr().cast::<u8>(), pairs_data.len() * 4)
    };
    let packed_pairs =
        Array::from_bytes(pairs_bytes, &[krot as i32, half_hidden as i32], Dtype::I32)
            .expect("packed_pairs");

    // cos=1.0, sin=0.0 everywhere → identity rotation.
    let cos_data = vec![1.0_f32; krot * half_hidden];
    let sin_data = vec![0.0_f32; krot * half_hidden];
    let cos_theta = make_f32_array(&cos_data, &[krot as i32, half_hidden as i32])
        .astype(Dtype::F16, Device::Gpu)
        .expect("cos f16");
    let sin_theta = make_f32_array(&sin_data, &[krot as i32, half_hidden as i32])
        .astype(Dtype::F16, Device::Gpu)
        .expect("sin f16");

    // channel_scales=1.0 → no rescaling.
    let cs_data = vec![1.0_f32; hidden];
    let channel_scales = make_f32_array(&cs_data, &[hidden as i32])
        .astype(Dtype::F16, Device::Gpu)
        .expect("channel_scales f16");

    let x_bf16 = x.astype(Dtype::Bf16, Device::Gpu).expect("x bf16");

    let out = paro_rotate_gpu(
        &x_bf16,
        &packed_pairs,
        &cos_theta,
        &sin_theta,
        &channel_scales,
        krot,
        group_size,
        Device::Gpu,
    )
    .expect("paro_rotate_gpu identity");

    let out_vals = to_f32_vec(&out);
    assert_eq!(out_vals.len(), batch * hidden);

    // With identity rotation and unit channel scales, output must equal input.
    for (i, (&got, &expected)) in out_vals.iter().zip(x_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 0.02,
            "identity: element {i}: expected {expected:.4}, got {got:.4}"
        );
    }
}
