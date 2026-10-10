use super::*;
use crate::{Array, Device, Dtype};

fn f32_array(vals: &[f32]) -> Array {
    let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[vals.len() as i32], Dtype::F32).expect("from_bytes")
}

fn to_vec(a: &Array) -> Vec<f32> {
    let f = a.astype(Dtype::F32, Device::Cpu).expect("astype");
    f.eval().expect("materialize");
    f.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `sqrt` matches `f32::sqrt` element-wise (CPU stream, no Metal claim).
#[test]
fn sqrt_matches_scalar_reference() {
    let input = [0.0f32, 1e-12, 1.0, 2.0, 4.0, 9.0, 25.0, 1234.5];
    let out = to_vec(&sqrt(&f32_array(&input), Device::Cpu).expect("sqrt"));
    assert_eq!(out.len(), input.len());
    for (o, i) in out.iter().zip(&input) {
        let want = i.sqrt();
        assert!(
            (o - want).abs() <= 1e-6 * want.max(1.0),
            "sqrt({i}) = {o}, want {want}"
        );
    }
}

/// L2-normalize via `sqrt(sum(x^2)+eps)` stays within 1e-6 of the
/// pre-existing `exp(0.5*ln(.))` identity (the path jina-v4 pooling
/// replaces) and equals the scalar reference norm.
#[test]
fn sqrt_l2_norm_equiv_to_exp_ln_identity() {
    let raw = [3.0f32, 4.0, 0.0, 12.0, -5.0, 0.5];
    let x = f32_array(&raw);
    let sq = multiply(&x, &x, Device::Cpu).expect("sq");
    let s = sum_axis(&sq, 0, Device::Cpu).expect("sum"); // scalar
    let s = add(&s, &scalar_f32(1e-12), Device::Cpu).expect("floor");

    let via_sqrt = to_vec(&sqrt(&s, Device::Cpu).expect("sqrt"))[0];
    let half_ln = multiply(
        &log(&s, Device::Cpu).expect("ln"),
        &scalar_f32(0.5),
        Device::Cpu,
    )
    .expect("half_ln");
    let via_exp_ln = to_vec(&exp(&half_ln, Device::Cpu).expect("exp"))[0];

    assert!(
        (via_sqrt - via_exp_ln).abs() < 1e-6,
        "sqrt={via_sqrt} vs exp/ln={via_exp_ln}"
    );
    let want_norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((via_sqrt - want_norm).abs() <= 1e-4 * want_norm);
}

// ── gelu (exact erf) ──────────────────────────────────────────────────────
//
// Reference values from PyTorch `torch.nn.GELU()(torch.tensor([...]))`:
// gelu(0) = 0.0
// gelu(1) ≈ 0.8413447
// gelu(-1) ≈ -0.1586553
// gelu(2) ≈ 1.9544997
//
// These differ from the tanh-approx (`gelu_tanh`) by up to ~0.001 near the
// tails — the test below verifies that divergence is measurable.

/// Exact erf GELU matches PyTorch `nn.GELU()` reference values within 1e-5.
#[test]
fn gelu_exact_matches_pytorch_reference() {
    // PyTorch reference (nn.GELU, approximate='none'):
    // gelu(0)=0, gelu(1)≈0.8413447, gelu(-1)≈-0.1586553, gelu(2)≈1.9544997
    let inputs = [0.0f32, 1.0, -1.0, 2.0];
    let expected = [0.0f32, 0.841_344_7, -0.158_655_3, 1.954_499_7];

    let x = f32_array(&inputs);
    let out = to_vec(&gelu(&x, Device::Cpu).expect("gelu"));

    assert_eq!(out.len(), inputs.len());
    for (i, (got, want)) in out.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "gelu({}): got {got}, want {want} (diff {})",
            inputs[i],
            (got - want).abs()
        );
    }
}

/// Exact GELU differs measurably from `gelu_tanh` at the documented tolerance.
///
/// At x=1 the tanh-approx gives ≈0.8411919 vs exact ≈0.8413447 (Δ≈0.00015).
/// At x=2 the tanh-approx gives ≈1.9544125 vs exact ≈1.9544997 (Δ≈0.00009).
/// Confirms the two ops are distinct and cannot be silently swapped.
#[test]
fn gelu_exact_differs_from_gelu_tanh() {
    let inputs = [1.0f32, -1.0, 2.0];
    let x = f32_array(&inputs);

    let exact_out = to_vec(&gelu(&x, Device::Cpu).expect("gelu"));
    let tanh_out = to_vec(&gelu_tanh(&x, Device::Cpu).expect("gelu_tanh"));

    // They must differ by at least 1e-4 (well beyond float noise) on at least
    // one input — confirming separate code paths.
    let max_diff = exact_out
        .iter()
        .zip(&tanh_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "gelu and gelu_tanh produced identical outputs (max_diff={max_diff}); \
         exact erf path may not be exercised"
    );
}

/// Activations return the dtype they were given.
///
/// `gelu` and `gelu_tanh` are composed from cached **f32** scalar constants, so
/// every step after the first runs at f32 for a bf16 input. Returning that f32
/// promotes whatever the activation feeds — in an FFN that is the `multiply` by
/// `up`, the down-projection's GEMV, the residual add, and the rest of the
/// layer. Nothing errors; the graph just doubles in width.
///
/// `silu`, `tanh`, `softmax_precise` and `pad` are checked alongside them as
/// controls: they route through single MLX ops that preserve dtype already, and
/// naming them here records which ops this was actually verified for rather
/// than implying a blanket property.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture construction on values built in this fn; a panic here is a test bug"
)]
fn activations_return_the_input_dtype() {
    let device = Device::Cpu;
    let x = Array::from_bytes(&[0u8; 16], &[4], Dtype::F32)
        .unwrap()
        .astype(Dtype::Bf16, device)
        .unwrap();
    assert_eq!(x.dtype(), Dtype::Bf16, "fixture must start at bf16");

    for (name, got) in [
        ("gelu_tanh", gelu_tanh(&x, device).unwrap().dtype()),
        ("gelu", gelu(&x, device).unwrap().dtype()),
        ("silu", silu(&x, device).unwrap().dtype()),
        ("tanh", tanh(&x, device).unwrap().dtype()),
        (
            "softmax_precise",
            softmax_precise(&x, -1, device).unwrap().dtype(),
        ),
        ("pad", pad(&x, &[0], &[1], &[1], device).unwrap().dtype()),
    ] {
        assert_eq!(
            got,
            Dtype::Bf16,
            "{name} returned {got:?} for a bf16 input — an activation that widens \
             its output promotes every op it feeds"
        );
    }
}

/// MLX transforms a last axis of `n = m * 2^k`, `m` in {1, 12, 20, 28}, only.
/// It refuses some other sizes only at evaluation (17408) and silently
/// mis-decomposes others (1000 = 20 * 50, 240 = 12 * 20). The wrapper must
/// refuse every size that is not `m * 2^k` before it builds the graph. A
/// refused array is never evaluated here.
#[test]
fn hadamard_transform_refuses_a_size_that_is_not_m_times_a_power_of_two() {
    for n in [17_408, 1000, 240, 5 * 1024 + 1] {
        let r = hadamard_transform(&f32_array(&vec![1.0; n]), None, Device::Cpu);
        assert!(
            matches!(&r, Err(rmlx_core::error::Error::Mlx(m)) if m.contains(&n.to_string())),
            "last axis {n}: want Error::Mlx naming the size, got {:?}",
            r.err()
        );
    }
}

/// The wrapper reads only the shape, so a lazy input that is never evaluated
/// is enough. MLX decomposes `36 * 2^k` as `12 * (3 * 2^k)` and its CPU path
/// then crashes the process; `k > 26` is past MLX's limit.
#[test]
fn hadamard_transform_checks_m_and_k_on_a_lazy_input() {
    let lazy = |n: i32| zeros(&[1, n], Dtype::F32, Device::Cpu).expect("a lazy zeros node");
    for n in [36 * 1024, 44 * 1024, 1 << 27, 12 << 27] {
        assert!(
            hadamard_transform(&lazy(n), None, Device::Cpu).is_err(),
            "last axis {n} must be refused"
        );
    }
    for n in [1 << 26, 28 << 26] {
        assert!(
            hadamard_transform(&lazy(n), None, Device::Cpu).is_ok(),
            "last axis {n} must be accepted"
        );
    }
}

/// At every supported size `H / sqrt(n)` is orthogonal and its `[0][0]` entry
/// is `+1 / sqrt(n)`. Both facts hold for MLX's unnormalised Sloane tables
/// (`m` in {12, 20, 28}) as well as for the Sylvester ones, so the test does
/// not depend on the table layout. Parseval on a fixed pseudo-random vector
/// checks the orthogonality on more than one direction.
fn check_hadamard_norm_and_first_entry(device: Device) {
    for n in [1024, 12 * 512, 20 * 256, 28 * 1024] {
        let ones = hadamard_transform(&f32_array(&vec![1.0; n]), None, device)
            .expect("a supported size builds");
        let sum_sq: f64 = to_vec(&ones).iter().map(|v| f64::from(*v).powi(2)).sum();
        assert!(
            (sum_sq - n as f64).abs() <= 1e-4 * n as f64,
            "last axis {n}: |H 1 / sqrt(n)|^2 is {sum_sq}, want {n}"
        );

        let e0: Vec<f32> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let column =
            hadamard_transform(&f32_array(&e0), None, device).expect("a supported size builds");
        let first = f64::from(to_vec(&column).first().copied().unwrap_or(f32::NAN));
        let want = 1.0 / (n as f64).sqrt();
        assert!(
            (first - want).abs() <= 1e-6 * want,
            "last axis {n}: (H e_0 / sqrt(n))[0] is {first}, want {want}"
        );

        let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ n as u64;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect();
        let x_sq: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
        let hx = hadamard_transform(&f32_array(&x), None, device).expect("a supported size builds");
        let hx_sq: f64 = to_vec(&hx).iter().map(|v| f64::from(*v).powi(2)).sum();
        assert!(
            (hx_sq - x_sq).abs() <= 1e-5 * x_sq,
            "last axis {n}: |H x / sqrt(n)|^2 is {hx_sq}, |x|^2 is {x_sq}"
        );
    }
}

#[test]
fn hadamard_transform_keeps_the_norm_and_maps_e0_to_plus_one_over_sqrt_n() {
    check_hadamard_norm_and_first_entry(Device::Cpu);
}

#[test]
#[ignore = "Metal: GPU twin of hadamard_transform_keeps_the_norm_and_maps_e0_to_plus_one_over_sqrt_n; needs the Metal claim (make gpu-test)"]
fn hadamard_transform_keeps_the_norm_and_maps_e0_to_plus_one_over_sqrt_n_on_gpu() {
    check_hadamard_norm_and_first_entry(Device::Gpu);
}
