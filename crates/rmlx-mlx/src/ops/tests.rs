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
