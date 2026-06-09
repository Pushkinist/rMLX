//! Dtype-lock regression tests for the Gemma4 activation stream.
//!
//! Background: Gemma4's `gelu_tanh` uses process-global f32 arithmetic
//! constants, so a bf16 gate fed through the fused GeGLU / PLI-GeGLU
//! activations is silently promoted to f32. On the mxfp8 path that promoted
//! the whole residual stream — and through Q/K/V projections, the global
//! (full-attention) `--kv-quant none` KV cache — to f32, roughly doubling KV
//! residency vs the bf16 expectation. The same widening came from the
//! embed-scale and per-layer-input scale constants when wrapped as strong-F32
//! scalars.
//!
//! The fix keeps the stream at the model dtype: the fused activation closures
//! restore the gate dtype on their output, and the scale constants adopt the
//! operand dtype before multiplying. These tests pin that invariant — a future
//! re-promotion of the Gemma4 stream to f32 makes them RED.
//!
//! All tests are CPU-constructible (no model, no GPU) and run as part of CI.

use rmlx_mlx::{multiply, scalar_f32, Array, Device, Dtype};

use super::kernels::{geglu_fused, pli_gelu_fused};

/// Build a small CPU array of the given dtype from f32 source values.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture builder; an Array construction failure is the test failing"
)]
fn cpu_array(values: &[f32], shape: &[i32], dtype: Dtype) -> Array {
    let f32_arr = Array::from_f32_slice(values, shape).unwrap();
    if dtype == Dtype::F32 {
        f32_arr
    } else {
        f32_arr.astype(dtype, Device::Cpu).unwrap()
    }
}

/// A bf16 gate fed through `geglu_fused` must yield a bf16 output. If the
/// dtype-restoring cast is removed, `gelu_tanh`'s f32 constants promote the
/// output to f32 and this assertion fails — locking the global-KV-stays-bf16
/// fix.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; a fused-closure error is the test failing"
)]
fn geglu_fused_bf16_gate_stays_bf16() {
    let gate = cpu_array(&[0.5, -1.0, 2.0, 0.25], &[1, 1, 4], Dtype::Bf16);
    let up = cpu_array(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 4], Dtype::Bf16);

    let out = geglu_fused(&gate, &up, Device::Cpu).unwrap();
    out.eval().unwrap();

    assert_eq!(
        out.dtype(),
        Dtype::Bf16,
        "geglu_fused must keep a bf16 gate at bf16 (f32 re-promotion would \
         widen the residual stream and the global KV cache)"
    );
}

/// An f32 gate fed through `geglu_fused` must stay f32 — the dtype-restoring
/// cast is a no-op when the gate is already f32, so the f32 path is unchanged.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; a fused-closure error is the test failing"
)]
fn geglu_fused_f32_gate_stays_f32() {
    let gate = cpu_array(&[0.5, -1.0, 2.0, 0.25], &[1, 1, 4], Dtype::F32);
    let up = cpu_array(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 4], Dtype::F32);

    let out = geglu_fused(&gate, &up, Device::Cpu).unwrap();
    out.eval().unwrap();

    assert_eq!(
        out.dtype(),
        Dtype::F32,
        "geglu_fused must leave f32 unchanged"
    );
}

/// A bf16 gate fed through `pli_gelu_fused` must yield a bf16 output — same
/// invariant as `geglu_fused`, applied to the per-layer-input gating path.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; a fused-closure error is the test failing"
)]
fn pli_gelu_fused_bf16_gate_stays_bf16() {
    let gate = cpu_array(&[0.5, -1.0, 2.0, 0.25], &[1, 1, 4], Dtype::Bf16);
    let per_layer = cpu_array(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 4], Dtype::Bf16);

    let out = pli_gelu_fused(&gate, &per_layer, Device::Cpu).unwrap();
    out.eval().unwrap();

    assert_eq!(
        out.dtype(),
        Dtype::Bf16,
        "pli_gelu_fused must keep a bf16 gate at bf16 (f32 re-promotion would \
         widen the residual stream and the global KV cache)"
    );
}

/// An f32 gate fed through `pli_gelu_fused` must stay f32 — f32 path unchanged.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; a fused-closure error is the test failing"
)]
fn pli_gelu_fused_f32_gate_stays_f32() {
    let gate = cpu_array(&[0.5, -1.0, 2.0, 0.25], &[1, 1, 4], Dtype::F32);
    let per_layer = cpu_array(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 4], Dtype::F32);

    let out = pli_gelu_fused(&gate, &per_layer, Device::Cpu).unwrap();
    out.eval().unwrap();

    assert_eq!(
        out.dtype(),
        Dtype::F32,
        "pli_gelu_fused must leave f32 unchanged"
    );
}

/// The scale-constant sites (embed-scale, per-layer-input scales) multiply a
/// bf16 operand by a scalar built from `scalar_f32`. A strong-F32 scalar
/// promotes the product to f32; adopting the operand dtype (the fix) keeps it
/// bf16. This pins the `astype(operand.dtype(), ...)` discipline at those
/// sites: a bf16 operand × dtype-adopted scale → bf16.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; an op error is the test failing"
)]
fn dtype_adopted_scale_keeps_bf16_operand_bf16() {
    let operand = cpu_array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], Dtype::Bf16);

    // Strong-F32 scalar (the pre-fix shape) would promote the product to f32.
    let strong = scalar_f32(2.0);
    let promoted = multiply(&operand, &strong, Device::Cpu).unwrap();
    promoted.eval().unwrap();
    assert_eq!(
        promoted.dtype(),
        Dtype::F32,
        "sanity: a strong-F32 scalar promotes a bf16 operand to f32 (this is \
         the bug the scale-site fix avoids)"
    );

    // Fix shape: the scale adopts the operand dtype before multiplying.
    let adopted = scalar_f32(2.0)
        .astype(operand.dtype(), Device::Cpu)
        .unwrap();
    let kept = multiply(&operand, &adopted, Device::Cpu).unwrap();
    kept.eval().unwrap();
    assert_eq!(
        kept.dtype(),
        Dtype::Bf16,
        "a dtype-adopted scale must keep a bf16 operand at bf16"
    );
}
