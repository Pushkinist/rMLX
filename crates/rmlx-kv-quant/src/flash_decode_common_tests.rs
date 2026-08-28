//! Contract tests for [`pad_norms_to_device_floor`].
//!
//! The floor it enforces is currently **inert** — the MLX abort it was measured
//! against no longer reproduces on the pinned build (see the module doc). That
//! is exactly why the function needs a test of its own: with every downstream
//! mutation check documented inert, deleting or stubbing this function failed
//! nothing, and a guard no test can kill is a guard nobody can trust when the
//! MLX bump that revives the abort lands.
//!
//! So these assert the function's own contract — pad below the floor, no-op at
//! and above it, element width preserved either way — and never the MLX
//! heuristic, which is not this crate's property to pin.

use super::{pad_norms_to_device_floor, NORMS_DEVICE_MIN};
use crate::storage::KV_SIDEBAND_DTYPE;
use rmlx_mlx::{Array, Device, Dtype};

/// Build a flat `[n]` array of `1.0, 2.0, …` at `dtype`.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a rejected cast or allocation is the failure this test wants to report by name"
)]
fn ramp(n: usize, dtype: Dtype) -> Array {
    let vals: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let a = Array::from_f32_slice(&vals, &[n as i32]).expect("build the f32 ramp");
    if dtype == Dtype::F32 {
        a
    } else {
        a.astype(dtype, Device::Cpu).expect("cast the ramp")
    }
}

/// Read a flat array back as `f32`, whatever width it is stored at.
#[allow(
    clippy::expect_used,
    reason = "test oracle: an eval or readback failure is the failure this test wants to report by name"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test oracle: chunks_exact guarantees the slice length the conversion needs"
)]
fn read_f32(a: &Array) -> Vec<f32> {
    let widened = if a.dtype() == Dtype::F32 {
        a.try_clone().expect("clone")
    } else {
        a.astype(Dtype::F32, Device::Cpu)
            .expect("widen for readback")
    };
    widened.eval().expect("eval");
    widened
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Below the floor: the result is exactly `NORMS_DEVICE_MIN` long, carries the
/// input as its prefix, and is zero past it.
///
/// The zero tail is the half that matters beyond the length: each kernel's
/// per-tile loop bound is the real `kv_seq` from its `dims` buffer, so the pad
/// is allocated and never read — but a pad that carried garbage would turn any
/// future off-by-one in that bound into a plausible norm rather than a crash.
#[test]
fn a_short_norms_plane_is_zero_padded_up_to_the_floor() {
    let tok_count: i64 = 3;
    let padded =
        pad_norms_to_device_floor(ramp(tok_count as usize, Dtype::F32), tok_count, Device::Cpu)
            .expect("pad below the floor");

    assert_eq!(
        padded.shape(),
        vec![NORMS_DEVICE_MIN as i32],
        "a plane below the floor must come back at exactly the floor"
    );
    let got = read_f32(&padded);
    assert_eq!(
        &got[..tok_count as usize],
        &[1.0_f32, 2.0, 3.0],
        "the real norms must survive the pad unchanged and in order"
    );
    assert!(
        got[tok_count as usize..].iter().all(|&v| v == 0.0),
        "the pad must be zeros, got {:?}",
        &got[tok_count as usize..]
    );
}

/// At and above the floor the function is a no-op — the same length back, the
/// same values, no allocation of a wider buffer.
#[test]
fn a_norms_plane_at_or_above_the_floor_is_returned_unchanged() {
    for tok_count in [NORMS_DEVICE_MIN, NORMS_DEVICE_MIN + 7] {
        let out =
            pad_norms_to_device_floor(ramp(tok_count as usize, Dtype::F32), tok_count, Device::Cpu)
                .expect("no-op at or above the floor");
        assert_eq!(
            out.shape(),
            vec![tok_count as i32],
            "tok_count={tok_count} is at or above the floor and must not be padded"
        );
        let got = read_f32(&out);
        assert_eq!(got.len(), tok_count as usize);
        assert_eq!(
            got[0], 1.0,
            "tok_count={tok_count}: values must be untouched"
        );
    }
}

/// The pad preserves the plane's element width.
///
/// Production hands this function a plane at
/// [`KV_SIDEBAND_DTYPE`] and binds the result to a kernel that declares that
/// dtype; MLX binds by the array's own dtype with no conversion, so a pad that
/// widened would not fail — it would reinterpret the bytes and decode garbage.
/// Asserted at both widths so the test states the property rather than the
/// current constant.
#[test]
fn the_pad_preserves_the_planes_element_width() {
    for dtype in [Dtype::F32, KV_SIDEBAND_DTYPE] {
        let tok_count: i64 = 2;
        let padded =
            pad_norms_to_device_floor(ramp(tok_count as usize, dtype), tok_count, Device::Cpu)
                .expect("pad below the floor");
        assert_eq!(
            padded.dtype(),
            dtype,
            "the pad must not change the plane's stored width"
        );
        assert_eq!(padded.shape(), vec![NORMS_DEVICE_MIN as i32]);
    }
}
