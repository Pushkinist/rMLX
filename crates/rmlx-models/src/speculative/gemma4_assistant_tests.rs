use super::*;

// ---------------------------------------------------------------------------
// Mask-mode selection (issue #24: "additive" is NOT a valid mlx-c mode).
// ---------------------------------------------------------------------------

/// The mlx-c Metal SDPA kernel accepts only these mask_mode strings. A drafter
/// layer must never emit anything else (the old "additive" string aborted every
/// decode step).
const KERNEL_ACCEPTED_MODES: [&str; 3] = ["causal", "array", ""];

#[test]
fn windowed_layer_uses_array_mode() {
    // A present window-bias mask routes through "array" (NOT "additive").
    let bias = make_bias();
    let (mode, arr) = swa_sdpa_mode(Some(&bias));
    assert_eq!(mode, "array");
    assert!(arr.is_some());
    assert!(KERNEL_ACCEPTED_MODES.contains(&mode));
}

#[test]
fn global_or_covered_layer_uses_empty_mode() {
    // Full-attention layers (and sliding layers whose window covers all KV)
    // pass no mask: empty mode, no array.
    let (mode, arr) = swa_sdpa_mode(None);
    assert_eq!(mode, "");
    assert!(arr.is_none());
    assert!(KERNEL_ACCEPTED_MODES.contains(&mode));
}

#[test]
fn no_additive_mode_ever_emitted() {
    // Regression guard for issue #24: neither branch may select "additive".
    let bias = make_bias();
    for m in [swa_sdpa_mode(Some(&bias)).0, swa_sdpa_mode(None).0] {
        assert_ne!(
            m, "additive",
            "issue #24: mlx-c rejects mask_mode 'additive'"
        );
        assert!(KERNEL_ACCEPTED_MODES.contains(&m));
    }
}

// ---------------------------------------------------------------------------
// Window-bias values — pure mirror of build_swa_mask's banded loop.
// ---------------------------------------------------------------------------

/// Pure mirror of the additive bidirectional-window bias built in
/// `build_swa_mask`. Locks the value convention: 0.0 in-window, -1e30 out of
/// window (NOT f32::NEG_INFINITY — a fully-masked softmax row over -inf is NaN,
/// matching the verifier `crate::layers::build_swa_*` builders).
#[allow(
    clippy::indexing_slicing,
    reason = "bias sized query_len*kv_len; (qi,ki) loop bounds keep the index in range"
)]
fn window_bias(q_off: i32, query_len: i32, kv_len: i32, window: i32) -> Vec<f32> {
    let mut bias = vec![0.0f32; (query_len * kv_len) as usize];
    for qi in 0..query_len {
        let q = q_off + qi;
        for ki in 0..kv_len {
            let dist = q - ki;
            if !(dist > -window && dist < window) {
                bias[(qi * kv_len + ki) as usize] = -1e30;
            }
        }
    }
    bias
}

#[allow(
    clippy::expect_used,
    reason = "test fixture: tiny constant-shape array, failure is a test bug"
)]
fn make_bias() -> Array {
    let data = window_bias(0, 1, 4, 2);
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, &[1, 1, 1, 4], Dtype::F32).expect("bias array")
}

// Exact bit-pattern compare avoids the float_cmp lint while still asserting
// the precise constants the kernel sees (these cells are assigned literals).
fn allowed(v: f32) -> bool {
    v.to_bits() == 0.0f32.to_bits()
}
fn masked(v: f32) -> bool {
    v.to_bits() == (-1e30f32).to_bits()
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "fixed-shape bias [1,1,1,6]; indices are in-bounds by construction"
)]
fn bias_masks_outside_window_with_finite_penalty() {
    // window=2, query at offset 5, kv_len 6: keys within (q-window, q+window)
    // exclusive are allowed (0.0), the rest get -1e30.
    let bias = window_bias(5, 1, 6, 2);
    // q=5: dist = 5-ki. allowed when -2 < dist < 2 => ki in {4,5} (dist 1,0).
    assert!(allowed(bias[4]));
    assert!(allowed(bias[5]));
    // ki=3 -> dist 2 (not < window) => masked. ki=0 far past => masked.
    assert!(masked(bias[3]));
    assert!(masked(bias[0]));
    // No -inf / NaN anywhere (NaN-avoidance invariant vs f32::NEG_INFINITY).
    assert!(bias.iter().all(|v| v.is_finite()));
}

#[test]
fn bias_all_allowed_when_window_covers_kv() {
    // window large enough to cover every key => all-zero bias.
    let bias = window_bias(0, 1, 4, 64);
    assert!(bias.iter().all(|v| allowed(*v)));
}
