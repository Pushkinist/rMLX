//! Tests for `QuantV` — focused on the codebook override surface.

use super::QuantV;

// ── value_codebook field ──────────────────────────────────────────────────────

/// `QuantV::from_cpu_blocks` must initialise `value_codebook` to `None`.
#[test]
fn from_cpu_blocks_value_codebook_is_none() {
    let qv = QuantV::from_cpu_blocks(Vec::new(), vec![1, 1, 0, 32], 4);
    assert!(
        qv.value_codebook.is_none(),
        "from_cpu_blocks must start with value_codebook = None"
    );
}

/// Constructing `QuantV` with `value_codebook = Some(...)` stores the
/// codebook on the struct — the field is accessible after construction.
#[test]
fn value_codebook_stored_after_construction() {
    let codebook = vec![
        -2.717_667_f32,
        -2.052_138,
        -1.600_802_4,
        -1.239_959,
        -0.928_244_7,
        -0.645_875_33,
        -0.381_178_23,
        -0.126_046_94,
        0.126_046_94,
        0.381_178_23,
        0.645_875_33,
        0.928_244_7,
        1.239_959,
        1.600_802_4,
        2.052_138,
        2.717_667,
    ];
    let qv = QuantV {
        blocks: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: vec![1, 1, 0, 32],
        bits: 4,
        max_seq: 0,
        high_precision_indices: None,
        value_codebook: Some(codebook.clone()),
        value_codebook_gpu: None,
        use_tcq: false,
    };
    assert_eq!(
        qv.value_codebook.as_deref(),
        Some(codebook.as_slice()),
        "value_codebook must be stored exactly as provided"
    );
}

/// `try_deep_clone` must propagate `value_codebook`.
#[test]
fn try_deep_clone_propagates_value_codebook() {
    let cb = vec![-1.5_f32, -0.5, 0.5, 1.5];
    let qv = QuantV {
        blocks: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: vec![1, 1, 0, 32],
        bits: 2,
        max_seq: 0,
        high_precision_indices: None,
        value_codebook: Some(cb.clone()),
        value_codebook_gpu: None,
        use_tcq: false,
    };
    let cloned = qv.try_deep_clone().expect("try_deep_clone must succeed");
    assert_eq!(
        cloned.value_codebook.as_deref(),
        Some(cb.as_slice()),
        "try_deep_clone must propagate value_codebook"
    );
}
