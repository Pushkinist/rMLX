//! Qwen2 integration tests.

#![allow(clippy::cloned_instead_of_copied)]

use rmlx_mlx::{Device, Dtype};

use super::loader::load_from_path;

/// Integration test: load ReaderLM-v2 snapshot (Qwen2, g64 b4 affine).
///
/// Skips if snapshot absent. Run explicitly:
/// cargo test -p rmlx-models integration_qwen2_readerlm -- --ignored
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn integration_qwen2_readerlm() {
    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_READERLM_V2").map(std::path::PathBuf::from)
    else {
        eprintln!("integration_qwen2_readerlm: skipping: RMLX_TEST_MODEL_READERLM_V2 not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        eprintln!(
            "integration_qwen2_readerlm: snapshot absent at {}, skipping",
            model_dir.display()
        );
        return;
    }

    let model = load_from_path(model_dir).expect("load_from_path failed");

    // Qwen2 BOS = 151643.
    let logits = model
        .forward_seq(&[151643], Device::Gpu)
        .expect("forward_seq failed");
    logits.eval().expect("logits eval");

    let vocab = model.cfg.vocab_size as i32;
    let logits_flat = logits.reshape(&[1, vocab], Device::Gpu).expect("reshape");
    logits_flat.eval().expect("logits_flat eval");

    assert_eq!(
        logits_flat.shape(),
        vec![1, vocab],
        "unexpected logits shape"
    );

    // Cast to f32 for NaN / range checks.
    let lf32 = logits_flat
        .astype(Dtype::F32, Device::Cpu)
        .expect("cast to f32");
    lf32.eval().expect("f32 eval");
    let bytes = lf32.to_bytes().expect("to_bytes");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let nan_count = values.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "NaN in logits: {nan_count}");

    let max_logit = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_logit > 5.0 && max_logit < 100.0,
        "max_logit {max_logit:.2} out of expected range [5, 100]"
    );
    eprintln!("qwen2 ReaderLM-v2 forward probe: max_logit={max_logit:.2}");
}
