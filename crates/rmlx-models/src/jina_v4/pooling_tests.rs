use super::*;
use crate::jina_v4::{load_from_path, JinaV4, JinaV4Task};

const HIDDEN: usize = 2048;
const PROJ_DIM: usize = 128;
const MATRYOSHKA: [usize; 5] = [128, 256, 512, 1024, 2048];

fn jina_v4_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from)
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn load() -> Option<(JinaV4, Device)> {
    let Some(dir_buf) = jina_v4_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_JINA_V4 not set");
        return None;
    };
    let dir = dir_buf.as_path();
    if !dir.join("adapters/adapter_model.safetensors").exists() {
        eprintln!("SKIP: jina-v4 snapshot absent at {}", dir.display());
        return None;
    }
    let m = load_from_path(dir).expect("load jina-v4");
    // CPU keeps the test free of the single-MLX-process Metal claim.
    Some((m, Device::Cpu))
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// A short non-padded token sequence (well within vocab 151936).
fn ids() -> Vec<i64> {
    vec![9707, 25, 419, 374, 264, 1273, 11652, 13]
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn unit_norm_helper_is_exact() {
    // Pure-math sanity for l2_normalize_last (no model needed): a known
    // vector normalizes to unit length within tolerance.
    let dev = Device::Cpu;
    let raw: Vec<f32> = vec![3.0, 4.0, 0.0, 12.0, -5.0];
    let bytes = raw
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<u8>>();
    let a = Array::from_bytes(&bytes, &[1, raw.len() as i32], Dtype::F32).unwrap();
    let n = l2_normalize_last(&a, dev).unwrap();
    let out = to_f32_vec(&n).unwrap();
    assert!((l2(&out) - 1.0).abs() < 1e-4, "‖·‖₂={}", l2(&out));
    // Direction preserved: out == raw / ‖raw‖.
    let norm = l2(&raw);
    for (o, r) in out.iter().zip(&raw) {
        assert!((o - r / norm).abs() < 1e-4);
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn validate_truncate_dim_accepts_and_rejects() {
    for &d in &MATRYOSHKA {
        assert!(validate_truncate_dim(d, &MATRYOSHKA).is_ok(), "{d} valid");
    }
    // Not in the matryoshka set -> clear error.
    for bad in [64usize, 100, 384, 768, 1536, 2049, 0] {
        let e = validate_truncate_dim(bad, &MATRYOSHKA).unwrap_err();
        let msg = format!("{e}");
        assert!(
            msg.contains("invalid truncate_dim") && msg.contains(&bad.to_string()),
            "bad dim {bad} must yield a clear error, got: {msg}"
        );
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn single_vector_full_is_unit_norm_finite_deterministic() {
    let Some((m, dev)) = load() else { return };
    let v1 = m.embed_single(&ids(), dev, None).expect("embed_single");
    assert_eq!(v1.len(), HIDDEN, "full single-vec length == hidden");
    assert!(v1.iter().all(|x| x.is_finite()), "all finite");
    assert!((l2(&v1) - 1.0).abs() < 1e-4, "‖v‖₂ ≈ 1, got {}", l2(&v1));
    // Deterministic: same input + same task -> bit-identical.
    let v2 = m.embed_single(&ids(), dev, None).expect("embed_single 2");
    assert_eq!(v1, v2, "single-vector must be deterministic");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn single_vector_matryoshka_all_dims_renormed() {
    let Some((m, dev)) = load() else { return };
    for &d in &MATRYOSHKA {
        let v = m
            .embed_single(&ids(), dev, Some(d))
            .unwrap_or_else(|e| panic!("embed_single dim {d}: {e}"));
        assert_eq!(v.len(), d, "matryoshka length == {d}");
        assert!(v.iter().all(|x| x.is_finite()), "dim {d} finite");
        assert!(
            (l2(&v) - 1.0).abs() < 1e-4,
            "dim {d} re-normed: ‖·‖₂ ≈ 1, got {}",
            l2(&v)
        );
    }
    // Invalid dim rejected with a clear error (no panic, no garbage).
    let e = m.embed_single(&ids(), dev, Some(384)).unwrap_err();
    assert!(
        format!("{e}").contains("invalid truncate_dim"),
        "invalid matryoshka dim must be rejected, got: {e}"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn multi_vector_shape_and_row_unit_norm() {
    let Some((m, dev)) = load() else { return };
    let mv = m.embed_multi(&ids(), dev).expect("embed_multi");
    assert_eq!(mv.len(), ids().len(), "multi-vec rows == seq");
    for (i, row) in mv.iter().enumerate() {
        assert_eq!(row.len(), PROJ_DIM, "row {i} width == 128");
        assert!(row.iter().all(|x| x.is_finite()), "row {i} finite");
        assert!(
            (l2(row) - 1.0).abs() < 1e-4,
            "row {i} ‖·‖₂ ≈ 1, got {}",
            l2(row)
        );
    }
    // Deterministic.
    let mv2 = m.embed_multi(&ids(), dev).expect("embed_multi 2");
    assert_eq!(mv, mv2, "multi-vector must be deterministic");
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn task_sensitivity_single_and_multi() {
    let Some((mut m, dev)) = load() else { return };
    let seq = ids();

    m.apply_task(JinaV4Task::Retrieval).expect("retrieval");
    let s_ret = m.embed_single(&seq, dev, None).expect("single retrieval");
    let mv_ret = m.embed_multi(&seq, dev).expect("multi retrieval");

    m.apply_task(JinaV4Task::Code).expect("code");
    let s_code = m.embed_single(&seq, dev, None).expect("single code");
    let mv_code = m.embed_multi(&seq, dev).expect("multi code");

    // LoRA flows through pooling + projector: retrieval != code.
    assert_ne!(
        s_ret, s_code,
        "single-vector must differ between retrieval and code (decoder LoRA)"
    );
    assert_ne!(
        mv_ret, mv_code,
        "multi-vector must differ between retrieval and code (decoder + projector LoRA)"
    );
    // Re-applying retrieval reproduces the original (clean replace).
    m.apply_task(JinaV4Task::Retrieval)
        .expect("retrieval again");
    assert_eq!(
        s_ret,
        m.embed_single(&seq, dev, None).expect("single retrieval 2"),
        "task switch is a clean replace (no residue)"
    );
}
