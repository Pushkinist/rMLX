use super::*;

fn jina_v4_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from)
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn load() -> Option<(JinaV4Text, Device)> {
    let Some(dir_buf) = jina_v4_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_JINA_V4 not set");
        return None;
    };
    let dir = dir_buf.as_path();
    if !dir.exists() {
        eprintln!("SKIP: jina-v4 snapshot absent at {}", dir.display());
        return None;
    }
    let cfg = super::super::JinaV4Config::from_file(&dir.join("config.json"))
        .expect("parse jina-v4 config.json");
    let tower = load_text_tower(dir, &cfg.text_config).expect("load jina-v4 text tower");
    // CPU keeps the test free of the single-MLX-process Metal claim and
    // works on any machine; the forward graph is device-agnostic.
    Some((tower, Device::Cpu))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn materialize_f32(arr: &Array) -> Vec<f32> {
    let f = arr.astype(Dtype::F32, Device::Cpu).expect("cast f32");
    f.eval().expect("materialize f32");
    f.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn forward_hidden_shape_and_finite() {
    let Some((tower, dev)) = load() else {
        return;
    };
    // Short arbitrary token sequence (well within vocab 151936).
    let ids: Vec<i64> = vec![9707, 25, 419, 374, 264, 1273, 11652, 13];
    let h = tower.forward_hidden(&ids, dev).expect("forward_hidden");
    h.eval().expect("eval hidden");

    assert_eq!(
        h.shape(),
        vec![1, ids.len() as i32, 2048],
        "post-norm hidden must be [1, seq, 2048]"
    );

    let vals = materialize_f32(&h);
    assert_eq!(vals.len(), ids.len() * 2048, "value count");
    let bad = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(bad, 0, "non-finite (NaN/Inf) values in hidden: {bad}");
    // Sanity: post-RMSNorm activations are not all zero.
    let max_abs = vals.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max_abs > 0.0, "hidden states all zero (max_abs={max_abs})");
    eprintln!(
        "jina-v4 forward_hidden: shape={:?} max_abs={max_abs:.4}",
        h.shape()
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn forward_hidden_deterministic() {
    let Some((tower, dev)) = load() else {
        return;
    };
    let ids: Vec<i64> = vec![40, 1079, 264, 4108, 6573, 13];
    let a = tower.forward_hidden(&ids, dev).expect("forward a");
    a.eval().expect("eval a");
    let b = tower.forward_hidden(&ids, dev).expect("forward b");
    b.eval().expect("eval b");

    let va = materialize_f32(&a);
    let vb = materialize_f32(&b);
    assert_eq!(va.len(), vb.len(), "length mismatch");
    assert_eq!(
        va, vb,
        "same input must yield bit-identical hidden states (deterministic forward)"
    );
}
