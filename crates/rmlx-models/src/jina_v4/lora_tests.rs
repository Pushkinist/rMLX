use super::*;
use rmlx_mlx::{Device, Dtype};

fn model_dir() -> Option<std::path::PathBuf> {
    let Some(p) = std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from) else {
        eprintln!("SKIP: RMLX_TEST_MODEL_JINA_V4 not set");
        return None;
    };
    if p.join("adapters/adapter_model.safetensors").exists() {
        Some(p)
    } else {
        eprintln!("SKIP: jina-v4 adapter absent at {}/adapters", p.display());
        None
    }
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn build_text(dir: &Path) -> JinaV4Text {
    let cfg = super::super::JinaV4Config::from_file(&dir.join("config.json"))
        .expect("parse jina-v4 config.json");
    super::super::model::load_text_tower(dir, &cfg.text_config).expect("load jina-v4 text tower")
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
    f.eval().expect("materialize");
    f.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn task_name_roundtrip() {
    for t in JinaV4Task::ALL {
        assert_eq!(JinaV4Task::from_name(t.name()).unwrap(), t);
    }
    assert_eq!(JinaV4Task::DEFAULT, JinaV4Task::Retrieval);
    assert!(JinaV4Task::from_name("nope").is_err());
    // jina names use a hyphen, not an underscore.
    assert!(JinaV4Task::from_name("text_matching").is_err());
}

/// DoD (a): full key coverage + factor shapes + no visual / config sanity.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn adapter_key_coverage_and_shapes() {
    let Some(dir) = model_dir() else { return };
    let num_layers = 36usize;
    let ad = JinaV4Adapters::load(&dir, num_layers).expect("load adapters");

    // r = alpha = 32 => scaling exactly 1.0.
    assert_eq!(ad.config().r, 32, "rank r");
    assert_eq!(ad.config().lora_alpha, 32, "lora_alpha");
    assert!(
        (ad.scaling() - 1.0).abs() < 1e-9,
        "scaling = alpha/r must be 1.0, got {}",
        ad.scaling()
    );
    assert_eq!(
        ad.config().exclude_modules.as_deref(),
        Some(".*visual.*"),
        "exclude_modules regex"
    );

    // Exact decoder cell count: 36 x 7 x 3 = 756 (A,B) pairs.
    assert_eq!(
        ad.decoder_pair_count(),
        36 * 7 * 3,
        "decoder (layer,proj,task) pairs"
    );
    // Projector retained: 3 task pairs (one per task), not applied here.
    assert_eq!(
        ad.projector_pair_count(),
        3,
        "multi_vector_projector task pairs"
    );

    // Every (layer, proj, task) present + A is [32, in], B is [out, 32].
    for layer in 0..num_layers {
        for proj in ProjId::ALL {
            for task in JinaV4Task::ALL {
                let fp = ad
                    .decoder
                    .get(&(layer, proj, task))
                    .unwrap_or_else(|| panic!("missing {layer}/{proj:?}/{task:?}"));
                let a = fp.a.shape();
                let b = fp.b.shape();
                assert_eq!(a.len(), 2, "A rank for {layer}/{proj:?}");
                assert_eq!(b.len(), 2, "B rank for {layer}/{proj:?}");
                assert_eq!(a[0], 32, "A must be [32, in] for {layer}/{proj:?}");
                assert_eq!(b[1], 32, "B must be [out, 32] for {layer}/{proj:?}");
                // A.in == matching projection input width sanity:
                // q/k/v/o/gate/up read 2048; down reads 11008.
                let expect_in = match proj {
                    ProjId::DownProj => 11008,
                    _ => 2048,
                };
                assert_eq!(a[1], expect_in, "A in-features for {layer}/{proj:?}");
            }
        }
    }

    // Projector LoRA shapes: A [32, 2048], B [128, 32] (2048->128).
    for task in JinaV4Task::ALL {
        let fp = ad.projector.get(&task).expect("projector task pair");
        assert_eq!(fp.a.shape(), vec![32, 2048], "projector A");
        assert_eq!(fp.b.shape(), vec![128, 32], "projector B");
    }
}

/// DoD (b): tasks differentiate; no-LoRA != any task; deterministic.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn task_differentiation_and_determinism() {
    let Some(dir) = model_dir() else { return };
    let dev = Device::Cpu; // device-agnostic; no Metal claim
    let ad = JinaV4Adapters::load(&dir, 36).expect("load adapters");
    let ids: Vec<i64> = vec![9707, 25, 419, 374, 264, 1273, 11652, 13];

    // Baseline: no LoRA active.
    let mut text = build_text(&dir);
    text.clear_all_loras();
    let base = text.forward_hidden(&ids, dev).expect("forward base");
    base.eval().expect("eval base");
    let v_base = materialize_f32(&base);
    assert!(
        v_base.iter().all(|x| x.is_finite()),
        "base hidden must be finite"
    );

    let mut per_task: Vec<(JinaV4Task, Vec<f32>)> = Vec::new();
    for task in JinaV4Task::ALL {
        let mut t = build_text(&dir);
        ad.apply_task(&mut t, task).expect("apply task");
        let h = t.forward_hidden(&ids, dev).expect("forward task");
        h.eval().expect("eval task");
        let v = materialize_f32(&h);
        assert_eq!(v.len(), v_base.len(), "shape parity");
        assert!(
            v.iter().all(|x| x.is_finite()),
            "task {} hidden must be finite",
            task.name()
        );
        // Determinism: same task, fresh tower -> bit-identical.
        let mut t2 = build_text(&dir);
        ad.apply_task(&mut t2, task).expect("apply task again");
        let h2 = t2.forward_hidden(&ids, dev).expect("forward task 2");
        h2.eval().expect("eval task 2");
        assert_eq!(
            v,
            materialize_f32(&h2),
            "task {} must be deterministic",
            task.name()
        );
        per_task.push((task, v));
    }

    // No-LoRA differs from every task (deltas are actually applied).
    for (task, v) in &per_task {
        assert_ne!(
            *v,
            v_base,
            "task {} hidden must differ from no-LoRA baseline",
            task.name()
        );
    }
    // All three tasks differ pairwise.
    for i in 0..per_task.len() {
        for j in (i + 1)..per_task.len() {
            assert_ne!(
                per_task[i].1,
                per_task[j].1,
                "tasks {} and {} must produce different hidden states",
                per_task[i].0.name(),
                per_task[j].0.name()
            );
        }
    }

    eprintln!(
        "jina-v4 LoRA: base + {} tasks all finite, pairwise-distinct, deterministic",
        per_task.len()
    );
}
