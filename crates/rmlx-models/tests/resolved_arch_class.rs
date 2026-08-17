//! Declared vs resolved architecture, and the KV guard that must key off the
//! resolved one.
//!
//! A checkpoint's `architectures[0]` is model-side data that nothing
//! validates. The Qwen3.5 loader ignores it for the dense-vs-sparse-MoE
//! decision — it probes the per-layer tensor witness
//! (`mlp.switch_mlp.gate_proj.weight`) instead — so the declaration and the
//! model that actually gets built can disagree. Where they do, any safety
//! predicate still reading the declaration is reasoning about a model that
//! does not exist.
//!
//! The predicate that matters here is the Qwen-MoE K-side codec guard: rotor /
//! iso / low-bit-K codecs were measured to destroy perplexity on Qwen sparse
//! MoE, and the guard rejects them. Keyed on the declaration, a snapshot that
//! declares dense while shipping MoE tensors runs the disaster path to
//! completion with no error — the run succeeds and only the output is wrong.
//!
//! These tests load real snapshots and are `#[ignore]`d for that reason; they
//! skip gracefully when the snapshot is absent.
//!
//! ```text
//! RMLX_O_MODELS_ROOT=/path/to/open-models \
//!   cargo test -p rmlx-models --test resolved_arch_class -- --ignored --test-threads=1
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

use std::path::{Path, PathBuf};

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;

/// Resolve a snapshot by env override first, then by slug under
/// `RMLX_O_MODELS_ROOT`. Returns `None` (with a SKIP note) when absent.
fn snapshot(env_key: &str, slug: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_key) {
        let p = PathBuf::from(p);
        if p.join("config.json").is_file() {
            return Some(p);
        }
    }
    if let Ok(root) = std::env::var("RMLX_O_MODELS_ROOT") {
        let p = PathBuf::from(root).join(slug);
        if p.join("config.json").is_file() {
            return Some(p);
        }
    }
    eprintln!("SKIP: snapshot {slug} not found (set {env_key} or RMLX_O_MODELS_ROOT)");
    None
}

/// Read `architectures[0]` straight from the snapshot's `config.json`.
fn declared_arch(model_dir: &Path) -> String {
    let data = std::fs::read(model_dir.join("config.json")).expect("read config.json");
    let v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
    v.get("architectures")
        .and_then(|a| a.get(0))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Build a snapshot that is byte-identical to `src` except that
/// `architectures[0]` is replaced by `declared`. Weights are symlinked, so this
/// costs no disk and no copy time.
///
/// This is the bypass under test: the tensors still say sparse MoE, the
/// declaration says something else.
fn relabelled_snapshot(src: &Path, declared: &str, dst: &Path) {
    std::fs::create_dir_all(dst).expect("create relabelled snapshot dir");
    for entry in std::fs::read_dir(src).expect("read snapshot dir") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name();
        if name == std::ffi::OsStr::new("config.json") {
            continue;
        }
        std::os::unix::fs::symlink(entry.path(), dst.join(&name)).expect("symlink weight file");
    }
    let data = std::fs::read(src.join("config.json")).expect("read config.json");
    let mut v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
    v.as_object_mut()
        .expect("config.json is a JSON object")
        .insert("architectures".to_owned(), serde_json::json!([declared]));
    std::fs::write(
        dst.join("config.json"),
        serde_json::to_vec_pretty(&v).expect("serialise config.json"),
    )
    .expect("write relabelled config.json");
}

/// Every K-side codec the Qwen-MoE guard exists to reject.
fn k_side_disaster_codecs() -> Vec<KvQuant> {
    vec![
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
        // The payload-bearing rotor variants — the only rotor codecs that
        // reach the fused-QK kernel, so the reachable half of the family.
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::PlanarK,
        KvQuant::TurboSym3,
        KvQuant::TurboSym4,
    ]
}

/// A Qwen3.5 checkpoint with no sparse-MoE tensors must report the dense class,
/// whatever the enum variant it shares with the MoE path is called.
///
/// Both Qwen3.5 arch strings load into one `Architecture` variant, so reporting
/// the variant's name labelled every dense snapshot as MoE — in tracing, in the
/// bench header, and in the metrics rows those runs wrote.
#[test]
#[ignore = "loads a real multi-GB snapshot"]
fn dense_qwen3_5_snapshot_resolves_to_the_dense_class() {
    let Some(dir) = snapshot(
        "RMLX_TEST_MODEL_ORNITH_9B",
        "sahilchachra__ornith-1.0-9b-mxfp8-mlx",
    ) else {
        return;
    };

    let model = arch::load_model(&dir, Device::Cpu, &arch::LoadOpts::default()).expect("load");

    assert_eq!(
        model.arch_class(),
        "Qwen3_5ForConditionalGeneration",
        "a checkpoint with no switch_mlp tensors resolves to dense; reporting the \
         MoE class mislabels it everywhere arch_class() is consumed"
    );

    // The guard is scoped to sparse MoE, so a genuinely dense model keeps
    // access to the K-side codecs. This is the half of the behaviour that must
    // NOT change.
    for kq in k_side_disaster_codecs() {
        assert!(
            model.validate_kv_quant(kq).is_ok(),
            "{kq} must stay available on a dense Qwen3.5 model"
        );
    }
}

/// The bypass: MoE tensors, dense declaration. The guard must still fire.
///
/// Keyed on `architectures[0]` this run completes normally on the measured
/// PPL-disaster path.
#[test]
#[ignore = "loads a real multi-GB snapshot"]
fn mislabelled_moe_snapshot_cannot_bypass_the_k_side_guard() {
    let Some(src) = snapshot(
        "RMLX_TEST_MODEL_QWEN36",
        "mlx-community__Qwen3.6-35B-A3B-8bit",
    ) else {
        return;
    };

    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("declares-dense-ships-moe");
    relabelled_snapshot(&src, "Qwen3_5ForConditionalGeneration", &dir);

    assert_eq!(
        declared_arch(&dir),
        "Qwen3_5ForConditionalGeneration",
        "test fixture must actually carry the mismatched declaration"
    );

    let model = arch::load_model(&dir, Device::Cpu, &arch::LoadOpts::default()).expect("load");

    assert_eq!(
        model.arch_class(),
        "Qwen3_5MoeForConditionalGeneration",
        "the resolved class must follow the tensors, not the declaration"
    );

    for kq in k_side_disaster_codecs() {
        assert!(
            model.validate_kv_quant(kq).is_err(),
            "{kq} reached a sparse-MoE model through a dense declaration"
        );
    }

    // K stays 8-bit here, so the guard has nothing to say — a refusal would be
    // over-broad rather than safe.
    assert!(model.validate_kv_quant(KvQuant::K8V8).is_ok());
    assert!(model.validate_kv_quant(KvQuant::K8V4).is_ok());
}

/// Control: the same guard on a snapshot whose declaration is honest. This is
/// the configuration that was already covered, and it must not regress.
#[test]
#[ignore = "loads a real multi-GB snapshot"]
fn correctly_declared_moe_snapshot_is_still_guarded() {
    let Some(dir) = snapshot(
        "RMLX_TEST_MODEL_QWEN36",
        "mlx-community__Qwen3.6-35B-A3B-8bit",
    ) else {
        return;
    };

    assert_eq!(declared_arch(&dir), "Qwen3_5MoeForConditionalGeneration");

    let model = arch::load_model(&dir, Device::Cpu, &arch::LoadOpts::default()).expect("load");

    assert_eq!(model.arch_class(), "Qwen3_5MoeForConditionalGeneration");
    for kq in k_side_disaster_codecs() {
        assert!(
            model.validate_kv_quant(kq).is_err(),
            "{kq} must be rejected"
        );
    }
}
