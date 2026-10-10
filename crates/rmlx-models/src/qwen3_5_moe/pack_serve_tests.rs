//! `load_model` serving a Hadamard pack, and the checks that refuse a pack or a
//! drafter it cannot serve.

use super::*;

/// The ids of the end-to-end forward pass.
const TINY_IDS: [u32; 5] = [3, 0, 7, 2, 5];

fn no_transform(rows: &[f64], _signs: &[f32], _block: usize) -> Vec<f64> {
    rows.to_vec()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| f64::max(m, x.abs()))
}

/// The logits at every position of `load_model(dir)` on `device`.
fn logits_on(dir: &Path, device: rmlx_mlx::Device) -> Result<Vec<f64>> {
    use crate::hadamard::hadamard_tests::{read_f32, widen};
    let arch = crate::arch::load_model(dir, device, &crate::arch::LoadOpts::default())?;
    let crate::arch::Architecture::Qwen3_5Moe(model) = arch else {
        panic!("the snapshot is a Qwen3.5 model");
    };
    let logits = model.forward_seq_logits_all(&TINY_IDS, device)?;
    assert_eq!(
        logits.shape(),
        vec![1, TINY_IDS.len() as i32, TINY_VOCAB as i32]
    );
    Ok(widen(&read_f32(&logits)?))
}

/// One arm of the seam: the pack and the bound its logits must meet.
struct Seam {
    pack: TinyPack,
    device: rmlx_mlx::Device,
    /// The dtype of the oracle's unfolded modules, which sets its activation
    /// dtype.
    oracle_dtype: rmlx_mlx::Dtype,
    /// Bound on `max |pack - oracle| / max |oracle|`.
    bound: f64,
    /// Each control must miss by more than this many bounds.
    control_margin: f64,
}

impl Seam {
    /// `load_model` takes the pack and builds a model whose logits are those
    /// of the plain model the contract unfolds to: the oracle, built on the
    /// host from the dequantized codes, the signs of each width and the
    /// Sylvester matrix, then loaded through the plain path. Both controls,
    /// the codes with no transform and with the other direction, miss.
    #[allow(
        clippy::unwrap_used,
        reason = "test fixture: tempdir failures should abort the test loudly"
    )]
    fn check(&self) -> Result<()> {
        use crate::hadamard::hadamard_tests::{max_abs_diff, oracle_forward, oracle_inverse};
        let unfolded = |unfold: Unfold| -> Result<Vec<f64>> {
            let dir = tempfile::tempdir().unwrap();
            write_unfolded(dir.path(), self.pack.gdn, unfold, self.oracle_dtype);
            logits_on(dir.path(), self.device)
        };
        let dir = tempfile::tempdir().unwrap();
        self.pack.write(dir.path());
        let got = logits_on(dir.path(), self.device)?;
        let want = unfolded(oracle_inverse)?;
        let scale = max_abs(&want);
        let err = max_abs_diff(&got, &want) / scale;
        println!("pack vs unfolded oracle: {err:.3e} of max |logit| {scale:.3}");
        assert!(
            err <= self.bound,
            "pack vs unfolded oracle: {err:.3e} > {}",
            self.bound
        );
        let controls: [(&str, Unfold); 2] = [
            ("no transform", no_transform),
            ("forward direction", oracle_forward),
        ];
        for (control, unfold) in controls {
            let miss = max_abs_diff(&got, &unfolded(unfold)?) / scale;
            println!("control `{control}`: {miss:.3e}");
            assert!(
                miss > self.control_margin * self.bound,
                "control `{control}` misses by {miss:.3e}, not more than {} x {}",
                self.control_margin,
                self.bound
            );
        }
        Ok(())
    }
}

#[test]
fn load_model_serves_a_pack_as_the_plain_model_its_contract_unfolds() {
    crate::test_child::run_child(
        "qwen3_5_moe::loader::pack_loader_tests::pack_serve_tests::\
         load_model_serves_a_pack_as_the_plain_model_its_contract_unfolds_child",
    );
}

/// The seam on the CPU, with the GPU forbidden, over the attention-only pack
/// (the gated-delta-net recurrence is a Metal kernel). A wiring check, not a
/// precision check: the MLX CPU `quantized_matmul` accumulates a bf16 input
/// in bf16, so the pack misses the F32 oracle by 0.283 of the largest logit,
/// and by 0.016 when the inner matmul is given an f32 input. The bound sits
/// above 0.283. Measured controls: no transform 24.8, the other direction
/// 0.989, which is 2.5 bounds.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn load_model_serves_a_pack_as_the_plain_model_its_contract_unfolds_child() -> Result<()> {
    if !crate::test_child::started_by_parent() {
        return Ok(());
    }
    rmlx_mlx::forbid_gpu()?;
    Seam {
        pack: ATTENTION_ONLY,
        device: rmlx_mlx::Device::Cpu,
        oracle_dtype: rmlx_mlx::Dtype::F32,
        bound: 0.4,
        control_margin: 2.0,
    }
    .check()?;
    println!("{}", crate::test_child::CHILD_DONE);
    Ok(())
}

/// The seam on the GPU, over the pack with its gated-delta-net layer: the
/// Metal `quantized_matmul` accumulates in float, so the bound is near the
/// bf16 activation floor. The oracle's modules are bf16, so both arms run
/// bf16 activations through the same kernels.
#[test]
#[ignore = "GPU Metal: loads and runs two synthetic models on Device::Gpu"]
fn load_model_serves_a_pack_as_the_plain_model_its_contract_unfolds_on_gpu() -> Result<()> {
    Seam {
        pack: PACK,
        device: rmlx_mlx::Device::Gpu,
        oracle_dtype: rmlx_mlx::Dtype::Bf16,
        bound: 0.03,
        control_margin: 10.0,
    }
    .check()
}

/// Rewrite `config.json` in `dir` through `edit`.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and JSON failures should abort the test loudly"
)]
fn edit_config(dir: &Path, edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) {
    let path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(config.as_object_mut().unwrap());
    std::fs::write(path, config.to_string()).unwrap();
}

/// `load_model` refuses a pack it cannot serve before any tensor loads,
/// naming the fact and the file that holds it, never an unsupported or empty
/// architecture.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir failures should abort the test loudly"
)]
fn load_model_refuses_a_pack_it_cannot_serve_by_its_fact() {
    type Edit = Box<dyn Fn(&Path)>;
    let config = |edit: fn(&mut serde_json::Map<String, serde_json::Value>)| -> Edit {
        Box::new(move |dir: &Path| edit_config(dir, edit))
    };
    let cases: Vec<(&str, TinyPack, Edit, &[&str])> = vec![
        (
            "a base_model_type with no class",
            PACK,
            config(|c| {
                c.insert("base_model_type".into(), "llama".into());
            }),
            &["config.json", "base_model_type", "\"llama\""],
        ),
        (
            "a declared class the fold does not serve",
            PACK,
            config(|c| {
                c.insert(
                    "architectures".into(),
                    serde_json::json!(["Qwen3ForCausalLM"]),
                );
            }),
            &["config.json", "Qwen3ForCausalLM"],
        ),
        (
            "the sparse MoE Qwen3.5 class over the dense backbone",
            PACK,
            config(|c| {
                c.insert(
                    "architectures".into(),
                    serde_json::json!(["Qwen3_5MoeForConditionalGeneration"]),
                );
            }),
            &["config.json", "Qwen3_5MoeForConditionalGeneration"],
        ),
        (
            "the Qwen3.5 class over another backbone",
            PACK,
            config(|c| {
                c.insert(
                    "architectures".into(),
                    serde_json::json!(["Qwen3_5ForConditionalGeneration"]),
                );
                c.insert("base_model_type".into(), "qwen3_5_moe".into());
            }),
            &["config.json", "base_model_type", "\"qwen3_5_moe\""],
        ),
        (
            "a ParoQuant checkpoint",
            PACK,
            config(|c| {
                c.insert(
                    "quantization_config".into(),
                    serde_json::json!({"quant_method": "paroquant"}),
                );
            }),
            &["config.json", "quantization_config", "paroquant"],
        ),
        (
            "a sign vector that is not the contract's",
            TinyPack {
                broken_signs: true,
                ..PACK
            },
            Box::new(|_: &Path| {}),
            &[
                "model.safetensors",
                "model.layers.0.mlp.up_proj.signs",
                "differs",
            ],
        ),
        (
            "no contract file",
            PACK,
            Box::new(|dir: &Path| std::fs::remove_file(dir.join("hadamard.json")).unwrap()),
            &["hadamard.json"],
        ),
        (
            "sign tensors and no hadamard_config",
            PACK,
            config(|c| {
                c.remove("hadamard_config");
            }),
            &["model.safetensors", ".signs", "hadamard_config"],
        ),
    ];
    for (what, pack, edit, needles) in cases {
        let dir = tempfile::tempdir().unwrap();
        pack.write(dir.path());
        edit(dir.path());
        let msg = crate::arch::load_model(
            dir.path(),
            rmlx_mlx::Device::Cpu,
            &crate::arch::LoadOpts::default(),
        )
        .err()
        .map_or_else(String::new, |e| e.to_string());
        assert!(msg.contains("is a Hadamard pack"), "{what}: {msg:?}");
        for needle in needles {
            assert!(msg.contains(needle), "{what}: no {needle:?} in {msg:?}");
        }
        for wrong in ["not yet supported", "(empty)"] {
            assert!(!msg.contains(wrong), "{what}: {msg:?}");
        }
    }
}

/// A one-layer EAGLE-3 drafter for the synthetic verifier: width
/// [`TINY_WIDTH`], verifier vocabulary [`TINY_VOCAB`], and `draft_vocab`
/// rows in its head. A smaller draft vocabulary carries a `d2t` table, which
/// makes the drafter's restricted-vocabulary hot path active. The loader
/// checks the shapes of `fc` and of the head only, so the other tensors are
/// one value each.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
)]
fn write_eagle3_drafter(dir: &Path, draft_vocab: usize) {
    let f32_tensor = |name: &str, shape: Vec<usize>| -> RawTensor {
        let n = shape.iter().product::<usize>();
        let values = uniform(5400, n);
        (
            name.to_owned(),
            safetensors::Dtype::F32,
            shape,
            le32(&values),
        )
    };
    let mut tensors = vec![
        f32_tensor("fc.weight", vec![TINY_WIDTH, 3 * TINY_WIDTH]),
        f32_tensor("lm_head.weight", vec![draft_vocab, TINY_WIDTH]),
    ];
    for name in [
        "layers.0.hidden_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "norm.weight",
    ] {
        tensors.push(f32_tensor(name, vec![1]));
    }
    if draft_vocab != TINY_VOCAB {
        let d2t: Vec<u8> = (0..draft_vocab)
            .flat_map(|i| (i as i64).to_le_bytes())
            .collect();
        tensors.push((
            "d2t".to_owned(),
            safetensors::Dtype::I64,
            vec![draft_vocab],
            d2t,
        ));
    }
    write_model_shard(dir, &tensors);
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLMEagle3"],
        "hidden_size": TINY_WIDTH, "num_attention_heads": TINY_HEADS,
        "num_key_value_heads": 1, "head_dim": TINY_HEAD_DIM,
        "draft_vocab_size": draft_vocab, "vocab_size": TINY_VOCAB,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 1, 1]}
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

/// The EAGLE-3 drafter cannot be built over a pack verifier when its
/// restricted-vocabulary head is active: that head multiplies folded output
/// rows with the final hidden state, which is wrong with no error. Controls:
/// the same drafter over the plain verifier, and a full-vocabulary drafter
/// over the pack, both load.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir failures should abort the test loudly"
)]
fn an_eagle3_drafter_with_hot_ids_is_refused_over_a_pack_verifier() -> Result<()> {
    use crate::speculative::eagle3::Eagle3Drafter;
    let pack = crate::arch::Architecture::Qwen3_5Moe(PACK.load()?);
    let plain = crate::arch::Architecture::Qwen3_5Moe(PLAIN.load()?);
    let device = rmlx_mlx::Device::Cpu;
    let hot = tempfile::tempdir().unwrap();
    write_eagle3_drafter(hot.path(), TINY_VOCAB / 2);
    let full = tempfile::tempdir().unwrap();
    write_eagle3_drafter(full.path(), TINY_VOCAB);

    let r = Eagle3Drafter::load(hot.path(), &pack, &[], device);
    assert!(
        matches!(&r, Err(Error::Model(m)) if m.contains("Hadamard")),
        "a hot-path drafter over the pack: {:?}",
        r.err()
    );
    let drafter = Eagle3Drafter::load(hot.path(), &plain, &[], device)?;
    assert!(drafter.hot_path_active(), "the control drafter has hot ids");
    let drafter = Eagle3Drafter::load(full.path(), &pack, &[], device)?;
    assert!(!drafter.hot_path_active(), "a full-vocabulary drafter");
    Ok(())
}
