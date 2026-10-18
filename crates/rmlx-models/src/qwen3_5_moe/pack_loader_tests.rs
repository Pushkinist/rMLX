//! Hadamard pack loading from a one-layer synthetic snapshot (`TinyPack`).
//!
//! The cases load the snapshot through `load_from_path` on the CPU: no Metal,
//! no model of any size.

use std::path::Path;

use rmlx_mlx::Array;

use rmlx_core::error::{Error, Result};

use super::super::decoder_layer::MlpBlock;
use super::super::layers::layers_tests::Folded;
use super::super::layers::Linear;
use super::super::model::Qwen3_5MoeText;
use crate::hadamard::hadamard_tests::{signs, uniform};

/// Synthetic Qwen3.5 snapshot with the shapes of a model that runs a forward
/// pass: hidden 512, a dense MLP of width 1024 on each layer, and one
/// full-attention layer with a gated query (2 heads, 1 KV head, `head_dim`
/// 256). With `gdn`, a gated-delta-net layer comes first (1 key head of 128,
/// 4 value heads of 256) and the attention layer is layer 1
/// (`full_attention_interval` 2); its `in_proj_a` and `in_proj_b` are dense
/// F32, as in the real pack. Affine 2-bit with group 128, each module's codes
/// from [`Folded::synthetic`] ([`tiny_folded`]); the input widths are 512 and
/// 1024. With `contract`, a Prism Hadamard pack (block 512, so a 1024-wide
/// input is two blocks) around it: the contract file, the `modules` records
/// and a `.signs` tensor per module.
#[derive(Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "test fixture: each flag is one independent fact of the snapshot"
)]
struct TinyPack {
    contract: bool,
    /// A gated-delta-net layer before the attention layer.
    gdn: bool,
    /// An extra quantized module the Qwen3.5 loader builds no layer for.
    extra_module: bool,
    /// One `.signs` value that differs from the contract's vector.
    broken_signs: bool,
    base_model_type: &'static str,
    /// A switch-MLP expert tensor in the layer: the witness the loader reads
    /// a sparse MoE layer from.
    moe_witness: bool,
}

const PACK: TinyPack = TinyPack {
    contract: true,
    gdn: true,
    extra_module: false,
    broken_signs: false,
    base_model_type: "qwen3_5",
    moe_witness: false,
};
const PLAIN: TinyPack = TinyPack {
    contract: false,
    ..PACK
};
/// The pack with its attention layer only: the shape a CPU forward runs, as
/// the gated-delta-net recurrence is a Metal kernel.
const ATTENTION_ONLY: TinyPack = TinyPack { gdn: false, ..PACK };

/// The hidden size, and the input width of every module but the two below.
const TINY_WIDTH: usize = 512;
/// The input width of `down_proj` and `out_proj`: two Hadamard blocks.
const TINY_WIDE: usize = 1024;
const TINY_BLOCK: usize = 512;
/// The rows of the head and of the embedding table.
const TINY_VOCAB: usize = 8;
const TINY_HEADS: usize = 2;
const TINY_HEAD_DIM: usize = 256;
const GDN_KEY_HEADS: usize = 1;
const GDN_KEY_DIM: usize = 128;
const GDN_VALUE_HEADS: usize = 4;
const GDN_VALUE_DIM: usize = 256;
const GDN_CONV_DIM: usize = 2 * GDN_KEY_HEADS * GDN_KEY_DIM + GDN_VALUE_HEADS * GDN_VALUE_DIM;
const TINY_EXTRA: &str = "model.layers.0.mlp.extra_proj";

/// The layer that carries full attention.
fn attention_layer(gdn: bool) -> usize {
    usize::from(gdn)
}

/// Module paths under the `language_model.` namespace, the head first (the
/// contract reader takes its tensor prefix from the first record).
fn tiny_modules(gdn: bool) -> Vec<String> {
    let mut modules = vec!["lm_head".to_owned(), "model.embed_tokens".to_owned()];
    if gdn {
        for p in ["in_proj_qkv", "in_proj_z", "out_proj"] {
            modules.push(format!("model.layers.0.linear_attn.{p}"));
        }
    }
    let a = attention_layer(gdn);
    for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        modules.push(format!("model.layers.{a}.self_attn.{p}"));
    }
    for layer in 0..=a {
        for p in ["gate_proj", "up_proj", "down_proj"] {
            modules.push(format!("model.layers.{layer}.mlp.{p}"));
        }
    }
    modules
}

/// The input width of module `path`.
fn tiny_width(path: &str) -> usize {
    match path.rsplit('.').next() {
        Some("down_proj" | "out_proj") => TINY_WIDE,
        _ => TINY_WIDTH,
    }
}

/// The contract's sign vector of `width`.
fn tiny_signs(width: usize) -> Vec<f32> {
    signs(5000 + width as u64, width)
}

/// The output rows of module `path`: the query projection carries the query
/// and its gate.
fn tiny_rows(path: &str) -> usize {
    match path.rsplit('.').next() {
        Some("lm_head" | "embed_tokens") => TINY_VOCAB,
        Some("q_proj") => 2 * TINY_HEADS * TINY_HEAD_DIM,
        Some("k_proj" | "v_proj") => TINY_HEAD_DIM,
        Some("in_proj_qkv") => GDN_CONV_DIM,
        Some("in_proj_z") => GDN_VALUE_HEADS * GDN_VALUE_DIM,
        Some("gate_proj" | "up_proj") => TINY_WIDE,
        _ => TINY_WIDTH,
    }
}

/// The quantized table of module `path`, seeded by its path.
fn tiny_folded(path: &str) -> Folded {
    let seed = path.bytes().fold(5100u64, |h, b| {
        h.wrapping_mul(31).wrapping_add(u64::from(b))
    });
    Folded::synthetic(seed, tiny_rows(path), tiny_width(path))
}

/// The unquantized tensors: name, shape, values. The norms are 1; the
/// gated-delta-net parameters are small values from `uniform`.
fn tiny_floats(gdn: bool, moe_witness: bool) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let ones = |n: usize| vec![1.0f32; n];
    let small = |seed: u64, n: usize, scale: f32| -> Vec<f32> {
        uniform(seed, n).iter().map(|v| v * scale).collect()
    };
    let a = attention_layer(gdn);
    let mut floats = vec![(
        "model.norm.weight".to_owned(),
        vec![TINY_WIDTH],
        ones(TINY_WIDTH),
    )];
    for layer in 0..=a {
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            floats.push((
                format!("model.layers.{layer}.{norm}.weight"),
                vec![TINY_WIDTH],
                ones(TINY_WIDTH),
            ));
        }
    }
    for norm in ["q_norm", "k_norm"] {
        floats.push((
            format!("model.layers.{a}.self_attn.{norm}.weight"),
            vec![TINY_HEAD_DIM],
            ones(TINY_HEAD_DIM),
        ));
    }
    if gdn {
        let la = "model.layers.0.linear_attn";
        let (hv, conv) = (GDN_VALUE_HEADS, GDN_CONV_DIM);
        floats.extend([
            (
                format!("{la}.conv1d.weight"),
                vec![conv, 4, 1],
                small(5601, conv * 4, 0.5),
            ),
            (format!("{la}.A_log"), vec![hv], small(5602, hv, 1.0)),
            (format!("{la}.dt_bias"), vec![hv], small(5603, hv, 1.0)),
            (
                format!("{la}.norm.weight"),
                vec![GDN_VALUE_DIM],
                ones(GDN_VALUE_DIM),
            ),
            (
                format!("{la}.in_proj_a.weight"),
                vec![hv, TINY_WIDTH],
                small(5604, hv * TINY_WIDTH, 0.05),
            ),
            (
                format!("{la}.in_proj_b.weight"),
                vec![hv, TINY_WIDTH],
                small(5605, hv * TINY_WIDTH, 0.05),
            ),
        ]);
    }
    if moe_witness {
        floats.push((
            "model.layers.0.mlp.switch_mlp.gate_proj.weight".to_owned(),
            vec![TINY_WIDTH],
            ones(TINY_WIDTH),
        ));
    }
    floats
}

fn tiny_text_config(gdn: bool) -> serde_json::Value {
    serde_json::json!({
        "num_hidden_layers": attention_layer(gdn) + 1, "hidden_size": TINY_WIDTH,
        "num_attention_heads": TINY_HEADS, "num_key_value_heads": 1,
        "head_dim": TINY_HEAD_DIM, "vocab_size": TINY_VOCAB,
        "intermediate_size": TINY_WIDE,
        "full_attention_interval": attention_layer(gdn) + 1,
        "linear_num_key_heads": GDN_KEY_HEADS, "linear_key_head_dim": GDN_KEY_DIM,
        "linear_num_value_heads": GDN_VALUE_HEADS, "linear_value_head_dim": GDN_VALUE_DIM,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": false,
        "rope_parameters": {"rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25}
    })
}

/// One tensor of a synthetic shard: name, dtype, shape, little-endian bytes.
type RawTensor = (String, safetensors::Dtype, Vec<usize>, Vec<u8>);

fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The unquantized tensors as F32 shard entries.
fn float_tensors(gdn: bool, moe_witness: bool) -> Vec<RawTensor> {
    tiny_floats(gdn, moe_witness)
        .into_iter()
        .map(|(name, shape, values)| {
            (
                format!("language_model.{name}"),
                safetensors::Dtype::F32,
                shape,
                le32(&values),
            )
        })
        .collect()
}

/// Write `tensors` as `model.safetensors` in `dir`.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
)]
fn write_model_shard(dir: &Path, tensors: &[RawTensor]) {
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(n, dt, shape, bytes)| {
            let view = safetensors::tensor::TensorView::new(*dt, shape.clone(), bytes).unwrap();
            (n.clone(), view)
        })
        .collect();
    let bytes = safetensors::serialize(views, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

/// Folded rows, signs, block -> the rows of the plain model.
type Unfold = fn(&[f64], &[f32], usize) -> Vec<f64>;

/// The plain snapshot a pack's contract unfolds to: the same model with
/// every folded module stored as `dtype` rows `unfold(dequant(row), signs of
/// its width, block)`. For the contract's arithmetic `unfold` is the inverse
/// transform: a linear `W' . H (D x)` is `(D H W'^T)^T x` because `H` is
/// symmetric, and an embedding row is `D H E'[id]`.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O, serialization and host array failures should abort the test loudly"
)]
fn write_unfolded(dir: &Path, gdn: bool, unfold: Unfold, dtype: rmlx_mlx::Dtype) {
    let mut tensors: Vec<RawTensor> = tiny_modules(gdn)
        .iter()
        .map(|m| {
            let folded = tiny_folded(m);
            let width = folded.width;
            let rows: Vec<f32> = unfold(&folded.dequant(), &tiny_signs(width), TINY_BLOCK)
                .iter()
                .map(|v| *v as f32)
                .collect();
            let shape = vec![folded.rows, width];
            let (st_dtype, bytes) = if dtype == rmlx_mlx::Dtype::Bf16 {
                let a = Array::from_f32_slice(&rows, &[rows.len() as i32])
                    .unwrap()
                    .astype(dtype, rmlx_mlx::Device::Cpu)
                    .unwrap();
                (safetensors::Dtype::BF16, a.to_bytes().unwrap())
            } else {
                (safetensors::Dtype::F32, le32(&rows))
            };
            (format!("language_model.{m}.weight"), st_dtype, shape, bytes)
        })
        .collect();
    tensors.extend(float_tensors(gdn, false));
    write_model_shard(dir, &tensors);
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "tie_word_embeddings": false,
        "text_config": tiny_text_config(gdn)
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

impl TinyPack {
    #[allow(
        clippy::unwrap_used,
        reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
    )]
    fn write(self, dir: &Path) {
        let mut modules = tiny_modules(self.gdn);
        if self.extra_module {
            modules.push(TINY_EXTRA.to_owned());
        }
        let le16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut tensors: Vec<RawTensor> = Vec::new();
        for m in &modules {
            let name = format!("language_model.{m}");
            let folded = tiny_folded(m);
            let groups = folded.width / 128;
            tensors.push((
                format!("{name}.weight"),
                safetensors::Dtype::U32,
                vec![folded.rows, folded.width / 16],
                folded.words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            ));
            for (part, values) in [("scales", &folded.scales), ("biases", &folded.biases)] {
                tensors.push((
                    format!("{name}.{part}"),
                    safetensors::Dtype::F16,
                    vec![folded.rows, groups],
                    le16(values),
                ));
            }
            if self.contract {
                let mut s = tiny_signs(folded.width);
                if self.broken_signs && m == "model.layers.0.mlp.up_proj" {
                    if let Some(v) = s.get_mut(5) {
                        *v = -*v;
                    }
                }
                tensors.push((
                    format!("{name}.signs"),
                    safetensors::Dtype::F32,
                    vec![folded.width],
                    le32(&s),
                ));
            }
        }
        tensors.extend(float_tensors(self.gdn, self.moe_witness));
        write_model_shard(dir, &tensors);

        let text_config = tiny_text_config(self.gdn);
        let quantization = serde_json::json!({"group_size": 128, "bits": 2, "mode": "affine"});
        let config = if self.contract {
            let records: Vec<_> = modules
                .iter()
                .map(|m| {
                    serde_json::json!({"path": m, "block": TINY_BLOCK,
                        "embedding": m == "model.embed_tokens", "dtype": "float16"})
                })
                .collect();
            let names = |embedding: bool| -> Vec<String> {
                modules
                    .iter()
                    .filter(|m| (*m == "model.embed_tokens") == embedding)
                    .map(|m| format!("language_model.{m}.weight"))
                    .collect()
            };
            let contract = serde_json::json!({
                "prism.hadamard.version": 1,
                "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
                "prism.hadamard.axis": "input-last-dimension",
                "prism.hadamard.sign_mode": "explicit",
                "prism.hadamard.gdn_v_grouped": true,
                "prism.hadamard.block_size": TINY_BLOCK,
                "prism.hadamard.sign_widths": [TINY_WIDTH, TINY_WIDE],
                "prism.hadamard.sign_values": ([tiny_signs(TINY_WIDTH), tiny_signs(TINY_WIDE)].concat()),
                "prism.hadamard.weight_names": names(false),
                "prism.hadamard.inverse_weight_names": names(true),
            });
            std::fs::write(dir.join("hadamard.json"), contract.to_string()).unwrap();
            serde_json::json!({
                "model_type": "prism_hadamard_qwen35", "base_model_type": self.base_model_type,
                "hadamard_config": "hadamard.json",
                "components": {"text": true, "vision": false, "mtp": false},
                "tie_word_embeddings": false, "quantization": quantization,
                "modules": records, "text_config": text_config
            })
        } else {
            serde_json::json!({
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5", "tie_word_embeddings": false,
                "quantization": quantization, "text_config": text_config
            })
        };
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    }

    /// `load_from_path` over this snapshot in a fresh temp directory.
    #[allow(
        clippy::unwrap_used,
        reason = "test fixture: tempdir failures should abort the test loudly"
    )]
    fn load(self) -> Result<Qwen3_5MoeText> {
        let dir = tempfile::tempdir().unwrap();
        self.write(dir.path());
        super::load_from_path(dir.path())
    }
}

/// The text of the error `pack` loads with, or the empty string.
fn load_error(pack: TinyPack) -> String {
    pack.load()
        .err()
        .map_or_else(String::new, |e| e.to_string())
}

/// Whether each layer the contract names was built as a Hadamard variant or
/// as the plain quantized one: head, embedding, the gated-delta-net and
/// attention projections, the MLP projections.
fn folded_layers(model: &Qwen3_5MoeText) -> Vec<(&'static str, bool)> {
    use super::super::decoder_layer::AttnBlock;
    use super::super::layers::Embedding;
    let lin = |l: &Linear| matches!(l, Linear::Hadamard { .. });
    let mut out = vec![
        ("lm_head", model.lm_head.as_ref().is_some_and(lin)),
        (
            "embed_tokens",
            matches!(model.embed_tokens, Embedding::Hadamard { .. }),
        ),
    ];
    for layer in &model.layers {
        match &layer.attn {
            AttnBlock::Full(a) => out.extend([
                ("q_proj", lin(&a.q_proj)),
                ("k_proj", lin(&a.k_proj)),
                ("v_proj", lin(&a.v_proj)),
                ("o_proj", lin(&a.o_proj)),
            ]),
            AttnBlock::Linear(g) => out.extend([
                ("in_proj_qkv", lin(&g.in_proj_qkv)),
                ("in_proj_z", lin(&g.in_proj_z)),
                ("out_proj", lin(&g.out_proj)),
            ]),
        }
        if let MlpBlock::Dense(m) = &layer.mlp {
            out.extend([
                ("gate_proj", lin(&m.gate_proj)),
                ("up_proj", lin(&m.up_proj)),
                ("down_proj", lin(&m.down_proj)),
            ]);
        }
    }
    out
}

/// The dtype of each gated-delta-net gate projection built as a plain
/// linear layer, or `None` for any other variant.
fn gate_projections(model: &Qwen3_5MoeText) -> Vec<(&'static str, Option<rmlx_mlx::Dtype>)> {
    use super::super::decoder_layer::AttnBlock;
    let plain = |l: &Linear| {
        if let Linear::Plain { weight } = l {
            Some(weight.dtype())
        } else {
            None
        }
    };
    model
        .layers
        .iter()
        .filter_map(|layer| match &layer.attn {
            AttnBlock::Linear(g) => Some([
                ("in_proj_a", plain(&g.in_proj_a)),
                ("in_proj_b", plain(&g.in_proj_b)),
            ]),
            AttnBlock::Full(_) => None,
        })
        .flatten()
        .collect()
}

/// The loader folds every contract module of a pack, those of the
/// gated-delta-net layer included, and no module of the plain snapshot. The
/// dense gate projections `in_proj_a` / `in_proj_b` stay plain F32 in both: a
/// bf16 cast of them raises the error of the delta-net gate above the bf16
/// activation floor (`docs/WEIGHT_QUANTS.md` §10.5).
#[test]
fn the_loader_folds_every_module_of_a_hadamard_pack_and_no_module_of_a_plain_snapshot() -> Result<()>
{
    for pack in [PACK, PLAIN, ATTENTION_ONLY] {
        let model = pack.load()?;
        let layers = folded_layers(&model);
        assert_eq!(
            layers.len(),
            tiny_modules(pack.gdn).len(),
            "every module was found"
        );
        for (layer, folded) in layers {
            assert_eq!(
                folded, pack.contract,
                "{layer}: folded is {folded}, pack is {}",
                pack.contract
            );
        }
        let gates = gate_projections(&model);
        assert_eq!(gates.len(), if pack.gdn { 2 } else { 0 });
        for (name, dtype) in gates {
            assert_eq!(dtype, Some(rmlx_mlx::Dtype::F32), "{name}: plain F32");
        }
    }
    Ok(())
}

#[test]
fn the_loader_refuses_a_pack_module_it_builds_no_layer_for_and_a_broken_contract() {
    let msg = load_error(TinyPack {
        extra_module: true,
        ..PACK
    });
    assert!(
        msg.contains("built no layer") && msg.contains(TINY_EXTRA),
        "an unbuilt contract module: {msg:?}"
    );
    let msg = load_error(TinyPack {
        broken_signs: true,
        ..PACK
    });
    assert!(
        msg.contains("model.layers.0.mlp.up_proj.signs") && msg.contains("differs"),
        "a sign tensor that is not the contract's vector: {msg:?}"
    );
}

/// The layers `load_from_path` builds from a pack against the oracle of the
/// pack's own codes, block and signs, at both input widths: the contract
/// reaches the fold, each width gets its own sign vector, a 1024-wide input
/// is two blocks, and the loader's bf16 scale cast keeps the values (the
/// fixture's scales are exact in bf16). Values with f32 input only: with
/// bf16 input and bf16 scales the MLX CPU `quantized_matmul` accumulates in
/// bf16, which is not the oracle's arithmetic. A bf16 input keeps its dtype.
#[test]
fn the_layers_loaded_from_a_pack_match_the_oracle() -> Result<()> {
    use super::super::decoder_layer::AttnBlock;
    use super::super::layers::layers_tests::{check_embedding, check_linear};
    let model = PACK.load()?;
    let f32_only = [rmlx_mlx::Dtype::F32];
    let (Some(AttnBlock::Linear(gdn)), Some(AttnBlock::Full(attn))) = (
        model.layers.first().map(|l| &l.attn),
        model.layers.get(1).map(|l| &l.attn),
    ) else {
        panic!("layer 0 is a gated delta net, layer 1 full attention");
    };
    let Some(MlpBlock::Dense(mlp)) = model.layers.first().map(|l| &l.mlp) else {
        panic!("the synthetic layer is dense");
    };
    for (path, built) in [
        ("model.layers.1.self_attn.q_proj", &attn.q_proj),
        ("model.layers.0.linear_attn.in_proj_qkv", &gdn.in_proj_qkv),
        ("model.layers.0.linear_attn.out_proj", &gdn.out_proj),
        ("model.layers.0.mlp.down_proj", &mlp.down_proj),
    ] {
        let folded = tiny_folded(path);
        let s = tiny_signs(folded.width);
        check_linear(path, built, &folded, &s, TINY_BLOCK, &[2], 5300, &f32_only)?;
        let x = Array::from_f32_slice(&uniform(5310, 2 * folded.width), &[2, folded.width as i32])?
            .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
        let y = built.forward(&x, rmlx_mlx::Device::Cpu)?;
        assert_eq!(
            y.dtype(),
            rmlx_mlx::Dtype::Bf16,
            "{path}: a bf16 activation stays bf16 through the folded layer"
        );
    }
    let path = "model.embed_tokens";
    check_embedding(
        path,
        &model.embed_tokens,
        &tiny_folded(path),
        &tiny_signs(TINY_WIDTH),
        TINY_BLOCK,
        &[3, 0, 2, 1],
    )
}

#[test]
fn the_loader_refuses_a_pack_on_another_base_model_type() {
    let msg = load_error(TinyPack {
        base_model_type: "qwen3_5_moe",
        ..PACK
    });
    assert!(
        msg.contains("base_model_type") && msg.contains("qwen3_5_moe"),
        "a pack whose base model type the loader does not fold: {msg:?}"
    );
}

#[test]
fn the_loader_refuses_a_pack_with_a_sparse_moe_layer() {
    let msg = load_error(TinyPack {
        moe_witness: true,
        ..PACK
    });
    assert!(
        msg.contains("Hadamard pack with a sparse MoE layer"),
        "a pack layer with a switch-MLP witness: {msg:?}"
    );
}

#[test]
fn a_hadamard_verifier_refuses_the_restricted_vocab_head() -> Result<()> {
    let verifier = crate::arch::Architecture::Qwen3_5Moe(PACK.load()?);
    let r = verifier.refuse_restricted_vocab_head();
    assert!(
        matches!(&r, Err(Error::Model(m)) if m.contains("Hadamard")),
        "a pack verifier: {r:?}"
    );
    let plain = crate::arch::Architecture::Qwen3_5Moe(PLAIN.load()?);
    let r = plain.refuse_restricted_vocab_head();
    assert!(r.is_ok(), "control, the plain snapshot: {r:?}");
    Ok(())
}

#[cfg(test)]
#[path = "pack_serve_tests.rs"]
mod pack_serve_tests;

#[cfg(test)]
#[path = "first_logprobs_tests.rs"]
mod first_logprobs_tests;
