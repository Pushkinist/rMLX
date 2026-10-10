//! `HadamardContract::read` on a synthetic pack: block 512, input widths 1024
//! and 1536, three forward modules and one inverse (embedding) module, under
//! the mlx-vlm tensor prefix `language_model.`.
//!
//! The observable is the parsed contract, and for each broken fact the error
//! variant with the file and key it names. Every case changes one fact of the
//! valid pack; a case that passes, or that is refused for another fact, is red.

use std::path::Path;

use serde_json::{json, Value};

use super::{HadamardContract, HadamardContractError};

const PREFIX: &str = "language_model.";
const CONFIG: &str = "config.json";
const CONTRACT: &str = "hadamard.json";
const INDEX: &str = "model.safetensors.index.json";
const SHARD: &str = "model.safetensors";
/// Sorts after `SHARD`, so a scan that stops at the first shard misses it.
const SHARD_2: &str = "shard-2.safetensors";
/// Sorts before `SHARD` and is written after every other shard.
const SHARD_0: &str = "a-shard.safetensors";
const BLOCK: usize = 512;
const NARROW: usize = 2 * BLOCK;
const WIDE: usize = 3 * BLOCK;

fn sign_vector(width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| {
            if (i * 7 + width).is_multiple_of(3) {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

/// `(path, embedding, input width in blocks)` of each module, in `modules`
/// order.
const MODULES: [(&str, bool, usize); 4] = [
    ("lm_head", false, 2),
    ("model.layers.0.mlp.up_proj", false, 2),
    ("model.layers.0.mlp.down_proj", false, 3),
    ("model.embed_tokens", true, 2),
];

struct Tensor {
    shard: &'static str,
    name: String,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn f32_tensor(name: &str, values: &[f32]) -> Tensor {
    Tensor {
        shard: SHARD,
        name: name.to_owned(),
        dtype: safetensors::Dtype::F32,
        shape: vec![values.len()],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// A two-byte-per-value tensor of zeros, standing in for affine `.scales`.
fn f16_tensor(name: &str, shape: &[usize]) -> Tensor {
    Tensor {
        shard: SHARD,
        name: name.to_owned(),
        dtype: safetensors::Dtype::F16,
        shape: shape.to_vec(),
        data: vec![0; 2 * shape.iter().product::<usize>()],
    }
}

fn signs_name(module: &str) -> String {
    format!("{PREFIX}{module}.signs")
}

struct Pack {
    config: Value,
    contract: Value,
    tensors: Vec<Tensor>,
}

impl Pack {
    fn valid() -> Self {
        Self::with_block(BLOCK)
    }

    /// The valid pack at `block`, with input widths `2 * block` and `3 * block`.
    fn with_block(block: usize) -> Self {
        let (narrow, wide) = (2 * block, 3 * block);
        let modules: Vec<Value> = MODULES
            .iter()
            .map(|(path, embedding, _)| {
                json!({"path": path, "block": block, "embedding": embedding, "dtype": "float16"})
            })
            .collect();
        let names = |embedding: bool| -> Vec<String> {
            MODULES
                .iter()
                .filter(|m| m.1 == embedding)
                .map(|m| format!("{PREFIX}{}.weight", m.0))
                .collect()
        };
        let mut tensors: Vec<Tensor> = MODULES
            .iter()
            .map(|(path, _, n)| f32_tensor(&signs_name(path), &sign_vector(n * block)))
            .collect();
        tensors.extend(MODULES.iter().map(|(path, _, n)| {
            f16_tensor(&format!("{PREFIX}{path}.scales"), &[4, n * block / 128])
        }));
        tensors.push(f32_tensor("language_model.model.norm.weight", &[1.0; 8]));
        Pack {
            config: json!({
                "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5",
                "hadamard_config": CONTRACT,
                "gdn_activation_layout": "grouped",
                "components": {"text": true, "vision": false, "mtp": false},
                "tie_word_embeddings": false,
                "text_config": {"tie_word_embeddings": false},
                "modules": modules,
            }),
            contract: json!({
                "prism.hadamard.version": 1,
                "prism.hadamard.block_size": block,
                "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
                "prism.hadamard.axis": "input-last-dimension",
                "prism.hadamard.sign_mode": "explicit",
                "prism.hadamard.weight_names": names(false),
                "prism.hadamard.inverse_weight_names": names(true),
                "prism.hadamard.sign_widths": [narrow, wide],
                "prism.hadamard.sign_values": ([sign_vector(narrow), sign_vector(wide)].concat()),
                "prism.hadamard.gdn_v_grouped": true,
            }),
            tensors,
        }
    }

    /// Writes the files. A pack with tensors in more than one shard also gets
    /// a `model.safetensors.index.json` naming the shard of each tensor;
    /// `SHARD_0` is written last.
    fn write(&self, dir: &Path) {
        std::fs::write(dir.join(CONFIG), self.config.to_string()).unwrap();
        std::fs::write(dir.join(CONTRACT), self.contract.to_string()).unwrap();
        let weight_map: serde_json::Map<String, Value> = self
            .tensors
            .iter()
            .map(|t| (t.name.clone(), json!(t.shard)))
            .collect();
        if self.tensors.iter().any(|t| t.shard != SHARD) {
            let index = json!({"metadata": {}, "weight_map": weight_map});
            std::fs::write(dir.join(INDEX), index.to_string()).unwrap();
        }
        for shard in [SHARD, SHARD_2, SHARD_0] {
            let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = self
                .tensors
                .iter()
                .filter(|t| t.shard == shard)
                .map(|t| {
                    let view =
                        safetensors::tensor::TensorView::new(t.dtype, t.shape.clone(), &t.data)
                            .unwrap();
                    (t.name.clone(), view)
                })
                .collect();
            if !views.is_empty() {
                std::fs::write(
                    dir.join(shard),
                    safetensors::serialize(views, None).unwrap(),
                )
                .unwrap();
            }
        }
    }

    fn read(&self) -> Result<Option<HadamardContract>, HadamardContractError> {
        let dir = tempfile::tempdir().unwrap();
        self.write(dir.path());
        HadamardContract::read(dir.path())
    }

    fn key(&mut self, key: &str) -> &mut Value {
        &mut self.contract[format!("prism.hadamard.{key}")]
    }

    fn drop_key(&mut self, key: &str) {
        self.contract
            .as_object_mut()
            .unwrap()
            .remove(&format!("prism.hadamard.{key}"));
    }

    fn signs_tensor(&mut self, module: &str) -> &mut Tensor {
        let name = signs_name(module);
        self.tensors.iter_mut().find(|t| t.name == name).unwrap()
    }
}

/// `(variant, file, key)` of an error: what each refusal case asserts on.
fn fact(e: &HadamardContractError) -> (&'static str, &str, &str) {
    match e {
        HadamardContractError::Unreadable { file, .. } => ("Unreadable", file, ""),
        HadamardContractError::Missing { file, key } => ("Missing", file, key),
        HadamardContractError::Unsupported { file, key, .. } => ("Unsupported", file, key),
        HadamardContractError::Invalid { file, key, .. } => ("Invalid", file, key),
    }
}

#[test]
fn a_valid_pack_parses_into_its_facts() {
    let c = Pack::valid()
        .read()
        .expect("the valid pack is accepted")
        .expect("a pack, not a plain snapshot");
    assert_eq!(c.block, BLOCK);
    assert_eq!(c.base_model_type, "qwen3_5");
    assert_eq!(c.tensor_prefix, PREFIX);
    assert_eq!(
        c.signs
            .iter()
            .map(|(w, s)| (*w, s.clone()))
            .collect::<Vec<_>>(),
        vec![(NARROW, sign_vector(NARROW)), (WIDE, sign_vector(WIDE))]
    );
    let forward: Vec<(&str, usize)> = c.forward.iter().map(|(k, w)| (k.as_str(), *w)).collect();
    assert_eq!(
        forward,
        vec![
            ("language_model.lm_head", NARROW),
            ("language_model.model.layers.0.mlp.down_proj", WIDE),
            ("language_model.model.layers.0.mlp.up_proj", NARROW),
        ]
    );
    let inverse: Vec<(&str, usize)> = c.inverse.iter().map(|(k, w)| (k.as_str(), *w)).collect();
    assert_eq!(inverse, vec![("language_model.model.embed_tokens", NARROW)]);
}

#[test]
fn a_snapshot_without_hadamard_config_is_not_a_pack() {
    let mut pack = Pack::valid();
    pack.config
        .as_object_mut()
        .unwrap()
        .remove("hadamard_config");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(CONFIG), pack.config.to_string()).unwrap();
    assert_eq!(HadamardContract::read(dir.path()), Ok(None));
}

/// A `.signs` tensor with no `hadamard_config` in `config.json` is a pack
/// whose contract cannot be read: it is refused by the shard and the tensor
/// that hold it, in any visible `*.safetensors` file of the directory. A
/// hidden file, and a file that does not parse, feed no weights and change
/// nothing.
#[test]
fn a_sign_tensor_without_hadamard_config_is_refused_by_its_shard_and_name() {
    let unpacked = |pack: &mut Pack| {
        pack.config
            .as_object_mut()
            .unwrap()
            .remove("hadamard_config");
    };
    let mut pack = Pack::valid();
    unpacked(&mut pack);
    let e = pack.read().unwrap_err();
    let (variant, file, key) = fact(&e);
    assert_eq!((variant, file), ("Invalid", SHARD), "{e}");
    assert_eq!(key.rsplit('.').next(), Some("signs"), "{e}");
    assert!(e.to_string().contains("hadamard_config"), "{e}");

    let mut pack = Pack::valid();
    unpacked(&mut pack);
    pack.tensors.retain(|t| t.name == signs_name(DOWN));
    pack.signs_tensor(DOWN).shard = SHARD_2;
    let e = pack.read().unwrap_err();
    assert_eq!(fact(&e), ("Invalid", SHARD_2, signs_name(DOWN).as_str()));

    let dir = tempfile::tempdir().unwrap();
    let mut pack = Pack::valid();
    unpacked(&mut pack);
    pack.tensors
        .retain(|t| t.name.rsplit('.').next() != Some("signs"));
    pack.write(dir.path());
    let signs = f32_tensor(&signs_name(DOWN), &[1.0; 4]);
    let view = safetensors::tensor::TensorView::new(signs.dtype, signs.shape, &signs.data).unwrap();
    let hidden = safetensors::serialize([(signs.name.as_str(), view)], None).unwrap();
    std::fs::write(dir.path().join("._model.safetensors"), hidden).unwrap();
    std::fs::write(dir.path().join("stray.safetensors"), b"not a header").unwrap();
    assert_eq!(HadamardContract::read(dir.path()), Ok(None));

    let mut pack = Pack::valid();
    unpacked(&mut pack);
    pack.tensors
        .retain(|t| t.name.rsplit('.').next() != Some("signs"));
    pack.tensors
        .push(f32_tensor("language_model.model.num_signs", &[1.0; 4]));
    assert_eq!(
        pack.read(),
        Ok(None),
        "a name whose last component only ends in `signs`"
    );
}

#[test]
fn a_snapshot_without_config_json_is_unreadable() {
    let dir = tempfile::tempdir().unwrap();
    let e = HadamardContract::read(dir.path()).unwrap_err();
    assert_eq!(fact(&e), ("Unreadable", CONFIG, ""));
}

/// The contract is checked on the shards the load reads: the shards that
/// `model.safetensors.index.json` names, or `model.safetensors` alone when
/// there is no index. A file that the load does not read is never opened,
/// whether it is hidden (`._model.safetensors`, an AppleDouble copy) or
/// visible and broken. A named shard that does not parse is refused, and so
/// is an index that does not parse. With no index the refusal names
/// `model.safetensors`, the file the load tries.
#[test]
fn the_shards_read_are_the_shards_that_load() {
    use std::os::unix::fs::PermissionsExt;
    let lock = |path: &Path, mode| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    };
    for split in [false, true] {
        let mut pack = Pack::valid();
        if split {
            pack.signs_tensor(DOWN).shard = SHARD_2;
        }
        let dir = tempfile::tempdir().unwrap();
        pack.write(dir.path());
        assert_eq!(dir.path().join(INDEX).exists(), split);
        let hidden = dir.path().join("._model.safetensors");
        let broken = dir.path().join("broken.safetensors");
        for unread in [&hidden, &broken] {
            std::fs::write(unread, b"not a header").unwrap();
            lock(unread, 0o000);
        }
        let r = HadamardContract::read(dir.path());
        for unread in [&hidden, &broken] {
            lock(unread, 0o600);
        }
        let c = r.unwrap_or_else(|e| panic!("split {split}: {e}"));
        let c = c.expect("a pack, not a plain snapshot");
        assert_eq!(
            c.forward.get(&format!("{PREFIX}{DOWN}")),
            Some(&WIDE),
            "split {split}"
        );
    }

    let mut pack = Pack::valid();
    pack.signs_tensor(DOWN).shard = SHARD_2;
    let dir = tempfile::tempdir().unwrap();
    pack.write(dir.path());
    std::fs::write(dir.path().join(SHARD_2), b"not a header").unwrap();
    let e = HadamardContract::read(dir.path()).unwrap_err();
    assert_eq!(fact(&e), ("Unreadable", SHARD_2, ""));

    std::fs::write(dir.path().join(INDEX), b"not an index").unwrap();
    let e = HadamardContract::read(dir.path()).unwrap_err();
    assert_eq!(fact(&e), ("Unreadable", INDEX, ""));

    let dir = tempfile::tempdir().unwrap();
    Pack::valid().write(dir.path());
    std::fs::write(dir.path().join(SHARD), b"not a header").unwrap();
    let e = HadamardContract::read(dir.path()).unwrap_err();
    assert_eq!(
        fact(&e),
        ("Unreadable", SHARD, ""),
        "no index: the shard tried"
    );

    let empty = tempfile::tempdir().unwrap();
    let mut pack = Pack::valid();
    pack.tensors.clear();
    pack.write(empty.path());
    let e = HadamardContract::read(empty.path()).unwrap_err();
    assert_eq!(fact(&e), ("Unreadable", SHARD, ""), "no index, no shard");
}

/// A shard that cannot be opened is refused by its file name, and the error
/// text holds no part of the snapshot's directory path.
#[test]
fn an_unopenable_shard_is_unreadable_and_the_error_names_no_directory() {
    use std::os::unix::fs::PermissionsExt;
    let mut pack = Pack::valid();
    pack.signs_tensor(DOWN).shard = SHARD_2;
    let dir = tempfile::tempdir().unwrap();
    pack.write(dir.path());
    let locked = dir.path().join(SHARD_2);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    let r = HadamardContract::read(dir.path());
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o600)).unwrap();
    let e = r.unwrap_err();
    assert_eq!(
        e,
        HadamardContractError::Unreadable {
            file: SHARD_2.to_owned(),
            why: "cannot open or map the shard".to_owned(),
        }
    );
    let text = e.to_string();
    let dir_text = dir.path().display().to_string();
    assert!(
        !text.contains(&dir_text),
        "the error names the snapshot directory: {text}"
    );
}

/// Changes that keep the pack valid: each block the vendor runtime validated
/// (`runtime/artifact.py`; listed here, not read from the parser), a
/// quantized tensor outside the language-model prefix (the vision tower), and
/// no `gdn_activation_layout` key.
#[test]
fn each_valid_variation_is_accepted() {
    for block in [512, 1024, 2048, 4096] {
        let c = Pack::with_block(block)
            .read()
            .unwrap_or_else(|e| panic!("block {block}: {e}"))
            .expect("a pack, not a plain snapshot");
        assert_eq!(c.block, block);
        assert_eq!(
            c.signs.keys().copied().collect::<Vec<_>>(),
            vec![2 * block, 3 * block]
        );
    }

    let mut pack = Pack::valid();
    pack.tensors
        .push(f16_tensor("vision_tower.blocks.0.attn.qkv.scales", &[4, 8]));
    let r = pack.read();
    assert!(matches!(r, Ok(Some(_))), "vision tower scales: {r:?}");

    let mut pack = Pack::valid();
    pack.config
        .as_object_mut()
        .unwrap()
        .remove("gdn_activation_layout");
    let r = pack.read();
    assert!(matches!(r, Ok(Some(_))), "no gdn_activation_layout: {r:?}");
}

/// The vendor runtime writes `block` 0 for a quantized linear it did not
/// fold, with no `.signs`. rMLX does not implement such a module, and the
/// refusal says so.
#[test]
fn an_unfolded_module_is_refused_as_not_implemented() {
    let mut pack = Pack::valid();
    pack.config["modules"][1]["block"] = json!(0);
    let e = pack.read().unwrap_err();
    assert_eq!(fact(&e), ("Invalid", CONFIG, "modules[1].block"));
    let text = e.to_string();
    assert!(
        text.contains("block 0 (unfolded module) is not implemented"),
        "{text}"
    );
}

type Mutation = fn(&mut Pack);
type Want = (&'static str, String, String);
type Case = (&'static str, Mutation, Want);

/// A refusal by contract key `prism.hadamard.<key>`.
fn h(variant: &'static str, key: &str) -> Want {
    (variant, CONTRACT.into(), format!("prism.hadamard.{key}"))
}

/// A refusal by `config.json` key.
fn c(variant: &'static str, key: &str) -> Want {
    (variant, CONFIG.into(), key.into())
}

/// A refusal by shard file and tensor name.
fn s(variant: &'static str, file: &str, tensor: &str) -> Want {
    (variant, file.into(), tensor.into())
}

fn push_name(p: &mut Pack, list: &str, name: &str) {
    p.key(list).as_array_mut().unwrap().push(json!(name));
}

const DOWN: &str = "model.layers.0.mlp.down_proj";
const GATE: &str = "model.layers.0.mlp.gate_proj";

#[allow(
    clippy::too_many_lines,
    reason = "one entry per broken fact: the table is the test"
)]
fn contract_facts() -> Vec<Case> {
    vec![
        (
            "version 2",
            |p| *p.key("version") = json!(2),
            h("Unsupported", "version"),
        ),
        (
            "version 0",
            |p| *p.key("version") = json!(0),
            h("Unsupported", "version"),
        ),
        (
            "version as a string",
            |p| *p.key("version") = json!("1"),
            h("Invalid", "version"),
        ),
        (
            "version missing",
            |p| p.drop_key("version"),
            h("Missing", "version"),
        ),
        (
            "transform unknown",
            |p| *p.key("transform") = json!("random-orthogonal"),
            h("Unsupported", "transform"),
        ),
        (
            "transform missing",
            |p| p.drop_key("transform"),
            h("Missing", "transform"),
        ),
        (
            "axis other",
            |p| *p.key("axis") = json!("output-first-dimension"),
            h("Unsupported", "axis"),
        ),
        ("axis missing", |p| p.drop_key("axis"), h("Missing", "axis")),
        (
            "sign mode seeded",
            |p| *p.key("sign_mode") = json!("seeded"),
            h("Unsupported", "sign_mode"),
        ),
        (
            "sign mode missing",
            |p| p.drop_key("sign_mode"),
            h("Missing", "sign_mode"),
        ),
        (
            "gdn_v_grouped false",
            |p| *p.key("gdn_v_grouped") = json!(false),
            h("Unsupported", "gdn_v_grouped"),
        ),
        (
            "gdn_v_grouped missing reads as false",
            |p| p.drop_key("gdn_v_grouped"),
            h("Unsupported", "gdn_v_grouped"),
        ),
        (
            "gdn_v_grouped as the string false",
            |p| *p.key("gdn_v_grouped") = json!("false"),
            h("Invalid", "gdn_v_grouped"),
        ),
        (
            "block missing",
            |p| p.drop_key("block_size"),
            h("Missing", "block_size"),
        ),
        (
            "block 6",
            |p| *p.key("block_size") = json!(6),
            h("Unsupported", "block_size"),
        ),
        (
            "block 0",
            |p| *p.key("block_size") = json!(0),
            h("Unsupported", "block_size"),
        ),
        (
            "block 256, a power of two the vendor did not validate",
            |p| *p.key("block_size") = json!(256),
            h("Unsupported", "block_size"),
        ),
        (
            "block as a string",
            |p| *p.key("block_size") = json!("512"),
            h("Invalid", "block_size"),
        ),
        (
            "width not a multiple of the block",
            |p| {
                *p.key("sign_widths") = json!([NARROW, 1000]);
                *p.key("sign_values") = json!([sign_vector(NARROW), sign_vector(1000)].concat());
            },
            h("Invalid", "sign_widths"),
        ),
        (
            "width 0",
            |p| *p.key("sign_widths") = json!([NARROW, WIDE, 0]),
            h("Invalid", "sign_widths"),
        ),
        (
            "no width",
            |p| {
                *p.key("sign_widths") = json!([]);
                *p.key("sign_values") = json!([]);
            },
            h("Invalid", "sign_widths"),
        ),
        (
            "widths not an array",
            |p| *p.key("sign_widths") = json!("1024"),
            h("Invalid", "sign_widths"),
        ),
        (
            "width twice",
            |p| {
                *p.key("sign_widths") = json!([NARROW, NARROW]);
                *p.key("sign_values") = json!([sign_vector(NARROW), sign_vector(NARROW)].concat());
            },
            h("Invalid", "sign_widths"),
        ),
        (
            "widths that overflow their sum",
            |p| {
                *p.key("sign_widths") = json!([1u64 << 63, (1u64 << 63) + 1024]);
            },
            h("Invalid", "sign_widths"),
        ),
        (
            "widths missing",
            |p| p.drop_key("sign_widths"),
            h("Missing", "sign_widths"),
        ),
        (
            "a sign value of 0.5",
            |p| p.key("sign_values")[3] = json!(0.5),
            h("Invalid", "sign_values"),
        ),
        (
            "one sign value short",
            |p| {
                p.key("sign_values").as_array_mut().unwrap().pop();
            },
            h("Invalid", "sign_values"),
        ),
        (
            "one sign value too many",
            |p| {
                p.key("sign_values")
                    .as_array_mut()
                    .unwrap()
                    .push(json!(1.0));
            },
            h("Invalid", "sign_values"),
        ),
        (
            "sign values not an array",
            |p| *p.key("sign_values") = json!({}),
            h("Invalid", "sign_values"),
        ),
        (
            "sign values missing",
            |p| p.drop_key("sign_values"),
            h("Missing", "sign_values"),
        ),
        (
            "weight names missing",
            |p| p.drop_key("weight_names"),
            h("Missing", "weight_names"),
        ),
        (
            "a weight name twice",
            |p| {
                let first = p.key("weight_names")[0].clone();
                p.key("weight_names").as_array_mut().unwrap().push(first);
            },
            h("Invalid", "weight_names"),
        ),
        (
            "forward and inverse overlap",
            |p| {
                let embed = p.key("inverse_weight_names")[0].clone();
                p.key("weight_names").as_array_mut().unwrap().push(embed);
            },
            h("Invalid", "inverse_weight_names"),
        ),
        (
            "inverse names missing leaves the embedding record unmatched",
            |p| p.drop_key("inverse_weight_names"),
            c("Invalid", "modules[3].path"),
        ),
        (
            "a weight name with no module",
            |p| {
                push_name(
                    p,
                    "weight_names",
                    "language_model.model.layers.0.mlp.gate_proj.weight",
                );
            },
            h("Invalid", "weight_names"),
        ),
        (
            "a weight name that only ends like the first module's path",
            |p| push_name(p, "weight_names", "language_model.model.xlm_head.weight"),
            h("Invalid", "weight_names"),
        ),
        (
            "two weight names that end in the first module's path",
            |p| push_name(p, "weight_names", "language_model.extra.lm_head.weight"),
            c("Invalid", "modules[0].path"),
        ),
        (
            "a weight name under another prefix",
            |p| p.key("weight_names")[1] = json!("other.model.layers.0.mlp.up_proj.weight"),
            c("Invalid", "modules[1].path"),
        ),
        (
            "contract not a JSON object",
            |p| p.contract = json!("not an object"),
            ("Invalid", CONTRACT.into(), "prism.hadamard".into()),
        ),
    ]
}

#[allow(
    clippy::too_many_lines,
    reason = "one entry per broken fact: the table is the test"
)]
fn config_and_shard_facts() -> Vec<Case> {
    vec![
        (
            "contract file absent",
            |p| p.config["hadamard_config"] = json!("absent.json"),
            ("Unreadable", "absent.json".into(), String::new()),
        ),
        (
            "contract file outside the snapshot",
            |p| p.config["hadamard_config"] = json!("../hadamard.json"),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file by absolute path",
            |p| p.config["hadamard_config"] = json!("/tmp/hadamard.json"),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file in a subdirectory",
            |p| p.config["hadamard_config"] = json!("sub/hadamard.json"),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file named by the empty string",
            |p| p.config["hadamard_config"] = json!(""),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file named `.`",
            |p| p.config["hadamard_config"] = json!("."),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file named `..`",
            |p| p.config["hadamard_config"] = json!(".."),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file named with a trailing slash",
            |p| p.config["hadamard_config"] = json!("hadamard.json/"),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file named with a trailing `/.`",
            |p| p.config["hadamard_config"] = json!("hadamard.json/."),
            c("Invalid", "hadamard_config"),
        ),
        (
            "contract file named with a NUL byte",
            |p| p.config["hadamard_config"] = json!("hadamard\u{0}.json"),
            c("Invalid", "hadamard_config"),
        ),
        (
            "hadamard_config not a string",
            |p| p.config["hadamard_config"] = json!(5),
            c("Invalid", "hadamard_config"),
        ),
        (
            "base_model_type missing",
            |p| {
                p.config.as_object_mut().unwrap().remove("base_model_type");
            },
            c("Missing", "base_model_type"),
        ),
        (
            "base_model_type not a string",
            |p| p.config["base_model_type"] = json!(5),
            c("Invalid", "base_model_type"),
        ),
        (
            "mtp true",
            |p| p.config["components"]["mtp"] = json!(true),
            c("Unsupported", "components.mtp"),
        ),
        (
            "mtp as a string",
            |p| p.config["components"]["mtp"] = json!("yes"),
            c("Invalid", "components.mtp"),
        ),
        (
            "tied embeddings at the top level",
            |p| p.config["tie_word_embeddings"] = json!(true),
            c("Unsupported", "tie_word_embeddings"),
        ),
        (
            "tied embeddings in text_config",
            |p| p.config["text_config"]["tie_word_embeddings"] = json!(true),
            c("Unsupported", "text_config.tie_word_embeddings"),
        ),
        (
            "tie_word_embeddings not a bool",
            |p| p.config["tie_word_embeddings"] = json!("no"),
            c("Invalid", "tie_word_embeddings"),
        ),
        (
            "GDN activations not grouped",
            |p| p.config["gdn_activation_layout"] = json!("tiled"),
            c("Unsupported", "gdn_activation_layout"),
        ),
        (
            "GDN activation layout not a string",
            |p| p.config["gdn_activation_layout"] = json!(1),
            c("Invalid", "gdn_activation_layout"),
        ),
        (
            "modules missing",
            |p| {
                p.config.as_object_mut().unwrap().remove("modules");
            },
            c("Missing", "modules"),
        ),
        (
            "modules empty",
            |p| p.config["modules"] = json!([]),
            c("Invalid", "modules"),
        ),
        (
            "a module with no embedding flag",
            |p| {
                p.config["modules"][1]
                    .as_object_mut()
                    .unwrap()
                    .remove("embedding");
            },
            c("Missing", "modules[1].embedding"),
        ),
        (
            "module dtype bfloat16",
            |p| p.config["modules"][1]["dtype"] = json!("bfloat16"),
            c("Unsupported", "modules[1].dtype"),
        ),
        (
            "module block 1024",
            |p| p.config["modules"][1]["block"] = json!(1024),
            c("Invalid", "modules[1].block"),
        ),
        (
            "module path twice",
            |p| p.config["modules"][2]["path"] = json!("model.layers.0.mlp.up_proj"),
            c("Invalid", "modules[2].path"),
        ),
        (
            "module the contract does not name",
            |p| {
                p.config["modules"].as_array_mut().unwrap().push(
                    json!({"path": GATE, "block": BLOCK, "embedding": false, "dtype": "float16"}),
                );
            },
            c("Invalid", "modules[4].path"),
        ),
        (
            "linear module flagged as embedding",
            |p| p.config["modules"][1]["embedding"] = json!(true),
            c("Invalid", "modules[1].path"),
        ),
        (
            "signs tensor missing",
            |p| p.tensors.retain(|t| t.name != signs_name(DOWN)),
            s("Missing", SHARD, &signs_name(DOWN)),
        ),
        (
            "signs tensor f16",
            |p| {
                let t = p.signs_tensor(DOWN);
                t.dtype = safetensors::Dtype::F16;
                t.data.truncate(2 * WIDE);
            },
            s("Invalid", SHARD, &signs_name(DOWN)),
        ),
        (
            "signs tensor of a width with no vector",
            |p| {
                *p.signs_tensor(DOWN) = f32_tensor(&signs_name(DOWN), &sign_vector(2048));
            },
            s("Invalid", SHARD, &signs_name(DOWN)),
        ),
        (
            "signs tensor two-dimensional with a valid last axis",
            |p| p.signs_tensor(DOWN).shape = vec![1, WIDE],
            s("Invalid", SHARD, &signs_name(DOWN)),
        ),
        (
            "signs tensor differs from the contract",
            |p| p.signs_tensor(DOWN).data[3] ^= 0x80,
            s("Invalid", SHARD, &signs_name(DOWN)),
        ),
        (
            "signs tensor of no module",
            |p| {
                p.tensors
                    .push(f32_tensor(&signs_name(GATE), &sign_vector(NARROW)));
            },
            s("Invalid", SHARD, &signs_name(GATE)),
        ),
        (
            "signs tensor of no module in a second shard",
            |p| {
                let mut t = f32_tensor(&signs_name(GATE), &sign_vector(NARROW));
                t.shard = SHARD_2;
                p.tensors.push(t);
            },
            s("Invalid", SHARD_2, &signs_name(GATE)),
        ),
        (
            "the same signs tensor in two shards",
            |p| {
                let mut t = f32_tensor(&signs_name(DOWN), &sign_vector(WIDE));
                t.shard = SHARD_2;
                p.tensors.push(t);
            },
            s("Invalid", &format!("{SHARD}, {SHARD_2}"), &signs_name(DOWN)),
        ),
        (
            "the same signs tensor in a shard that sorts first and is written last",
            |p| {
                let mut t = f32_tensor(&signs_name(DOWN), &sign_vector(WIDE));
                t.shard = SHARD_0;
                p.tensors.push(t);
            },
            s("Invalid", &format!("{SHARD_0}, {SHARD}"), &signs_name(DOWN)),
        ),
        (
            "a quantized linear with no module record",
            |p| {
                p.tensors
                    .push(f16_tensor(&format!("{PREFIX}{GATE}.scales"), &[4, 8]));
            },
            s("Invalid", SHARD, &format!("{PREFIX}{GATE}.scales")),
        ),
    ]
}

#[test]
fn each_broken_fact_is_refused_by_name() {
    let mut wrong = Vec::new();
    let mut count = 0;
    for (what, mutate, (variant, file, key)) in
        contract_facts().into_iter().chain(config_and_shard_facts())
    {
        count += 1;
        let mut pack = Pack::valid();
        mutate(&mut pack);
        let want = (variant, file.as_str(), key.as_str());
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pack.read())) {
            Ok(Err(e)) if fact(&e) == want => {}
            Ok(got) => wrong.push(format!("{what}: want {want:?}, got {got:?}")),
            Err(_) => wrong.push(format!("{what}: want {want:?}, got a panic")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {count} cases not refused by the named fact:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
