//! A small synthetic Whisper snapshot for the weight-file tests: the same
//! tensors, written as `weights.safetensors` or as `weights.npz`.

use std::path::Path;

use crate::npz::npz_tests::{build_zip32_members, npy_bytes};
use crate::npz::WeightMap;
use rmlx_mlx::{Array, Dtype};

pub(super) const STATE: usize = 4;
pub(super) const TEXT_CTX: usize = 16;
const N_VOCAB: usize = 51_866;

/// One tensor: a safetensors dtype name, a shape and little-endian bytes.
pub(super) struct Tensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Tensor {
    pub(super) fn f32(name: &str, shape: &[usize], seed: usize) -> Self {
        let len: usize = shape.iter().product();
        let bytes = (0..len)
            .flat_map(|i| (((i * 7 + seed * 13) % 97) as f32 / 97.0 - 0.5).to_le_bytes())
            .collect();
        Self::new(name, "F32", shape, bytes)
    }

    pub(super) fn f16(name: &str, shape: &[usize], seed: usize) -> Self {
        let len: usize = shape.iter().product();
        let bytes = (0..len)
            .flat_map(|i| {
                let half = 0x3000_u16 + ((i * 7 + seed * 13) % 97) as u16;
                half.to_le_bytes()
            })
            .collect();
        Self::new(name, "F16", shape, bytes)
    }

    /// An `alignment_heads` entry of int64 (layer, head) pairs.
    pub(super) fn int64_pairs(pairs: &[(i64, i64)]) -> Self {
        let bytes = pairs
            .iter()
            .flat_map(|&(layer, head)| [layer, head])
            .flat_map(i64::to_le_bytes)
            .collect();
        Self::new("alignment_heads", "I64", &[pairs.len(), 2], bytes)
    }

    pub(super) fn new(name: &str, dtype: &'static str, shape: &[usize], bytes: Vec<u8>) -> Self {
        Self {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
            bytes,
        }
    }

    fn npy_descr(&self) -> &'static str {
        match self.dtype {
            "F16" => "<f2",
            "F32" => "<f4",
            "I64" => "<i8",
            "U32" => "<u4",
            "BOOL" => "|b1",
            "I8" => "|i1",
            other => panic!("no npy descr for {other}"),
        }
    }
}

/// The bytes of a safetensors file of `tensors`, in order.
pub(super) fn safetensors_bytes(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for t in tensors {
        header.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, offset + t.bytes.len()],
            }),
        );
        offset += t.bytes.len();
    }
    let mut header = serde_json::Value::Object(header).to_string().into_bytes();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}

/// The bytes of an npz file of `tensors`, in order.
pub(super) fn npz_bytes(tensors: &[Tensor]) -> Vec<u8> {
    let members: Vec<(String, Vec<u8>)> = tensors
        .iter()
        .map(|t| {
            let shape = match t.shape.as_slice() {
                [n] => format!("({n},)"),
                dims => format!(
                    "({})",
                    dims.iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            };
            (
                format!("{}.npy", t.name),
                npy_bytes(t.npy_descr(), &shape, &t.bytes),
            )
        })
        .collect();
    let refs: Vec<(&str, &[u8])> = members
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    build_zip32_members(&refs)
}

pub(super) fn write_safetensors(dir: &Path, tensors: &[Tensor]) {
    std::fs::write(dir.join("weights.safetensors"), safetensors_bytes(tensors)).unwrap();
}

pub(super) fn write_npz(dir: &Path, tensors: &[Tensor]) {
    std::fs::write(dir.join("weights.npz"), npz_bytes(tensors)).unwrap();
}

/// A `config.json` of the large-v3 sizes except the four given.
pub(super) fn config_json(
    n_mels: usize,
    n_vocab: usize,
    n_audio_ctx: usize,
    n_text_layer: usize,
) -> serde_json::Value {
    serde_json::json!({
        "n_mels": n_mels,
        "n_audio_ctx": n_audio_ctx,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": n_vocab,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": n_text_layer,
    })
}

/// A `config.json` for the tensors of this module: the large-v3 vocabulary
/// and audio context, a state of [`STATE`], two heads, a text context of
/// [`TEXT_CTX`] and the given number of blocks on each side.
pub(super) fn tiny_config_json(n_audio_layer: usize, n_text_layer: usize) -> serde_json::Value {
    let mut config = config_json(128, N_VOCAB, 1500, n_text_layer);
    for (field, value) in [
        ("n_audio_state", STATE),
        ("n_audio_head", 2),
        ("n_audio_layer", n_audio_layer),
        ("n_text_ctx", TEXT_CTX),
        ("n_text_state", STATE),
        ("n_text_head", 2),
    ] {
        config[field] = value.into();
    }
    config
}

/// The f32 tensors as a weight map.
pub(super) fn weight_map(tensors: &[Tensor]) -> WeightMap {
    tensors
        .iter()
        .map(|t| {
            let dims: Vec<i32> = t.shape.iter().map(|&dim| dim as i32).collect();
            let array = Array::from_bytes(&t.bytes, &dims, Dtype::F32).unwrap();
            (t.name.clone(), array)
        })
        .collect()
}

/// The tensors of one block: the given attentions, their norms, the MLP.
fn block_tensors(block: &str, attentions: &[&str], seed: &mut usize) -> Vec<Tensor> {
    let hidden = 2 * STATE;
    let mut out = Vec::new();
    let mut put = |name: String, shape: &[usize]| {
        *seed += 1;
        out.push(Tensor::f32(&name, shape, *seed));
    };
    for attention in attentions {
        for linear in ["query", "key", "value", "out"] {
            put(
                format!("{block}.{attention}.{linear}.weight"),
                &[STATE, STATE],
            );
            if linear != "key" {
                put(format!("{block}.{attention}.{linear}.bias"), &[STATE]);
            }
        }
    }
    let norms = ["attn_ln", "cross_attn_ln", "mlp_ln"];
    let used = norms
        .iter()
        .filter(|norm| **norm != "cross_attn_ln" || attentions.contains(&"cross_attn"));
    for norm in used {
        put(format!("{block}.{norm}.weight"), &[STATE]);
        put(format!("{block}.{norm}.bias"), &[STATE]);
    }
    put(format!("{block}.mlp1.weight"), &[hidden, STATE]);
    put(format!("{block}.mlp1.bias"), &[hidden]);
    put(format!("{block}.mlp2.weight"), &[STATE, hidden]);
    put(format!("{block}.mlp2.bias"), &[STATE]);
    out
}

/// Every tensor that the text decoder reads for [`tiny_config_json`] with
/// `blocks` text layers.
pub(super) fn decoder_tensors(blocks: usize) -> Vec<Tensor> {
    let mut seed = 100;
    let mut out = vec![
        Tensor::f32("decoder.token_embedding.weight", &[N_VOCAB, STATE], 1),
        Tensor::f32("decoder.positional_embedding", &[TEXT_CTX, STATE], 2),
        Tensor::f32("decoder.ln.weight", &[STATE], 3),
        Tensor::f32("decoder.ln.bias", &[STATE], 4),
    ];
    for block in 0..blocks {
        let prefix = format!("decoder.blocks.{block}");
        out.extend(block_tensors(&prefix, &["attn", "cross_attn"], &mut seed));
    }
    out
}

/// Every tensor that `WhisperModel::load` reads for [`tiny_config_json`] with
/// one block on each side, and the given `alignment_heads` entry.
pub(super) fn model_tensors(heads: Tensor) -> Vec<Tensor> {
    let mut seed = 200;
    let mut out = vec![
        Tensor::f32("encoder.conv1.weight", &[STATE, 3, 128], 5),
        Tensor::f32("encoder.conv1.bias", &[STATE], 6),
        Tensor::f32("encoder.conv2.weight", &[STATE, 3, STATE], 7),
        Tensor::f32("encoder.conv2.bias", &[STATE], 8),
        Tensor::f32("encoder.ln_post.weight", &[STATE], 9),
        Tensor::f32("encoder.ln_post.bias", &[STATE], 10),
    ];
    out.extend(block_tensors("encoder.blocks.0", &["attn"], &mut seed));
    out.extend(decoder_tensors(1));
    out.push(heads);
    out
}
