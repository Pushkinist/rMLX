//! BitNet model loader.
//!
//! Loads a `BitNetForCausalLM` snapshot from a model directory.
//!
//! ## Trit dequantization (load-time, once)
//!
//! BitNet weights are stored as U8 with 4 trits per byte (2 bits each, LSB
//! first):
//! - bits [1:0] = field 0
//! - bits [3:2] = field 1
//! - bits [5:4] = field 2
//! - bits [7:6] = field 3
//!
//! Encoding: ternary value = `raw - 1`, i.e. raw `0` → `-1`, `1` → `0`,
//! `2` → `+1` (`3` → `+2`, never present in valid ternary data). This matches
//! the `autobitlinear` offline-pack convention in HF transformers
//! `integrations/bitnet.py::unpack_weights` and the mlx-lm
//! `bitlinear_layers.py` Metal kernel (`(w & 3) - 1`).
//!
//! Row layout: the packed tensor is `U8 [N//4, K]`. Field `i` of packed row
//! `r` maps to **logical row `i * (N//4) + r`** — a strided interleave, NOT
//! a contiguous `r*4 + i` block. (mlx-lm kernel: output row
//! `row_idx + i * (out_features/4)`.)
//!
//! Each weight tensor has a sibling `*.weight_scale` (BF16 scalar `[1]`) that
//! is multiplied in at load time so the resulting BF16 matrix is fully scaled.
//! `linear_class: autobitlinear` ⇒ multiply (do not invert) the scale.
//!
//! Storage shape: `U8 [N//4, K]` → logical `[N, K]` after unpacking.
//!
//! ## Output
//!
//! The dequantized weight is stored as a BF16 `[N, K]` array and wrapped in
//! `BitLinear { weight }`. Forward pass uses plain BF16 matmul.

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::{Array, Device, Dtype};
use tracing::{debug, info};

use super::config::BitNetConfig;
use super::model::{BitLinear, BitNetAttention, BitNetDecoderLayer, BitNetMlp, BitNetText};
use crate::layers::RmsNorm;

// ---------------------------------------------------------------------------
// BF16 helpers
// ---------------------------------------------------------------------------

/// Encode a single f32 value to a 2-byte little-endian BF16 word.
///
/// BF16 = top 16 bits of IEEE 754 f32. We round-to-nearest-even by checking
/// whether the dropped mantissa bits warrant rounding up.
///
/// Hand-rolled — `half` is not a direct dependency of `rmlx-models` (it is
/// only a transitive dep via `rmlx-mlx`). Adding it as a direct dep just for
/// two scalar conversions done once at load time is not justified.
#[inline]
fn f32_to_bf16_bytes(x: f32) -> [u8; 2] {
    // Round-to-nearest-even: if the lower 16 bits of the f32 representation
    // are exactly 0x8000, round to even (add 1 to the upper bits if odd).
    let bits = x.to_bits();
    let lsb = bits & 0xFFFF;
    let bf16_bits: u16 = if lsb > 0x8000 || (lsb == 0x8000 && (bits & 0x10000) != 0) {
        // Round up.
        ((bits >> 16) as u16).wrapping_add(1)
    } else {
        (bits >> 16) as u16
    };
    bf16_bits.to_le_bytes()
}

/// Read a BF16 scalar from 2 little-endian bytes.
#[inline]
fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits(u32::from(bits) << 16)
}

// ---------------------------------------------------------------------------
// Trit dequantization
// ---------------------------------------------------------------------------

/// Dequantize a ternary U8 `[N//4, K]` weight tensor to BF16 `[N, K]`.
///
/// Each byte encodes 4 fields (2 bits each, LSB first).
/// Encoding: ternary value = `raw - 1` (raw `0`→`-1`, `1`→`0`, `2`→`+1`,
/// `3`→`+2`), scaled by `weight_scale`.
///
/// Field `i` of packed row `r` maps to logical row `i * packed_rows + r`
/// (strided interleave — see module docs / mlx-lm `bitlinear_layers.py`).
///
/// The `weight_scale` is a BF16 scalar `[1]` loaded from the sibling tensor.
/// It is multiplied into every value so the resulting matrix is fully scaled
/// and ready for plain BF16 matmul.
///
/// Returns a BF16 Array of shape `[n_rows * 4, n_cols]`.
fn dequant_trit_u8(
    u8_bytes: &[u8],
    packed_rows: usize,
    cols: usize,
    weight_scale: f32,
) -> Result<Array> {
    let n_rows = packed_rows.checked_mul(4).ok_or_else(|| {
        Error::Loader(format!("bitnet: shape overflow: packed_rows={packed_rows}"))
    })?;
    let n_elem = n_rows.checked_mul(cols).ok_or_else(|| {
        Error::Loader(format!(
            "bitnet: shape overflow: n_rows={n_rows} × cols={cols}"
        ))
    })?;

    // Each packed row has `cols` bytes; each byte unpacks to 4 output elements.
    let expected_bytes = packed_rows.checked_mul(cols).ok_or_else(|| {
        Error::Loader(format!(
            "bitnet: shape overflow: packed_rows={packed_rows} × cols={cols}"
        ))
    })?;
    if u8_bytes.len() != expected_bytes {
        return Err(Error::Loader(format!(
            "bitnet dequant_trit_u8: expected {expected_bytes} bytes, got {}",
            u8_bytes.len()
        )));
    }

    // Ternary value = raw - 1, scaled. Indexed by raw field [0..3].
    // raw 0 → -scale, 1 → 0, 2 → +scale, 3 → +2*scale (never in valid data).
    let val_bf16: [[u8; 2]; 4] = [
        f32_to_bf16_bytes(-weight_scale),
        f32_to_bf16_bytes(0.0_f32),
        f32_to_bf16_bytes(weight_scale),
        f32_to_bf16_bytes(2.0_f32 * weight_scale),
    ];

    let out_bytes = n_elem
        .checked_mul(2)
        .ok_or_else(|| Error::Loader(format!("bitnet: shape overflow: n_elem={n_elem} * 2")))?;
    let mut out = vec![0u8; out_bytes]; // BF16 = 2 bytes per element

    // Layout: logical output is [n_rows, cols], n_rows = packed_rows * 4.
    // Packed input is [packed_rows, cols].
    // For each packed_row r and col c:
    //   u8_bytes[r * cols + c] holds 4 fields; field t maps to logical row
    //   `t * packed_rows + r` (strided interleave — see module docs).
    for r in 0..packed_rows {
        for c in 0..cols {
            #[allow(
                clippy::indexing_slicing,
                reason = "r < packed_rows, c < cols, so r*cols+c < packed_rows*cols = u8_bytes.len()"
            )]
            let byte = u8_bytes[r * cols + c];

            for t in 0..4usize {
                let raw = ((byte >> (t * 2)) & 0x3) as usize;
                #[allow(
                    clippy::indexing_slicing,
                    reason = "raw is masked to 0..=3, val_bf16 has 4 entries"
                )]
                let bf16_val = val_bf16[raw];
                // Output index: logical row = t*packed_rows + r, col = c.
                let out_idx = (t * packed_rows + r) * cols + c;
                #[allow(
                    clippy::indexing_slicing,
                    reason = "out_idx = (t*packed_rows+r)*cols+c < n_rows*cols = n_elem; 2*out_idx+1 < 2*n_elem = out.len()"
                )]
                {
                    out[out_idx * 2] = bf16_val[0];
                    out[out_idx * 2 + 1] = bf16_val[1];
                }
            }
        }
    }

    Array::from_bytes(&out, &[n_rows as i32, cols as i32], Dtype::Bf16).map_err(|e| {
        Error::Loader(format!(
            "bitnet dequant_trit_u8: Array::from_bytes failed: {e}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a `BitNetForCausalLM` model from a snapshot directory.
///
/// Reads `config.json`, opens the safetensors shards, and dequantizes all
/// ternary (U8) linear weight tensors to BF16 at load time.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn load_from_path(model_dir: &Path) -> Result<BitNetText> {
    let cfg_raw = load_config(model_dir)?;
    let cfg = BitNetConfig::from_model_config(&cfg_raw)?;

    if !cfg.tie_word_embeddings {
        return Err(Error::Loader(
            "bitnet: tie_word_embeddings=false not supported (LM head untied)".into(),
        ));
    }

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        vocab_size = cfg.vocab_size,
        head_dim = cfg.head_dim,
        "BitNet: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    // Load a raw byte slice from any shard that has the named tensor.
    // Returns (bytes, dtype_str, shape).
    let load_raw = |name: &str| -> Result<(Vec<u8>, String, Vec<usize>)> {
        for (_filename, handle) in shards.iter() {
            let st = handle.safetensors().map_err(|e| {
                Error::Loader(format!("bitnet: opening shard while seeking '{name}': {e}"))
            })?;
            if let Ok(t) = st.tensor(name) {
                let bytes = t.data().to_vec();
                let dtype = format!("{:?}", t.dtype());
                let shape: Vec<usize> = t.shape().to_vec();
                return Ok((bytes, dtype, shape));
            }
        }
        Err(Error::Loader(format!(
            "bitnet: tensor '{name}' not found in any shard"
        )))
    };

    // Load a BF16 scalar weight_scale from its 2 bytes.
    let load_scale = |base: &str| -> Result<f32> {
        let name = format!("{base}.weight_scale");
        let (bytes, _, _) = load_raw(&name)?;
        if bytes.len() < 2 {
            return Err(Error::Loader(format!(
                "bitnet: weight_scale '{name}' has {} bytes, expected >=2",
                bytes.len()
            )));
        }
        Ok(bf16_bytes_to_f32([bytes[0], bytes[1]]))
    };

    // Load and dequantize a ternary linear layer.
    let load_bitlinear = |base: &str| -> Result<BitLinear> {
        let w_name = format!("{base}.weight");
        let (bytes, dtype, shape) = load_raw(&w_name)?;

        if dtype != "U8" {
            return Err(Error::Loader(format!(
                "bitnet: expected U8 for '{w_name}', got '{dtype}'"
            )));
        }
        if shape.len() != 2 {
            return Err(Error::Loader(format!(
                "bitnet: expected 2D shape for '{w_name}', got {shape:?}"
            )));
        }
        let packed_rows = shape[0];
        let cols = shape[1];

        let scale = load_scale(base)?;
        // dequant produces [out, in]; pre-transpose to [in, out] so forward
        // is a direct matmul with no per-call transpose on the hot path.
        let weight = dequant_trit_u8(&bytes, packed_rows, cols, scale)?;
        let weight_t = weight.transpose(&[1, 0], Device::Gpu).map_err(|e| {
            Error::Loader(format!("bitnet: transposing weight for '{w_name}': {e}"))
        })?;

        debug!(
            base,
            packed_rows,
            cols,
            n_rows = packed_rows * 4,
            scale,
            "BitNet: loaded ternary linear layer"
        );

        Ok(BitLinear { weight_t })
    };

    // Load an RmsNorm from `<name>.weight` as a BF16 array.
    let load_rms = |name: &str| -> Result<RmsNorm> {
        let wname = format!("{name}.weight");
        // RmsNorm weights are plain BF16 vectors.
        let arr = {
            let mut found = None;
            for (_filename, handle) in shards.iter() {
                let st = handle.safetensors().map_err(|e| {
                    Error::Loader(format!(
                        "bitnet: opening shard while seeking '{wname}': {e}"
                    ))
                })?;
                if let Ok(t) = st.tensor(&wname) {
                    let tv = rmlx_loader::TensorView {
                        name: &wname,
                        dtype: t.dtype(),
                        shape: t.shape().to_vec(),
                        bytes: t.data(),
                    };
                    found = Some(Array::from_safetensor_view(&tv)?);
                    break;
                }
            }
            found
                .ok_or_else(|| Error::Loader(format!("bitnet: norm tensor '{wname}' not found")))?
        };
        Ok(RmsNorm {
            weight: Some(arr),
            eps: cfg.rms_norm_eps,
        })
    };

    let pfx = "model";

    // Embedding table — plain BF16 [vocab, hidden].
    let embed_tokens = {
        let name = format!("{pfx}.embed_tokens.weight");
        let mut found = None;
        for (_filename, handle) in shards.iter() {
            let st = handle.safetensors().map_err(|e| {
                Error::Loader(format!("bitnet: opening shard while seeking '{name}': {e}"))
            })?;
            if let Ok(t) = st.tensor(&name) {
                let tv = rmlx_loader::TensorView {
                    name: &name,
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    bytes: t.data(),
                };
                found = Some(Array::from_safetensor_view(&tv)?);
                break;
            }
        }
        found.ok_or_else(|| Error::Loader("bitnet: embed_tokens.weight not found".to_owned()))?
    };

    // Final norm.
    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    // Decoder layers.
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    let scale = 1.0_f32 / (cfg.head_dim as f32).sqrt();

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");

        let attn = BitNetAttention {
            q_proj: load_bitlinear(&format!("{base}.self_attn.q_proj"))?,
            k_proj: load_bitlinear(&format!("{base}.self_attn.k_proj"))?,
            v_proj: load_bitlinear(&format!("{base}.self_attn.v_proj"))?,
            o_proj: load_bitlinear(&format!("{base}.self_attn.o_proj"))?,
            attn_sub_norm: load_rms(&format!("{base}.self_attn.attn_sub_norm"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale,
            rope_theta: cfg.rope_theta,
        };

        let mlp = BitNetMlp {
            gate_proj: load_bitlinear(&format!("{base}.mlp.gate_proj"))?,
            up_proj: load_bitlinear(&format!("{base}.mlp.up_proj"))?,
            down_proj: load_bitlinear(&format!("{base}.mlp.down_proj"))?,
            ffn_sub_norm: load_rms(&format!("{base}.mlp.ffn_sub_norm"))?,
        };

        layers.push(BitNetDecoderLayer {
            input_norm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });

        debug!(layer = i, "BitNet: loaded layer");
    }

    info!(
        total_layers = cfg.num_hidden_layers,
        "BitNet: all layers loaded"
    );

    // Pre-transpose embed_tokens [vocab, hidden] → [hidden, vocab] for the
    // tied LM head. Done once at load time to avoid a per-decode-step transpose.
    let embed_tokens_t = embed_tokens
        .transpose(&[1, 0], Device::Gpu)
        .map_err(|e| Error::Loader(format!("bitnet: transposing embed_tokens: {e}")))?;

    Ok(BitNetText {
        cfg,
        embed_tokens,
        embed_tokens_t,
        layers,
        final_norm,
        kv_bytes: crate::kv_bytes::KvBytesCounter::default(),
        model_sig: crate::prompt_cache::model_cache_sig(model_dir),
    })
}
