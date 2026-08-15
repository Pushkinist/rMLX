// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(trivial_numeric_casts, trivial_casts)]

//! Qwen2 model internals: weight types, layers, and forward pass.

#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::cloned_instead_of_copied,
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::struct_field_names,
    clippy::too_many_lines
)]

use rmlx_core::error::Result;
use rmlx_mlx::{add, multiply, rms_norm, rope, scaled_dot_product_attention, Array, Device, Dtype};
use tracing::debug;

use rmlx_kv_quant::KvCache;

use super::config::Qwen2Config;

// ---------------------------------------------------------------------------
// Local Linear + Embedding with optional additive bias
// ---------------------------------------------------------------------------
//
// Qwen2 q/k/v projections carry a `.bias` additive vector (separate from the
// quantization `.biases` sibling). We add it after the matmul. Local types
// mirror gemma3.rs approach to avoid touching layers.rs.

#[allow(missing_debug_implementations)]
pub(super) enum Linear {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>, // affine quantization biases
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Linear {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Linear::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
        }
    }
}

#[allow(missing_debug_implementations)]
pub(super) enum Embedding {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Embedding {
    fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => weight.take(ids, 0, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => qwen_embedding_lookup(
                ids,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                device,
            ),
        }
    }

    fn as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
        }
    }
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn qwen_embedding_lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    let cpu = Device::Cpu;
    let weight_rows = weight.take(ids, 0, cpu)?;
    let scales_rows = scales.take(ids, 0, cpu)?;
    let biases_rows = biases.map(|b| b.take(ids, 0, cpu)).transpose()?;

    let seq = ids.dim(0)? as usize;
    let mut eye_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        eye_data[i * seq + i] = 1.0;
    }
    let eye_bytes =
        unsafe { std::slice::from_raw_parts(eye_data.as_ptr().cast::<u8>(), eye_data.len() * 4) };
    let eye = Array::from_bytes(eye_bytes, &[seq as i32, seq as i32], Dtype::F32)?;
    let eye_bf16 = eye.astype(Dtype::Bf16, cpu)?;

    let result = rmlx_mlx::quantized_matmul(
        &eye_bf16,
        &weight_rows,
        &scales_rows,
        biases_rows.as_ref(),
        group_size,
        bits,
        mode,
        false,
        cpu,
    )?;
    if device == cpu {
        Ok(result)
    } else {
        result.astype(result.dtype(), device)
    }
}

// ---------------------------------------------------------------------------
// RmsNorm (plain-gamma, no +1 shift)
// ---------------------------------------------------------------------------

pub(super) struct RmsNorm {
    pub(super) weight: Array,
    pub(super) eps: f32,
}

impl RmsNorm {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, Some(&self.weight), self.eps, device)
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct Attention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Linear,
    pub(super) v_proj: Linear,
    pub(super) o_proj: Linear,
    // Additive bias for q/k/v (shape [out_features], bf16).
    // Present in Qwen2; absent in Qwen3. Always checked per-snapshot.
    pub(super) q_bias: Option<Array>,
    pub(super) k_bias: Option<Array>,
    pub(super) v_bias: Option<Array>,
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) scale: f32,
    pub(super) rope_theta: f32,
}

impl Attention {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        device: Device,
    ) -> Result<Array> {
        let shape = x.shape(); // [batch, seq, hidden]
        let batch = shape[0];
        let seq = shape[1];

        // Q: project → add bias → reshape → transpose → RoPE.
        let q = self.q_proj.forward(x, device)?;
        let q = if let Some(b) = &self.q_bias {
            add(&q, b, device)?
        } else {
            q
        };
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]

        // K.
        let k = self.k_proj.forward(x, device)?;
        let k = if let Some(b) = &self.k_bias {
            add(&k, b, device)?
        } else {
            k
        };
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let k = k.transpose(&[0, 2, 1, 3], device)?;

        // V.
        let v = self.v_proj.forward(x, device)?;
        let v = if let Some(b) = &self.v_bias {
            add(&v, b, device)?
        } else {
            v
        };
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // Full RoPE over entire head_dim.
        let rope_dims = self.head_dim as i32;
        let q = rope(&q, rope_dims, false, self.rope_theta, 1.0, offset, device)?;
        let k = rope(&k, rope_dims, false, self.rope_theta, 1.0, offset, device)?;

        // GQA: MLX's fast SDPA kernel handles head broadcasting natively when
        // `n_q_heads % n_kv_heads == 0`. Skip the manual `repeat_kv` expand to
        // avoid materializing K/V at the larger `n_q_heads` shape in the
        // cache — that inflates KV memory by the GQA ratio for nothing.
        let _ = repeat_kv;

        // Universal dispatch (Task 5/9): a single `KvCache::update_and_sdpa`
        // call covers every cache variant (Mixed, K8V4-flash, legacy).
        //
        // Mask discipline: explicit additive mask only for `mask_mode == "array"`
        // (chunked prefill). Cast to query dtype to avoid MLX mismatch.
        let mask_mode = crate::layers::pick_attn_mask_mode(offset, seq);
        let q_dtype = q.dtype();
        let chunked_mask_owned: Option<Array> = if mask_mode == "array" {
            Some(crate::layers::build_chunked_prefill_mask(
                offset, seq, device,
            )?)
        } else {
            None
        };
        let cast_mask_owned: Option<Array> = match &chunked_mask_owned {
            Some(m) if m.dtype() != q_dtype => Some(m.astype(q_dtype, device)?),
            _ => None,
        };
        let additive_mask: Option<&Array> =
            cast_mask_owned.as_ref().or(chunked_mask_owned.as_ref());

        let out = if let Some(c) = cache {
            c.update_and_sdpa(&q, &k, &v, self.scale, mask_mode, additive_mask, device)?
        } else {
            // No cache — run SDPA directly on the pre-RoPE K/V.
            scaled_dot_product_attention(&q, &k, &v, self.scale, mask_mode, additive_mask, device)?
        };
        let out = out.transpose(&[0, 2, 1, 3], device)?;
        let out = out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;
        self.o_proj.forward(&out, device)
    }
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    let s = x.shape();
    let (b, kv_h, seq, d) = (s[0], s[1], s[2], s[3]);
    let x5 = rmlx_mlx::expand_dims(x, 2, device)?;
    let bc = rmlx_mlx::broadcast_to(&x5, &[b, kv_h, repeat as i32, seq, d], device)?;
    bc.reshape(&[b, kv_h * repeat as i32, seq, d], device)
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct Mlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = self.gate_proj.forward(x, device)?;
        let gate = rmlx_mlx::silu(&gate, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct DecoderLayer {
    pub(super) input_norm: RmsNorm,
    pub(super) post_attn_norm: RmsNorm,
    pub(super) attn: Attention,
    pub(super) mlp: Mlp,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        device: Device,
    ) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let h = self.attn.forward(&h, offset, cache, device)?;
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let h = self.post_attn_norm.forward(&h, device)?;
        let h = self.mlp.forward(&h, device)?;
        add(&residual, &h, device)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model struct — private weight fields; public API is forward_seq() and forward_multimodal(); adding a field requires updating load_weights and the Qwen2 loader"
)]
/// Qwen2ForCausalLM model weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct Qwen2Text {
    /// Parsed model configuration.
    pub cfg: Qwen2Config,
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: RmsNorm,
    /// `None` when `tie_word_embeddings = true`.
    pub(super) lm_head: Option<Linear>,
    /// Resident-KV byte total of this instance's last generation, paired with a
    /// store sequence. Per model instance, never per arch — two models of the
    /// same architecture must not write each other's figure.
    pub(crate) kv_bytes: crate::kv_bytes::KvBytesCounter,
    /// Stable identity of the snapshot this instance was loaded from, folded
    /// into the prompt-cache key. The prompt cache is one static per arch, so
    /// without it a second model of the same arch serves its K/V from this
    /// one's slots. See [`crate::prompt_cache::cache_seed`].
    pub(crate) model_sig: u64,
}

impl Qwen2Text {
    /// Full-sequence forward pass (no KV cache).
    ///
    /// Returns logits for the last position, shape `[1, 1, vocab_size]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Forward pass with optional KV cache.
    /// When `caches` is `Some`, each entry corresponds to one decoder layer.
    /// When `None`, behaves exactly as `forward_seq`.
    pub fn forward_seq_with_cache(
        &self,
        ids: &[u32],
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;
        self.forward_arr(&ids_arr, seq as i32, caches, device)
    }

    /// Forward pass with token IDs already in an MLX `Array`.
    ///
    /// Used by the async-pipelined decode loop so the next forward can chain
    /// on top of the prior step's `argmax` Array without forcing a CPU sync
    /// via `to_bytes()`. Mirrors `qwen3::Qwen3Text::forward_arr`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        // Qwen2's quantized embedding lookup uses weight.take(ids, 0, cpu) —
        // a CPU-side gather that needs the token IDs in CPU-readable memory.
        // When `ids_arr` comes from an async GPU argmax (the pipeline path),
        // we must force eval so the GPU has written the values to unified memory
        // before the CPU gather reads them. This is a no-op for the non-pipelined
        // path (ids_arr already evaluated) and a sync barrier for the pipelined
        // path (GPU argmax → CPU unified memory).
        ids_arr.eval()?;

        let base_offset = caches
            .as_ref()
            .and_then(|cs| cs.first())
            .map_or(0, |c| c.offset());

        let h = self.embed_tokens.forward(ids_arr, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let mut h = h;
        match caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "qwen2 forward layer");
                    h = layer.forward(&h, base_offset, None, device)?;
                }
            }
            Some(cs) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "qwen2 forward layer (cached)");
                    h = layer.forward(&h, base_offset, Some(&mut cs[i]), device)?;
                }
            }
        }

        let h = self.final_norm.forward(&h, device)?;

        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - 1, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, 1, hidden], device)?;

        match &self.lm_head {
            Some(lm) => lm.forward(&h_last, device),
            None => self.embed_tokens.as_linear(&h_last, device),
        }
    }
}
