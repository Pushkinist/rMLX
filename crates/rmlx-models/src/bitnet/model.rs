//! BitNet full model forward pass.
//!
//! Implements `BitNetForCausalLM` — a 30-layer Llama-style decoder with:
//! - Ternary (int2-packed) linear weights dequantized to BF16 at load time.
//! - Sub-norms (`attn_sub_norm`, `ffn_sub_norm`) — RMSNorm applied between
//!   activations and the output projection in each sub-block.
//! - Relu2 activation (`max(x, 0)^2`) in the FFN.
//! - Tied LM head (embed_tokens weight == lm_head weight).

// unsafe_code: MLX Array zero-copy view — slice::from_raw_parts byte-reinterpret.
#![allow(unsafe_code)]

use std::mem::size_of_val;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{matmul, multiply, rope, scaled_dot_product_attention, Array, Device, Dtype};

use crate::layers::RmsNorm;
use rmlx_kv_quant::KvCache;

use super::config::BitNetConfig;

// ---------------------------------------------------------------------------
// BitLinear — plain BF16 (dequantized at load time)
// ---------------------------------------------------------------------------

/// A single ternary linear layer, dequantized to BF16 at load time.
///
/// Stored as a pre-transposed BF16 weight `[in, out]` so that `forward` is a
/// direct matmul without a per-call transpose. `weight_scale` is baked in at
/// load time.
#[allow(missing_debug_implementations)]
pub(super) struct BitLinear {
    /// BF16 weight `[in, out]` — pre-transposed and scaled at load time.
    pub(super) weight_t: Array,
}

impl BitLinear {
    /// `x`: `[batch, seq, in_features]` or `[batch, in_features]`.
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        matmul(x, &self.weight_t, device)
    }
}

// ---------------------------------------------------------------------------
// BitNet MLP block (gate-up-down with sub-norm)
// ---------------------------------------------------------------------------

/// BitNet FFN block.
///
/// Layout (from `modeling_bitnet.py`):
/// 1. `gate = gate_proj(x)`
/// 2. `up = up_proj(x)`
/// 3. `gated = relu2(gate) * up`
/// 4. `gated = ffn_sub_norm(gated)`  ← unique to BitNet
/// 5. `out = down_proj(gated)`
#[allow(missing_debug_implementations)]
pub(super) struct BitNetMlp {
    /// Gate projection (ternary).
    pub(super) gate_proj: BitLinear,
    /// Up projection (ternary).
    pub(super) up_proj: BitLinear,
    /// Down projection (ternary).
    pub(super) down_proj: BitLinear,
    /// Sub-norm applied after `relu2(gate) * up`, before `down_proj`.
    pub(super) ffn_sub_norm: RmsNorm,
}

impl BitNetMlp {
    /// BitNet FFN forward. Implements relu2 inline because this block also
    /// applies `ffn_sub_norm` between the activation and `down_proj`, which the
    /// shared `layers::Mlp` cannot express.
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        // relu2(gate) = max(gate, 0)^2
        let gate = self.gate_proj.forward(x, device)?;
        let up = self.up_proj.forward(x, device)?;

        let gate_act = {
            // relu2: max(x,0)^2 — zero adopts gate dtype to avoid f32 promotion.
            use rmlx_mlx::{maximum, scalar_f32};
            let zero = scalar_f32(0.0).astype(gate.dtype(), device)?;
            let pos = maximum(&gate, &zero, device)?;
            multiply(&pos, &pos, device)?
        };
        let gated = multiply(&gate_act, &up, device)?;

        // Apply ffn_sub_norm before down_proj.
        let gated = self.ffn_sub_norm.forward(&gated, device)?;

        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// BitNet attention block
// ---------------------------------------------------------------------------

/// BitNet self-attention.
///
/// GQA (20 Q heads, 5 KV heads), head_dim=128, RoPE theta=500_000.
///
/// Sub-norm (`attn_sub_norm`) is applied to the concatenated attention output
/// **before** `o_proj`:
/// 1. Q = q_proj(x)
/// 2. K = k_proj(x), V = v_proj(x)
/// 3. Apply RoPE to Q, K
/// 4. attn_out = SDPA(Q, K, V)
/// 5. attn_out = reshape + attn_sub_norm(attn_out)   ← unique to BitNet
/// 6. out = o_proj(attn_out)
#[allow(missing_debug_implementations)]
pub(super) struct BitNetAttention {
    pub(super) q_proj: BitLinear,
    pub(super) k_proj: BitLinear,
    pub(super) v_proj: BitLinear,
    pub(super) o_proj: BitLinear,
    /// Sub-norm applied to concatenated attention output before o_proj.
    pub(super) attn_sub_norm: RmsNorm,
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    /// Attention scale: `1 / sqrt(head_dim)`.
    pub(super) scale: f32,
    pub(super) rope_theta: f32,
}

impl BitNetAttention {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        device: Device,
    ) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];

        // Project Q, K, V.
        let q = self.q_proj.forward(x, device)?;
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]

        let k = self.k_proj.forward(x, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let k = k.transpose(&[0, 2, 1, 3], device)?;

        let v = self.v_proj.forward(x, device)?;
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // RoPE (full rotation — no partial dims).
        let rope_dims = self.head_dim as i32;
        let q = rope(&q, rope_dims, false, self.rope_theta, 1.0, offset, device)?;
        let k = rope(&k, rope_dims, false, self.rope_theta, 1.0, offset, device)?;

        // Causal mask mode.
        let mask_holder: Option<Array>;
        let mask_mode: &str;
        let mode = crate::layers::pick_attn_mask_mode(offset, seq);
        if mode == "array" {
            mask_holder = Some(crate::layers::build_chunked_prefill_mask(
                offset, seq, device,
            )?);
            mask_mode = "array";
        } else {
            mask_holder = None;
            mask_mode = mode;
        }
        let mask_ref = mask_holder.as_ref();

        // SDPA with optional KV cache.
        let attn_out = if let Some(c) = cache {
            c.update_and_sdpa(&q, &k, &v, self.scale, mask_mode, mask_ref, device)?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, mask_mode, mask_ref, device)?
        };

        // Transpose + reshape back to [B, S, hidden].
        let attn_out = attn_out.transpose(&[0, 2, 1, 3], device)?;
        let attn_out =
            attn_out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        // attn_sub_norm before o_proj (BitNet-specific).
        let attn_out = self.attn_sub_norm.forward(&attn_out, device)?;

        self.o_proj.forward(&attn_out, device)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// BitNet decoder layer.
///
/// Standard Llama-style pre-norm residual layout:
/// 1. h = input_norm(x)
/// 2. h = attn(h)
/// 3. x = x + h
/// 4. h = post_attn_norm(x)
/// 5. h = mlp(h)
/// 6. x = x + h
#[allow(missing_debug_implementations)]
pub(super) struct BitNetDecoderLayer {
    pub(super) input_norm: RmsNorm,
    pub(super) post_attn_norm: RmsNorm,
    pub(super) attn: BitNetAttention,
    pub(super) mlp: BitNetMlp,
}

impl BitNetDecoderLayer {
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        device: Device,
    ) -> Result<Array> {
        // Attention sub-layer.
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let h = self.attn.forward(&h, offset, cache, device)?;
        let x = rmlx_mlx::add(&residual, &h, device)?;

        // FFN sub-layer. `x` is owned here so we move it into `residual`
        // instead of cloning — the norm gets a reference, the add consumes residual.
        let residual = x;
        let h = self.post_attn_norm.forward(&residual, device)?;
        let h = self.mlp.forward(&h, device)?;
        rmlx_mlx::add(&residual, &h, device)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// `BitNetForCausalLM` weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct BitNetText {
    /// Parsed model configuration.
    pub cfg: BitNetConfig,
    /// Token embedding table (BF16, plain) — `[vocab, hidden]`, used for input lookup.
    pub(super) embed_tokens: Array,
    /// Transposed embedding table `[hidden, vocab]` — used for the tied LM head.
    /// Pre-computed at load time to avoid a per-call transpose on every decode step.
    pub(super) embed_tokens_t: Array,
    /// Decoder layers.
    pub(super) layers: Vec<BitNetDecoderLayer>,
    /// Final RMSNorm.
    pub(super) final_norm: RmsNorm,
    // No lm_head — always tied to embed_tokens in this model.
}

impl BitNetText {
    /// Run a full-sequence forward pass (no KV cache).
    ///
    /// `ids`: token ids for the sequence.
    /// Returns logits for the **last** position only, shape `[1, 1, vocab_size]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Run a forward pass with optional KV cache.
    pub fn forward_seq_with_cache(
        &self,
        ids: &[u32],
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        debug_assert_eq!(ids_i32.len(), seq);
        let ids_bytes = unsafe {
            std::slice::from_raw_parts(
                ids_i32.as_ptr().cast::<u8>(),
                size_of_val(ids_i32.as_slice()),
            )
        };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;
        self.forward_arr(&ids_arr, seq as i32, caches, device)
    }

    /// Forward pass with token IDs as an MLX Array.
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        if seq <= 0 {
            return Err(Error::Model(format!(
                "bitnet forward: seq must be >= 1, got {seq}"
            )));
        }
        // Embedding lookup: take rows from embed_tokens.
        // embed_tokens: [vocab, hidden] → lookup → [seq, hidden] → [1, seq, hidden]
        let h = self.embed_tokens.take(ids_arr, 0, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        self.forward_h(h, seq, caches, device)
    }

    /// Shared decoder trunk + LM head over a precomputed hidden state.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward_h(
        &self,
        h: Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let base_offset = caches
            .as_ref()
            .and_then(|cs| cs.first())
            .map_or(0, KvCache::offset);

        // Decoder layers.
        let mut h = h;
        match caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    tracing::debug!(layer = i, "bitnet forward layer");
                    h = layer.forward(&h, base_offset, None, device)?;
                }
            }
            Some(cs) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    tracing::debug!(layer = i, "bitnet forward layer (cached)");
                    h = layer.forward(&h, base_offset, Some(&mut cs[i]), device)?;
                }
            }
        }

        // Final norm.
        let h = self.final_norm.forward(&h, device)?;

        // Extract last-position hidden: [1, seq, hidden] → [1, 1, hidden].
        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - 1, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, 1, hidden], device)?;

        // LM head: tied to embed_tokens. Multiply [1, 1, hidden] @ [hidden, vocab].
        // embed_tokens_t is pre-transposed at load time — no per-call transpose.
        matmul(&h_last, &self.embed_tokens_t, device)
    }
}
