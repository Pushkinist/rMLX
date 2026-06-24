//! Gemma3 full model.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;
use rmlx_mlx::{divide, multiply, scalar_f32, tanh, Array, Device, Dtype};
use tracing::debug;

use rmlx_kv_quant::KvCache;

use super::config::Gemma3TextConfig;
use super::decoder_layer::DecoderLayer;
use super::layers::{Embedding, Linear};

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Gemma3ForConditionalGeneration text decoder weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct Gemma3Text {
    /// Parsed model configuration.
    pub cfg: Gemma3TextConfig,
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: super::layers::RmsNormShifted,
    pub(super) lm_head: Option<Linear>, // None when weight-tied to embed_tokens
}

impl Gemma3Text {
    /// Count the affine `.scales` / `.biases` sibling tensors that the loader
    /// actually materialised into this model (embed_tokens + every decoder
    /// projection + optional lm_head). Each `Quantized` variant contributes one
    /// `.scales` always, plus one `.biases` when present.
    ///
    /// Test-only introspection for the loader sibling-parity invariant: the
    /// snapshot's `model.safetensors.index.json` omits all `.scales` / `.biases`
    /// entries, so this count must equal the header-truth sibling count to prove
    /// the scan-only loader did not silently drop any index-omitted sibling.
    ///
    /// `#[doc(hidden)]` + `pub`: reachable from the `tests/` integration target
    /// (`#[cfg(test)]` would not be, since that target links the crate as an
    /// external dependency). Pure read-only counting — no behavioral effect.
    #[doc(hidden)]
    pub fn loaded_sibling_count(&self) -> usize {
        fn linear_siblings(l: &Linear) -> usize {
            match l {
                Linear::Plain { .. } => 0,
                Linear::Quantized { biases, .. } => 1 + usize::from(biases.is_some()),
            }
        }
        fn embedding_siblings(e: &Embedding) -> usize {
            match e {
                Embedding::Plain { .. } => 0,
                Embedding::Quantized { biases, .. } => 1 + usize::from(biases.is_some()),
            }
        }

        let mut n = embedding_siblings(&self.embed_tokens);
        if let Some(lm) = &self.lm_head {
            n += linear_siblings(lm);
        }
        for layer in &self.layers {
            n += linear_siblings(&layer.attn.q_proj);
            n += linear_siblings(&layer.attn.k_proj);
            n += linear_siblings(&layer.attn.v_proj);
            n += linear_siblings(&layer.attn.o_proj);
            n += linear_siblings(&layer.mlp.gate_proj);
            n += linear_siblings(&layer.mlp.up_proj);
            n += linear_siblings(&layer.mlp.down_proj);
        }
        n
    }

    /// Run a full-sequence forward pass (no KV cache).
    ///
    /// `ids`: all token ids in the sequence.
    /// Returns logits for the **last** position only, shape `[1, 1, vocab_size]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Run a forward pass with optional KV cache.
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
    /// Used by the async-pipelined decode loop so the next forward
    /// can chain on top of the prior step's `argmax` Array without forcing a
    /// CPU sync via `to_bytes()`. Mirrors `qwen3::Qwen3Text::forward_arr`.
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        // Embed tokens → [1, seq, hidden].
        let h = self.embed_tokens.forward(ids_arr, device)?;
        // Embedding scale: sqrt(hidden_size). Adopt activation dtype so a
        // strong-f32 scalar does not upcast the embedding stream.
        // Reference: gemma3_text.py Gemma3Model.__call__ line 190:
        // `h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)`
        let embed_scale =
            scalar_f32((self.cfg.hidden_size as f32).sqrt()).astype(h.dtype(), device)?;
        let h = multiply(&h, &embed_scale, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;
        self.forward_h(h, seq, caches, device)
    }

    /// Forward pass from precomputed, already-scaled `inputs_embeds`.
    ///
    /// `embeds`: `[1, seq, hidden]` — scaled text embeddings with image
    /// (vision) features already scattered at the image-token positions
    /// (mirrors mlx-vlm `get_input_embeddings` → `language_model(inputs_embeds=…)`).
    /// Gemma3 has no per-layer-input gating, so unlike Gemma4 there is no masked
    /// ids array — the scatter-merged `embeds` is fed straight to the trunk.
    ///
    /// Returns logits for the last position, `[1, 1, vocab]`.
    pub fn forward_arr_embeds(
        &self,
        embeds: Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        self.forward_h(embeds, seq, caches, device)
    }

    /// Shared decoder trunk + LM head over a precomputed scaled hidden state.
    ///
    /// `h`: `[1, seq, hidden]` scaled embeddings (text path scales inside
    /// [`forward_arr`]; the image path passes scatter-merged embeds via
    /// [`forward_arr_embeds`]).
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
            .map_or(0, |c| c.offset());

        // Decoder layers.
        let mut h = h;
        match caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "gemma3 forward layer");
                    h = layer.forward(&h, base_offset, None, device)?;
                }
            }
            Some(cs) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "gemma3 forward layer (cached)");
                    h = layer.forward(&h, base_offset, Some(&mut cs[i]), device)?;
                }
            }
        }

        // Final norm (shifted-gamma).
        let h = self.final_norm.forward(&h, device)?;

        // Extract last-position hidden: [1, seq, hidden] → [1, 1, hidden].
        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - 1, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, 1, hidden], device)?;

        // Logit projection.
        let logits = match &self.lm_head {
            Some(lm) => lm.forward(&h_last, device)?,
            None => self.embed_tokens.as_linear(&h_last, device)?,
        };

        // Final logit softcapping (optional — null in medgemma). Adopt the
        // logit dtype so a strong-f32 cap scalar does not upcast the stream.
        let logits = if let Some(cap) = self.cfg.final_logit_softcapping {
            let cap_arr = scalar_f32(cap).astype(logits.dtype(), device)?;
            let scaled = divide(&logits, &cap_arr, device)?;
            let t = tanh(&scaled, device)?;
            multiply(&t, &cap_arr, device)?
        } else {
            logits
        };

        Ok(logits)
    }
}
