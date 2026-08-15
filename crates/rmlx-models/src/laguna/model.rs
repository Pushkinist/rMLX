//! Laguna full model.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::redundant_closure_for_method_calls)]
use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};
use tracing::debug;

use rmlx_kv_quant::KvCache;

use super::config::LagunaConfig;
use super::decoder_layer::DecoderLayer;
use super::layers::{Embedding, Linear};

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// LagunaForCausalLM model weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct LagunaText {
    /// Parsed model configuration.
    pub cfg: LagunaConfig,
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: super::layers::RmsNorm,
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

impl LagunaText {
    /// Full-sequence forward pass (no KV cache).
    ///
    /// Returns logits for the last position, shape [1, 1, vocab_size].
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Forward pass with optional KV cache.
    /// When `caches` is `Some`, each entry corresponds to one decoder layer.
    /// When `None`, behaves exactly as `forward_seq`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
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

        let base_offset = caches
            .as_ref()
            .and_then(|cs| cs.first())
            .map_or(0, |c| c.offset());

        let h = self.embed_tokens.forward(&ids_arr, device)?;
        let h = h.reshape(&[1, seq as i32, self.cfg.hidden_size as i32], device)?;

        let mut h = h;
        match caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "laguna forward layer");
                    h = layer.forward(&h, base_offset, None, device)?;
                }
            }
            Some(cs) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "laguna forward layer (cached)");
                    h = layer.forward(&h, base_offset, Some(&mut cs[i]), device)?;
                }
            }
        }

        let h = self.final_norm.forward(&h, device)?;
        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(
            &[0, (seq as i32) - 1, 0],
            &[1, seq as i32, hidden],
            &[1, 1, 1],
            device,
        )?;
        let h_last = h_last.reshape(&[1, 1, hidden], device)?;

        match &self.lm_head {
            Some(lm) => lm.forward(&h_last, device),
            None => self.embed_tokens.as_linear(&h_last, device),
        }
    }
}
