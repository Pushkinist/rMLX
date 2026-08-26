//! `MapleText` model struct and forward pass.
//!
//! Embeddings live at `model.word_embeddings` (not `embed_tokens`). Prefill
//! builds one full-attention mask and one SWA-512 mask, then picks per layer.
//! SWA layers use a rotating KV ring (`sliding_window`); full-attention
//! layers use unbounded KV. No GatedDeltaNet.
//!
//! `MapleDecoderLayer::forward` contract (owned by attention/decoder):
//! `forward(x, offset, cache, prebuilt_mask, device)`. When `prebuilt_mask`
//! is `Some`, attention must use `mask_mode="array"` with that mask. When
//! `None`, fall back to `crate::layers::pick_attn_mask_mode`.

// unsafe_code: none — token ids go through `Array::from_i32_slice`.
#![allow(clippy::redundant_closure_for_method_calls, clippy::ref_option)]
use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device};

use crate::layers::{Embedding, Linear};
use rmlx_kv_quant::{KvCache, KvQuant};

use super::config::MapleConfig;
use super::decoder_layer::MapleDecoderLayer;
use super::rms::MapleRmsNorm;

/// `MapleForCausalLM` weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct MapleText {
    /// Parsed checkpoint `config.json`.
    pub cfg: MapleConfig,
    /// `model.word_embeddings` (not `embed_tokens`).
    pub(super) embed: Embedding,
    /// Decoder stack (SWA and full-attention layers mixed).
    pub(super) layers: Vec<MapleDecoderLayer>,
    /// Final RMSNorm before the LM head (`MapleRMSNorm`: fp32 weight multiply).
    pub(super) norm: MapleRmsNorm,
    /// `None` when `tie_word_embeddings`.
    pub(super) lm_head: Option<Linear>,
    /// Resident-KV byte total of this instance's last generation.
    pub(crate) kv_bytes: crate::kv_bytes::KvBytesCounter,
    /// Snapshot identity folded into the prompt-cache key.
    pub(crate) model_sig: u64,
}

impl MapleText {
    /// Decoder depth as `usize` (config stores `i32`).
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers.max(0) as usize
    }

    /// Per-layer KV caches: rotating `max=sliding_window` on SWA layers,
    /// unbounded `KvCache` on full-attention layers. Mirrors maple.py
    /// `Model.make_cache`.
    #[must_use]
    pub fn make_cache(
        &self,
        kv_quant: KvQuant,
        initial_max_seq: i32,
        max_seq_ceiling: i32,
    ) -> Vec<KvCache> {
        let n_layers = self.n_layers();
        let window = self.cfg.sliding_window;
        crate::kv_cache::kv_layer_quants(n_layers, kv_quant)
            .into_iter()
            .enumerate()
            .map(|(i, q)| {
                let sliding = if self.cfg.is_swa_layer(i) {
                    Some(window)
                } else {
                    None
                };
                KvCache::with_quant_max_seq_window(q, initial_max_seq, sliding)
                    .with_max_seq_ceiling(max_seq_ceiling)
                    .with_layer_idx(i)
            })
            .collect()
    }

    /// Full-sequence forward (no KV cache). Last-position logits `[1, 1, vocab]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Forward with optional per-layer KV caches.
    pub fn forward_seq_with_cache(
        &self,
        ids: &[u32],
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        if seq == 0 {
            return Err(rmlx_core::error::Error::Model(
                "maple forward_seq: empty ids".to_owned(),
            ));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_arr = Array::from_i32_slice(&ids_i32, &[seq as i32])?;
        self.forward_arr(&ids_arr, seq as i32, caches, device)
    }

    /// One-token decode step for [`crate::decode_loop::pipelined_decode`].
    pub fn forward_step(
        &self,
        ids_arr: &Array,
        caches: &mut [KvCache],
        device: Device,
    ) -> Result<Array> {
        self.forward_arr(ids_arr, 1, Some(caches), device)
    }

    /// Forward from token ids already on-device. `seq` is the length of
    /// `ids_arr`. Last-position logits `[1, 1, vocab]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "layer index bounded by layers.len(); caches are one-per-layer"
    )]
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        if seq < 1 {
            return Err(rmlx_core::error::Error::Model(format!(
                "maple forward_arr: seq={seq} < 1"
            )));
        }

        let (full_offset, swa_offset, swa_rotating) = match caches.as_ref() {
            Some(cs) => {
                let ga = self.first_full_idx();
                let sw = self.first_swa_idx();
                (
                    ga.and_then(|i| cs.get(i)).map_or(0, KvCache::offset),
                    sw.and_then(|i| cs.get(i)).map_or(0, KvCache::offset),
                    sw.and_then(|i| cs.get(i)).is_some_and(KvCache::is_rotating),
                )
            }
            None => (0, 0, false),
        };
        let (full_mask, swa_mask) =
            self.build_masks(seq, full_offset, swa_offset, swa_rotating, device)?;

        let hidden = self.cfg.hidden_size;
        let h = self.embed.forward(ids_arr, device)?;
        let mut h = h.reshape(&[1, seq, hidden], device)?;

        match caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    let mask = if self.cfg.is_swa_layer(i) {
                        swa_mask.as_ref()
                    } else {
                        full_mask.as_ref()
                    };
                    h = layer.forward(&h, 0, None, mask, device)?;
                }
            }
            Some(cs) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    let mask = if self.cfg.is_swa_layer(i) {
                        swa_mask.as_ref()
                    } else {
                        full_mask.as_ref()
                    };
                    let offset = cs.get(i).map_or(0, KvCache::offset);
                    h = layer.forward(&h, offset, Some(&mut cs[i]), mask, device)?;
                }
            }
        }

        let h = self.norm.forward(&h, device)?;
        let h_last = h.slice(&[0, seq - 1, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, 1, hidden], device)?;
        self.logits_from_hidden(&h_last, device)
    }

    /// LM-head (or tied embedding) projection. `hidden`: `[1, n, H]`.
    pub fn logits_from_hidden(&self, hidden: &Array, device: Device) -> Result<Array> {
        match &self.lm_head {
            Some(lm) => lm.forward(hidden, device),
            None => self.embed.as_linear(hidden, device),
        }
    }

    fn first_swa_idx(&self) -> Option<usize> {
        (0..self.n_layers()).find(|&i| self.cfg.is_swa_layer(i))
    }

    fn first_full_idx(&self) -> Option<usize> {
        (0..self.n_layers()).find(|&i| !self.cfg.is_swa_layer(i))
    }

    /// One full-attn mask + one SWA-512 mask, shared across layers of this
    /// forward (maple.py `MapleModel.__call__`).
    fn build_masks(
        &self,
        seq: i32,
        full_offset: i32,
        swa_offset: i32,
        swa_rotating: bool,
        device: Device,
    ) -> Result<(Option<Array>, Option<Array>)> {
        let window = self.cfg.sliding_window.max(0) as usize;
        let full = if self.first_full_idx().is_some()
            && crate::layers::pick_attn_mask_mode(full_offset, seq) == "array"
        {
            Some(crate::layers::build_chunked_prefill_mask(
                full_offset,
                seq,
                device,
            )?)
        } else {
            None
        };

        let swa = if self.first_swa_idx().is_none() {
            None
        } else if seq == 1 {
            if swa_rotating {
                None
            } else {
                crate::layers::build_swa_decode_mask(swa_offset + seq, window, device)?
            }
        } else {
            let effective = if swa_rotating {
                swa_offset.min(self.cfg.sliding_window - 1)
            } else {
                swa_offset
            };
            Some(crate::layers::build_swa_prefill_mask(
                effective, seq, window, device,
            )?)
        };
        Ok((full, swa))
    }
}
