//! `Gemma4Text` model struct and forward pass.
//!
//! [`Gemma4Text`] holds the full model weights and orchestrates the
//! per-layer decode: embedding lookup, altup gating (PerLayerInput),
//! decoder-layer stack (attention + MoE), final norm, and LM-head projection.
//! The forward pass handles both prefill (full-sequence) and decode
//! (single-token) modes transparently through the KV cache.
//!
//! # Public API
//!
//! - [`Gemma4Text`] — model struct, constructed by [`super::loader::load_from_path`].

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::manual_let_else, clippy::redundant_closure_for_method_calls)]
// trivial_casts: `k as &Array` coercions are required to coerce from owned reference to trait-object
// slice in heterogeneous iterator contexts; rustc does not accept a plain reference without the cast.
#![allow(trivial_casts)]
use rmlx_core::error::Result;
use rmlx_mlx::{multiply, scalar_f32, Array, Device, Dtype};
use tracing::debug;

use crate::layers::Embedding;
use rmlx_kv_quant::{KvCache, SharedKvOut};

use super::config::Gemma4TextConfig;
use super::decoder_layer::DecoderLayer;
use super::layers::softcap_fused;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the sequence offset that should be used as the RoPE `base_offset`
/// for this forward call.
///
/// Gemma4 layer 0 is always a `SlidingAttention` layer whose cache becomes
/// `KvStorage::None` after SSD hydration (the rotating ring buffer cannot be
/// spilled). That cache carries the block's `seq_len` as `self.offset` so the
/// SWA decode can compute correct write positions, but [`KvCache::offset`] on
/// such a cache does **not** reflect how many K/V tokens are actually usable
/// for attention (the answer is zero). Using it for RoPE base_offset would
/// silently produce incorrect token positions.
///
/// We therefore skip `has_persistent_cache() == false` caches and read from
/// the first full-attention cache (which always has real quantised K/V data).
/// If no cache has persistent data (all fresh / all SWA-None), fall back to
/// `cs[0].offset()` which is 0 on a cold start.
fn cache_base_offset(caches: Option<&[KvCache]>) -> i32 {
    let Some(cs) = caches else {
        return 0;
    };
    cs.iter()
        .find(|c| c.has_persistent_cache())
        .or_else(|| cs.first())
        .map_or(0, |c| c.offset())
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Gemma4ForConditionalGeneration text decoder weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct Gemma4Text {
    /// Parsed model configuration.
    pub cfg: Gemma4TextConfig,
    pub(super) embed_tokens: Embedding,
    pub(super) embed_tokens_per_layer: Option<Embedding>,
    pub(super) per_layer_model_proj: Option<crate::layers::Linear>,
    pub(super) per_layer_proj_norm: Option<crate::layers::RmsNorm>,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: crate::layers::RmsNorm,
    /// `previous_kvs[i]` = index of the layer whose KV layer `i` should use.
    pub(super) previous_kvs: Vec<usize>,
}

impl Gemma4Text {
    /// full-sequence forward returning logits at **every** position.
    ///
    /// Used by the offline PPL scorer (`ppl::compute_ppl`) to read the
    /// per-position log-likelihood of the actual prompt token.
    ///
    /// Returns an `Array` of shape `[1, seq, vocab_size]`. No KV cache — the
    /// scorer slides a fresh window each call; reusing the decode-loop cache
    /// would constrain absolute-position embeddings.
    ///
    /// Mirrors `qwen3::Qwen3Text::forward_seq_logits_all`. Implements the full
    /// Gemma4 trunk (embed_scale, per-layer gating, shared-KV, final norm,
    /// tied LM head, logit softcapping) without slicing to a subset of positions.
    /// Structurally similar to `forward_h`'s no-cache branch; the only mathematical
    /// delta is that we run `as_linear` on the full `[1, seq, hidden]` tensor
    /// instead of slicing to `[1, 1, hidden]` first.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_seq_logits_all(&self, ids: &[u32], device: Device) -> Result<Array> {
        let seq = ids.len();
        if seq == 0 {
            return Err(rmlx_core::error::Error::Other(
                "forward_seq_logits_all: empty prompt".to_owned(),
            ));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        // SAFETY: `ids_i32` is a Vec<i32> of length `seq`; reinterpreting as &[u8] of
        // length `seq * 4` is sound (i32 has stricter alignment than u8, no padding,
        // and the slice does not outlive `ids_i32`).
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        // Embed + scale. Mirrors forward_arr.
        let h_raw = self.embed_tokens.forward(&ids_arr, device)?;
        // Match mlx-lm dtype discipline: the embed-scale constant adopts the
        // activation dtype (bf16 for mxfp8) instead of a strong-F32 scalar.
        // A strong-F32 scalar would promote the whole residual stream to f32,
        // which then propagates through Q/K/V projections, attention, and the
        // global KV cache (doubling its residency on the None path). mlx-lm
        // multiplies by a Python float (weak type) / a `mx.array(_, bf16)`,
        // keeping the stream at the model dtype.
        let embed_scale =
            scalar_f32((self.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
        let h = multiply(&h_raw, &embed_scale, device)?;
        let h = h.reshape(&[1, seq as i32, self.cfg.hidden_size as i32], device)?;

        // Per-layer inputs.
        let per_layer_inputs = self.compute_per_layer_inputs(&ids_arr, &h, device)?;

        // KV accumulation for shared-KV layers (same pattern as forward_arr/forward_h).
        let mut stored_kvs: Vec<Option<SharedKvOut>> =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        let mut h = h;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            debug!(layer = layer_idx, "gemma4 forward_seq_logits_all layer");
            let prev_idx = self.previous_kvs[layer_idx];
            let shared_kv = if prev_idx == layer_idx {
                None
            } else {
                stored_kvs[prev_idx].as_ref()
            };
            let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
            let (new_h, new_kv) = layer.forward(
                &h, shared_kv, per_layer, 0, // base_offset = 0 (fresh forward, no cache)
                None, false, // no cache → no rotating cache
                device,
            )?;
            h = new_h;
            if let Some(kv) = new_kv {
                stored_kvs[layer_idx] = Some(kv);
            }
        }

        // Final norm over full [1, seq, hidden].
        let h = self.final_norm.forward(&h, device)?;

        // Tied LM head: embed_tokens.as_linear on full [1, seq, hidden] → [1, seq, vocab].
        let logits = self.embed_tokens.as_linear(&h, device)?;

        // Final logit softcapping.
        apply_softcap(&logits, self.cfg.final_logit_softcapping, device)
    }

    /// Run a single-token forward pass (no KV cache — allocates fresh KV each call).
    ///
    /// `token_id`: the input token (u32 cast to i32).
    /// Returns logits shape: `[1, 1, vocab_size]`.
    pub fn forward_one(&self, token_id: u32, device: Device) -> Result<Array> {
        self.forward_seq(&[token_id], device)
    }

    /// Run a full-sequence forward pass (no KV cache).
    ///
    /// `ids`: all token ids in the sequence (u32 cast to i32 each).
    /// Returns logits for the **last** position only, shape `[1, 1, vocab_size]`.
    ///
    /// Called by `generate_greedy` with the growing prefix; intentionally O(n²)
    /// since this is the smoke probe, not the production generation path.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Run a forward pass with optional KV cache.
    ///
    /// When `caches` is `Some`, each element corresponds to one decoder layer.
    /// The offset for RoPE is read from `caches[0].offset()` (all caches advance together).
    /// When `None`, behaves exactly as `forward_seq` (no caching).
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

    /// Forward pass returning logits at the **last K positions**.
    ///
    /// Speculative-decoding scaffold. Returns shape
    /// `[1, K, vocab_size]` — one logit row per position in `ids[seq-K..seq]`.
    /// `K` must satisfy `1 <= K <= ids.len()`.
    ///
    /// No KV cache in this path — the cache-using version wires verifier/draft
    /// cache truncation.
    pub fn forward_seq_last_k(&self, ids: &[u32], k: usize, device: Device) -> Result<Array> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_seq_last_k: k={k} out of range for seq={seq}"
            )));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;
        self.forward_arr_last_k(&ids_arr, seq as i32, k as i32, None, device)
    }

    /// Cache-using forward returning logits at the **last K positions**.
    ///
    /// Like `forward_seq_last_k` but reads + writes the provided per-layer
    /// `caches`. Used by speculative decoding to feed `K+1` new tokens
    /// (1 carry-token + K draft tokens) into the verifier's persistent
    /// cache in one forward call. Cache offset advances by `ids.len()`.
    ///
    /// Returns shape `[1, k, vocab_size]`.
    pub fn forward_seq_last_k_with_cache(
        &self,
        ids: &[u32],
        k: usize,
        caches: &mut [KvCache],
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_seq_last_k_with_cache: k={k} out of range for seq={seq}"
            )));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;
        self.forward_arr_last_k(&ids_arr, seq as i32, k as i32, Some(caches), device)
    }

    /// Forward pass with token IDs already in an MLX `Array`.
    ///
    /// Used by the async-pipelined decode loop so the next forward
    /// can chain on top of the prior step's `argmax` Array without forcing a
    /// CPU sync via `to_bytes()`. Mirrors `gemma3::Gemma3Text::forward_arr` and
    /// `qwen3::Qwen3Text::forward_arr`.
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        // Embed tokens → [1, seq, hidden]; scale by hidden_size^0.5.
        let h_raw = self.embed_tokens.forward(ids_arr, device)?;
        // Match mlx-lm dtype discipline: the embed-scale constant adopts the
        // activation dtype (bf16 for mxfp8) instead of a strong-F32 scalar.
        // A strong-F32 scalar would promote the whole residual stream to f32,
        // which then propagates through Q/K/V projections, attention, and the
        // global KV cache (doubling its residency on the None path). mlx-lm
        // multiplies by a Python float (weak type) / a `mx.array(_, bf16)`,
        // keeping the stream at the model dtype.
        let embed_scale =
            scalar_f32((self.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
        let h = multiply(&h_raw, &embed_scale, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;
        self.forward_h(h, ids_arr, seq, caches, device)
    }

    /// forward pass from precomputed, already-scaled `inputs_embeds`.
    ///
    /// `embeds`: `[1, seq, hidden]` — the scaled text embeddings with image
    /// (vision) features already scattered at the image-token positions
    /// (mirrors mlx-vlm `get_input_embeddings` → `language_model(inputs_embeds=…)`).
    /// `ids_arr`: `[seq]` token ids with the multimodal positions **masked to
    /// 0** (mlx-vlm zeroes image/audio tokens before `get_per_layer_inputs`);
    /// used only for the per-layer-input gating, not for embedding.
    ///
    /// Returns logits for the last position, `[1, 1, vocab]`.
    pub fn forward_arr_embeds(
        &self,
        embeds: Array,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        self.forward_h(embeds, ids_arr, seq, caches, device)
    }

    /// Shared decoder trunk + LM head over a precomputed scaled hidden state.
    ///
    /// `h`: `[1, seq, hidden]` scaled embeddings (text path scales inside
    /// [`forward_arr`]; the image path passes scatter-merged embeds via
    /// [`forward_arr_embeds`]). `ids_arr`: `[seq]` ids for per-layer gating.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward_h(
        &self,
        h: Array,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        // Current sequence offset (0 when no cache).
        let base_offset = cache_base_offset(caches.as_deref());

        // Per-position per-layer inputs.
        // ids_arr is [seq], h is [1, seq, hidden] — both span the full call.
        let per_layer_inputs = self.compute_per_layer_inputs(ids_arr, &h, device)?;

        // KV storage: index = previous_kvs[layer_idx], value = (K, V) from that layer.
        let mut stored_kvs: Vec<Option<SharedKvOut>> =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        let mut h = h;
        // Pre-compute "is rotating" flag per layer. For shared-KV
        // layers, the K/V comes from the source layer's cache, which dictates
        // the K shape. The flag drives the SWA mask path in attention.
        let kv_is_rotating: Vec<bool> = (0..self.cfg.num_hidden_layers)
            .map(|i| {
                let src = self.previous_kvs[i];
                caches
                    .as_ref()
                    .and_then(|cs| cs.get(src))
                    .is_some_and(|c| c.is_rotating())
            })
            .collect();
        // Split caches mutably so we can index per layer.
        match caches {
            None => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let prev_idx = self.previous_kvs[layer_idx];
                    let shared_kv = if prev_idx == layer_idx {
                        None
                    } else {
                        stored_kvs[prev_idx].as_ref()
                    };
                    let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
                    let (new_h, new_kv) = layer.forward(
                        &h,
                        shared_kv,
                        per_layer,
                        base_offset,
                        None,
                        kv_is_rotating[layer_idx],
                        device,
                    )?;
                    h = new_h;
                    if let Some(kv) = new_kv {
                        stored_kvs[layer_idx] = Some(kv);
                    }
                }
            }
            Some(cs) => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let prev_idx = self.previous_kvs[layer_idx];
                    let shared_kv = if prev_idx == layer_idx {
                        None
                    } else {
                        stored_kvs[prev_idx].as_ref()
                    };
                    let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
                    // Shared-KV layers don't have their own cache entry; pass None.
                    let cache = if prev_idx == layer_idx {
                        Some(&mut cs[layer_idx])
                    } else {
                        None
                    };
                    let (new_h, new_kv) = layer.forward(
                        &h,
                        shared_kv,
                        per_layer,
                        base_offset,
                        cache,
                        kv_is_rotating[layer_idx],
                        device,
                    )?;
                    h = new_h;
                    if let Some(kv) = new_kv {
                        stored_kvs[layer_idx] = Some(kv);
                    }
                }
            }
        }

        // Final norm.
        let h = self.final_norm.forward(&h, device)?;

        // Extract last-position hidden state: [1, seq, hidden] → [1, 1, hidden].
        // This is the position whose next-token logits we want.
        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - 1, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, 1, hidden], device)?;

        // Logit projection (tied embeddings: embed_tokens.as_linear).
        let logits = self.embed_tokens.as_linear(&h_last, device)?;

        // Final logit softcapping: tanh(logits / cap) * cap.
        let cap = self.cfg.final_logit_softcapping;
        let logits = apply_softcap(&logits, cap, device)?;

        Ok(logits)
    }

    /// Forward pass returning logits at the last `k` positions.
    ///
    /// Mirrors `forward_arr` body but slices `[seq-k..seq]` instead of
    /// `[seq-1..seq]` before the LM-head. Returns shape `[1, k, vocab]`.
    ///
    /// Scaffold for speculative decoding. Always runs without KV cache —
    /// the cache-using sibling handles the verifier/draft case.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_arr_last_k(
        &self,
        ids_arr: &Array,
        seq: i32,
        k: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        if k < 1 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_arr_last_k: k={k} out of range for seq={seq}"
            )));
        }

        let base_offset = cache_base_offset(caches.as_deref());

        let h_raw = self.embed_tokens.forward(ids_arr, device)?;
        // Match mlx-lm dtype discipline: the embed-scale constant adopts the
        // activation dtype (bf16 for mxfp8) instead of a strong-F32 scalar.
        // A strong-F32 scalar would promote the whole residual stream to f32,
        // which then propagates through Q/K/V projections, attention, and the
        // global KV cache (doubling its residency on the None path). mlx-lm
        // multiplies by a Python float (weak type) / a `mx.array(_, bf16)`,
        // keeping the stream at the model dtype.
        let embed_scale =
            scalar_f32((self.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
        let h = multiply(&h_raw, &embed_scale, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let per_layer_inputs = self.compute_per_layer_inputs(ids_arr, &h, device)?;

        let mut stored_kvs: Vec<Option<SharedKvOut>> =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        let mut h = h;
        // Rotating-cache flag per layer (see forward_arr).
        let kv_is_rotating: Vec<bool> = (0..self.cfg.num_hidden_layers)
            .map(|i| {
                let src = self.previous_kvs[i];
                caches
                    .as_ref()
                    .and_then(|cs| cs.get(src))
                    .is_some_and(|c| c.is_rotating())
            })
            .collect();
        match caches {
            None => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let prev_idx = self.previous_kvs[layer_idx];
                    let shared_kv = if prev_idx == layer_idx {
                        None
                    } else {
                        stored_kvs[prev_idx].as_ref()
                    };
                    let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
                    let (new_h, new_kv) = layer.forward(
                        &h,
                        shared_kv,
                        per_layer,
                        base_offset,
                        None,
                        kv_is_rotating[layer_idx],
                        device,
                    )?;
                    h = new_h;
                    if let Some(kv) = new_kv {
                        stored_kvs[layer_idx] = Some(kv);
                    }
                }
            }
            Some(cs) => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let prev_idx = self.previous_kvs[layer_idx];
                    let shared_kv = if prev_idx == layer_idx {
                        None
                    } else {
                        stored_kvs[prev_idx].as_ref()
                    };
                    let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
                    let cache = if prev_idx == layer_idx {
                        Some(&mut cs[layer_idx])
                    } else {
                        None
                    };
                    let (new_h, new_kv) = layer.forward(
                        &h,
                        shared_kv,
                        per_layer,
                        base_offset,
                        cache,
                        kv_is_rotating[layer_idx],
                        device,
                    )?;
                    h = new_h;
                    if let Some(kv) = new_kv {
                        stored_kvs[layer_idx] = Some(kv);
                    }
                }
            }
        }

        let h = self.final_norm.forward(&h, device)?;

        // Slice the last `k` positions: [1, seq, hidden] → [1, k, hidden].
        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, k, hidden], device)?;

        let logits = self.embed_tokens.as_linear(&h_last, device)?;

        let cap = self.cfg.final_logit_softcapping;
        let logits = apply_softcap(&logits, cap, device)?;

        Ok(logits)
    }

    /// Cache-using forward returning the **pre-final-norm** hidden states at
    /// the last `k` positions (— MTP conditioning signal).
    ///
    /// This is the trunk output *before* `final_norm` and the LM head — the
    /// exact signal the qwen3.5 MTP drafter conditions on (mlx-vlm
    /// `_mtp_verify_without_logits` calls `lm.model(..., skip_final_norm=True)`
    /// and the MTP head re-derives logits from it via
    /// `speculative_logits_from_hidden`). Returns shape `[1, k, hidden]`.
    ///
    /// Reads + writes the provided per-layer `caches` exactly like
    /// `forward_arr_last_k`; the cache offset advances by `ids.len()`. The only
    /// difference from the logit path is that this stops at the trunk output and
    /// skips `final_norm` + LM-head + softcap.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_hidden_states(
        &self,
        ids: &[u32],
        k: usize,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_hidden_states: k={k} out of range for seq={seq}"
            )));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        let seq = seq as i32;
        let k = k as i32;
        let base_offset = cache_base_offset(caches.as_deref());

        let h_raw = self.embed_tokens.forward(&ids_arr, device)?;
        // Match mlx-lm dtype discipline: the embed-scale constant adopts the
        // activation dtype (bf16 for mxfp8) instead of a strong-F32 scalar.
        // A strong-F32 scalar would promote the whole residual stream to f32,
        // which then propagates through Q/K/V projections, attention, and the
        // global KV cache (doubling its residency on the None path). mlx-lm
        // multiplies by a Python float (weak type) / a `mx.array(_, bf16)`,
        // keeping the stream at the model dtype.
        let embed_scale =
            scalar_f32((self.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
        let h = multiply(&h_raw, &embed_scale, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let per_layer_inputs = self.compute_per_layer_inputs(&ids_arr, &h, device)?;

        let mut stored_kvs: Vec<Option<SharedKvOut>> =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        let mut h = h;
        let kv_is_rotating: Vec<bool> = (0..self.cfg.num_hidden_layers)
            .map(|i| {
                let src = self.previous_kvs[i];
                caches
                    .as_ref()
                    .and_then(|cs| cs.get(src))
                    .is_some_and(|c| c.is_rotating())
            })
            .collect();
        match caches {
            None => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let prev_idx = self.previous_kvs[layer_idx];
                    let shared_kv = if prev_idx == layer_idx {
                        None
                    } else {
                        stored_kvs[prev_idx].as_ref()
                    };
                    let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
                    let (new_h, new_kv) = layer.forward(
                        &h,
                        shared_kv,
                        per_layer,
                        base_offset,
                        None,
                        kv_is_rotating[layer_idx],
                        device,
                    )?;
                    h = new_h;
                    if let Some(kv) = new_kv {
                        stored_kvs[layer_idx] = Some(kv);
                    }
                }
            }
            Some(cs) => {
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let prev_idx = self.previous_kvs[layer_idx];
                    let shared_kv = if prev_idx == layer_idx {
                        None
                    } else {
                        stored_kvs[prev_idx].as_ref()
                    };
                    let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
                    let cache = if prev_idx == layer_idx {
                        Some(&mut cs[layer_idx])
                    } else {
                        None
                    };
                    let (new_h, new_kv) = layer.forward(
                        &h,
                        shared_kv,
                        per_layer,
                        base_offset,
                        cache,
                        kv_is_rotating[layer_idx],
                        device,
                    )?;
                    h = new_h;
                    if let Some(kv) = new_kv {
                        stored_kvs[layer_idx] = Some(kv);
                    }
                }
            }
        }

        // Pre-final-norm trunk output — NO final_norm, NO LM-head, NO softcap.
        // Slice the last `k` positions: [1, seq, hidden] → [1, k, hidden].
        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        h_last.reshape(&[1, k, hidden], device)
    }

    /// Like [`forward_hidden_states`] but also returns the verifier's
    /// representative per-layer-type K/V for the Gemma4-assistant MTP drafter
    ///. The assistant drafter shares the verifier's K/V rather than
    /// keeping its own cache: every drafter `sliding_attention` layer attends
    /// over the verifier's **last** sliding-layer K/V, and the drafter's
    /// `full_attention` layer attends over the verifier's **last** full-layer
    /// K/V (mirrors mlx-vlm `_mtp_shared_kv_from_prompt_cache`, which keys the
    /// shared dict by `layer.layer_type` and is overwritten last-wins).
    ///
    /// Returns `(hidden[1,k,H], sliding_kv, full_kv, kv_offset)` where each
    /// `(K, V)` is `[1, n_kv_heads, kv_len, head_dim]` (post-RoPE, as stored in
    /// the verifier cache) and `kv_offset` is the verifier sequence position of
    /// the cache *before* this call (the absolute position the draft tokens are
    /// placed at). The caches advance by `ids.len()` exactly as in
    /// `forward_hidden_states`.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_hidden_states_shared_kv(
        &self,
        ids: &[u32],
        k: usize,
        caches: &mut [KvCache],
        device: Device,
    ) -> Result<(Array, (Array, Array), (Array, Array), i32)> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_hidden_states_shared_kv: k={k} out of range for seq={seq}"
            )));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        let seq = seq as i32;
        let k = k as i32;
        let base_offset = cache_base_offset(Some(caches));

        let h_raw = self.embed_tokens.forward(&ids_arr, device)?;
        // Match mlx-lm dtype discipline: the embed-scale constant adopts the
        // activation dtype (bf16 for mxfp8) instead of a strong-F32 scalar.
        // A strong-F32 scalar would promote the whole residual stream to f32,
        // which then propagates through Q/K/V projections, attention, and the
        // global KV cache (doubling its residency on the None path). mlx-lm
        // multiplies by a Python float (weak type) / a `mx.array(_, bf16)`,
        // keeping the stream at the model dtype.
        let embed_scale =
            scalar_f32((self.cfg.hidden_size as f32).sqrt()).astype(h_raw.dtype(), device)?;
        let h = multiply(&h_raw, &embed_scale, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let per_layer_inputs = self.compute_per_layer_inputs(&ids_arr, &h, device)?;

        let mut stored_kvs: Vec<Option<SharedKvOut>> =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        let mut h = h;
        let kv_is_rotating: Vec<bool> = (0..self.cfg.num_hidden_layers)
            .map(|i| {
                let src = self.previous_kvs[i];
                caches.get(src).is_some_and(|c| c.is_rotating())
            })
            .collect();
        let cs = &mut *caches;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prev_idx = self.previous_kvs[layer_idx];
            let shared_kv = if prev_idx == layer_idx {
                None
            } else {
                stored_kvs[prev_idx].as_ref()
            };
            let per_layer = per_layer_inputs.as_ref().map(|pli| &pli[layer_idx]);
            let cache = if prev_idx == layer_idx {
                Some(&mut cs[layer_idx])
            } else {
                None
            };
            let (new_h, new_kv) = layer.forward(
                &h,
                shared_kv,
                per_layer,
                base_offset,
                cache,
                kv_is_rotating[layer_idx],
                device,
            )?;
            h = new_h;
            if let Some(kv) = new_kv {
                stored_kvs[layer_idx] = Some(kv);
            }
        }

        // Representative shared K/V: highest-index cache-holding layer of each
        // type (last-wins, matching the Python dict-overwrite semantics).
        // This speculative-decode entry point returns bf16 (K, V) to the
        // drafter machinery. The fused-quant shared-KV decode share surfaces
        // quant 3-tuples instead of bf16 on the Mixed global path, which this
        // bf16 contract cannot consume — so only `Bf16` payloads are picked.
        // A `MixedQuant` payload falls through to the loud "no … K/V captured"
        // error below rather than silently mis-feeding the drafter; fused-quant
        // KV share + Gemma4 speculative decode is unsupported in this slice.
        let pick = |want: super::config::LayerType| -> Option<(Array, Array)> {
            for i in (0..self.cfg.num_hidden_layers).rev() {
                if self.cfg.layer_types[i] == want {
                    if let Some(SharedKvOut::Bf16(kk, vv)) = &stored_kvs[i] {
                        return Some((kk.try_clone().ok()?, vv.try_clone().ok()?));
                    }
                }
            }
            None
        };
        let sliding_kv = pick(super::config::LayerType::SlidingAttention).ok_or_else(|| {
            rmlx_core::error::Error::Model(
                "forward_hidden_states_shared_kv: no sliding-layer K/V captured".into(),
            )
        })?;
        let full_kv = pick(super::config::LayerType::FullAttention).ok_or_else(|| {
            rmlx_core::error::Error::Model(
                "forward_hidden_states_shared_kv: no full-layer K/V captured".into(),
            )
        })?;

        let hidden = self.cfg.hidden_size as i32;
        let h_last = h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, k, hidden], device)?;
        Ok((h_last, sliding_kv, full_kv, base_offset))
    }

    /// Apply the final RMSNorm to a pre-final-norm hidden (— MTP drafter
    /// conditioning). Mirrors mlx-vlm `speculative_draft_hidden(h) =
    /// model.norm(h)`: the Gemma4-assistant drafter conditions on the *normed*
    /// trunk hidden, not the raw pre-norm hidden. `[1, n, hidden]` in/out.
    pub fn apply_final_norm(&self, hidden: &Array, device: Device) -> Result<Array> {
        self.final_norm.forward(hidden, device)
    }

    /// Target-scaled input embedding of a single token id (— MTP drafter).
    ///
    /// Returns `embed_tokens(tok) * embed_scale` (`embed_scale = sqrt(hidden)`),
    /// shape `[1, 1, hidden]`. The Gemma4-assistant drafter feeds this
    /// (concatenated with the verifier hidden) into its `pre_projection`.
    ///
    /// mlx-vlm `bind()` resolves `inner = target.language_model.model`
    /// (`Gemma4TextModel`) and reads `inner.embed_scale = hidden_size**0.5`
    /// (≈39.19 for E2B), then applies it to the target-token embed:
    /// `tok_embed = self._input_embed(tok) * self._input_embed_scale`. The prior
    /// "Gemma4 has none → scale 1.0" claim was wrong — the un-scaled embed left
    /// the `b`-token conditioning ~40× too small, collapsing the MTP accept
    /// rate to ~0. Verified by numeric diff vs the mlx-vlm reference drafter:
    /// scale 1.0 → drafter argmax 7001 (reject); scale sqrt(1536) → 5279 (==
    /// verifier prediction → accept).
    pub fn embed_token_raw(&self, tok: u32, device: Device) -> Result<Array> {
        let ids = [tok as i32];
        let ids_bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[1], Dtype::I32)?;
        let e = self.embed_tokens.forward(&ids_arr, device)?;
        let e = e.reshape(&[1, 1, self.cfg.hidden_size as i32], device)?;
        let scale = scalar_f32((self.cfg.hidden_size as f32).sqrt());
        multiply(&e, &scale, device)
    }

    /// Re-derive logits from a pre-final-norm hidden state.
    ///
    /// Inverse of the tail dropped by [`forward_hidden_states`]: applies
    /// `final_norm` → tied LM head (`embed_tokens.as_linear`) → final-logit
    /// softcap. Mirrors mlx-vlm `speculative_logits_from_hidden`, used by the
    /// MTP deferred-greedy acceptance walk to score draft positions against the
    /// verifier without re-running the trunk. `hidden`: `[1, n, hidden]`,
    /// returns `[1, n, vocab]`.
    pub fn logits_from_hidden(&self, hidden: &Array, device: Device) -> Result<Array> {
        let h = self.final_norm.forward(hidden, device)?;
        let logits = self.embed_tokens.as_linear(&h, device)?;
        apply_softcap(&logits, self.cfg.final_logit_softcapping, device)
    }

    /// Compute per-position per-layer inputs.
    ///
    /// `ids`: [seq] — all token ids in this forward call.
    /// `h`: [1, seq, hidden] — all hidden states (post-embedding).
    ///
    /// Returns `None` if the model has no per-layer input embeddings.
    /// Otherwise returns `Some(Vec<Array>)` of length `n_layers`,
    /// each element shape `[1, seq, hidden_per_layer]` — per-position gating
    /// for that layer.
    ///
    /// Per-position is required for KV-cache equivalence: the K/V cached
    /// during prefill must match the K/V at the same positions in a
    /// full-sequence forward pass. The earlier Stage-1 simplification
    /// (last-token broadcast over all positions) made cached K differ from
    /// fresh K and broke prefill+decode.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn compute_per_layer_inputs(
        &self,
        ids: &Array,
        h: &Array,
        device: Device,
    ) -> Result<Option<Vec<Array>>> {
        let (embed_per_layer, proj, norm) = match (
            &self.embed_tokens_per_layer,
            &self.per_layer_model_proj,
            &self.per_layer_proj_norm,
        ) {
            (Some(e), Some(p), Some(n)) => (e, p, n),
            _ => return Ok(None),
        };

        let n = self.cfg.num_hidden_layers;
        let d = self.cfg.hidden_size_per_layer_input;
        let seq = ids.shape()[0];

        // Per-layer-input scale constants adopt the operand dtype (bf16 for
        // mxfp8), mirroring mlx-lm where these scales are Python floats (weak
        // types that do not promote the activation). A strong-F32 scalar here
        // would promote the per-layer-input tensor to f32, which then promotes
        // the residual stream through the per-layer-input gating `add` in the
        // decoder layer — and from there into Q/K/V and the global KV cache.
        //
        // Token embedding for per-layer: [seq, n * d] → reshape [1, seq, n, d].
        let raw = embed_per_layer.forward(ids, device)?;
        let scale = scalar_f32((d as f32).sqrt()).astype(raw.dtype(), device)?;
        let raw = multiply(&raw, &scale, device)?;
        let raw = raw.reshape(&[1, seq, n as i32, d as i32], device)?;

        // Model projection: h is [1, seq, hidden] → proj → [1, seq, n * d].
        // Ref: mlx-lm/mlx_lm/models/gemma4_text.py _project_per_layer_inputs:
        // per_layer_projection = norm(proj * scale)
        // return (per_layer_projection + per_layer_inputs) * 2^-0.5
        let proj_out = proj.forward(h, device)?;
        let proj_scale = scalar_f32((self.cfg.hidden_size as f32).powf(-0.5))
            .astype(proj_out.dtype(), device)?;
        let proj_out = multiply(&proj_out, &proj_scale, device)?;
        let proj_out = proj_out.reshape(&[1, seq, n as i32, d as i32], device)?;

        let proj_normed = norm.forward(&proj_out, device)?;

        let combined = rmlx_mlx::add(&proj_normed, &raw, device)?;
        let inv_sqrt2 = scalar_f32((-0.5f32).exp2()).astype(combined.dtype(), device)?;
        let combined = multiply(&combined, &inv_sqrt2, device)?;

        // Split into per-layer slices: each [1, seq, d].
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let sliced = combined.slice(
                &[0, 0, i as i32, 0],
                &[1, seq, (i + 1) as i32, d as i32],
                &[1, 1, 1, 1],
                device,
            )?;
            let sliced = sliced.reshape(&[1, seq, d as i32], device)?;
            result.push(sliced);
        }

        Ok(Some(result))
    }
}

/// Logit softcap: `tanh(x / cap) * cap`.
///
/// Thin wrapper around `softcap_fused` (mx.compile-fused). Identical math
/// to mlx-lm's `@partial(mx.compile, shapeless=True) logit_softcap`. See
/// `gemma4/layers.rs::softcap_fused` for the compile-cache pattern.
pub(super) fn apply_softcap(logits: &Array, cap: f32, device: Device) -> Result<Array> {
    softcap_fused(logits, cap, device)
}
