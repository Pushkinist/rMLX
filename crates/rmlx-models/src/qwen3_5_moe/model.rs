//! `Qwen3_5MoeText` model struct and forward pass.
//!
//! [`Qwen3_5MoeText`] holds the full model weights for the
//! `Qwen3_5MoeForConditionalGeneration` (and PARO-dense) architecture and
//! orchestrates per-layer decode: embedding, decoder stack (GDN or attention
//! + dense/sparse MoE), final RMSNorm, and LM-head projection.
//!
//! # Public API
//!
//! - [`Qwen3_5MoeText`] — model struct, constructed by
//!   [`super::loader::load_from_path`] or [`super::loader::load_from_path_paro`].

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::redundant_closure_for_method_calls, clippy::ref_option)]
use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{DecoderLayer, MlpBlock};
use super::layers::{Embedding, Linear, RmsNorm};
use rmlx_kv_quant::{KvCache, LinearAttnCache};

// ---------------------------------------------------------------------------
// Hot-head cache
// ---------------------------------------------------------------------------

/// Cached result of gathering the draft-vocab row subset from the LM head
/// (or tied `embed_tokens`) weight.
///
/// Built once on the first call to `hot_logits_from_final_hidden`; reused on
/// every subsequent call. The cache is valid for the lifetime of the model
/// because `hot_ids` is fixed at drafter load time and the LM head weights
/// are immutable after loading.
///
/// Mirrors mlx-vlm's `_hot_lm_head_cache_key` / `w_hot` cached attribute on
/// `Eagle3Rounds`.
pub(super) struct CachedHotHead {
    /// Gathered weight rows `[draft_vocab, H]` (plain) or packed `[draft_vocab, H/pack]`
    /// (quantized). Ready to be used directly in the restricted matmul.
    pub(super) w_hot: Array,
    /// Scales for quantized heads (`[draft_vocab, n_groups]`), `None` for plain.
    pub(super) s_hot: Option<Array>,
    /// Biases for quantized heads, `None` for plain or absent-bias quantized.
    pub(super) b_hot: Option<Array>,
    /// Original `group_size` and `bits` — kept for the quantized matmul call.
    pub(super) group_size: i32,
    pub(super) bits: i32,
    /// Whether this is a quantized (true) or plain (false) head.
    pub(super) is_quantized: bool,
    /// Mode string for quantized matmul (`"affine"` etc.), empty for plain.
    pub(super) mode: String,
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Qwen3_5MoeForConditionalGeneration model weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct Qwen3_5MoeText {
    /// Parsed model configuration.
    pub cfg: Qwen3_5MoeConfig,
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: RmsNorm,
    pub(super) lm_head: Option<Linear>,
    /// Cached gathered LM-head rows for the EAGLE-3 restricted-vocab hot-path.
    /// Populated lazily on the first call to `hot_logits_from_final_hidden`.
    pub(super) cached_hot_head: std::sync::OnceLock<CachedHotHead>,
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

impl Qwen3_5MoeText {
    /// Whether any decoder layer actually resolved to a sparse-MoE MLP block.
    ///
    /// This is the *built* truth, not a declaration: the loader selects
    /// dense-vs-MoE per layer from the checkpoint's tensor witness
    /// (`mlp.switch_mlp.gate_proj.weight`), so a config that names the wrong
    /// architecture cannot change the answer. Both Qwen3.5 arch strings share
    /// this one model struct, so callers that need to distinguish sparse MoE
    /// from dense SwiGLU must ask here rather than read `architectures[0]`.
    pub fn has_sparse_moe_layers(&self) -> bool {
        self.layers
            .iter()
            .any(|l| matches!(l.mlp, MlpBlock::Moe(_)))
    }

    /// The arch class this instance actually resolved to.
    ///
    /// Single source of the dense-vs-MoE name for this family:
    /// `Architecture::arch_class()` delegates here, and the generate path uses
    /// it for its `arch` tracing field so one run cannot emit two different
    /// values for the same model.
    pub fn arch_class(&self) -> &'static str {
        if self.has_sparse_moe_layers() {
            "Qwen3_5MoeForConditionalGeneration"
        } else {
            "Qwen3_5ForConditionalGeneration"
        }
    }

    /// Full-sequence forward pass (no KV cache).
    /// Returns logits for the last position, shape `[1, 1, vocab_size]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, None, device)
    }

    /// Forward pass with optional KV + linear-attention caches.
    ///
    /// `kv_caches` — one `KvCache` per decoder layer (only consulted by
    /// `FullAttention` layers; ignored by `GatedDeltaNet`). When `None`,
    /// FullAttention layers recompute K/V from scratch.
    ///
    /// `lin_caches` — one `LinearAttnCache` per decoder layer (only consulted
    /// by `GatedDeltaNet` layers; ignored by `FullAttention`). When `None`,
    /// linear-attention layers start from a zero conv tail and zero delta
    /// state on every call.
    pub fn forward_seq_with_cache(
        &self,
        ids: &[u32],
        kv_caches: Option<&mut [KvCache]>,
        lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;
        self.forward_arr(&ids_arr, seq as i32, kv_caches, lin_caches, device)
    }

    /// Forward pass with token IDs already in an MLX Array. Used by the
    /// async-pipelined decode loop so the next forward can chain on top of
    /// the prior step's argmax without a CPU sync.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        kv_caches: Option<&mut [KvCache]>,
        lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        // base_offset for RoPE must come from a FullAttention layer's KvCache.
        // GatedDeltaNet (linear-attn) layers never advance KvCache::offset, so
        // reading kv_caches[0] (which is GDN at layer 0) returns 0 forever even
        // mid-decode. Use the first FullAttention layer (full_attention_interval - 1).
        let fa_idx = self.cfg.full_attention_interval.saturating_sub(1);
        let base_offset = kv_caches
            .as_ref()
            .and_then(|cs| cs.get(fa_idx))
            .map_or(0, |c| c.offset());

        // Build the chunked-prefill array mask ONCE per forward call (when
        // applicable) and share it across all 10 FullAttention layers.
        // Saves N-1 redundant mask Vec allocs + GPU uploads per chunk.
        let shared_mask: Option<Array> =
            if crate::layers::pick_attn_mask_mode(base_offset, seq) == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    base_offset,
                    seq,
                    device,
                )?)
            } else {
                None
            };

        let h = self.embed_tokens.forward(ids_arr, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let mut h = h;
        // We index into both caches by layer; using two `Option<&mut [...]>`
        // keeps the borrow checker happy without an enum wrapper.
        // Per-layer trace! removed from the hot path: 40 layers × decode
        // step = 40 log emissions per token, JSON-serialized to disk under
        // the default debug,rmlx=trace filter. Net cost was non-trivial at
        // 4096-token bench (164k log lines). Layer-level tracing remains
        // available via tracing::Span if/when needed.
        match (kv_caches, lin_caches) {
            (None, None) => {
                for layer in &self.layers {
                    h = layer.forward(&h, base_offset, None, None, shared_mask.as_ref(), device)?;
                }
            }
            (Some(kv), None) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        Some(&mut kv[i]),
                        None,
                        shared_mask.as_ref(),
                        device,
                    )?;
                }
            }
            (None, Some(lin)) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        None,
                        Some(&mut lin[i]),
                        shared_mask.as_ref(),
                        device,
                    )?;
                }
            }
            (Some(kv), Some(lin)) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        Some(&mut kv[i]),
                        Some(&mut lin[i]),
                        shared_mask.as_ref(),
                        device,
                    )?;
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

    /// Cache-using forward returning logits at the **last `k` positions**
    /// (L36 Phase 4 — speculative verifier path).
    ///
    /// Mirrors `forward_arr` exactly but slices `[seq-k..seq]` before the
    /// LM head instead of `[seq-1..seq]`, returning shape `[1, k, vocab]`.
    /// Reads + writes BOTH the per-layer `kv_caches` (FullAttention) and the
    /// recurrent `lin_caches` (GatedDeltaNet) so a single call advances the
    /// full hybrid state by `ids.len()` positions. Used by speculative
    /// decoding to feed `K+1` new tokens (1 carry-token + K draft tokens)
    /// through the verifier's persistent cache in one forward.
    ///
    /// This is additive: the plain `forward_arr` / `forward_seq` decode path
    /// is untouched. The last-K slice over the same `ids` (with the same
    /// pre-state) produces logits identical to a plain `forward_arr` per
    /// position — only the slice width differs.
    pub fn forward_seq_last_k_with_cache(
        &self,
        ids: &[u32],
        k: usize,
        kv_caches: &mut [KvCache],
        lin_caches: Option<&mut [LinearAttnCache]>,
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
        self.forward_arr_last_k(
            &ids_arr, seq as i32, k as i32, kv_caches, lin_caches, device,
        )
    }

    /// Forward pass returning logits at the last `k` positions, reading +
    /// writing both KV and recurrent caches.
    ///
    /// Body mirrors `forward_arr` (same hybrid-cache layer loop, same RoPE
    /// offset / chunked-prefill mask logic) but slices `[seq-k..seq]` before
    /// the LM head. Returns shape `[1, k, vocab]`. `k == 1` is exactly the
    /// `forward_arr` decode path.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_arr_last_k(
        &self,
        ids_arr: &Array,
        seq: i32,
        k: i32,
        kv_caches: &mut [KvCache],
        lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        if k < 1 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_arr_last_k: k={k} out of range for seq={seq}"
            )));
        }

        // RoPE base offset: read from the first FullAttention layer's KvCache
        // (GDN layers never advance KvCache::offset). Mirrors `forward_arr`.
        let fa_idx = self.cfg.full_attention_interval.saturating_sub(1);
        let base_offset = kv_caches.get(fa_idx).map_or(0, |c| c.offset());

        let shared_mask: Option<Array> =
            if crate::layers::pick_attn_mask_mode(base_offset, seq) == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    base_offset,
                    seq,
                    device,
                )?)
            } else {
                None
            };

        let h = self.embed_tokens.forward(ids_arr, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let mut h = h;
        match lin_caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        Some(&mut kv_caches[i]),
                        None,
                        shared_mask.as_ref(),
                        device,
                    )?;
                }
            }
            Some(lin) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        Some(&mut kv_caches[i]),
                        Some(&mut lin[i]),
                        shared_mask.as_ref(),
                        device,
                    )?;
                }
            }
        }

        let h = self.final_norm.forward(&h, device)?;
        let hidden = self.cfg.hidden_size as i32;

        // Slice the last `k` positions: [1, seq, hidden] → [1, k, hidden].
        let h_last = h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, k, hidden], device)?;

        match &self.lm_head {
            Some(lm) => lm.forward(&h_last, device),
            None => self.embed_tokens.as_linear(&h_last, device),
        }
    }

    /// Multi-layer hidden capture for the DFlash drafter.
    ///
    /// Forwards `ids` through the hybrid stack reading + writing both caches
    /// (exactly like [`forward_arr_last_k`]) but, instead of slicing the final
    /// logits, captures the **residual-stream output** of each layer in
    /// `capture_layer_ids` and concatenates them along the feature axis.
    /// Mirrors the mlx-vlm `qwen3_5` `Qwen3_5Model.__call__(capture_layer_ids=...)`
    /// path (`hidden_sink.append(h)` AFTER layer `i`), then the DFlash round-loop
    /// does `mx.concatenate(verify_out.hidden_states, axis=-1)`.
    ///
    /// Returns the captured-and-concatenated hidden at the **last `k`**
    /// positions: `[1, k, len(capture_layer_ids) * hidden]`. No final-norm is
    /// applied (the drafter's `fc`/`hidden_norm` consume the raw per-layer
    /// residual stream, matching the Python which appends `h` pre-`self.norm`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_hidden_states_multi(
        &self,
        ids: &[u32],
        k: usize,
        capture_layer_ids: &[usize],
        kv_caches: &mut [KvCache],
        lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_hidden_states_multi: k={k} out of range for seq={seq}"
            )));
        }
        if capture_layer_ids.is_empty() {
            return Err(rmlx_core::error::Error::Model(
                "forward_hidden_states_multi: capture_layer_ids empty".into(),
            ));
        }

        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        let seq = seq as i32;
        let k = k as i32;
        let fa_idx = self.cfg.full_attention_interval.saturating_sub(1);
        let base_offset = kv_caches.get(fa_idx).map_or(0, |c| c.offset());

        let shared_mask: Option<Array> =
            if crate::layers::pick_attn_mask_mode(base_offset, seq) == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    base_offset,
                    seq,
                    device,
                )?)
            } else {
                None
            };

        let h = self.embed_tokens.forward(&ids_arr, device)?;
        let mut h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let hidden = self.cfg.hidden_size as i32;
        // Capture the last-k slice of the residual stream output of each
        // captured layer, in the order requested (matches the Python which
        // appends in layer order and concatenates that list along axis -1).
        let mut captures: Vec<(usize, Array)> = Vec::with_capacity(capture_layer_ids.len());
        let capture_set: std::collections::HashSet<usize> =
            capture_layer_ids.iter().copied().collect();

        match lin_caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        Some(&mut kv_caches[i]),
                        None,
                        shared_mask.as_ref(),
                        device,
                    )?;
                    if capture_set.contains(&i) {
                        let slice =
                            h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
                        captures.push((i, slice.reshape(&[1, k, hidden], device)?));
                    }
                }
            }
            Some(lin) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        base_offset,
                        Some(&mut kv_caches[i]),
                        Some(&mut lin[i]),
                        shared_mask.as_ref(),
                        device,
                    )?;
                    if capture_set.contains(&i) {
                        let slice =
                            h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
                        captures.push((i, slice.reshape(&[1, k, hidden], device)?));
                    }
                }
            }
        }

        // Re-order captures into the requested `capture_layer_ids` order (the
        // loop above visits in ascending layer index; the drafter's `fc` was
        // trained on the config-listed order).
        let mut ordered: Vec<&Array> = Vec::with_capacity(capture_layer_ids.len());
        for want in capture_layer_ids {
            match captures.iter().find(|(idx, _)| idx == want) {
                Some((_, arr)) => ordered.push(arr),
                None => {
                    return Err(rmlx_core::error::Error::Model(format!(
                        "forward_hidden_states_multi: capture layer {want} out of \
                         range (num_hidden_layers={})",
                        self.layers.len()
                    )))
                }
            }
        }
        rmlx_mlx::concatenate(&ordered, -1, device)
    }

    /// Combined verify forward for the DFlash round-loop: one cached
    /// pass returning BOTH the last-k logits AND the concatenated multi-layer
    /// hidden capture, so the verifier runs only once per round (mirrors the
    /// Python `lm(verify_input, capture_layer_ids=...)` returning
    /// `verify_out.logits` + `verify_out.hidden_states`).
    ///
    /// Returns `(logits[1,k,vocab], concat_hidden[1,k,len(ids)*hidden])`.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_verify_capture(
        &self,
        ids: &[u32],
        k: usize,
        capture_layer_ids: &[usize],
        kv_caches: &mut [KvCache],
        mut lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<(Array, Array)> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_verify_capture: k={k} out of range for seq={seq}"
            )));
        }
        if capture_layer_ids.is_empty() {
            return Err(rmlx_core::error::Error::Model(
                "forward_verify_capture: capture_layer_ids empty".into(),
            ));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        let seq = seq as i32;
        let k = k as i32;
        let hidden = self.cfg.hidden_size as i32;
        let fa_idx = self.cfg.full_attention_interval.saturating_sub(1);
        let base_offset = kv_caches.get(fa_idx).map_or(0, |c| c.offset());

        let shared_mask: Option<Array> =
            if crate::layers::pick_attn_mask_mode(base_offset, seq) == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    base_offset,
                    seq,
                    device,
                )?)
            } else {
                None
            };

        let h = self.embed_tokens.forward(&ids_arr, device)?;
        let mut h = h.reshape(&[1, seq, hidden], device)?;

        let capture_set: std::collections::HashSet<usize> =
            capture_layer_ids.iter().copied().collect();
        let mut captures: Vec<(usize, Array)> = Vec::with_capacity(capture_layer_ids.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let lin_ref = lin_caches.as_deref_mut().map(|l| &mut l[i]);
            h = layer.forward(
                &h,
                base_offset,
                Some(&mut kv_caches[i]),
                lin_ref,
                shared_mask.as_ref(),
                device,
            )?;
            if capture_set.contains(&i) {
                let slice = h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
                captures.push((i, slice.reshape(&[1, k, hidden], device)?));
            }
        }

        // Logits from the final-normed last-k hidden.
        let h_normed = self.final_norm.forward(&h, device)?;
        let h_last = h_normed.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, k, hidden], device)?;
        let logits = self.logits_from_final_hidden(&h_last, device)?;

        // Concat captures in requested order.
        let mut ordered: Vec<&Array> = Vec::with_capacity(capture_layer_ids.len());
        for want in capture_layer_ids {
            match captures.iter().find(|(idx, _)| idx == want) {
                Some((_, arr)) => ordered.push(arr),
                None => {
                    return Err(rmlx_core::error::Error::Model(format!(
                        "forward_verify_capture: capture layer {want} out of range \
                         (num_hidden_layers={})",
                        self.layers.len()
                    )))
                }
            }
        }
        let concat = rmlx_mlx::concatenate(&ordered, -1, device)?;
        Ok((logits, concat))
    }

    /// Combined verify forward with final-normed hidden returned (hot-path).
    ///
    /// Same as [`forward_verify_capture`] but additionally returns
    /// `final_hidden[1,k,H]` — the final-RMSNorm'd verifier hidden at all k
    /// verify positions, before the LM head projection.
    ///
    /// `final_hidden` is the input for two hot-path operations:
    ///
    /// 1. Restricted matmul: `hot_logits_from_final_hidden(&final_hidden, hot_ids)`
    /// 2. Full-vocab correction at a single position `p`: slice
    ///
    /// No full-vocab `[1, k, vocab]` tensor is materialised — the caller is
    /// responsible for computing full-vocab logits only at the single correction
    /// position, matching the Python `_eagle3_verify_target_hot` reference.
    ///
    /// Returns `(concat_hidden[1,k,n_aux*H], final_hidden[1,k,H])`.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_verify_capture_hot(
        &self,
        ids: &[u32],
        k: usize,
        capture_layer_ids: &[usize],
        kv_caches: &mut [KvCache],
        mut lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<(Array, Array)> {
        let seq = ids.len();
        if k == 0 || k > seq {
            return Err(rmlx_core::error::Error::Model(format!(
                "forward_verify_capture_hot: k={k} out of range for seq={seq}"
            )));
        }
        if capture_layer_ids.is_empty() {
            return Err(rmlx_core::error::Error::Model(
                "forward_verify_capture_hot: capture_layer_ids empty".into(),
            ));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        let seq = seq as i32;
        let k = k as i32;
        let hidden = self.cfg.hidden_size as i32;
        let fa_idx = self.cfg.full_attention_interval.saturating_sub(1);
        let base_offset = kv_caches.get(fa_idx).map_or(0, |c| c.offset());

        let shared_mask: Option<Array> =
            if crate::layers::pick_attn_mask_mode(base_offset, seq) == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    base_offset,
                    seq,
                    device,
                )?)
            } else {
                None
            };

        let h = self.embed_tokens.forward(&ids_arr, device)?;
        let mut h = h.reshape(&[1, seq, hidden], device)?;

        let capture_set: std::collections::HashSet<usize> =
            capture_layer_ids.iter().copied().collect();
        let mut captures: Vec<(usize, Array)> = Vec::with_capacity(capture_layer_ids.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let lin_ref = lin_caches.as_deref_mut().map(|l| &mut l[i]);
            h = layer.forward(
                &h,
                base_offset,
                Some(&mut kv_caches[i]),
                lin_ref,
                shared_mask.as_ref(),
                device,
            )?;
            if capture_set.contains(&i) {
                let slice = h.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
                captures.push((i, slice.reshape(&[1, k, hidden], device)?));
            }
        }

        // Final-normed hidden at the last-k positions (: also returned).
        let h_normed = self.final_norm.forward(&h, device)?;
        let h_last = h_normed.slice(&[0, seq - k, 0], &[1, seq, hidden], &[1, 1, 1], device)?;
        let h_last = h_last.reshape(&[1, k, hidden], device)?;
        // No full-vocab logits here — the caller computes them for a single
        // correction position only (see `logits_from_hidden`), matching the
        // Python `_eagle3_verify_target_hot` reference.

        // Concat aux captures in requested order.
        let mut ordered: Vec<&Array> = Vec::with_capacity(capture_layer_ids.len());
        for want in capture_layer_ids {
            match captures.iter().find(|(idx, _)| idx == want) {
                Some((_, arr)) => ordered.push(arr),
                None => {
                    return Err(rmlx_core::error::Error::Model(format!(
                        "forward_verify_capture_hot: capture layer {want} out of range \
                         (num_hidden_layers={})",
                        self.layers.len()
                    )))
                }
            }
        }
        let concat = rmlx_mlx::concatenate(&ordered, -1, device)?;
        // h_last is [1, k, hidden] — final-normed hidden for hot-path restricted
        // matmul and single-position full-vocab correction.
        Ok((concat, h_last))
    }

    /// One-pass hidden capture + last-position logits for a single chunk.
    ///
    /// Runs the full hybrid layer stack over `ids` updating `kv_caches` and
    /// `lin_caches` as usual, but:
    ///
    /// - captures the residual-stream output at `capture_layer_ids` for **all**
    ///   positions (shape `[1, seq, n_aux*hidden]`) — needed for the drafter KV
    ///   prefill which conditions on every prompt position, and
    /// - applies final-norm + LM head only to the **last single position**
    ///   (shape `[1, 1, vocab]`) — avoiding a `[1, seq, vocab]` materialisation.
    ///
    /// This is the inner kernel of the chunked-prefill path; it runs the model
    /// exactly once, making it safe to call for every chunk without double-updating
    /// the caches.
    ///
    /// Returns `(logits[1,1,vocab], concat_hidden[1,seq,n_aux*hidden])`.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward_chunk_capture_last_logit(
        &self,
        ids: &[u32],
        capture_layer_ids: &[usize],
        kv_caches: &mut [KvCache],
        lin_caches: Option<&mut [LinearAttnCache]>,
        device: Device,
    ) -> Result<(Array, Array)> {
        let seq = ids.len();
        debug_assert!(!ids.is_empty());
        debug_assert!(!capture_layer_ids.is_empty());

        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        let seq_i = seq as i32;
        let hidden = self.cfg.hidden_size as i32;
        let fa_idx = self.cfg.full_attention_interval.saturating_sub(1);
        let base_offset = kv_caches.get(fa_idx).map_or(0, |c| c.offset());

        let shared_mask: Option<Array> =
            if crate::layers::pick_attn_mask_mode(base_offset, seq_i) == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    base_offset,
                    seq_i,
                    device,
                )?)
            } else {
                None
            };

        let h_embed = self.embed_tokens.forward(&ids_arr, device)?;
        let mut h = h_embed.reshape(&[1, seq_i, hidden], device)?;

        let capture_set: std::collections::HashSet<usize> =
            capture_layer_ids.iter().copied().collect();
        let mut captures: Vec<(usize, Array)> = Vec::with_capacity(capture_layer_ids.len());

        let mut lin_opt: Option<&mut [LinearAttnCache]> = lin_caches;
        for (i, layer) in self.layers.iter().enumerate() {
            let lin_ref = lin_opt.as_deref_mut().map(|l| &mut l[i]);
            h = layer.forward(
                &h,
                base_offset,
                Some(&mut kv_caches[i]),
                lin_ref,
                shared_mask.as_ref(),
                device,
            )?;
            if capture_set.contains(&i) {
                // Capture ALL positions for this aux layer (the drafter prefill
                // conditions on every prompt position, so we need the full slice).
                captures.push((i, h.try_clone()?));
            }
        }

        // Logits: final-norm on the last position only, then LM head.
        // Avoids materialising a [1, seq, vocab] tensor.
        let h_normed = self.final_norm.forward(&h, device)?;
        let h_last_norm =
            h_normed.slice(&[0, seq_i - 1, 0], &[1, seq_i, hidden], &[1, 1, 1], device)?;
        let h_last_norm = h_last_norm.reshape(&[1, 1, hidden], device)?;
        let last_logits = self.logits_from_final_hidden(&h_last_norm, device)?;

        // Re-order aux captures and slice to [1, seq, H] each.
        let mut ordered: Vec<Array> = Vec::with_capacity(capture_layer_ids.len());
        for want in capture_layer_ids {
            match captures.iter().find(|(idx, _)| idx == want) {
                Some((_, arr)) => {
                    let slice = arr.reshape(&[1, seq_i, hidden], device)?;
                    ordered.push(slice);
                }
                None => {
                    return Err(rmlx_core::error::Error::Model(format!(
                        "forward_chunk_capture_last_logit: capture layer {want} out of range"
                    )))
                }
            }
        }
        let refs: Vec<&Array> = ordered.iter().collect();
        let concat_hidden = rmlx_mlx::concatenate(&refs, -1, device)?;

        Ok((last_logits, concat_hidden))
    }

    /// Chunked variant of [`forward_verify_capture`] for long prompts.
    ///
    /// Splits the prompt into consecutive chunks of at most `chunk_size` tokens
    /// and runs each chunk through the verifier separately, accumulating the
    /// KV/GDN caches normally. Concatenates the per-chunk `concat_hidden`
    /// slices along the sequence axis to produce a single `[1, n, n_aux*hidden]`
    /// tensor covering all prompt positions — identical to what a single-shot
    /// `forward_verify_capture(..., k=n)` would return.
    ///
    /// Unlike the single-shot path, logits are materialised only for the **last
    /// position of the last chunk** (shape `[1, 1, vocab]`), and the per-layer
    /// aux hidden captures are flushed (`.eval()`) after every non-final chunk,
    /// so the peak Metal command-buffer footprint is bounded by a single chunk.
    ///
    /// For Qwen3.6-MoE with `hidden=2048`, `vocab=248320`, `n_aux=3`, a
    /// chunk of 1024 tokens costs at most:
    /// - aux hidden: 1024 × 3 × 2048 × 2 B ≈ 12 MB
    /// - logits (last chunk only): 1 × 248320 × 2 B ≈ 0.5 MB
    ///
    /// Far below the 4-5 s Metal watchdog budget that a 4096-token single-shot
    /// would exceed (~2 GB logits alone).
    ///
    /// Returns `(logits[1,1,vocab], concat_hidden[1,n,n_aux*hidden])`.
    ///
    /// When `ids.len() <= chunk_size` the entire prompt is a single chunk
    /// (no concatenation overhead).
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_verify_capture_chunked(
        &self,
        ids: &[u32],
        capture_layer_ids: &[usize],
        kv_caches: &mut [KvCache],
        lin_caches: Option<&mut [LinearAttnCache]>,
        chunk_size: usize,
        device: Device,
    ) -> Result<(Array, Array)> {
        let n = ids.len();
        if n == 0 {
            return Err(rmlx_core::error::Error::Model(
                "forward_verify_capture_chunked: empty ids".into(),
            ));
        }
        if capture_layer_ids.is_empty() {
            return Err(rmlx_core::error::Error::Model(
                "forward_verify_capture_chunked: capture_layer_ids empty".into(),
            ));
        }

        let chunk_size = chunk_size.max(1);
        let mut lin_opt: Option<&mut [LinearAttnCache]> = lin_caches;
        let mut hidden_chunks: Vec<Array> = Vec::new();
        let mut pos = 0usize;

        while pos < n {
            let end = (pos + chunk_size).min(n);
            let chunk = &ids[pos..end];
            let chunk_n = chunk.len();
            let is_last = end == n;

            tracing::debug!(pos, end, chunk_n, is_last, "eagle3 prefill chunk");

            let (last_logits_opt, hidden_chunk) = if is_last {
                // Final chunk: one pass — all aux hidden + last-position logits only.
                let (lg, hid) = self.forward_chunk_capture_last_logit(
                    chunk,
                    capture_layer_ids,
                    kv_caches,
                    lin_opt.as_deref_mut(),
                    device,
                )?;
                (Some(lg), hid)
            } else {
                // Non-final chunk: hidden capture only, no logits.
                let hid = self.forward_hidden_states_multi(
                    chunk,
                    chunk_n,
                    capture_layer_ids,
                    kv_caches,
                    lin_opt.as_deref_mut(),
                    device,
                )?;
                (None, hid)
            };

            // Materialise each chunk's GPU work before the next chunk so Metal
            // can reclaim intermediate buffers.
            hidden_chunk.eval()?;
            hidden_chunks.push(hidden_chunk);

            if let Some(last_logits) = last_logits_opt {
                let refs: Vec<&Array> = hidden_chunks.iter().collect();
                let full_hidden = if refs.len() == 1 {
                    refs[0].try_clone()?
                } else {
                    rmlx_mlx::concatenate(&refs, 1, device)?
                };
                return Ok((last_logits, full_hidden));
            }

            pos = end;
        }

        // Unreachable: `is_last` is true at the final iteration and returns early.
        Err(rmlx_core::error::Error::Model(
            "forward_verify_capture_chunked: internal: loop exited without last chunk".into(),
        ))
    }

    /// Raw input-token embedding (seam 3, DFlash drafter).
    ///
    /// Returns `[1, n, hidden]` — the plain `embed_tokens` lookup with NO
    /// scale. The Qwen3.5 verifier's `embed_tokens` is a bare `nn.Embedding`
    /// (no `embed_scale` attribute), so the DFlash `bind()` resolves
    /// `embed_scale = 1.0` — the drafter consumes the unscaled embedding.
    /// (Contrast Gemma4 , which scales by `sqrt(hidden)`.)
    pub fn embed_token_ids(&self, ids: &[i32], device: Device) -> Result<Array> {
        let n = ids.len();
        let ids_bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), n * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[n as i32], Dtype::I32)?;
        let h = self.embed_tokens.forward(&ids_arr, device)?;
        h.reshape(&[1, n as i32, self.cfg.hidden_size as i32], device)
    }

    /// Re-derive logits from a hidden state via the LM head.
    ///
    /// `hidden`: `[1, n, hidden]` → `[1, n, vocab]`. Uses the tied
    /// `embed_tokens.as_linear` when `lm_head` is absent (mirrors `forward_arr`).
    pub fn logits_from_hidden(&self, hidden: &Array, device: Device) -> Result<Array> {
        let h = self.final_norm.forward(hidden, device)?;
        self.logits_from_final_hidden(&h, device)
    }

    /// The same, from a hidden state the caller has already final-normed.
    ///
    /// The two entry points exist because a speculative loop holds one or the
    /// other depending on where in the verify pass it captured: naming which is
    /// which is what keeps a missing — or doubled — `final_norm` from becoming a
    /// silent reweighting of the vocabulary.
    pub fn logits_from_final_hidden(&self, hidden: &Array, device: Device) -> Result<Array> {
        match &self.lm_head {
            Some(lm) => lm.forward(hidden, device),
            None => self.embed_tokens.as_linear(hidden, device),
        }
    }

    /// Restricted-vocab logits for the EAGLE-3 hot-path.
    ///
    /// Computes `hidden @ W_hot.T` where `W_hot` contains only the `hot_ids`
    /// rows of the LM head weight — the draft-vocabulary subset (32 000 out of
    /// 248 320 rows). This is the same computation as `logits_from_hidden` but
    /// restricted to the draft vocabulary, reducing the logit materialisation by
    /// a factor of 248320/32000 ≈ 7.8×.
    ///
    /// `hidden`: `[1, k, H]` (final-normed, from `forward_verify_capture_hot`).
    /// `hot_ids`: `[draft_vocab_size]` i32 indices into the LM head row axis
    /// (= `arange(draft_vocab_size) + d2t.astype(i32)`, precomputed by the drafter).
    ///
    /// Returns `[1, k, draft_vocab_size]`.
    ///
    /// Handles both plain and quantized LM heads (and the tied-embedding case).
    ///
    /// # Caching
    ///
    /// The gather of `w_hot` (and `s_hot`/`b_hot` for quantized heads) is expensive
    /// (~80 MB at native bit-width). This method caches the gathered rows in
    /// `self.cached_hot_head` after the first call and reuses them on every
    /// subsequent call. The cache is valid for the model lifetime because `hot_ids`
    /// is fixed at drafter load time and LM head weights are immutable.
    ///
    /// Mirrors mlx-vlm's `_hot_lm_head_cache_key` / `w_hot` caching on
    /// `Eagle3Rounds`.
    ///
    /// # Note on `lm_head` vs `embed_tokens`
    ///
    /// rMLX prefers `lm_head` over `embed_tokens` (un-tied case), unlike mlx-vlm
    /// which always uses `lm.model.embed_tokens`. Numerically equivalent for
    /// Qwen3.5-MoE (which has tied weights); rMLX is correct on un-tied checkpoints
    /// (mlx-vlm would be wrong there).
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    pub fn hot_logits_from_final_hidden(
        &self,
        hidden: &Array,
        hot_ids: &Array,
        device: Device,
    ) -> Result<Array> {
        // Build (or retrieve) the cached hot-head weight rows.
        // `OnceLock::get_or_try_init` is not yet stable; use get + set instead.
        // On a rare concurrent first call, two gathers may run; the loser is
        // discarded — both results are identical (same weights, same hot_ids).
        let cache = if let Some(c) = self.cached_hot_head.get() {
            c
        } else {
            let built = build_cached_hot_head(&self.lm_head, &self.embed_tokens, hot_ids, device)?;
            let _ = self.cached_hot_head.set(built);
            self.cached_hot_head
                .get()
                .expect("cached_hot_head was just set")
        };

        // Run the restricted matmul using the cached gathered rows.
        if cache.is_quantized {
            rmlx_mlx::quantized_matmul(
                hidden,
                &cache.w_hot,
                cache
                    .s_hot
                    .as_ref()
                    .expect("quantized hot-head: s_hot missing"),
                cache.b_hot.as_ref(),
                cache.group_size,
                cache.bits,
                &cache.mode,
                true,
                device,
            )
        } else {
            // hidden [1,k,H] @ w_hot.T [H,draft_vocab] → [1,k,draft_vocab].
            rmlx_mlx::matmul(hidden, &cache.w_hot.transpose(&[1, 0], device)?, device)
        }
    }
}

/// Build a `CachedHotHead` by gathering the draft-vocab row subset from
/// the LM head (or tied embed_tokens) weight at `hot_ids` positions.
///
/// Called once per model instance via `OnceLock::get_or_try_init`.
fn build_cached_hot_head(
    lm_head: &Option<Linear>,
    embed_tokens: &Embedding,
    hot_ids: &Array,
    device: Device,
) -> Result<CachedHotHead> {
    match lm_head {
        Some(lm) => match lm {
            Linear::Plain { weight } => {
                let w_hot = weight.take(hot_ids, 0, device)?;
                w_hot.eval()?;
                Ok(CachedHotHead {
                    w_hot,
                    s_hot: None,
                    b_hot: None,
                    group_size: 0,
                    bits: 0,
                    is_quantized: false,
                    mode: String::new(),
                })
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => {
                let w_hot = weight.take(hot_ids, 0, device)?;
                let s_hot = scales.take(hot_ids, 0, device)?;
                let b_hot = biases
                    .as_ref()
                    .map(|b| b.take(hot_ids, 0, device))
                    .transpose()?;
                w_hot.eval()?;
                s_hot.eval()?;
                if let Some(b) = &b_hot {
                    b.eval()?;
                }
                Ok(CachedHotHead {
                    w_hot,
                    s_hot: Some(s_hot),
                    b_hot,
                    group_size: *group_size,
                    bits: *bits,
                    is_quantized: true,
                    mode: mode.clone(),
                })
            }
            Linear::Paro { .. } => {
                // PARO lm_head is not expected in practice (PARO is for weight
                // quantization, not output heads). Fall back to full-vocab logits.
                Err(rmlx_core::error::Error::Model(
                    "hot_logits_from_final_hidden: Paro lm_head not supported for \
                     restricted-vocab hot-path (not expected in practice)"
                        .into(),
                ))
            }
        },
        None => {
            // Tied embedding: take draft-vocab rows from embed_tokens weight.
            // Note: rMLX prefers `lm_head` over `embed_tokens` (un-tied case),
            // unlike mlx-vlm which always uses `lm.model.embed_tokens`. Equivalent
            // for Qwen3.5-MoE which has tied weights.
            match embed_tokens {
                Embedding::Plain { weight } => {
                    let w_hot = weight.take(hot_ids, 0, device)?;
                    w_hot.eval()?;
                    Ok(CachedHotHead {
                        w_hot,
                        s_hot: None,
                        b_hot: None,
                        group_size: 0,
                        bits: 0,
                        is_quantized: false,
                        mode: String::new(),
                    })
                }
                Embedding::Quantized {
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                    mode,
                } => {
                    let w_hot = weight.take(hot_ids, 0, device)?;
                    let s_hot = scales.take(hot_ids, 0, device)?;
                    let b_hot = biases
                        .as_ref()
                        .map(|b| b.take(hot_ids, 0, device))
                        .transpose()?;
                    w_hot.eval()?;
                    s_hot.eval()?;
                    if let Some(b) = &b_hot {
                        b.eval()?;
                    }
                    Ok(CachedHotHead {
                        w_hot,
                        s_hot: Some(s_hot),
                        b_hot,
                        group_size: *group_size,
                        bits: *bits,
                        is_quantized: true,
                        mode: mode.clone(),
                    })
                }
            }
        }
    }
}
