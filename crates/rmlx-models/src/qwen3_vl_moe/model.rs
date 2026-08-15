// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Qwen3-VL-MoE text decoder model struct + forward pass.
//!
//! Plain Qwen3-MoE GQA stack with 3D interleaved M-RoPE. The vision tower and
//! image branch are wired by [`super::vision`] / the generator; this struct is
//! the text-decode core (also exercises text-only generation directly).

#![allow(clippy::redundant_closure_for_method_calls)]
use std::mem::size_of_val;

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use crate::layers::{build_chunked_prefill_mask, pick_attn_mask_mode, RmsNorm};
use rmlx_kv_quant::KvCache;

use super::attention::Attention;
use super::config::Qwen3VlMoeTextConfig;
use super::layers::{Embedding, Linear};
use super::moe::{DenseMlp, SparseMoeBlock};
use super::mrope::{build_interleaved_mrope_tables, RopeIndex3D};

#[allow(missing_debug_implementations)]
pub(super) enum MlpBlock {
    Moe(Box<SparseMoeBlock>),
    Dense(DenseMlp),
}

impl MlpBlock {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            MlpBlock::Moe(m) => m.forward(x, device),
            MlpBlock::Dense(m) => m.forward(x, device),
        }
    }
}

#[allow(missing_debug_implementations)]
pub(super) struct DecoderLayer {
    pub(super) input_layernorm: RmsNorm,
    pub(super) post_attention_layernorm: RmsNorm,
    pub(super) self_attn: Attention,
    pub(super) mlp: MlpBlock,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: Option<&mut KvCache>,
        prebuilt_mask: Option<&Array>,
        mask_mode: &str,
        device: Device,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward(x, device)?;
        let attn =
            self.self_attn
                .forward(&normed, cos, sin, cache, prebuilt_mask, mask_mode, device)?;
        let h = rmlx_mlx::add(x, &attn, device)?;
        let normed2 = self.post_attention_layernorm.forward(&h, device)?;
        let mlp = self.mlp.forward(&normed2, device)?;
        rmlx_mlx::add(&h, &mlp, device)
    }
}

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model struct — private weight fields; public API is forward_seq() and forward_multimodal(); adding a field requires updating load_weights and the qwen3_vl_moe loader"
)]
/// Qwen3VLMoeForConditionalGeneration text decoder weights.
#[allow(missing_debug_implementations)]
pub struct Qwen3VlMoeText {
    /// Parsed text-decoder configuration.
    pub cfg: Qwen3VlMoeTextConfig,
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) final_norm: RmsNorm,
    pub(super) lm_head: Option<Linear>,
    /// Resident-KV byte total of this instance's last generation, paired with a
    /// store sequence. Per model instance, never per arch — two models of the
    /// same architecture must not write each other's figure.
    pub(crate) kv_bytes: crate::kv_bytes::KvBytesCounter,
}

/// Full Qwen3-VL-MoE model: the text decoder plus the image-token / vision
/// metadata the [`crate::arch::Architecture`] variant exposes. The vision tower
/// itself is loaded separately by the server (mirroring the Gemma4/Gemma3
/// `VisionBundle` pattern) so text-only requests pay no vision-load cost.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model struct — fields are the complete Qwen3-VL-MoE top-level model contract; adding a field requires updating the qwen3_vl_moe loader and Architecture dispatch"
)]
#[allow(missing_debug_implementations)]
pub struct Qwen3VlMoe {
    /// Text decoder model.
    pub text: Qwen3VlMoeText,
    /// Token id for `<image>` soft tokens.
    pub image_token_id: i64,
    /// Vision tower spatial merge factor.
    pub spatial_merge_size: usize,
    /// Maximum supported sequence length from config.
    pub max_position_embeddings: u32,
}

impl Qwen3VlMoeText {
    /// Text-only forward: sequential 3D positions (t==h==w==index+offset).
    /// Returns logits for the last position, `[1, 1, vocab]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Text-only forward with optional per-layer KV cache.
    pub fn forward_seq_with_cache(
        &self,
        ids: &[u32],
        kv_caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;

        // Base offset from the cache (prior tokens already consumed).
        let base_offset = kv_caches
            .as_ref()
            .and_then(|cs| cs.first())
            .map_or(0, |c| c.offset());

        // Sequential 3D positions for this chunk: position = base_offset + k.
        let positions: Vec<i64> = (0..seq as i64)
            .map(|k| i64::from(base_offset) + k)
            .collect();
        let pos = RopeIndex3D {
            t: positions.clone(),
            h: positions.clone(),
            w: positions,
        };
        self.forward_arr(&ids_arr, seq as i32, &pos, base_offset, kv_caches, device)
    }

    /// Core forward over an MLX id array with explicit 3D positions.
    ///
    /// `pos` holds the (t,h,w) position id for each of the `seq` tokens in this
    /// chunk (for text these are all equal; for image spans they differ — see
    /// [`super::mrope::get_rope_index`]). `base_offset` is the KV-cache offset
    /// the chunk starts at (drives the attention mask mode).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        pos: &RopeIndex3D,
        base_offset: i32,
        kv_caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        // Build the per-token interleaved M-RoPE cos/sin tables [seq, head_dim].
        let (cos_v, sin_v) = build_interleaved_mrope_tables(
            pos,
            self.cfg.head_dim,
            f64::from(self.cfg.rope_theta),
            &self.cfg.mrope_section,
        )?;
        let hd = self.cfg.head_dim as i32;
        let cos = f32_to_bf16_arr(&cos_v, &[1, 1, seq, hd], device)?;
        let sin = f32_to_bf16_arr(&sin_v, &[1, 1, seq, hd], device)?;

        let mask_mode = pick_attn_mask_mode(base_offset, seq);
        let shared_mask: Option<Array> = if mask_mode == "array" {
            Some(build_chunked_prefill_mask(base_offset, seq, device)?)
        } else {
            None
        };

        let h = self.embed_tokens.forward(ids_arr, device)?;
        let mut h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        match kv_caches {
            Some(kv) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(
                        &h,
                        &cos,
                        &sin,
                        Some(&mut kv[i]),
                        shared_mask.as_ref(),
                        mask_mode,
                        device,
                    )?;
                }
            }
            None => {
                for layer in &self.layers {
                    h = layer.forward(
                        &h,
                        &cos,
                        &sin,
                        None,
                        shared_mask.as_ref(),
                        mask_mode,
                        device,
                    )?;
                }
            }
        }

        let h = self.final_norm.forward(&h, device)?;
        // Last position only.
        let last = h.slice(
            &[0, seq - 1, 0],
            &[1, seq, self.cfg.hidden_size as i32],
            &[1, 1, 1],
            device,
        )?;
        match &self.lm_head {
            Some(lin) => lin.forward(&last, device),
            None => self.embed_as_linear(&last, device),
        }
    }

    /// Embed token ids into the LM hidden space, `[1, seq, hidden]`. Used by the
    /// image branch to build `inputs_embeds` before the vision-feature scatter.
    pub(crate) fn embed_ids(&self, ids: &[u32], device: Device) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), seq * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq as i32], Dtype::I32)?;
        let h = self.embed_tokens.forward(&ids_arr, device)?;
        h.reshape(&[1, seq as i32, self.cfg.hidden_size as i32], device)
    }

    /// Chunked image-branch prefill: forward precomputed `inputs_embeds` `[1,
    /// seq, hidden]` (text embeddings with the vision features already scattered
    /// at the image-token positions) with explicit 3D M-RoPE `pos`, optionally
    /// injecting `deepstack_embeds[k]` additively at `visual_positions` after
    /// decoder layer `k` (mirrors `language.py::_deepstack_process`: the first
    /// `len(deepstack_embeds)` layers get an injection).
    ///
    /// The prompt is encoded in `prefill_chunk`-token slices so a long image
    /// prompt (thousands of image soft tokens) does not run a single multi-
    /// thousand-token forward in one Metal command buffer (the ~10s GPU
    /// watchdog). Each chunk advances the per-layer [`KvCache`] offset; the
    /// downstream chunked-prefill SDPA mask path engages automatically for
    /// `base_offset > 0`. Deepstack visual injection is applied per-layer to the
    /// subset of `visual_positions` that fall inside the current chunk.
    ///
    /// `pos` holds the 3D M-RoPE position id for every token; `visual_positions`
    /// is the (contiguous) image-token run, aligned 1:1 with each
    /// `deepstack_embeds[k]` row. Returns logits for the final position
    /// `[1, 1, vocab]`.
    ///
    /// `prefill_chunk` must be ≥ 1; a value ≥ `seq` collapses to a single
    /// forward.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn forward_embeds_chunked(
        &self,
        inputs_embeds: &Array,
        seq: i32,
        pos: &RopeIndex3D,
        deepstack_embeds: &[Array],
        visual_positions: &[usize],
        prefill_chunk: usize,
        kv_caches: &mut [KvCache],
        device: Device,
    ) -> Result<Array> {
        let chunk = prefill_chunk.max(1) as i32;
        let n_deep = deepstack_embeds.len();
        let hid = self.cfg.hidden_size as i32;
        // Contiguous image-token run: [vis_lo, vis_hi). Empty when there are no
        // visual tokens (pure-text augmented prompt — defensive).
        let vis_lo = visual_positions.first().copied().unwrap_or(0);
        let vis_hi = vis_lo + visual_positions.len();

        let mut last_logits: Option<Array> = None;
        let mut start = 0_i32;
        while start < seq {
            let end = (start + chunk).min(seq);
            let clen = end - start;

            // Slice this chunk's embeds and 3D positions.
            let chunk_embeds =
                inputs_embeds.slice(&[0, start, 0], &[1, end, hid], &[1, 1, 1], device)?;
            let chunk_pos = RopeIndex3D {
                t: pos.t[start as usize..end as usize].to_vec(),
                h: pos.h[start as usize..end as usize].to_vec(),
                w: pos.w[start as usize..end as usize].to_vec(),
            };

            // Visual positions inside this chunk, mapped to chunk-local indices,
            // and the matching deepstack rows. The run is contiguous, so the
            // intersection [lo, hi) is a single sub-run.
            let lo = (start as usize).max(vis_lo);
            let hi = (end as usize).min(vis_hi);
            let chunk_visual_positions: Vec<usize> = if lo < hi {
                (lo - start as usize..hi - start as usize).collect()
            } else {
                Vec::new()
            };

            let (cos_v, sin_v) = build_interleaved_mrope_tables(
                &chunk_pos,
                self.cfg.head_dim,
                f64::from(self.cfg.rope_theta),
                &self.cfg.mrope_section,
            )?;
            let hd = self.cfg.head_dim as i32;
            let cos = f32_to_bf16_arr(&cos_v, &[1, 1, clen, hd], device)?;
            let sin = f32_to_bf16_arr(&sin_v, &[1, 1, clen, hd], device)?;

            let base_offset = kv_caches[0].offset();
            tracing::debug!(
                chunk_start = start,
                chunk_end = end,
                clen,
                base_offset,
                "qwen3_vl_moe image prefill chunk"
            );
            let mask_mode = pick_attn_mask_mode(base_offset, clen);
            let shared_mask: Option<Array> = if mask_mode == "array" {
                Some(build_chunked_prefill_mask(base_offset, clen, device)?)
            } else {
                None
            };

            let mut h = chunk_embeds;
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(
                    &h,
                    &cos,
                    &sin,
                    Some(&mut kv_caches[i]),
                    shared_mask.as_ref(),
                    mask_mode,
                    device,
                )?;
                if i < n_deep && lo < hi {
                    // Deepstack rows aligned 1:1 with visual_positions; slice the
                    // [lo - vis_lo, hi - vis_lo) rows for this chunk.
                    let row_lo = (lo - vis_lo) as i32;
                    let row_hi = (hi - vis_lo) as i32;
                    let ds = &deepstack_embeds[i];
                    let ds_chunk = ds.slice(&[row_lo, 0], &[row_hi, hid], &[1, 1], device)?;
                    h = super::image::deepstack_inject(
                        &h,
                        &ds_chunk,
                        &chunk_visual_positions,
                        device,
                    )?;
                }
            }

            // Flush the per-chunk command buffer under the ~10s Metal watchdog.
            // Eval the chunk hidden directly (forces this chunk's full forward,
            // including the K/V writes) so a long image prompt does not fold all
            // chunks into one buffer. Non-final chunks then skip the lm_head.
            h.eval()?;
            if end < seq {
                for c in kv_caches.iter() {
                    c.eval_prefill_state()?;
                }
            } else {
                let h = self.final_norm.forward(&h, device)?;
                let last = h.slice(&[0, clen - 1, 0], &[1, clen, hid], &[1, 1, 1], device)?;
                let logits = match &self.lm_head {
                    Some(lin) => lin.forward(&last, device)?,
                    None => self.embed_as_linear(&last, device)?,
                };
                last_logits = Some(logits);
            }

            start = end;
        }

        last_logits.ok_or_else(|| {
            rmlx_core::error::Error::Model(
                "qwen3_vl_moe forward_embeds_chunked: empty prompt produced no logits".into(),
            )
        })
    }

    fn embed_as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        // tie_word_embeddings path (target snapshot has tie=false, so unused,
        // but kept for completeness).
        match &self.embed_tokens {
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

fn f32_to_bf16_arr(data: &[f32], shape: &[i32], device: Device) -> Result<Array> {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
    let a = Array::from_bytes(bytes, shape, Dtype::F32)?;
    a.astype(Dtype::Bf16, device)
}
