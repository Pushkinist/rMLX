//! Gemma4 layer implementations: Attention, PerLayerInput, and MoE block.
//!
//! Contains the three compute-heavy layer types for `Gemma4ForConditionalGeneration`:
//! the multi-head attention block with per-head q/k norms and SWA support,
//! the `PerLayerInput` gating projection, and the sparse MoE block with a
//! router + per-expert MLPs. All types are `pub(crate)` — they are wired
//! together in [`super::model`] and not part of the crate's public surface.
//!
//! # Key types (pub(crate))
//!
//! - `Attention` — multi-head attention with GQA, per-head q/k norms, and
//!   proportional RoPE; dispatches to [`KvCache::update_and_sdpa`].
//! - `PerLayerInputProjection` — gating projection consumed by every decoder
//!   layer (Gemma4-specific altup architecture).
//! - `Gemma4MoE` — sparse MoE block: router softmax + top-K expert dispatch.
//! - [`build_proportional_rope_freqs`] — constructs the frequency tensor for
//!   proportional (Gemma4-style) RoPE.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::implicit_clone,
    clippy::struct_field_names,
    clippy::too_many_lines
)]
use rmlx_core::error::{Error, Result};
use rmlx_mlx::{
    broadcast_to, expand_dims, rope, rope_with_freqs, scaled_dot_product_attention, Array, Device,
    Dtype,
};

use crate::layers::{Linear, RmsNorm};
use rmlx_kv_quant::mixed_quant::mixed_quantized_sdpa;
use rmlx_kv_quant::{KvCache, SharedKvOut};

use super::config::LayerType;

pub(super) mod kernels;
pub(super) mod moe;

#[cfg(test)]
#[path = "mask_tests.rs"]
mod mask_tests;

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod kernels_tests;

// Re-exports used by decoder_layer.rs and model.rs (siblings of layers/).
pub(super) use kernels::{geglu_fused, qk_norm_fused, softcap_fused};
pub(super) use moe::{Gemma4Experts, Gemma4MoeBlock, Gemma4Router, PerLayerInput};

// ---------------------------------------------------------------------------
// ProportionalRoPE frequency table builder
// ---------------------------------------------------------------------------

/// Build the precomputed frequency table for ProportionalRoPE (Gemma4 full-attention).
///
/// Reference: `mlx_lm/models/rope_utils.py` `ProportionalRoPE.__init__` (line ~215):
/// ```python
/// exponents = mx.arange(0, rotated_dims, 2) / dims # / global_head_dim, not rotated_dims
/// freqs = mx.concatenate([
/// factor * (base ** exponents), # grows: 1.0 .. ~30 for Gemma4
/// mx.full(((dims - rotated_dims) // 2,), mx.inf), # unrotated suffix → no rotation
/// ])
/// ```
///
/// Returns a 1-D F32 array of shape `[global_head_dim / 2]`.
/// Indices 0..`rotated_dims/2` hold `theta^(2*i / global_head_dim)` (values ≥ 1.0).
/// Indices `rotated_dims/2..global_head_dim/2` hold `+inf` (unrotated pairs).
///
/// For Gemma4 4B (global_head_dim=512, rotated_dims=128, theta=1e6):
/// - 64 rotated values: 1.0, ~1.06, ..., ~30.0
/// - 192 inf values
/// - total shape [256]
pub(crate) fn build_proportional_rope_freqs(
    global_head_dim: usize,
    rotated_dims: usize,
    theta: f32,
) -> Result<Array> {
    let half_rotated = rotated_dims / 2; // 64 for Gemma4
    let half_unrotated = (global_head_dim - rotated_dims) / 2; // 192 for Gemma4
    let total = global_head_dim / 2; // 256 for Gemma4

    let mut freqs = Vec::with_capacity(total);

    // Rotated half: theta^(2*i / global_head_dim)
    for i in 0..half_rotated {
        let exponent = (2 * i) as f32 / global_head_dim as f32;
        freqs.push(theta.powf(exponent));
    }
    // Unrotated suffix: +inf tells the MLX kernel to skip these pairs.
    for _ in 0..half_unrotated {
        freqs.push(f32::INFINITY);
    }

    // Convert to bytes for Array::from_bytes.
    let bytes: Vec<u8> = freqs.iter().flat_map(|&v| v.to_le_bytes()).collect();

    Array::from_bytes(&bytes, &[total as i32], Dtype::F32)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct Attention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Option<Linear>,
    pub(super) v_proj: Option<Linear>,
    pub(super) o_proj: Linear,
    pub(super) q_norm: RmsNorm,
    pub(super) k_norm: Option<RmsNorm>,
    pub(super) v_norm: RmsNorm, // RMSNormNoScale (weight=None)
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) layer_type: LayerType,
    /// SWA window size in tokens. Only relevant when `layer_type == SlidingAttention`.
    /// Set from `sliding_window` in the model config (e.g. 1024 for 31B, 512 for e4b).
    pub(super) sliding_window: usize,
    /// Sliding-attention RoPE: `dims` to rotate and `theta`. Used when
    /// `layer_type == SlidingAttention`. MLX computes freqs from theta internally.
    pub(super) rope_dims: i32,
    pub(super) rope_theta: f32,
    /// Full-attention ProportionalRoPE frequency table, cached at build time.
    ///
    /// Shape: `[global_head_dim / 2]` = `[256]` for the 4B model.
    /// Layout: `[freqs_rotated (64 values) | +inf (192 values)]`.
    /// `freqs_rotated[i] = theta^(2*i / global_head_dim)` for i in 0..64.
    /// The +inf tail instructs the MLX kernel to leave the unrotated pairs
    /// untouched (standard ProportionalRoPE / mlx-lm convention).
    ///
    /// Present only on `LayerType::FullAttention` layers; `None` for sliding.
    /// Computed once in the loader (not per-forward-call) via
    /// `build_proportional_rope_freqs`.
    pub(super) proportional_rope_freqs: Option<Array>,
}

impl Attention {
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        shared_kv: Option<&SharedKvOut>,
        offset: i32,
        cache: Option<&mut KvCache>,
        kv_is_rotating: bool,
        device: Device,
    ) -> Result<(Array, Option<SharedKvOut>)> {
        let shape = x.shape(); // [batch, seq, hidden]
        let batch = shape[0];
        let seq = shape[1];

        // Rotating-cache flag is passed in by the caller. Shared-KV
        // layers (`prev_kvs[i] != i`) get it from the source layer's cache,
        // since their own cache slot is unused (`cache: None`) but their K
        // shape is still rotating-format (`min(window-1, prev_offset) + S`).
        let attn_is_rotating = kv_is_rotating;

        // Q projection + reshape. Per-head q_norm fuses with k_norm via
        // qk_norm_fused on non-shared-KV layers (fused QK-norm). On shared-KV layers
        // the upstream K is already normed and we run q_norm alone.
        let q = self.q_proj.forward(x, device)?;
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;

        // RoPE dispatch:
        // SlidingAttention: standard rope(dims=head_dim=256, theta=10000). MLX computes
        // freqs internally as 1/theta^(2i/dims). Correct for sliding layers.
        // FullAttention: ProportionalRoPE — freqs = theta^(2i / global_head_dim=512)
        // with +inf for the unrotated suffix. dims = global_head_dim = 512 (NOT 128).
        // Using dims=128 would divide the exponent by 128 instead of 512, giving
        // ~27 000× higher frequencies at the top rotated dim — the prior bug.
        // K/V — either shared from previous layer or computed here.
        // Shared-KV layers do NOT push to their own cache; they read from the
        // upstream layer's already-concatenated KV (passed in via shared_kv).
        // Q on shared-KV path: q_norm alone (cannot fuse — no fresh K).
        // Q on non-shared path: qk_norm_fused with K (see else branch).
        //
        // Cache-holding layers route SDPA through
        // `KvCache::update_and_sdpa_returning_kv` — the universal wrapper's
        // shared-KV variant that returns the accumulated `(K, V)` so the
        // source-of-shared-KV layers can also fill `new_kv` for downstream
        // consumers. `cache` is consumed here; below we branch on whether the
        // wrapper has already run (`attn_out_holder`) or we still need a
        // direct `scaled_dot_product_attention` on shared / cacheless K/V.
        let mut attn_out_holder: Option<Array> = None;
        let (q, k, v, new_kv) = if let Some(shared) = shared_kv {
            // Shared KV: q_norm runs alone, then transpose + RoPE on Q.
            let q = self.q_norm.forward(&q, device)?;
            let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]
            let q = match self.layer_type {
                LayerType::SlidingAttention => rope(
                    &q,
                    self.rope_dims,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                    device,
                )?,
                LayerType::FullAttention => {
                    let freqs = self.proportional_rope_freqs.as_ref().expect(
                        "Attention(FullAttention): proportional_rope_freqs is None — build bug",
                    );
                    rope_with_freqs(&q, self.head_dim as i32, false, 1.0, offset, freqs, device)?
                }
            };
            match shared {
                SharedKvOut::Bf16(sk, sv) => (q, sk.try_clone()?, sv.try_clone()?, None),
                SharedKvOut::MixedQuant {
                    k,
                    v,
                    k_rotation,
                    k_bits,
                    v_bits,
                    k_group_size,
                    v_group_size,
                } => {
                    // Fused-quant shared-KV consumer: attend the producer's
                    // quant store directly via `mixed_quantized_sdpa` instead of
                    // a dequantized bf16 SDPA. This path is only produced on the
                    // GLOBAL (FullAttention) decode branch (seq == 1), where the
                    // attention mask is empty ("") — so no additive mask is
                    // needed and no causal/array mask rework applies. GQA head
                    // broadcast is handled inside `mixed_quantized_sdpa`.
                    if seq != 1 {
                        return Err(Error::Model(format!(
                            "Gemma4 fused-quant shared-KV consumer reached with seq={seq} \
                             (expected decode seq==1; quant share is decode-only)"
                        )));
                    }
                    let attn_out = mixed_quantized_sdpa(
                        &q,
                        k,
                        v,
                        1.0,
                        None,
                        *k_group_size,
                        *k_bits,
                        *v_group_size,
                        *v_bits,
                        k_rotation.as_ref(),
                        device,
                    )?;
                    attn_out_holder = Some(attn_out);
                    // The consumer's own SDPA already ran (`attn_out_holder` is
                    // Some), so the k/v slots are unused; placeholder clones of q
                    // keep the tuple well-typed without allocating fresh tensors.
                    let k_unused = q.try_clone()?;
                    let v_unused = q.try_clone()?;
                    (q, k_unused, v_unused, None)
                }
            }
        } else {
            let k_proj = self.k_proj.as_ref().ok_or_else(|| {
                Error::Model("Attention: has_kv=false but no shared_kv provided".to_owned())
            })?;
            let v_proj = self
                .v_proj
                .as_ref()
                .ok_or_else(|| Error::Model("Attention: has_kv=false but no v_proj".to_owned()))?;

            let k = k_proj.forward(x, device)?;
            let k = k.reshape(
                &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
                device,
            )?;
            // Fuse Q+K per-head RMSNorm into one compiled Metal program.
            // Both q_norm and k_norm have weight=Some on non-shared-KV layers
            // (RMSNormNoScale only applies to v_norm).
            let k_norm = self
                .k_norm
                .as_ref()
                .expect("Attention: shared_kv=None but k_norm is None — load bug");
            let q_w = self.q_norm.weight.as_ref().expect(
                "Attention: q_norm.weight is None — gemma4 q_norm requires a learned scale",
            );
            let k_w = k_norm.weight.as_ref().expect(
                "Attention: k_norm.weight is None — gemma4 k_norm requires a learned scale",
            );
            let (q_n, k_n) = qk_norm_fused(&q, &k, q_w, k_w, self.q_norm.eps, device)?;
            let q = q_n.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]
            let q = match self.layer_type {
                LayerType::SlidingAttention => rope(
                    &q,
                    self.rope_dims,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                    device,
                )?,
                LayerType::FullAttention => {
                    let freqs = self.proportional_rope_freqs.as_ref().expect(
                        "Attention(FullAttention): proportional_rope_freqs is None — build bug",
                    );
                    rope_with_freqs(&q, self.head_dim as i32, false, 1.0, offset, freqs, device)?
                }
            };
            let k = k_n.transpose(&[0, 2, 1, 3], device)?;
            let k = match self.layer_type {
                LayerType::SlidingAttention => rope(
                    &k,
                    self.rope_dims,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                    device,
                )?,
                LayerType::FullAttention => {
                    let freqs = self.proportional_rope_freqs.as_ref().expect(
                        "Attention(FullAttention): proportional_rope_freqs is None — build bug",
                    );
                    rope_with_freqs(&k, self.head_dim as i32, false, 1.0, offset, freqs, device)?
                }
            };

            let v = v_proj.forward(x, device)?;
            let v = v.reshape(
                &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
                device,
            )?;
            let v = self.v_norm.forward(&v, device)?;
            let v = v.transpose(&[0, 2, 1, 3], device)?;

            // Mask must be built before the cache update so the wrapper can run
            // update + SDPA in one shot. The mask's key dim MUST equal the
            // post-update K seq dim the SDPA actually attends, otherwise mlx-c
            // `scaled_dot_product_attention` rejects the broadcast (issue #32:
            // mask `(1,1,5,kv+1)` vs scores `(1,8,5,kv)`).
            //
            // The post-update K length is `producer_offset + seq` (non-rotating)
            // or the ring-capped `min(max_size-1, producer_offset) + seq`
            // (rotating). `producer_offset` is the **cache-holding layer's own**
            // `offset()`, which can drift from the model-wide `offset`
            // (`cache_base_offset`, picked from the first full-attention cache)
            // by one position across a speculative verify-block rollback: the
            // rotating sliding cache that drives the round's `v_target` rolls
            // back with no-op semantics once it wraps, desyncing its reported
            // offset from the non-rotating producer. RoPE still uses the
            // model-wide `offset` (absolute position); only the mask's key dim
            // is bound to the producer's own K length. See
            // `update_and_sdpa_returning_kv` (rotating short-circuit) for the
            // sibling Mixed-path fix this generalises.
            if let Some(c) = cache {
                let producer_offset = c.offset();
                let effective_offset = producer_effective_offset(
                    producer_offset,
                    attn_is_rotating,
                    self.sliding_window,
                );
                let total_kv_len_pre = effective_offset + seq;
                let (mask_holder_pre, mask_mode_pre) = build_attn_mask(
                    self.layer_type,
                    seq,
                    effective_offset,
                    total_kv_len_pre,
                    attn_is_rotating,
                    self.sliding_window,
                    device,
                )?;
                // Cache-holding layer: route through the shared-KV variant of
                // the universal wrapper. Returns (out, shared_kv) so the
                // accumulated K/V can be handed to downstream consumer layers
                // via `stored_kvs[layer_idx]` in `gemma4/model.rs`. The wrapper
                // surfaces bf16 K/V for None/K8V4/K8V8/Planar/SWA/prefill, and —
                // on the Mixed GLOBAL decode path — the live quant 3-tuples so
                // the consumer attends the quant store directly (no bf16
                // re-inflation of the O(ctx) global KV).
                let (attn_out, shared) = c.update_and_sdpa_returning_shared_kv(
                    &q,
                    &k,
                    &v,
                    1.0,
                    mask_mode_pre,
                    mask_holder_pre.as_ref(),
                    device,
                )?;
                // Guard (issue #32): the array-mode mask's key dim must equal
                // the K seq dim the SDPA just attended. A mismatch is a sizing
                // bug, not user input — fail loudly here rather than let a
                // later layer hit the opaque mlx-c broadcast error. Only the
                // bf16-surfacing path exposes a K seq dim to check; the quant
                // path's mask was already validated inside the wrapper's SDPA.
                if let (Some(mask), SharedKvOut::Bf16(k_full, _)) =
                    (mask_holder_pre.as_ref(), &shared)
                {
                    let mask_kv = mask.shape()[3];
                    let k_seq = k_full.shape()[2];
                    if mask_kv != k_seq {
                        return Err(Error::Model(format!(
                            "Gemma4 attention mask key dim {mask_kv} != K seq dim {k_seq} \
                             (layer_type={:?}, seq={seq}, producer_offset={producer_offset})",
                            self.layer_type
                        )));
                    }
                }
                attn_out_holder = Some(attn_out);
                // Producer's own SDPA already ran inside the wrapper
                // (`attn_out_holder` is Some), so the `k`/`v` slots below are
                // unused for this branch — placeholder clones of the freshly
                // projected K/V keep the tuple well-typed. `new_kv` carries the
                // shared payload downstream.
                (q, k.try_clone()?, v.try_clone()?, Some(shared))
            } else {
                // No-cache forward (e.g. eval path with `caches: None`). The
                // freshly-computed K is exactly `offset + seq` long, so size the
                // mask from the model-wide `offset` directly.
                let effective_offset = if attn_is_rotating {
                    offset.min(self.sliding_window as i32 - 1)
                } else {
                    offset
                };
                let total_kv_len_pre = effective_offset + seq;
                let (mask_holder_pre, mask_mode_pre) = build_attn_mask(
                    self.layer_type,
                    seq,
                    effective_offset,
                    total_kv_len_pre,
                    attn_is_rotating,
                    self.sliding_window,
                    device,
                )?;
                let k_new = k.try_clone()?;
                let v_new = v.try_clone()?;
                let attn_out = scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    1.0,
                    mask_mode_pre,
                    mask_holder_pre.as_ref(),
                    device,
                )?;
                attn_out_holder = Some(attn_out);
                (q, k, v, Some(SharedKvOut::Bf16(k_new, v_new)))
            }
        };

        // GQA: MLX's fast SDPA kernel handles head broadcasting natively when
        // `n_q_heads % n_kv_heads == 0` — even with array-mode masks (SWA banded /
        // chunked-prefill). Skip the manual `repeat_kv` expand to avoid two
        // broadcast+reshape ops per attention call (2 ops × n_layers per
        // decode step). Reference impl preserved below for documentation; same
        // optimisation already shipped on Gemma3 using the same SDPA
        // kernel and SWA mask logic.
        let _ = repeat_kv;

        // SDPA: cache-holding and no-cache-non-shared branches dispatched
        // inline above (cache-holding via `update_and_sdpa_returning_kv`,
        // no-cache via direct SDPA). The shared-KV consumer branch reaches
        // this point with `attn_out_holder == None`; run direct SDPA here on
        // the upstream-shared K/V.
        //
        // Mask selection per layer type:
        // SlidingAttention:
        // - prefill (seq > 1): banded-causal SWA mask [1,1,seq,offset+seq].
        // - decode (seq == 1), total_kv_len <= window: no mask ("").
        // - decode (seq == 1), total_kv_len > window: SWA decode mask [1,1,1,total_kv_len].
        // FullAttention:
        // - prefill, offset == 0: "causal".
        // - prefill, offset > 0: explicit chunked-prefill mask [1,1,seq,offset+seq].
        // - decode: "" (no mask).
        let attn_out = if let Some(out) = attn_out_holder {
            out
        } else {
            // Shared-KV consumer branch: K is fetched from the producer layer
            // (`shared_kv`), not freshly projected here. Its seq dim
            // (`k.shape()[2]`) is the ground truth for how many keys SDPA
            // attends. Size the mask's key dim from that actual K length, NOT
            // from the model-wide `offset` (#32 part 2): across a speculative
            // partial-accept verify rollback the producer cache is rolled back
            // while the model-wide `offset` (`cache_base_offset`, taken from the
            // first full-attention cache that was NOT rolled back) is not, so
            // `offset` can be one-or-more positions above the real K length.
            // Sizing the band from `offset` then yields a mask one key too wide
            // → the opaque mlx-c SDPA broadcast crash.
            //
            // `effective_offset = total_kv_len - seq` (K length minus the
            // newly-projected query tokens) is the symmetric counterpart of the
            // producer-branch `producer_effective_offset`. In the non-spec path
            // (no rollback) `total_kv_len == offset + seq` exactly (rotating:
            // `total_kv_len == offset.min(window-1) + seq`), so
            // `total_kv_len - seq` reproduces the IDENTICAL effective offset the
            // old `offset`-based sizing produced — including the rotating cap,
            // which is already baked into the producer's K length. No regression
            // on ordinary prefill/decode; the value only diverges (correctly)
            // when a rollback desynced `offset` from the K it shares.
            let total_kv_len = k.shape()[2]; // [B, kv_heads, total_kv, D]
            if total_kv_len < seq {
                return Err(Error::Model(format!(
                    "Gemma4 consumer/shared-KV K seq dim {total_kv_len} < query seq {seq} \
                     (layer_type={:?}, offset={offset})",
                    self.layer_type
                )));
            }
            let effective_offset = consumer_effective_offset(total_kv_len, seq);
            let (mask_holder, mask_mode) = build_attn_mask(
                self.layer_type,
                seq,
                effective_offset,
                total_kv_len,
                attn_is_rotating,
                self.sliding_window,
                device,
            )?;
            // Guard (issue #32 part 2): the array-mode consumer mask's key dim
            // must equal the K seq dim the SDPA is about to attend. Fail loudly
            // on any future off-by-one rather than surfacing the opaque mlx-c
            // broadcast error from a later frame.
            if let Some(mask) = mask_holder.as_ref() {
                let mask_kv = mask.shape()[3];
                if mask_kv != total_kv_len {
                    return Err(Error::Model(format!(
                        "Gemma4 consumer/shared-KV mask key dim {mask_kv} != K seq dim \
                         {total_kv_len} (layer_type={:?}, seq={seq}, offset={offset})",
                        self.layer_type
                    )));
                }
            }
            scaled_dot_product_attention(&q, &k, &v, 1.0, mask_mode, mask_holder.as_ref(), device)?
        };
        // [B, H, S, D] -> [B, S, H*D]
        let attn_out = attn_out.transpose(&[0, 2, 1, 3], device)?;
        let attn_out =
            attn_out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        let out = self.o_proj.forward(&attn_out, device)?;
        Ok((out, new_kv))
    }
}

/// Compute the effective mask offset from a **producer cache's own offset**
/// (not the model-wide `cache_base_offset`).
///
/// This is the single place that implements the #32 fix: at a speculative
/// verify-block step the rotating sliding cache that drove the round's
/// `v_target` can desync from the non-rotating full-attention producer by one
/// position across a partial-accept rollback. Using the producer's `c.offset()`
/// (not the model-wide `offset` argument) and capping it for rotating caches
/// keeps the mask key dim equal to the post-update K seq dim.
///
/// Extracted as a named helper so the selection is covered by
/// `mask_tests::guard_invariant_*` without requiring a full model forward pass.
#[cfg_attr(test, allow(dead_code))]
pub(super) fn producer_effective_offset(
    producer_offset: i32,
    attn_is_rotating: bool,
    sliding_window: usize,
) -> i32 {
    if attn_is_rotating {
        producer_offset.min(sliding_window as i32 - 1)
    } else {
        producer_offset
    }
}

/// Compute the effective mask offset for a **consumer / shared-KV branch**.
///
/// The consumer branch does not own a cache; it attends a shared K tensor
/// handed from the producer layer. The K seq dim (`total_kv_len = k.shape()[2]`)
/// is the ground truth. `total_kv_len - seq` is the effective offset: how many
/// keys precede the newly-projected query tokens. In the non-spec path
/// (`total_kv_len == offset + seq`) this reproduces the same value as the
/// old `offset`-based sizing; it only diverges — correctly — when a
/// partial-accept rollback desynced `offset` from the actual K length.
///
/// Extracted as a named helper symmetric to `producer_effective_offset` so
/// `mask_tests::guard_invariant_consumer_*` can call it directly and go RED
/// if the call site in `Attention::forward` is reverted to `offset`-based sizing.
#[cfg_attr(test, allow(dead_code))]
pub(super) fn consumer_effective_offset(total_kv_len: i32, seq: i32) -> i32 {
    total_kv_len - seq
}

/// Build the per-step additive mask + mask-mode for Gemma4 attention.
///
/// `effective_offset` is the K-side offset *seen by SDPA* (capped at
/// `sliding_window - 1` when the source cache is rotating). `total_kv_len` is
/// the post-update K shape (`effective_offset + seq` for the wrapper call,
/// `k.shape()[2]` for direct SDPA on shared K/V — they agree).
///
/// Returns `(mask_array_opt, mask_mode_str)`. `mask_mode` is one of
/// `"causal"`, `"array"`, `""` per the SDPA contract.
#[allow(clippy::too_many_arguments)]
fn build_attn_mask(
    layer_type: LayerType,
    seq: i32,
    effective_offset: i32,
    total_kv_len: i32,
    attn_is_rotating: bool,
    sliding_window: usize,
    device: Device,
) -> Result<(Option<Array>, &'static str)> {
    match layer_type {
        LayerType::SlidingAttention => {
            if seq == 1 {
                if attn_is_rotating {
                    // Rotating cache caps K at `sliding_window`, so the single
                    // decode query may attend everything. No mask needed
                    // (mirrors mlx-lm `make_mask` returning None when
                    // `window_size == max_size`).
                    Ok((None, ""))
                } else {
                    let mask =
                        crate::layers::build_swa_decode_mask(total_kv_len, sliding_window, device)?;
                    let mode = if mask.is_some() { "array" } else { "" };
                    Ok((mask, mode))
                }
            } else {
                // SWA prefill — banded-causal mask sized by the
                // capped effective offset.
                let mask = Some(crate::layers::build_swa_prefill_mask(
                    effective_offset,
                    seq,
                    sliding_window,
                    device,
                )?);
                Ok((mask, "array"))
            }
        }
        LayerType::FullAttention => {
            let mode = crate::layers::pick_attn_mask_mode(effective_offset, seq);
            let mask = if mode == "array" {
                Some(crate::layers::build_chunked_prefill_mask(
                    effective_offset,
                    seq,
                    device,
                )?)
            } else {
                None
            };
            Ok((mask, mode))
        }
    }
}

/// Expand K/V from [B, kv_heads, S, D] to [B, q_heads, S, D] by repeating.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    if repeat == 1 {
        return x.try_clone();
    }
    let shape = x.shape(); // [B, kv_heads, S, D]
    let (b, kv_h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    // Expand to [B, kv_heads, 1, S, D] then broadcast to [B, kv_heads, repeat, S, D].
    let x5 = expand_dims(x, 2, device)?;
    let x_bc = broadcast_to(&x5, &[b, kv_h, repeat as i32, s, d], device)?;
    // Reshape to [B, kv_heads * repeat, S, D].
    x_bc.reshape(&[b, kv_h * repeat as i32, s, d], device)
}

// ---------------------------------------------------------------------------
// MLP — uses crate::layers::Mlp (see import above).
// ---------------------------------------------------------------------------
//
// Gemma4 uses GeGLU: down_proj(gelu_tanh(gate_proj(x)) * up_proj(x)).
// The shared Mlp type with Activation::GeluTanh implements this exactly.
// Plus an mx.compile-fused `geglu` (gelu_tanh(gate) * up) helper below — used
// by `gemma4::decoder_layer::DecoderLayer::forward` in place of the unfused
// `Mlp::forward`. Mirrors mlx-lm `gemma4_text.py::geglu`.
