// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Gemma4 Conformer audio tower (`audio_tower.*`).
//!
//! Faithful port of `mlx_vlm/models/gemma4/audio.py`. Pipeline (batch=1):
//!
//! log-mel `[B, T, 128]` (from )
//! -> SSCP: 2× (Conv2d stride-2 + LayerNorm(channels) + ReLU) -> flatten(F·C)
//! -> `input_proj_linear` -> `[B, T_sub, hidden=1024]`
//! -> 12 Conformer blocks: FFW1(macaron 0.5) -> Attention -> LightConv1d
//! -> FFW2(macaron 0.5) -> clip -> RMSNorm
//! -> optional `output_proj` Linear -> `[B, T_sub, output_proj_dims=1536]`
//!
//! ## Deltas vs a stock Conformer (CRITICAL — applied here)
//!
//! 1. **Macaron FFW with `residual_weight=0.5`.** `out = residual + ff(x)*0.5`.
//! 2. **Chunked local attention.** Queries reshaped into blocks of `chunk_size`;
//!    each block attends a `context_size = chunk + left + right` window extracted
//!    per block. `attention_context_right=0`, left=13 on the e4b snapshot.
//! 3. **Relative position embedding.** Sinusoidal positions projected by
//!    `relative_k_proj`, added to q·k logits via the `_relative_shift` trick.
//!    Per-head `per_dim_scale` (softplus) on the query.
//! 4. **Logit softcap.** `tanh(logits / 50) * 50` before masking + softmax.
//! 5. **Invalid-logit masking.** Block-validity (from the padding mask) AND the
//!    local causal/window mask gate logits to `-1e9`.
//! 6. **LightConv1d.** norm -> linear(2×) -> GLU -> causal depthwise conv1d
//!    -> conv_norm -> silu -> linear, residual add.
//! 7. **ClippableLinear** everywhere a checkpoint Linear carries the four
//!    `input/output_{min,max}` clamp scalars (reused from `vision.rs`).
//!
//! The tower runs in **float32** (every weight upcast at load), matching the
//! vision-tower precedent: there is no decode loop and attention is f32 upstream.

#![allow(
    clippy::items_after_statements,
    clippy::maybe_infinite_iter,
    clippy::unused_self
)]
use std::mem::size_of_val;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{
    add, clip, concatenate, conv1d, conv2d, matmul, maximum, multiply, pad, rms_norm, scalar_f32,
    softmax, softplus, sum_axis_keepdims, tanh, tril, where_cond, Array, Device, Dtype,
};
use tracing::info;

use super::config::Gemma4AudioConfig;
use super::vision::{ClipBounds, ClippableLinear};

// ---------------------------------------------------------------------------
// Small host helpers (mirroring vision.rs)
// ---------------------------------------------------------------------------

#[inline]
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes; from_bytes copies immediately.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

#[inline]
fn i32_bytes(v: &[i32]) -> &[u8] {
    // SAFETY: i32 is 4 bytes; from_bytes copies immediately.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

/// `relu(x)` via `maximum(x, 0)`.
fn relu(x: &Array, device: Device) -> Result<Array> {
    let zero = scalar_f32(0.0);
    maximum(x, &zero, device)
}

// ---------------------------------------------------------------------------
// AudioRMSNorm — RMSNorm with learned weight (no offset), eps from config.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct AudioRmsNorm {
    weight: Array,
    eps: f32,
}

impl AudioRmsNorm {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, Some(&self.weight), self.eps, device)
    }
}

/// `nn.LayerNorm(dim, bias=False)` over the last axis: `(x - mean) / sqrt(var
/// + eps) * weight`. var is the population variance (ddof=0).
#[allow(missing_debug_implementations)]
struct ChannelLayerNorm {
    weight: Array,
    eps: f32,
}

impl ChannelLayerNorm {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let last = (x.ndim() - 1) as i32;
        let dim = x.shape()[x.ndim() - 1] as f32;
        let inv_n = scalar_f32(1.0 / dim);
        let mean = multiply(&sum_axis_keepdims(x, last, device)?, &inv_n, device)?;
        let centered = add(x, &multiply(&mean, &scalar_f32(-1.0), device)?, device)?;
        let sq = multiply(&centered, &centered, device)?;
        let var = multiply(&sum_axis_keepdims(&sq, last, device)?, &inv_n, device)?;
        let denom = rmlx_mlx::sqrt(&add(&var, &scalar_f32(self.eps), device)?, device)?;
        let normed = rmlx_mlx::divide(&centered, &denom, device)?;
        multiply(&normed, &self.weight, device)
    }
}

// ---------------------------------------------------------------------------
// SSCPConvBlock: Conv2d(stride 2, no pad) on a (1,1,1,1)-padded input ->
// LayerNorm(channels) -> ReLU. Returns downsampled features + downsampled mask.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct SscpConvBlock {
    conv_w: Array, // [C_out, 3, 3, C_in]
    norm: ChannelLayerNorm,
    time_stride: i32,
}

impl SscpConvBlock {
    /// `x`: `[B, T, F, C]`. `mask`: `[B, T]` (true = invalid/padding).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, x: &Array, mask: &Array, device: Device) -> Result<(Array, Array)> {
        let shp = x.shape();
        let (b, t, f, c) = (shp[0], shp[1], shp[2], shp[3]);

        // Zero out invalid positions: x = where(mask[:, :, None, None], 0, x).
        let mask_b = mask.reshape(&[b, t, 1, 1], device)?;
        let zeros = rmlx_mlx::zeros(&[b, t, f, c], Dtype::F32, device)?;
        let mask_bool = rmlx_mlx::greater_equal(&mask_b, &scalar_f32(0.5), device)?;
        let mask_bcast = rmlx_mlx::broadcast_to(&mask_bool, &[b, t, f, c], device)?;
        let x = where_cond(&mask_bcast, &zeros, x, device)?;

        // Symmetric (1,1) padding on T and F dims (axes 1 and 2).
        let x = pad(&x, &[1, 2], &[1, 1], &[1, 1], device)?;

        // Conv2d stride (2,2), no padding.
        let x = conv2d(&x, &self.conv_w, (2, 2), (0, 0), (1, 1), 1, device)?;

        // Downsample mask by time stride, clamp to conv output length.
        let t_out = x.shape()[1];
        let mask_t = mask.shape()[1];
        let strided = mask.slice(&[0, 0], &[b, mask_t], &[1, self.time_stride], device)?;
        let strided_len = strided.shape()[1];
        let keep = t_out.min(strided_len);
        let out_mask = strided.slice(&[0, 0], &[b, keep], &[1, 1], device)?;

        let x = self.norm.forward(&x, device)?;
        let x = relu(&x, device)?;
        Ok((x, out_mask))
    }
}

// ---------------------------------------------------------------------------
// SubSampleConvProjection (SSCP): layer0 -> layer1 -> flatten(F·C) -> Linear.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct SubSampleConvProjection {
    layer0: SscpConvBlock,
    layer1: SscpConvBlock,
    input_proj_w: Array, // [hidden, F*C]
}

impl SubSampleConvProjection {
    /// `audio_mel`: `[B, T, F_in]`. `mask`: `[B, T]`. Returns `([B, T_sub,
    /// hidden], mask[B, T_sub])`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, audio_mel: &Array, mask: &Array, device: Device) -> Result<(Array, Array)> {
        let x = rmlx_mlx::expand_dims(audio_mel, 3, device)?; // [B, T, F, 1]
        let (x, mask) = self.layer0.forward(&x, mask, device)?;
        let (x, mask) = self.layer1.forward(&x, &mask, device)?;

        let s = x.shape();
        let (b, t, f, c) = (s[0], s[1], s[2], s[3]);
        let x = x.reshape(&[b, t, f * c], device)?;
        let x = matmul(&x, &self.input_proj_w.transpose(&[1, 0], device)?, device)?;
        Ok((x, mask))
    }
}

// ---------------------------------------------------------------------------
// ConformerFeedForward — macaron, residual_weight scaling.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct ConformerFeedForward {
    pre_layer_norm: AudioRmsNorm,
    ffw_layer_1: ClippableLinear,
    ffw_layer_2: ClippableLinear,
    post_layer_norm: AudioRmsNorm,
    gradient_clipping: f32,
    residual_weight: f32,
}

impl ConformerFeedForward {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let residual = x;
        let lo = scalar_f32(-self.gradient_clipping);
        let hi = scalar_f32(self.gradient_clipping);
        let h = clip(x, &lo, &hi, device)?;
        let h = self.pre_layer_norm.forward(&h, device)?;
        let h = self.ffw_layer_1.forward(&h, device)?;
        let h = rmlx_mlx::silu(&h, device)?;
        let h = self.ffw_layer_2.forward(&h, device)?;
        let h = clip(&h, &lo, &hi, device)?;
        let h = self.post_layer_norm.forward(&h, device)?;
        let scaled = multiply(&h, &scalar_f32(self.residual_weight), device)?;
        add(residual, &scaled, device)
    }
}

// ---------------------------------------------------------------------------
// AudioAttention — chunked local attention + relative position + softcap.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct AudioAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    post: ClippableLinear,
    relative_k_proj_w: Array, // [num_heads*head_dim, hidden]
    per_dim_scale: Array,     // [head_dim]
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_past_horizon: usize,
    max_future_horizon: usize,
    context_size: usize,
    invalid_logits_value: f32,
    softcap: f32,
    q_scale: f32,
    k_scale: f32,
    /// Sinusoidal inv-timescales `[hidden/2]`, precomputed once at load.
    inv_timescales: Vec<f64>,
}

impl AudioAttention {
    /// `[B, T, ...]` -> `[B, num_blocks, chunk_size, ...]` (right-pad on T).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn convert_to_block(&self, x: &Array, device: Device) -> Result<Array> {
        let s = x.shape();
        let (b, t) = (s[0], s[1]);
        let rest = &s[2..];
        let cs = self.chunk_size as i32;
        let num_blocks = (t + cs - 1) / cs;
        let pad_len = num_blocks * cs - t;
        let x = if pad_len > 0 {
            pad(x, &[1], &[0], &[pad_len], device)?
        } else {
            x.try_clone()?
        };
        let mut shape = vec![b, num_blocks, cs];
        shape.extend_from_slice(rest);
        x.reshape(&shape, device)
    }

    /// `[B, T, ...]` -> `[B, num_blocks, context_size, ...]` via padded gather.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn extract_block_context(&self, x: &Array, device: Device) -> Result<Array> {
        let pad_left = self.max_past_horizon as i32;
        let pad_right = (self.max_future_horizon + self.chunk_size - 1) as i32;
        let x = pad(x, &[1], &[pad_left], &[pad_right], device)?;
        let s = x.shape();
        let (b, t_padded) = (s[0], s[1]);
        let rest = &s[2..];
        let cs = self.chunk_size as i32;
        let ctx = self.context_size as i32;
        let num_blocks = (t_padded - ctx) / cs + 1;

        // Flat gather indices [num_blocks * context_size] along axis 1.
        let mut idx = Vec::with_capacity((num_blocks * ctx) as usize);
        for blk in 0..num_blocks {
            let start = blk * cs;
            for off in 0..ctx {
                idx.push(start + off);
            }
        }
        let idx_arr = Array::from_bytes(i32_bytes(&idx), &[num_blocks * ctx], Dtype::I32)?;
        let gathered = x.take(&idx_arr, 1, device)?; // [B, nb*ctx, ...]
        let mut shape = vec![b, num_blocks, ctx];
        shape.extend_from_slice(rest);
        gathered.reshape(&shape, device)
    }

    /// Sinusoidal timing signal for `M` positions -> `[M, hidden]`
    /// (concat of sin and cos halves), matching `_get_timing_signal`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn timing_signal(&self, positions: &[f64], device: Device) -> Result<Array> {
        let m = positions.len();
        let half = self.inv_timescales.len();
        let mut sin_buf = vec![0.0f32; m * half];
        let mut cos_buf = vec![0.0f32; m * half];
        for (pi, &p) in positions.iter().enumerate() {
            for (j, &inv) in self.inv_timescales.iter().enumerate() {
                let t = p * inv;
                sin_buf[pi * half + j] = t.sin() as f32;
                cos_buf[pi * half + j] = t.cos() as f32;
            }
        }
        let sin_a = Array::from_bytes(f32_bytes(&sin_buf), &[m as i32, half as i32], Dtype::F32)?;
        let cos_a = Array::from_bytes(f32_bytes(&cos_buf), &[m as i32, half as i32], Dtype::F32)?;
        concatenate(&[&sin_a, &cos_a], 1, device) // [M, hidden]
    }

    /// `_relative_shift`: `[B, N, U, W, M]` -> `[B, N, U, W, C]`.
    #[allow(clippy::too_many_arguments)]
    fn relative_shift(
        &self,
        term_bd: &Array,
        b: i32,
        n: i32,
        u: i32,
        w: i32,
        c: i32,
        m: i32,
        device: Device,
    ) -> Result<Array> {
        let pad_amount = (c + 1) - m;
        let term = if pad_amount > 0 {
            pad(term_bd, &[4], &[0], &[pad_amount], device)?
        } else {
            term_bd.try_clone()?
        };
        let term = term.reshape(&[b, n, u, w * (c + 1)], device)?;
        let term = term.slice(&[0, 0, 0, 0], &[b, n, u, w * c], &[1, 1, 1, 1], device)?;
        term.reshape(&[b, n, u, w, c], device)
    }

    /// Relative-position logits: term_ac + term_bd, shape `[B, N, U, W, C]`.
    /// `query_blocks`: `[B, U, W, N, H]`, `key_blocks`: `[B, U, C, N, H]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn rel_pos_logits(
        &self,
        query_blocks: &Array,
        key_blocks: &Array,
        device: Device,
    ) -> Result<Array> {
        let qs = query_blocks.shape();
        let (b, u, w, n, h) = (qs[0], qs[1], qs[2], qs[3], qs[4]);
        let c = key_blocks.shape()[2];

        // pos_indices = arange(max_backward, -max_forward-1, -1).
        let max_backward = self.max_past_horizon as i64;
        let max_forward = self.max_future_horizon as i64;
        let positions: Vec<f64> = (0..)
            .map(|i| max_backward - i)
            .take_while(|&v| v >= -max_forward)
            .map(|v| v as f64)
            .collect();
        let m = positions.len() as i32;

        // sin_emb: [M, hidden] -> relative_k_proj -> [M, N*H] -> [M, N, H].
        let sin_emb = self.timing_signal(&positions, device)?;
        let sin_emb = matmul(
            &sin_emb,
            &self.relative_k_proj_w.transpose(&[1, 0], device)?,
            device,
        )?;
        let sin_emb = sin_emb.reshape(&[m, n, h], device)?;

        let queries_p = query_blocks.transpose(&[0, 3, 1, 2, 4], device)?; // [B,N,U,W,H]
        let keys_p = key_blocks.transpose(&[0, 3, 1, 4, 2], device)?; // [B,N,U,H,C]
        let term_ac = matmul(&queries_p, &keys_p, device)?; // [B,N,U,W,C]

        let sin_emb_t = sin_emb.transpose(&[1, 2, 0], device)?; // [N,H,M]
        let q_reshaped = queries_p.reshape(&[b, n, u * w, h], device)?;
        let term_bd = matmul(&q_reshaped, &sin_emb_t, device)?; // [B,N,U*W,M]
        let term_bd = term_bd.reshape(&[b, n, u, w, m], device)?;
        let term_bd = self.relative_shift(&term_bd, b, n, u, w, c, m, device)?;

        add(&term_ac, &term_bd, device)
    }

    /// `hidden_states`: `[B, T, hidden]`. `mask`: `[B, T]` (1.0 = invalid).
    /// `causal_valid_mask`: `[chunk_size, context_size]` (1.0 = valid).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(
        &self,
        hidden_states: &Array,
        mask: &Array,
        causal_valid_mask: &Array,
        device: Device,
    ) -> Result<Array> {
        let s = hidden_states.shape();
        let (b, t) = (s[0], s[1]);
        let nh = self.num_heads as i32;
        let hd = self.head_dim as i32;

        let q = self
            .q_proj
            .forward(hidden_states, device)?
            .reshape(&[b, t, nh, hd], device)?;
        let k = self
            .k_proj
            .forward(hidden_states, device)?
            .reshape(&[b, t, nh, hd], device)?;
        let v = self
            .v_proj
            .forward(hidden_states, device)?
            .reshape(&[b, t, nh, hd], device)?;

        // q *= q_scale * softplus(per_dim_scale); k *= k_scale.
        let scale = softplus(&self.per_dim_scale, device)?;
        let q = multiply(
            &q,
            &multiply(&scale, &scalar_f32(self.q_scale), device)?,
            device,
        )?;
        let k = multiply(&k, &scalar_f32(self.k_scale), device)?;

        let query_blocks = self.convert_to_block(&q, device)?; // [B, U, W, N, H]
        let key_blocks = self.extract_block_context(&k, device)?; // [B, U, C, N, H]
        let value_blocks = self.extract_block_context(&v, device)?; // [B, U, C, N, H]
        let u = query_blocks.shape()[1];

        // Block validity: valid = 1 - mask, extracted to [B, U, C].
        let mask_f = mask.astype(Dtype::F32, device)?;
        let valid = add(
            &scalar_f32(1.0),
            &multiply(&mask_f, &scalar_f32(-1.0), device)?,
            device,
        )?;
        let extracted_valid = self.extract_block_context(&valid, device)?; // [B, U, C]
        let ev = extracted_valid.shape();
        let cs = self.chunk_size as i32;
        let ctx = self.context_size as i32;
        // condition = ev[:,None,:,None,:] & causal[None,None,None,:,:] -> [B,N,U,W,C].
        let ev_b = extracted_valid.reshape(&[ev[0], 1, ev[1], 1, ev[2]], device)?;
        let causal_b = causal_valid_mask.reshape(&[1, 1, 1, cs, ctx], device)?;
        let cond = multiply(
            &rmlx_mlx::broadcast_to(&ev_b, &[ev[0], nh, u, cs, ctx], device)?,
            &rmlx_mlx::broadcast_to(&causal_b, &[ev[0], nh, u, cs, ctx], device)?,
            device,
        )?; // 1.0 = valid

        // logits = rel_pos; softcap; mask invalid.
        let logits = self.rel_pos_logits(&query_blocks, &key_blocks, device)?;
        let cap = scalar_f32(self.softcap);
        let inv_cap = scalar_f32(1.0 / self.softcap);
        let logits = multiply(
            &tanh(&multiply(&logits, &inv_cap, device)?, device)?,
            &cap,
            device,
        )?;
        let cond_bool = rmlx_mlx::greater_equal(&cond, &scalar_f32(0.5), device)?;
        let invalid = rmlx_mlx::broadcast_to(
            &scalar_f32(self.invalid_logits_value),
            &logits.shape(),
            device,
        )?;
        let logits = where_cond(&cond_bool, &logits, &invalid, device)?;

        let probs = softmax(&logits, -1, device)?; // [B, N, U, W, C]

        // context = einsum("bnuwc,bucnh->buwnh", probs, value_blocks) via
        // batched matmul over (b,u,n):
        // probs [B,N,U,W,C] -> [B,U,N,W,C] (transpose 0,2,1,3,4)
        // value [B,U,C,N,H] -> [B,U,N,C,H] (transpose 0,1,3,2,4)
        // mm -> [B,U,N,W,H] -> [B,U,W,N,H] (transpose 0,1,3,2,4)
        let probs_r = probs.transpose(&[0, 2, 1, 3, 4], device)?;
        let value_r = value_blocks.transpose(&[0, 1, 3, 2, 4], device)?;
        let ctx_mm = matmul(&probs_r, &value_r, device)?; // [B,U,N,W,H]
        let context = ctx_mm.transpose(&[0, 1, 3, 2, 4], device)?; // [B,U,W,N,H]

        let context = context.reshape(&[b, u * cs, nh, hd], device)?;
        let context = context.slice(&[0, 0, 0, 0], &[b, t, nh, hd], &[1, 1, 1, 1], device)?;
        let context = context.reshape(&[b, t, nh * hd], device)?;
        self.post.forward(&context, device)
    }
}

// ---------------------------------------------------------------------------
// ConformerLightConv1d — norm -> linear(2x) -> GLU -> causal depthwise conv1d
// -> conv_norm -> silu -> linear, residual add.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct ConformerLightConv1d {
    pre_layer_norm: AudioRmsNorm,
    linear_start: ClippableLinear,
    depthwise_conv_w: Array, // [hidden, kernel, 1]
    conv_norm: AudioRmsNorm,
    linear_end: ClippableLinear,
    gradient_clipping: f32,
    causal_padding: i32,
    hidden_size: i32,
}

impl ConformerLightConv1d {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let residual = x;
        let h = self.pre_layer_norm.forward(x, device)?;
        let h = self.linear_start.forward(&h, device)?;

        // GLU: split last dim in half, gate.
        let s = h.shape();
        let nd = h.ndim();
        let half = s[nd - 1] / 2;
        let starts = vec![0i32; nd];
        let mut stops = s.clone();
        stops[nd - 1] = half;
        let strides = vec![1i32; nd];
        let x1 = h.slice(&starts, &stops, &strides, device)?;
        let mut starts2 = vec![0i32; nd];
        starts2[nd - 1] = half;
        let x2 = h.slice(&starts2, &s, &strides, device)?;
        let h = multiply(&x1, &rmlx_mlx::sigmoid(&x2, device)?, device)?;

        // Causal left-pad on time axis (axis 1), then depthwise conv1d (groups=C).
        let h = pad(&h, &[1], &[self.causal_padding], &[0], device)?;
        let h = conv1d(
            &h,
            &self.depthwise_conv_w,
            1,
            0,
            1,
            self.hidden_size,
            device,
        )?;

        let lo = scalar_f32(-self.gradient_clipping);
        let hi = scalar_f32(self.gradient_clipping);
        let h = clip(&h, &lo, &hi, device)?;
        let h = self.conv_norm.forward(&h, device)?;
        let h = rmlx_mlx::silu(&h, device)?;
        let h = self.linear_end.forward(&h, device)?;
        add(&h, residual, device)
    }
}

// ---------------------------------------------------------------------------
// ConformerBlock — ff1 -> attn -> lconv -> ff2 -> clip -> norm_out.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct ConformerBlock {
    feed_forward1: ConformerFeedForward,
    self_attn: AudioAttention,
    lconv1d: ConformerLightConv1d,
    feed_forward2: ConformerFeedForward,
    norm_pre_attn: AudioRmsNorm,
    norm_post_attn: AudioRmsNorm,
    norm_out: AudioRmsNorm,
    gradient_clipping: f32,
}

impl ConformerBlock {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(
        &self,
        x: &Array,
        mask: &Array,
        causal_valid_mask: &Array,
        device: Device,
    ) -> Result<Array> {
        let lo = scalar_f32(-self.gradient_clipping);
        let hi = scalar_f32(self.gradient_clipping);

        let x = self.feed_forward1.forward(x, device)?;

        // Attention with pre/post norm + residual.
        let residual = &x;
        let h = clip(&x, &lo, &hi, device)?;
        let h = self.norm_pre_attn.forward(&h, device)?;
        let h = self
            .self_attn
            .forward(&h, mask, causal_valid_mask, device)?;
        let h = clip(&h, &lo, &hi, device)?;
        let h = self.norm_post_attn.forward(&h, device)?;
        let x = add(residual, &h, device)?;

        // Zero out invalid positions before lconv1d: x *= (1 - mask)[:, :, None].
        let s = x.shape();
        let (b, t) = (s[0], s[1]);
        let mask_f = mask.astype(Dtype::F32, device)?;
        let validity = add(
            &scalar_f32(1.0),
            &multiply(&mask_f, &scalar_f32(-1.0), device)?,
            device,
        )?
        .reshape(&[b, t, 1], device)?;
        let x = multiply(&x, &validity, device)?;

        let x = self.lconv1d.forward(&x, device)?;
        let x = self.feed_forward2.forward(&x, device)?;
        let x = clip(&x, &lo, &hi, device)?;
        self.norm_out.forward(&x, device)
    }
}

// ---------------------------------------------------------------------------
// AudioEncoder
// ---------------------------------------------------------------------------

/// Gemma4 Conformer audio tower (`audio_tower.*`). Forward consumes log-mel
/// features `[B, T, 128]` and produces `[B, T_sub, output_proj_dims]` audio
/// embeddings (fed to `embed_audio` / `MultimodalEmbedder` by ).
#[allow(missing_debug_implementations)]
pub struct AudioEncoder {
    cfg: Gemma4AudioConfig,
    subsample: SubSampleConvProjection,
    layers: Vec<ConformerBlock>,
    /// `output_proj` Linear (`[output_proj_dims, hidden]` weight + bias) when
    /// `output_proj_dims` is set.
    output_proj: Option<(Array, Array)>,
}

impl AudioEncoder {
    /// Parsed audio sub-config this tower was built from.
    pub fn config(&self) -> &Gemma4AudioConfig {
        &self.cfg
    }

    /// Build the `[chunk_size, context_size]` local causal+validity mask
    /// (1.0 = valid). Matches `_build_causal_valid_mask`.
    fn build_causal_valid_mask(&self, device: Device) -> Result<Array> {
        let chunk = self.cfg.attention_chunk_size as i32;
        let max_future = self.cfg.attention_context_right as i32;
        let max_past = self.cfg.attention_context_left.saturating_sub(1) as i32;
        let upper_diag = max_past + max_future;
        let ctx = chunk + max_past + max_future;

        let ones_cc = rmlx_mlx::broadcast_to(&scalar_f32(1.0), &[ctx, chunk], device)?;
        // lower_causal = tril(ones(context, chunk)).T -> [chunk, context].
        let lower_causal = tril(&ones_cc, 0, device)?.transpose(&[1, 0], device)?;
        let ones_cc2 = rmlx_mlx::broadcast_to(&scalar_f32(1.0), &[chunk, ctx], device)?;
        let upper_causal = tril(&ones_cc2, upper_diag, device)?; // [chunk, context]
                                                                 // mask = lower * upper (float 0/1; downstream uses >=0.5 as valid).
        multiply(&lower_causal, &upper_causal, device)
    }

    /// Run the SSCP + Conformer stack + optional output projection.
    /// `audio_mel`: `[B, T, 128]`. `audio_mel_mask`: `[B, T]` (1.0 = padding).
    /// Returns `[B, T_sub, output_proj_dims (or hidden)]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward(
        &self,
        audio_mel: &Array,
        audio_mel_mask: &Array,
        device: Device,
    ) -> Result<Array> {
        let (mut x, mut mask) = self.subsample.forward(audio_mel, audio_mel_mask, device)?;
        let causal_valid_mask = self.build_causal_valid_mask(device)?;

        for blk in &self.layers {
            x = blk.forward(&x, &mask, &causal_valid_mask, device)?;
        }

        if let Some((w, bias)) = &self.output_proj {
            x = add(
                &matmul(&x, &w.transpose(&[1, 0], device)?, device)?,
                bias,
                device,
            )?;
        }

        // Crop mask to output length, zero out invalid positions.
        let s = x.shape();
        let (b, t, hidden) = (s[0], s[1], s[2]);
        let keep = t.min(mask.shape()[1]);
        if mask.shape()[1] != keep {
            mask = mask.slice(&[0, 0], &[b, keep], &[1, 1], device)?;
        }
        let mask_b = mask.reshape(&[b, keep, 1], device)?;
        let mask_bool = rmlx_mlx::greater_equal(&mask_b, &scalar_f32(0.5), device)?;
        let mask_bool = if keep == t {
            rmlx_mlx::broadcast_to(&mask_bool, &[b, t, hidden], device)?
        } else {
            // T_sub from convs equals mask length on the no-padding path; guard
            // against off-by-one by broadcasting only over the kept rows.
            rmlx_mlx::broadcast_to(&mask_bool, &[b, keep, hidden], device)?
        };
        let zeros = rmlx_mlx::zeros(&[b, t, hidden], Dtype::F32, device)?;
        // mask_bool has `keep` rows; when keep==t (the common path) this is exact.
        if keep == t {
            where_cond(&mask_bool, &zeros, &x, device)
        } else {
            Ok(x)
        }
    }
}

// ---------------------------------------------------------------------------
// Loader — loads audio_tower.* from the main safetensors shards (f32).
// ---------------------------------------------------------------------------

/// Load the Gemma4 audio tower (`audio_tower.*`) from a snapshot directory.
pub fn load_audio_tower(model_dir: &Path, cfg: &Gemma4AudioConfig) -> Result<AudioEncoder> {
    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    fn load_f32(shards: &ShardSet, name: &str) -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle.safetensors()?;
            if let Ok(t) = st.tensor(name) {
                let tv = rmlx_loader::TensorView {
                    name,
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    bytes: t.data(),
                };
                let a = Array::from_safetensor_view(&tv)?;
                return a.astype(Dtype::F32, Device::Cpu);
            }
        }
        Err(Error::Loader(format!(
            "gemma4 audio: tensor '{name}' not found in any shard"
        )))
    }
    let has = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };
    let load_clip = |base: &str| -> Result<ClippableLinear> {
        let weight = load_f32(&shards, &format!("{base}.linear.weight"))?;
        let clip = if cfg.use_clipped_linears && has(&format!("{base}.input_min")) {
            Some(ClipBounds {
                input_min: load_f32(&shards, &format!("{base}.input_min"))?,
                input_max: load_f32(&shards, &format!("{base}.input_max"))?,
                output_min: load_f32(&shards, &format!("{base}.output_min"))?,
                output_max: load_f32(&shards, &format!("{base}.output_max"))?,
            })
        } else {
            None
        };
        Ok(ClippableLinear { weight, clip })
    };
    let load_rms = |name: &str| -> Result<AudioRmsNorm> {
        Ok(AudioRmsNorm {
            weight: load_f32(&shards, &format!("{name}.weight"))?,
            eps: cfg.rms_norm_eps,
        })
    };
    let load_ln = |name: &str| -> Result<ChannelLayerNorm> {
        Ok(ChannelLayerNorm {
            weight: load_f32(&shards, &format!("{name}.weight"))?,
            eps: cfg.rms_norm_eps,
        })
    };

    info!(
        layers = cfg.num_hidden_layers,
        hidden = cfg.hidden_size,
        heads = cfg.num_attention_heads,
        "gemma4: loading audio tower (f32)"
    );

    let ssp = "audio_tower.subsample_conv_projection";
    let subsample = SubSampleConvProjection {
        layer0: SscpConvBlock {
            conv_w: load_f32(&shards, &format!("{ssp}.layer0.conv.weight"))?,
            norm: load_ln(&format!("{ssp}.layer0.norm"))?,
            time_stride: 2,
        },
        layer1: SscpConvBlock {
            conv_w: load_f32(&shards, &format!("{ssp}.layer1.conv.weight"))?,
            norm: load_ln(&format!("{ssp}.layer1.norm"))?,
            time_stride: 2,
        },
        input_proj_w: load_f32(&shards, &format!("{ssp}.input_proj_linear.weight"))?,
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let b = format!("audio_tower.layers.{i}");
        let load_ff = |ff: &str| -> Result<ConformerFeedForward> {
            let base = format!("{b}.{ff}");
            Ok(ConformerFeedForward {
                pre_layer_norm: load_rms(&format!("{base}.pre_layer_norm"))?,
                ffw_layer_1: load_clip(&format!("{base}.ffw_layer_1"))?,
                ffw_layer_2: load_clip(&format!("{base}.ffw_layer_2"))?,
                post_layer_norm: load_rms(&format!("{base}.post_layer_norm"))?,
                gradient_clipping: cfg.gradient_clipping,
                residual_weight: cfg.residual_weight,
            })
        };
        let sa = format!("{b}.self_attn");
        let attn = build_attention(
            cfg,
            load_clip(&format!("{sa}.q_proj"))?,
            load_clip(&format!("{sa}.k_proj"))?,
            load_clip(&format!("{sa}.v_proj"))?,
            load_clip(&format!("{sa}.post"))?,
            load_f32(&shards, &format!("{sa}.relative_k_proj.weight"))?,
            load_f32(&shards, &format!("{sa}.per_dim_scale"))?,
        );
        let lc = format!("{b}.lconv1d");
        let lconv = ConformerLightConv1d {
            pre_layer_norm: load_rms(&format!("{lc}.pre_layer_norm"))?,
            linear_start: load_clip(&format!("{lc}.linear_start"))?,
            depthwise_conv_w: load_f32(&shards, &format!("{lc}.depthwise_conv1d.weight"))?,
            conv_norm: load_rms(&format!("{lc}.conv_norm"))?,
            linear_end: load_clip(&format!("{lc}.linear_end"))?,
            gradient_clipping: cfg.gradient_clipping,
            causal_padding: (cfg.conv_kernel_size - 1) as i32,
            hidden_size: cfg.hidden_size as i32,
        };
        layers.push(ConformerBlock {
            feed_forward1: load_ff("feed_forward1")?,
            self_attn: attn,
            lconv1d: lconv,
            feed_forward2: load_ff("feed_forward2")?,
            norm_pre_attn: load_rms(&format!("{b}.norm_pre_attn"))?,
            norm_post_attn: load_rms(&format!("{b}.norm_post_attn"))?,
            norm_out: load_rms(&format!("{b}.norm_out"))?,
            gradient_clipping: cfg.gradient_clipping,
        });
    }

    let output_proj = if cfg.output_proj_dims.is_some() && has("audio_tower.output_proj.weight") {
        Some((
            load_f32(&shards, "audio_tower.output_proj.weight")?,
            load_f32(&shards, "audio_tower.output_proj.bias")?,
        ))
    } else {
        None
    };

    info!(layers = cfg.num_hidden_layers, "gemma4: audio tower loaded");
    Ok(AudioEncoder {
        cfg: cfg.clone(),
        subsample,
        layers,
        output_proj,
    })
}

/// Construct an [`AudioAttention`], precomputing scalar scales + inv-timescales.
fn build_attention(
    cfg: &Gemma4AudioConfig,
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    post: ClippableLinear,
    relative_k_proj_w: Array,
    per_dim_scale: Array,
) -> AudioAttention {
    let num_heads = cfg.num_attention_heads;
    let hidden = cfg.hidden_size;
    let head_dim = hidden / num_heads;
    let max_past = cfg.attention_context_left.saturating_sub(1);
    let max_future = cfg.attention_context_right;
    let chunk = cfg.attention_chunk_size;
    let context_size = chunk + max_past + max_future;

    let q_scale = (head_dim as f32).powf(-0.5) / std::f32::consts::LN_2;
    let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;

    // Sinusoidal inv-timescales over hidden/2 channels (min=1, max=10000).
    let num_timescales = hidden / 2;
    let log_inc = (10000.0f64 / 1.0f64).ln() / (num_timescales.max(2) - 1) as f64;
    let inv_timescales: Vec<f64> = (0..num_timescales)
        .map(|i| (-(i as f64) * log_inc).exp())
        .collect();

    AudioAttention {
        q_proj,
        k_proj,
        v_proj,
        post,
        relative_k_proj_w,
        per_dim_scale,
        num_heads,
        head_dim,
        chunk_size: chunk,
        max_past_horizon: max_past,
        max_future_horizon: max_future,
        context_size,
        invalid_logits_value: cfg.attention_invalid_logits_value,
        softcap: cfg.attention_logit_cap,
        q_scale,
        k_scale,
        inv_timescales,
    }
}

#[cfg(test)]
mod tests;
