//! The DFlash 2 drafter's decoder stack.
//!
//! Ported from the z-lab MLX reference `DFlash2DraftModel.hidden_states` and
//! the `GroupedDynamicCausalConv` / `DFlashAttention` it drives. The block is
//! denoised in one non-autoregressive pass: the drafter attends over a window
//! of conditioning rows projected from the verifier's hidden states, plus the
//! whole block bidirectionally.
//!
//! # Position offsets
//!
//! The reference RoPEs its conditioning rows at the drafter cache's absolute
//! offset and the block at that offset plus the context length. This port
//! recomputes the conditioning K/V every call rather than caching them, so
//! every row of one call is rotated in the same pass at consecutive positions
//! and only the query-key difference survives into the attention scores. A
//! uniform shift of every position is therefore not observable, and the
//! context starts at zero here — mathematically the reference's answer at any
//! absolute offset. A caller that starts caching conditioning K/V across calls
//! loses that invariance and has to carry the absolute offset back in.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{
    add, arange, concatenate, greater_equal, maximum, multiply, rope, scalar_f32,
    scaled_dot_product_attention, where_cond, zeros, Array, Device, Dtype,
};

use super::{DFlash2Conv, DFlash2Drafter, DFlash2Layer, CONV_SIDES};

/// Additive bias applied to a blocked (query, key) pair — large-negative rather
/// than `-inf` so a fully blocked row still softmaxes to a finite distribution.
const BLOCKED_BIAS: f32 = -1e30;

/// Which of a convolution's two kernel sides is being applied: the one on the
/// sublayer's normed input, or the one on the sublayer's output.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvSide {
    /// Applied to the sublayer input, before attention or the MLP.
    Prepare = 0,
    /// Applied to the sublayer output, before the residual add.
    Finish = 1,
}

impl DFlash2Drafter {
    /// Run the decoder stack over a draft block and return its final hidden
    /// states, `[1, block_len, hidden]`.
    ///
    /// `block` is the block's input embeddings, `[1, block_len, hidden]` — the
    /// last verified token at position 0 followed by masked positions.
    /// `target_hidden` is the verifier's hidden states at `target_layer_ids`
    /// concatenated along the feature axis, `[1, ctx_len, n_targets * hidden]`,
    /// oldest row first. Rows older than the attention window are dropped here,
    /// so a caller may pass the whole conditioning history.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when either input's rank or width is not the one the
    /// config predicts, or when either is empty.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape axes are established by construction: `shaped` validates both inputs' rank at the entry point and every array below is reshaped from them"
    )]
    pub fn forward_hidden(&self, block: &Array, target_hidden: &Array) -> Result<Array> {
        let device = self.device;
        let hidden = self.cfg.hidden_size as i32;
        let block_len = shaped(block, "block", &[1, -1, hidden])?;
        let full_ctx = shaped(
            target_hidden,
            "target_hidden",
            &[1, -1, self.cfg.target_layer_ids.len() as i32 * hidden],
        )?;

        // `fc` and `hidden_norm` are row-wise, so dropping the unreachable rows
        // before them is the same result for less work.
        let keep = self.conditioning_rows();
        let ctx_len = full_ctx.min(keep);
        let trimmed = if full_ctx > keep {
            target_hidden.slice(
                &[0, full_ctx - keep, 0],
                &[1, full_ctx, target_hidden.shape()[2]],
                &[1, 1, 1],
                device,
            )?
        } else {
            target_hidden.try_clone()?
        };
        let projected = self.fc.forward(&trimmed, device)?;
        let h_ctx = self.hidden_norm.forward(&projected, device)?;

        // One mask for the stack: every layer of this drafter is a sliding
        // layer over the same two lengths.
        let mask = self.window_mask(block_len, ctx_len, block.dtype())?;

        let mut x = block.try_clone()?;
        for layer in &self.layers {
            x = self.layer_forward(layer, &x, &h_ctx, &mask)?;
        }
        tracing::debug!(
            block_len,
            ctx_len,
            dropped = full_ctx - ctx_len,
            layers = self.layers.len(),
            "DFlash2Drafter: block forward"
        );
        self.norm.forward(&x, device)
    }

    /// One decoder layer: a dynamic convolution wrapped around attention, then
    /// another wrapped around the MLP, both residual.
    ///
    /// Each convolution reads its kernel correction from the sublayer's *input*
    /// once and applies one side of it before the sublayer and the other after.
    fn layer_forward(
        &self,
        layer: &DFlash2Layer,
        x: &Array,
        h_ctx: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let device = self.device;

        let residual = x.try_clone()?;
        let normed = layer.input_layernorm.forward(x, device)?;
        let (attn_in, finish_kernel) = self.conv_prepare(&layer.attention_conv, &normed)?;
        let attn = self.attention(layer, &attn_in, h_ctx, mask)?;
        let attn = self.convolve(
            &layer.attention_conv,
            &attn,
            &finish_kernel,
            ConvSide::Finish,
        )?;
        let x = add(&residual, &attn, device)?;

        let residual = x.try_clone()?;
        let normed = layer.post_attention_layernorm.forward(&x, device)?;
        let (mlp_in, finish_kernel) = self.conv_prepare(&layer.mlp_conv, &normed)?;
        let mlp_out = layer.mlp.forward(&mlp_in, device)?;
        let mlp_out = self.convolve(&layer.mlp_conv, &mlp_out, &finish_kernel, ConvSide::Finish)?;
        add(&residual, &mlp_out, device)
    }

    /// Project one convolution's per-position kernel correction and apply the
    /// prepare side of it.
    ///
    /// Returns the convolved input and the finish side's correction, which the
    /// caller applies to the sublayer's output. Both sides come from this one
    /// projection of the sublayer's input — the finish kernel is not recomputed
    /// from the output.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape axes are established by construction: `shaped` validates both inputs' rank at the entry point and every array below is reshaped from them"
    )]
    fn conv_prepare(&self, conv: &DFlash2Conv, h: &Array) -> Result<(Array, Array)> {
        let device = self.device;
        let len = h.shape()[1];
        let taps = self.cfg.conv_kernel_size as i32;
        let groups = self.conv_groups();
        let dynamic = conv
            .kernel_projection
            .forward(h, device)?
            .reshape(&[1, len, CONV_SIDES as i32, taps, groups], device)?;
        let take = |side: ConvSide| -> Result<Array> {
            let s = side as i32;
            dynamic
                .slice(
                    &[0, 0, s, 0, 0],
                    &[1, len, s + 1, taps, groups],
                    &[1, 1, 1, 1, 1],
                    device,
                )?
                .reshape(&[1, len, taps, groups], device)
        };
        let prepared = self.convolve(conv, h, &take(ConvSide::Prepare)?, ConvSide::Prepare)?;
        Ok((prepared, take(ConvSide::Finish)?))
    }

    /// The two-tap dynamic depthwise convolution over the block.
    ///
    /// Per channel `c` in group `g = c / conv_group_size`, per block position
    /// `t`, summed over taps `k`:
    ///
    /// ```text
    /// out[t, c] = SUM_k (base[side, k, c] + correction[t, k, g]) * x[t - k, c]
    /// ```
    ///
    /// The base term is per channel and the correction per group. Taps reaching
    /// before the block read zero: the convolution is block-local and touches
    /// no context row and no cache.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape axes are established by construction: `shaped` validates both inputs' rank at the entry point and every array below is reshaped from them"
    )]
    fn convolve(
        &self,
        conv: &DFlash2Conv,
        x: &Array,
        correction: &Array,
        side: ConvSide,
    ) -> Result<Array> {
        let device = self.device;
        let len = x.shape()[1];
        let hidden = self.cfg.hidden_size as i32;
        let group_size = self.cfg.conv_group_size as i32;
        let groups = self.conv_groups();
        let taps = self.cfg.conv_kernel_size as i32;
        let s = side as i32;

        let blocks = x.reshape(&[1, len, groups, group_size], device)?;
        let per_group = correction.reshape(&[1, len, taps, groups, 1], device)?;
        let mut out = zeros(&[1, len, groups, group_size], x.dtype(), device)?;
        for tap in 0..taps {
            let values = shift_into_block(&blocks, tap, device)?;
            let base = conv
                .base_kernel
                .slice(&[s, tap, 0], &[s + 1, tap + 1, hidden], &[1, 1, 1], device)?
                .reshape(&[1, 1, groups, group_size], device)?
                .astype(x.dtype(), device)?;
            out = add(&out, &multiply(&base, &values, device)?, device)?;
            let correction = per_group
                .slice(
                    &[0, 0, tap, 0, 0],
                    &[1, len, tap + 1, groups, 1],
                    &[1, 1, 1, 1, 1],
                    device,
                )?
                .reshape(&[1, len, groups, 1], device)?;
            out = add(&out, &multiply(&correction, &values, device)?, device)?;
        }
        out.reshape(&[1, len, hidden], device)
    }

    /// Grouped-query attention over the conditioning window and the whole block.
    ///
    /// Queries come from the block; keys and values from the conditioning rows
    /// *and* the block. Conditioning K/V occupy positions `0..ctx_len` and the
    /// block follows at `ctx_len..`, which is what `mask` was built against.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape axes are established by construction: `shaped` validates both inputs' rank at the entry point and every array below is reshaped from them"
    )]
    fn attention(
        &self,
        layer: &DFlash2Layer,
        x: &Array,
        h_ctx: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let device = self.device;
        let heads = self.cfg.num_attention_heads as i32;
        let kv_heads = self.cfg.num_key_value_heads as i32;
        let head_dim = self.cfg.head_dim as i32;
        let len = x.shape()[1];
        let ctx_len = h_ctx.shape()[1];

        let heads_first = |a: &Array, rows: i32, count: i32| -> Result<Array> {
            a.reshape(&[1, rows, count, head_dim], device)?
                .transpose(&[0, 2, 1, 3], device)
        };

        let q = heads_first(&layer.q_proj.forward(x, device)?, len, heads)?;
        let q = layer.q_norm.forward(&q, device)?;
        let q = rope(
            &q,
            head_dim,
            false,
            self.cfg.rope_theta,
            1.0,
            ctx_len,
            device,
        )?;

        let ctx_k = heads_first(&layer.k_proj.forward(h_ctx, device)?, ctx_len, kv_heads)?;
        let ctx_k = layer.k_norm.forward(&ctx_k, device)?;
        let ctx_k = rope(&ctx_k, head_dim, false, self.cfg.rope_theta, 1.0, 0, device)?;
        let ctx_v = heads_first(&layer.v_proj.forward(h_ctx, device)?, ctx_len, kv_heads)?;

        let block_k = heads_first(&layer.k_proj.forward(x, device)?, len, kv_heads)?;
        let block_k = layer.k_norm.forward(&block_k, device)?;
        let block_k = rope(
            &block_k,
            head_dim,
            false,
            self.cfg.rope_theta,
            1.0,
            ctx_len,
            device,
        )?;
        let block_v = heads_first(&layer.v_proj.forward(x, device)?, len, kv_heads)?;

        let keys = concatenate(&[&ctx_k, &block_k], 2, device)?;
        let values = concatenate(&[&ctx_v, &block_v], 2, device)?;

        let scale = (head_dim as f32).powf(-0.5);
        let out =
            scaled_dot_product_attention(&q, &keys, &values, scale, "array", Some(mask), device)?;
        let out = out
            .transpose(&[0, 2, 1, 3], device)?
            .reshape(&[1, len, heads * head_dim], device)?;
        layer.o_proj.forward(&out, device)
    }

    /// The additive attention mask over `[conditioning ; block]`.
    ///
    /// A block query at index `t` sits at key position `ctx_len + t`. It reads
    /// every block key — the block is drafted bidirectionally, `is_causal` is
    /// false on this checkpoint — and every conditioning key inside its own
    /// window. Every block key is inside that window whenever the block is
    /// shorter than the window, so the union collapses to "inside the window,
    /// or in the block".
    fn window_mask(&self, block_len: i32, ctx_len: i32, dtype: Dtype) -> Result<Array> {
        let device = self.device;
        let keys = ctx_len + block_len;
        let key_pos =
            arange(0.0, f64::from(keys), 1.0, device)?.reshape(&[1, 1, 1, keys], device)?;

        // The oldest key query `t` may read: its own position minus the window,
        // plus one. i64 so a window reaching back before position 0 stays
        // well-defined instead of wrapping.
        let first = i64::from(ctx_len) - self.cfg.sliding_window as i64 + 1;
        let oldest = arange(
            first as f64,
            (first + i64::from(block_len)) as f64,
            1.0,
            device,
        )?
        .reshape(&[1, 1, block_len, 1], device)?;
        let in_window = greater_equal(&key_pos, &oldest, device)?;
        // f32-ok: the operand is a position index from `arange`, which is f32,
        // and the comparison's result is a boolean mask. Nothing here carries
        // the block's dtype, so casting to it would be a cast to the wrong
        // side.
        let in_block = greater_equal(&key_pos, &scalar_f32(ctx_len as f32), device)?;
        let allowed = maximum(&in_window, &in_block, device)?;

        let open = scalar_f32(0.0).astype(dtype, device)?;
        let blocked = scalar_f32(BLOCKED_BIAS).astype(dtype, device)?;
        where_cond(&allowed, &open, &blocked, device)
    }

    /// Conditioning rows a block query can reach: the window less the block
    /// position itself.
    ///
    /// A block query at index `t` sits at key position `ctx_len + t`, and the
    /// window lets it read back to `ctx_len + t - sliding_window + 1`. At
    /// `t = 0` — the shallowest query, and so the one that reaches back least
    /// far — the oldest readable key is `ctx_len - sliding_window + 1`, which is
    /// index 0 exactly when `ctx_len` is `sliding_window - 1`. One row more and
    /// the oldest is masked for every query in the block.
    ///
    /// `i32` because that is what an array axis is, and
    /// [`check_config`](super::check_config) refuses a window that does not fit
    /// in one — past that this subtraction wraps negative and
    /// [`Self::trim_conditioning`] slices from beyond its own end.
    pub(super) fn conditioning_rows(&self) -> i32 {
        self.cfg.sliding_window as i32 - 1
    }

    /// Drop the conditioning rows no block query can read.
    ///
    /// Two callers, and they are not interchangeable. [`Self::forward_hidden`]
    /// calls it to avoid projecting rows the mask will hide, which changes no
    /// output; the round loop calls it on the buffer it carries between rounds,
    /// where it is the **bound** on something that would otherwise grow by
    /// `len(target_layer_ids) * hidden_size` per emitted token forever. Only the
    /// first is visible in an answer, so a round loop passing its own row count
    /// could drift to any value at all and no test would move. It passes none.
    ///
    /// `hidden` is `[1, rows, len(target_layer_ids) * hidden_size]`, oldest row
    /// first. Returned unchanged when it is already short enough.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when the buffer is not the rank or the width this
    /// drafter's `fc` reads — a capture taken at another set of target layers
    /// has the same rank and the same row count, and would be projected as
    /// though it were this one.
    #[allow(
        clippy::indexing_slicing,
        reason = "each axis is read only after the rank has been compared against 3"
    )]
    pub(super) fn trim_conditioning(&self, hidden: &Array) -> Result<Array> {
        let device = self.device;
        let width = self.cfg.target_layer_ids.len() as i32 * self.cfg.hidden_size as i32;
        let shape = hidden.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != width {
            return Err(Error::Model(format!(
                "DFlash2Drafter: the conditioning buffer has shape {shape:?}, not the \
                 [1, rows, {width}] this drafter's target_layer_ids predict"
            )));
        }
        let keep = self.conditioning_rows();
        let rows = shape[1];
        if rows <= keep {
            return hidden.try_clone();
        }
        hidden.slice(&[0, rows - keep, 0], &[1, rows, width], &[1, 1, 1], device)
    }

    /// Append a round's committed conditioning rows to the buffer carried from
    /// the last one, bounded.
    ///
    /// Growing and bounding are one operation because the invariant is on the
    /// result, not on either step: a caller that appended and did not trim would
    /// produce the same tokens forever and grow without limit, and nothing in an
    /// answer would say so. There is no way to reach the first half alone.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when either side is not the rank or the width this
    /// drafter's `fc` reads.
    pub(super) fn extend_conditioning(&self, carried: &Array, committed: &Array) -> Result<Array> {
        let grown = concatenate(&[carried, committed], 1, self.device)?;
        self.trim_conditioning(&grown)
    }

    /// Channel groups the dynamic kernel's correction is shared over.
    fn conv_groups(&self) -> i32 {
        (self.cfg.hidden_size / self.cfg.conv_group_size) as i32
    }
}

/// `blocks` shifted `tap` positions later along the block axis, zero-filled
/// where the tap reaches before the block's first position.
#[allow(
    clippy::indexing_slicing,
    reason = "rank 4 by construction: the only caller reshapes its input to [1, len, groups, group_size] first"
)]
fn shift_into_block(blocks: &Array, tap: i32, device: Device) -> Result<Array> {
    if tap == 0 {
        return blocks.try_clone();
    }
    let shape = blocks.shape();
    let len = shape[1];
    let mut pad_shape = shape.clone();
    pad_shape[1] = tap.min(len);
    let pad = zeros(&pad_shape, blocks.dtype(), device)?;
    if tap >= len {
        return Ok(pad);
    }
    let kept = blocks.slice(
        &[0, 0, 0, 0],
        &[shape[0], len - tap, shape[2], shape[3]],
        &[1, 1, 1, 1],
        device,
    )?;
    concatenate(&[&pad, &kept], 1, device)
}

/// Check an input's rank and every axis `want` pins, and return the length of
/// the one axis it leaves free (`-1`).
fn shaped(a: &Array, name: &str, want: &[i32]) -> Result<i32> {
    let shape = a.shape();
    if shape.len() != want.len() {
        return Err(Error::Model(format!(
            "DFlash2Drafter: {name} has shape {shape:?}, not the rank-{} \
             {want:?} the config predicts",
            want.len()
        )));
    }
    let mut free = 0;
    for (axis, (&got, &wanted)) in shape.iter().zip(want).enumerate() {
        if wanted < 0 {
            if got < 1 {
                return Err(Error::Model(format!(
                    "DFlash2Drafter: {name} is empty along axis {axis}; the \
                     drafter has nothing to read"
                )));
            }
            free = got;
        } else if got != wanted {
            return Err(Error::Model(format!(
                "DFlash2Drafter: {name} has shape {shape:?}, not the {want:?} \
                 the config predicts"
            )));
        }
    }
    Ok(free)
}

#[cfg(test)]
#[path = "forward_tests.rs"]
mod forward_tests;
