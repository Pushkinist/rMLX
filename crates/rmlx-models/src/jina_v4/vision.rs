// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! jina-embeddings-v4 vision tower — Qwen2.5-VL ViT.
//!
//! Faithful port of the Qwen2.5-VL vision transformer
//! (`open-models/jinaai__jina-embeddings-v4/qwen2_5_vl.py`,
//! `Qwen2_5_VisionTransformerPretrainedModel`, eager/sdpa path) producing the
//! merged image embeddings `[num_merged_tokens, out_hidden_size=2048]` that
//! are scattered at the `<|image_pad|>` positions.
//!
//! Scope is the ViT in isolation: `patch_embed -> 2D-RoPE -> 32 blocks
//! (window/full attention dispatch + window permutation/reverse) ->
//! PatchMerger`. The image-feature merge / M-RoPE / image-span pooling are
//! NOT implemented here.
//!
//! ## jina-v4 deltas vs stock `mlx_vlm` (CRITICAL — applied here)
//!
//! 1. **`bias=True` trap.** jina-v4 vision carries additive bias on
//!    `blocks.N.mlp.{gate,up,down}_proj`, `blocks.N.attn.proj`, and
//!    `blocks.N.attn.qkv` (verified from the real safetensors header — 162
//!    `.bias` tensors under `visual.*`). Stock `mlx_vlm` uses `bias=False` on
//!    MLP + `attn.proj`. We load and apply every bias.
//! 2. **MLP is SwiGLU, not the stock 2-layer GELU MLP.** jina vision MLP is
//!    `Qwen2_5_VLMLP(bias=True)` (`hidden_act = "silu"`) with
//!    `gate_proj/up_proj/down_proj` — identical structure to the text tower's
//!    `Mlp` (`model.rs`), NOT the stock `mlx_vlm` `fc1/fc2` + GELU.
//! 3. **PatchEmbed Conv3d -> reshape+matmul.** kernel == stride (no overlap),
//!    so the conv is a plain matmul of the flattened patch against the
//!    flattened weight `[1280, 3*2*14*14] = [1280, 1176]` (== preprocess
//!    `feature_len`). No conv op.
//! 4. **PatchMerger** = `RMSNorm(ln_q)` -> `Linear(mlp.0)` -> **exact erf
//!    `rmlx_mlx::gelu`** -> `Linear(mlp.2)`; out dim == text hidden 2048.
//!    Biases on `mlp.0` / `mlp.2`.
//! 5. **2D vision RoPE** uses the NeoX `rotate_half` convention with a
//!    *precomputed, geometry-dependent* per-token angle table (`rot_pos_emb`,
//!    then re-ordered by the window permutation). MLX `rope*` kernels multiply
//!    `freq * row_index` internally and cannot express the arbitrary
//!    per-token (h,w) angle, so we precompute `cos`/`sin` on the host (pure
//!    geometry, no weights) and apply `x*cos + rotate_half(x)*sin` with
//!    element-wise ops.
//! 6. **No LoRA on vision** (`exclude_modules=.*visual.*`) — no adapter seam.
//!
//! Window vs full attention: `fullatt_block_indexes` blocks attend over all
//! patches (`cu_seqlens`); the rest are windowed (`cu_window_seqlens`). Both
//! are realized as a block-diagonal additive mask. Patches are permuted into
//! window order before the blocks and the **inverse permutation**
//! (argsort of `window_index`) is applied to the merger output, exactly as the
//! reference.

#![allow(
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::ref_option,
    clippy::struct_field_names
)]
use std::mem::size_of_val;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{
    add, concatenate, gelu, matmul, multiply, negative, rms_norm, scaled_dot_product_attention,
    silu, Array, Device, Dtype,
};
use tracing::{debug, info};

use super::config::JinaV4VisionConfig;
use super::preprocess::PixelValues;

// ---------------------------------------------------------------------------
// Plain bf16 Linear (+ optional additive bias). No LoRA seam — vision is
// explicitly excluded from jina's adapters (exclude_modules=.*visual.*).
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Linear {
    /// `[out_features, in_features]` bf16 weight (HF convention; transposed at
    /// matmul time, matching `model.rs`).
    weight: Array,
    /// Additive bias `[out_features]`. Present on every jina vision Linear
    /// except `patch_embed.proj` (the bias trap — see module docs).
    bias: Option<Array>,
}

impl Linear {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let y = matmul(x, &self.weight.transpose(&[1, 0], device)?, device)?;
        match &self.bias {
            Some(b) => add(&y, b, device),
            None => y.try_clone(),
        }
    }
}

struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        // Plain Qwen2RMSNorm (no `+1`), eps 1e-6 — same op the text tower uses.
        rms_norm(x, Some(&self.weight), self.eps, device)
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP (gate/up/down, bias=True, silu) — jina `Qwen2_5_VLMLP(bias=True)`
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = silu(&self.gate_proj.forward(x, device)?, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// Attention (full or windowed via a block-diagonal additive mask)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Attention {
    qkv: Linear,  // bias=True
    proj: Linear, // bias=True (the trap; stock mlx_vlm has bias=False)
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    /// `x`: `[seq, hidden]`. `cos`/`sin`: `[seq, head_dim]` precomputed RoPE
    /// tables (already in window order). `mask`: `[1, seq, seq]` additive
    /// block-diagonal mask (0 in-block, large-negative cross-block).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        device: Device,
    ) -> Result<Array> {
        let seq = x.shape()[0];
        let h = self.num_heads as i32;
        let d = self.head_dim as i32;

        // qkv -> [seq, 3, num_heads, head_dim] -> [3, seq, num_heads, head_dim]
        let qkv = self.qkv.forward(x, device)?;
        let qkv = qkv.reshape(&[seq, 3, h, d], device)?;
        let qkv = qkv.transpose(&[1, 0, 2, 3], device)?; // [3, seq, H, D]

        let q = qkv.slice(&[0, 0, 0, 0], &[1, seq, h, d], &[1, 1, 1, 1], device)?;
        let k = qkv.slice(&[1, 0, 0, 0], &[2, seq, h, d], &[1, 1, 1, 1], device)?;
        let v = qkv.slice(&[2, 0, 0, 0], &[3, seq, h, d], &[1, 1, 1, 1], device)?;
        let q = q.reshape(&[seq, h, d], device)?;
        let k = k.reshape(&[seq, h, d], device)?;
        let v = v.reshape(&[seq, h, d], device)?;

        // 2D vision RoPE (NeoX rotate_half) with precomputed per-token tables.
        // cos/sin are [seq, head_dim] -> broadcast over the head axis.
        let q = apply_rotary_pos_emb_vision(&q, cos, sin, device)?;
        let k = apply_rotary_pos_emb_vision(&k, cos, sin, device)?;

        // [seq, H, D] -> [1, H, seq, D] for SDPA (batch=1).
        let to_bhsd = |a: &Array| -> Result<Array> {
            let a = a.transpose(&[1, 0, 2], device)?; // [H, seq, D]
            a.reshape(&[1, h, seq, d], device)
        };
        let q = to_bhsd(&q)?;
        let k = to_bhsd(&k)?;
        let v = to_bhsd(&v)?;

        // Additive mask is [1, seq, seq]; SDPA broadcasts over the head axis.
        // mlx-c uses `"array"` for a caller-supplied additive mask array
        // (matches gemma4 layers.rs:314 — `"additive"` is rejected by 0.6.0).
        let out =
            scaled_dot_product_attention(&q, &k, &v, self.scale, "array", Some(mask), device)?;
        // [1, H, seq, D] -> [seq, H*D]
        let out = out.reshape(&[h, seq, d], device)?;
        let out = out.transpose(&[1, 0, 2], device)?; // [seq, H, D]
        let out = out.reshape(&[seq, h * d], device)?;
        self.proj.forward(&out, device)
    }
}

/// `q_embed = (x * cos) + (rotate_half(x) * sin)` — NeoX vision RoPE.
///
/// `x`: `[seq, num_heads, head_dim]`. `cos`/`sin`: `[seq, head_dim]`
/// (broadcast over the head axis via an inserted size-1 dim).
/// `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn apply_rotary_pos_emb_vision(
    x: &Array,
    cos: &Array,
    sin: &Array,
    device: Device,
) -> Result<Array> {
    let s = x.shape();
    let (seq, heads, d) = (s[0], s[1], s[2]);
    let half = d / 2;

    // [seq, head_dim] -> [seq, 1, head_dim] to broadcast over heads.
    let cos_b = cos.reshape(&[seq, 1, d], device)?;
    let sin_b = sin.reshape(&[seq, 1, d], device)?;

    let x1 = x.slice(&[0, 0, 0], &[seq, heads, half], &[1, 1, 1], device)?;
    let x2 = x.slice(&[0, 0, half], &[seq, heads, d], &[1, 1, 1], device)?;
    let rot = concatenate(&[&negative(&x2, device)?, &x1], 2, device)?;

    let a = multiply(x, &cos_b, device)?;
    let b = multiply(&rot, &sin_b, device)?;
    add(&a, &b, device)
}

// ---------------------------------------------------------------------------
// Vision block (pre-norm residuals; norm = plain RMSNorm)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Block {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        device: Device,
    ) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = self.norm1.forward(x, device)?;
        let h = self.attn.forward(&h, cos, sin, mask, device)?;
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let n = self.norm2.forward(&h, device)?;
        let m = self.mlp.forward(&n, device)?;
        add(&residual, &m, device)
    }
}

// ---------------------------------------------------------------------------
// PatchMerger: RMSNorm(ln_q) -> Linear(mlp.0) -> exact-GELU -> Linear(mlp.2)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct PatchMerger {
    ln_q: RmsNorm,      // RMSNorm over context_dim (1280)
    mlp0: Linear,       // [hidden_size, hidden_size] = [5120, 5120], bias=True
    mlp2: Linear,       // [out_hidden_size, hidden_size] = [2048, 5120], bias=True
    hidden_size: usize, // context_dim * spatial_merge_size^2
}

impl PatchMerger {
    /// `x`: `[seq, context_dim]`. Returns `[seq / merge_unit, out_hidden]`.
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let normed = self.ln_q.forward(x, device)?;
        let merged = normed.reshape(&[-1, self.hidden_size as i32], device)?;
        let h = self.mlp0.forward(&merged, device)?;
        // Exact erf GELU (NOT gelu_tanh) — PatchMerger fidelity requirement.
        let h = gelu(&h, device)?;
        self.mlp2.forward(&h, device)
    }
}

// ---------------------------------------------------------------------------
// Vision tower
// ---------------------------------------------------------------------------

/// jina-v4 Qwen2.5-VL vision tower. Loaded from the main safetensors shards
/// (`visual.*`). No LoRA seam (vision is excluded from jina adapters).
#[allow(missing_debug_implementations)]
pub struct JinaV4Vision {
    cfg: JinaV4VisionConfig,
    patch_embed_w: Array, // [hidden, feature_len] = [1280, 1176]
    blocks: Vec<Block>,
    merger: PatchMerger,
    /// `head_dim = hidden_size / num_heads`.
    head_dim: usize,
    /// `spatial_merge_size^2`.
    merge_unit: usize,
}

impl JinaV4Vision {
    /// Parsed vision sub-config this tower was built from.
    pub fn config(&self) -> &JinaV4VisionConfig {
        &self.cfg
    }

    /// Run the ViT over one preprocessed image.
    ///
    /// Returns `[num_merged_tokens, out_hidden_size]` where
    /// `num_merged_tokens = num_patches / spatial_merge_size^2`. The output
    /// rows are in the **original patch order** (the window permutation is
    /// reversed via `argsort(window_index)`), matching the reference.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward(&self, pv: &PixelValues, device: Device) -> Result<Array> {
        let grid = pv.grid;
        let merge = self.cfg.spatial_merge_size;
        if !grid.h.is_multiple_of(merge) || !grid.w.is_multiple_of(merge) {
            return Err(Error::Model(format!(
                "jina-v4 vision: grid {}x{} not divisible by spatial_merge_size {merge}",
                grid.h, grid.w
            )));
        }
        let num_patches = pv.num_patches;
        if !num_patches.is_multiple_of(self.merge_unit) {
            return Err(Error::Model(format!(
                "jina-v4 vision: num_patches {num_patches} not divisible by merge_unit {}",
                self.merge_unit
            )));
        }

        // ---- PatchEmbed (Conv3d kernel==stride -> reshape + matmul) ------
        // pixel_values is row-major [num_patches, feature_len] f32. Upload
        // and keep f32 — the whole ViT runs float32 (see load_array note;
        // matches jina's float32 image-embedding reference).
        let np = num_patches as i32;
        let fl = pv.feature_len as i32;
        let pv_bytes =
            unsafe { std::slice::from_raw_parts(pv.data.as_ptr().cast::<u8>(), pv.data.len() * 4) };
        let pv_arr = Array::from_bytes(pv_bytes, &[np, fl], Dtype::F32)?;
        // patch_embed_w: [hidden, feature_len] -> x @ W^T -> [num_patches, hidden]
        let mut h = matmul(
            &pv_arr,
            &self.patch_embed_w.transpose(&[1, 0], device)?,
            device,
        )?;

        // ---- geometry (host, pure — depends only on grid_thw) ------------
        let geo = compute_geometry(&self.cfg, grid.t, grid.h, grid.w, self.head_dim)?;
        let seq = num_patches as i32;

        // window-permute hidden: [seq, hid] -> [seq/mu, mu, hid] -> gather
        let hid = h.shape()[1];
        let h_grp = h.reshape(
            &[seq / self.merge_unit as i32, self.merge_unit as i32, hid],
            device,
        )?;
        let widx = Array::from_bytes(
            i32_bytes(&geo.window_index),
            &[geo.window_index.len() as i32],
            Dtype::I32,
        )?;
        let h_grp = h_grp.take(&widx, 0, device)?;
        h = h_grp.reshape(&[seq, hid], device)?;

        // RoPE cos/sin: built per-token on host (already in window order),
        // shape [seq, head_dim]. Keep f32 (the ViT runs float32).
        let cos = Array::from_bytes(
            f32_bytes(&geo.cos),
            &[seq, self.head_dim as i32],
            Dtype::F32,
        )?;
        let sin = Array::from_bytes(
            f32_bytes(&geo.sin),
            &[seq, self.head_dim as i32],
            Dtype::F32,
        )?;

        // Additive masks (full + window) — [1, seq, seq] bf16.
        let full_mask = build_block_mask(&geo.cu_seqlens, num_patches, device)?;
        let window_mask = build_block_mask(&geo.cu_window_seqlens, num_patches, device)?;

        debug!(
            num_patches,
            seq = num_patches,
            full_blocks = ?self.cfg.fullatt_block_indexes,
            "jina-v4 vision: dispatching {} blocks",
            self.blocks.len()
        );

        for (i, blk) in self.blocks.iter().enumerate() {
            let mask = if self.cfg.fullatt_block_indexes.contains(&i) {
                &full_mask
            } else {
                &window_mask
            };
            h = blk.forward(&h, &cos, &sin, mask, device)?;
        }

        // ---- PatchMerger + reverse window permutation --------------------
        let merged = self.merger.forward(&h, device)?; // [seq/mu, out_hidden]
        let ridx = Array::from_bytes(
            i32_bytes(&geo.reverse_index),
            &[geo.reverse_index.len() as i32],
            Dtype::I32,
        )?;
        merged.take(&ridx, 0, device)
    }
}

// ---------------------------------------------------------------------------
// Geometry — host-side, pure (mirrors qwen2_5_vl.py rot_pos_emb /
// get_window_index / forward exactly). No weights, no MLX.
// ---------------------------------------------------------------------------

struct Geometry {
    /// Per-token RoPE `cos` table, row-major `[seq * head_dim]` (window order).
    cos: Vec<f32>,
    /// Per-token RoPE `sin` table, row-major `[seq * head_dim]` (window order).
    sin: Vec<f32>,
    /// Window permutation over merge-groups, length `seq / merge_unit`.
    window_index: Vec<i32>,
    /// `argsort(window_index)` — inverse permutation, same length.
    reverse_index: Vec<i32>,
    /// Cumulative full-attention seqlens (block boundaries), patch units.
    cu_seqlens: Vec<usize>,
    /// Cumulative window-attention seqlens (de-duplicated consecutive),
    /// patch units.
    cu_window_seqlens: Vec<usize>,
}

/// Build the per-token RoPE angle table + window partition for one image.
///
/// Faithful host port of `qwen2_5_vl.py`:
/// - `Qwen2_5_VisionRotaryEmbedding`: `inv_freq[i] = theta^(-2i/dim_rot)`,
///   `freqs = outer(arange(max_grid), inv_freq)`; `dim_rot = head_dim/2`.
/// - `rot_pos_emb`: per-(h,w) pos-id reordering (the `spatial_merge_size`
///   block transpose), `rotary_pos_emb_full[pos_ids].flatten(1)` ->
///   `[num_patches, head_dim/2]`, then `cat(emb, emb)` -> cos/sin over
///   `head_dim`.
/// - `get_window_index` + the `forward` window reshape/gather.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn compute_geometry(
    cfg: &JinaV4VisionConfig,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
) -> Result<Geometry> {
    let merge = cfg.spatial_merge_size;
    let merge_unit = merge * merge;
    let theta = 10_000.0_f64;
    // VisionRotaryEmbedding(head_dim // 2): the rotary dim is head_dim/2, and
    // inv_freq has head_dim/4 entries (arange(0, dim, 2) over dim=head_dim/2).
    let dim_rot = head_dim / 2;
    let n_freq = dim_rot / 2; // == head_dim / 4
    let inv_freq: Vec<f64> = (0..n_freq)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / dim_rot as f64))
        .collect();

    // ---- rot_pos_emb pos_ids (single image) -----------------------------
    // hpos/wpos per the spatial_merge_size block transpose, flattened.
    let mut hpos = vec![0usize; grid_h * grid_w];
    let mut wpos = vec![0usize; grid_h * grid_w];
    {
        // hpos_ids = arange(h)[:,None].expand(h,w) reshaped
        // (h/m, m, w/m, m) -> permute(0,2,1,3) -> flatten
        let hb = grid_h / merge;
        let wb = grid_w / merge;
        let mut idx = 0usize;
        for a in 0..hb {
            for b in 0..wb {
                for i in 0..merge {
                    for j in 0..merge {
                        let hv = a * merge + i;
                        let wv = b * merge + j;
                        hpos[idx] = hv;
                        wpos[idx] = wv;
                        idx += 1;
                    }
                }
            }
        }
        debug_assert_eq!(idx, grid_h * grid_w);
    }

    // rotary_pos_emb_full[pos_ids].flatten(1): for each token, the angle
    // vector is concat(freqs[h] , freqs[w]) of length 2*n_freq == head_dim/2.
    // Then forward() does cat(emb, emb) -> head_dim, cos()/sin().
    let tokens_per_t = grid_h * grid_w;
    let seq = grid_t * tokens_per_t;
    let mut cos = vec![0.0_f32; seq * head_dim];
    let mut sin = vec![0.0_f32; seq * head_dim];
    for t in 0..grid_t {
        for p in 0..tokens_per_t {
            let tok = t * tokens_per_t + p;
            // angle[0..n_freq] = h_pos * inv_freq
            // angle[n_freq..2n] = w_pos * inv_freq (== head_dim/2 wide)
            // emb = cat(angle, angle) -> head_dim
            let hf = hpos[p] as f64;
            let wf = wpos[p] as f64;
            let base = tok * head_dim;
            for i in 0..n_freq {
                let ah = (hf * inv_freq[i]) as f32;
                let aw = (wf * inv_freq[i]) as f32;
                // first half of `emb` (== rotary_pos_emb)
                cos[base + i] = ah.cos();
                sin[base + i] = ah.sin();
                cos[base + n_freq + i] = aw.cos();
                sin[base + n_freq + i] = aw.sin();
                // second half (cat(emb, emb) duplicate)
                cos[base + dim_rot + i] = ah.cos();
                sin[base + dim_rot + i] = ah.sin();
                cos[base + dim_rot + n_freq + i] = aw.cos();
                sin[base + dim_rot + n_freq + i] = aw.sin();
            }
        }
    }

    // ---- get_window_index ----------------------------------------------
    let vit_merger_window = cfg.window_size / merge / cfg.patch_size;
    if vit_merger_window == 0 {
        return Err(Error::Model(
            "jina-v4 vision: vit_merger_window_size computed as 0".into(),
        ));
    }
    let mut window_index: Vec<i32> = Vec::new();
    let mut cu_window_seqlens: Vec<usize> = vec![0];
    let mut window_index_id = 0usize;
    {
        let llm_h = grid_h / merge;
        let llm_w = grid_w / merge;
        for _t in 0..grid_t {
            // index = arange(t*llm_h*llm_w).reshape(t, llm_h, llm_w) — here a
            // single t-slice.
            let pad_h = vit_merger_window - llm_h % vit_merger_window;
            let pad_w = vit_merger_window - llm_w % vit_merger_window;
            let nh = (llm_h + pad_h) / vit_merger_window;
            let nw = (llm_w + pad_w) / vit_merger_window;
            // index_padded[r][c] = r*llm_w + c if in-bounds else -100
            let val = |r: usize, c: usize| -> i64 {
                if r < llm_h && c < llm_w {
                    (r * llm_w + c) as i64
                } else {
                    -100
                }
            };
            // permute(0,1,3,2,4): iterate (wh, ww, ih, iw); collect non-pad,
            // and per-window seqlen.
            for wh in 0..nh {
                for ww in 0..nw {
                    let mut seqlen = 0usize;
                    for ih in 0..vit_merger_window {
                        for iw in 0..vit_merger_window {
                            let r = wh * vit_merger_window + ih;
                            let c = ww * vit_merger_window + iw;
                            let vv = val(r, c);
                            if vv != -100 {
                                window_index.push(vv as i32 + window_index_id as i32);
                                seqlen += 1;
                            }
                        }
                    }
                    let prev = *cu_window_seqlens.last().unwrap();
                    cu_window_seqlens.push(seqlen * merge_unit + prev);
                }
            }
            window_index_id += llm_h * llm_w;
        }
    }
    // torch.unique_consecutive on cu_window_seqlens.
    let mut cu_window_dedup: Vec<usize> = Vec::with_capacity(cu_window_seqlens.len());
    for &v in &cu_window_seqlens {
        if cu_window_dedup.last() != Some(&v) {
            cu_window_dedup.push(v);
        }
    }

    // ---- cu_seqlens (full attention) -----------------------------------
    // repeat_interleave(grid_h*grid_w, grid_t).cumsum, padded with a leading 0.
    let mut cu_seqlens: Vec<usize> = vec![0];
    {
        let mut acc = 0usize;
        for _t in 0..grid_t {
            acc += grid_h * grid_w;
            cu_seqlens.push(acc);
        }
    }

    // ---- apply window permutation to cos/sin (forward() reshape+gather) --
    // hidden/rope reshaped [seq/mu, mu, *] then indexed by window_index.
    let n_groups = seq / merge_unit;
    if window_index.len() != n_groups {
        return Err(Error::Model(format!(
            "jina-v4 vision: window_index len {} != seq/merge_unit {n_groups}",
            window_index.len()
        )));
    }
    let mut cos_w = vec![0.0_f32; seq * head_dim];
    let mut sin_w = vec![0.0_f32; seq * head_dim];
    for (dst_g, &src_g) in window_index.iter().enumerate() {
        let src_g = src_g as usize;
        for u in 0..merge_unit {
            let dst_tok = dst_g * merge_unit + u;
            let src_tok = src_g * merge_unit + u;
            let d = dst_tok * head_dim;
            let s = src_tok * head_dim;
            cos_w[d..d + head_dim].copy_from_slice(&cos[s..s + head_dim]);
            sin_w[d..d + head_dim].copy_from_slice(&sin[s..s + head_dim]);
        }
    }

    // reverse_index = argsort(window_index) (stable; values are a permutation).
    let mut order: Vec<i32> = (0..window_index.len() as i32).collect();
    order.sort_by_key(|&i| window_index[i as usize]);

    Ok(Geometry {
        cos: cos_w,
        sin: sin_w,
        window_index,
        reverse_index: order,
        cu_seqlens,
        cu_window_seqlens: cu_window_dedup,
    })
}

/// Build a `[1, seq, seq]` additive attention mask that is `0` inside each
/// `[cu[i-1], cu[i])` block and a large negative elsewhere (block-diagonal).
/// Mirrors the eager `Qwen2_5_VLVisionAttention` mask construction.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn build_block_mask(cu: &[usize], seq: usize, _device: Device) -> Result<Array> {
    let neg = -1.0e9_f32;
    let mut m = vec![neg; seq * seq];
    for w in cu.windows(2) {
        let (s, e) = (w[0], w[1]);
        for r in s..e {
            let row = r * seq;
            for c in s..e {
                m[row + c] = 0.0;
            }
        }
    }
    // f32 — the ViT runs float32 (mask dtype must match q/k/v).
    Array::from_bytes(f32_bytes(&m), &[1, seq as i32, seq as i32], Dtype::F32)
}

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

// ---------------------------------------------------------------------------
// Loader (single-scan over the main shards — mirrors model.rs)
// ---------------------------------------------------------------------------

/// Load the jina-v4 vision tower (`visual.*`) from `model_dir`'s main
/// safetensors shards. Pure bf16. Every Linear bias under `visual.*` (except
/// `patch_embed.proj`, which has none) is loaded and applied — the bias trap.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn load_vision_tower(
    model_dir: &Path,
    cfg: &JinaV4VisionConfig,
) -> Result<JinaV4Vision> {
    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    // The vision tower runs in **float32**. jina's reference image-embedding
    // path (`AutoModel.from_pretrained(torch_dtype=float32)`,
    // `get_image_features`) computes the ViT in float32, and over 32
    // transformer blocks bf16 accumulation drifts ~4% relative (cosine
    // ~0.999 per merged token → image embedding ~0.991–0.995, below the
    // 0.999 gate). The text tower stays bf16 (its parity holds at 0.99998 —
    // mean-pool averages residual bf16 noise and its reference matches);
    // vision is small here (no decode loop) so the f32 cost is negligible.
    // Every vision weight/bias is upcast to f32 at load; the forward keeps
    // pixel_values / RoPE tables / masks in f32 too.
    fn load_array(shards: &ShardSet, name: &str) -> Result<Array> {
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
            "jina-v4 vision: tensor '{name}' not found in any shard"
        )))
    }
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn has_tensor(shards: &ShardSet, name: &str) -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    }
    let load_linear = |base: &str| -> Result<Linear> {
        let weight = load_array(&shards, &format!("{base}.weight"))?;
        let bias_name = format!("{base}.bias");
        let bias = if has_tensor(&shards, &bias_name) {
            Some(load_array(&shards, &bias_name)?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    };
    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: load_array(&shards, &format!("{name}.weight"))?,
            eps: 1e-6,
        })
    };

    let head_dim = cfg.hidden_size / cfg.num_heads;
    let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;

    info!(
        depth = cfg.depth,
        hidden = cfg.hidden_size,
        intermediate = cfg.intermediate_size,
        num_heads = cfg.num_heads,
        head_dim,
        fullatt = ?cfg.fullatt_block_indexes,
        "jina-v4: loading vision tower (bf16, bias=true, no LoRA)"
    );

    // PatchEmbed: Conv3d weight [hidden, in_ch, tps, ps, ps] -> flatten to
    // [hidden, in_ch*tps*ps*ps] for the reshape+matmul (kernel == stride).
    let pe = load_array(&shards, "visual.patch_embed.proj.weight")?;
    let pe_shape = pe.shape();
    let hidden = pe_shape[0];
    let feat_len: i32 = pe_shape[1..].iter().product();
    let patch_embed_w = pe.reshape(&[hidden, feat_len], Device::Cpu)?;
    debug!(
        ?pe_shape,
        flat = ?[hidden, feat_len],
        "jina-v4 vision: patch_embed Conv3d -> flat matmul weight"
    );

    let mut blocks = Vec::with_capacity(cfg.depth);
    for i in 0..cfg.depth {
        let b = format!("visual.blocks.{i}");
        blocks.push(Block {
            norm1: load_rms(&format!("{b}.norm1"))?,
            norm2: load_rms(&format!("{b}.norm2"))?,
            attn: Attention {
                qkv: load_linear(&format!("{b}.attn.qkv"))?,
                proj: load_linear(&format!("{b}.attn.proj"))?,
                num_heads: cfg.num_heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
            },
            mlp: Mlp {
                gate_proj: load_linear(&format!("{b}.mlp.gate_proj"))?,
                up_proj: load_linear(&format!("{b}.mlp.up_proj"))?,
                down_proj: load_linear(&format!("{b}.mlp.down_proj"))?,
            },
        });
        debug!(block = i, "jina-v4 vision: loaded block");
    }

    let merger = PatchMerger {
        ln_q: load_rms("visual.merger.ln_q")?,
        mlp0: load_linear("visual.merger.mlp.0")?,
        mlp2: load_linear("visual.merger.mlp.2")?,
        hidden_size: cfg.hidden_size * merge_unit,
    };

    info!(total_blocks = cfg.depth, "jina-v4: vision tower loaded");
    Ok(JinaV4Vision {
        cfg: cfg.clone(),
        patch_embed_w,
        blocks,
        merger,
        head_dim,
        merge_unit,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;
