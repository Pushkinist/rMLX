// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Qwen3-VL-MoE vision tower (ViT) — faithful port of
//! `mlx-vlm/mlx_vlm/models/qwen3_vl_moe/vision.py::VisionModel`.
//!
//! Architecturally distinct from the Qwen2.5-VL ViT reused by jina-v4
//! ([`crate::jina_v4::vision`]):
//! - **No window attention.** Every block attends over all patches of one
//!   image (a single `cu_seqlens` block per image — full attention). There is
//!   no `fullatt_block_indexes` / window permutation.
//! - **LayerNorm with bias** (`norm1`/`norm2`), not RMSNorm.
//! - **2-layer GELU-tanh MLP** (`linear_fc1` -> GELU(tanh) -> `linear_fc2`),
//!   not SwiGLU.
//! - **Learned absolute position embeddings** bilinearly interpolated to the
//!   image grid (`fast_pos_embed_interpolate`), added to the patch embeddings
//!   *before* the blocks — in addition to the per-token 2D vision RoPE.
//! - **Deepstack mergers**: at vision layers in `deepstack_visual_indexes`
//!   (`[8, 16, 24]`) the post-block hidden is run through an extra
//!   `PatchMerger` (with `use_postshuffle_norm=true`) to produce a
//!   `deepstack_visual_embed` that the LM additively injects into the matching
//!   decoder layer (`language.py::_deepstack_process`).
//!
//! All vision weights are BF16 unquantized (`vision_tower.*`). The whole ViT
//! runs in **float32** for numerical fidelity (it runs once per image, not in
//! the decode hot loop) — mirroring the jina-v4 vision float32 decision.
//!
//! ## PatchEmbed (Conv3d kernel == stride -> reshape + matmul)
//!
//! `patch_embed.proj` is a `Conv3d(in_ch, hidden, kernel=stride=[tps, ps, ps])`.
//! Kernel == stride means no overlap, so the conv is a plain matmul of each
//! flattened patch (`in_ch * tps * ps * ps`) against the flattened weight. The
//! preprocessed `pixel_values` already arrive as
//! `[num_patches, in_ch*tps*ps*ps]`, so the patch embed is
//! `pixel_values @ W_flat^T + bias`.
//!
//! ## 2D vision RoPE
//!
//! `VisionRotaryEmbedding(head_dim // 2)`: `inv_freq[i] = theta^(-2i/dim_rot)`
//! with `dim_rot = head_dim/2`, `theta = 10000`. `rot_pos_emb` builds per-(h,w)
//! position ids (via the `spatial_merge_size` block transpose), looks up
//! `freq_table[h]` / `freq_table[w]`, concatenates them
//! (`[num_patches, head_dim/2]`), then `cat(emb, emb)` doubles to `head_dim`.
//! Applied as `x*cos + rotate_half(x)*sin` (NeoX convention).

use std::mem::size_of_val;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{
    add, concatenate, divide, gelu_tanh, matmul, multiply, negative, scaled_dot_product_attention,
    sqrt, subtract, sum_axis, Array, Device, Dtype,
};
use tracing::{debug, info};

use super::config::Qwen3VlMoeVisionConfig;

// ---------------------------------------------------------------------------
// Plain Linear (+ bias). Every Qwen3-VL vision Linear carries a bias.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Linear {
    /// `[out, in]` weight (HF convention; transposed at matmul time).
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let y = matmul(x, &self.weight.transpose(&[1, 0], device)?, device)?;
        match &self.bias {
            Some(b) => add(&y, b, device),
            None => Ok(y),
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNorm (weight + bias, over the last axis). rmlx-mlx exposes no
// `layer_norm` op; computed with reduction/elementwise ops (mirrors gemma3
// vision LayerNorm). Runs once per image, so the extra dispatches are cheap.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let mut keep = x.shape();
        let d = *keep.last().expect("layer_norm: empty shape");
        let inv_d = rmlx_mlx::scalar_f32(1.0 / d as f32); // f32-ok: Qwen3-VL-MoE vision tower runs entirely in f32
        *keep.last_mut().unwrap() = 1;

        let mean = multiply(
            &sum_axis(x, -1, device)?.reshape(&keep, device)?,
            &inv_d,
            device,
        )?;
        let xc = subtract(x, &mean, device)?;
        let var = multiply(
            &sum_axis(&multiply(&xc, &xc, device)?, -1, device)?.reshape(&keep, device)?,
            &inv_d,
            device,
        )?;
        let denom = sqrt(&add(&var, &rmlx_mlx::scalar_f32(self.eps), device)?, device)?; // f32-ok: Qwen3-VL-MoE vision tower f32
        let normed = divide(&xc, &denom, device)?;
        let scaled = multiply(&normed, &self.weight, device)?;
        add(&scaled, &self.bias, device)
    }
}

// ---------------------------------------------------------------------------
// 2-layer GELU-tanh MLP: linear_fc1 -> GELU(tanh) -> linear_fc2
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Mlp {
    linear_fc1: Linear,
    linear_fc2: Linear,
}

impl Mlp {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let h = self.linear_fc1.forward(x, device)?;
        let h = gelu_tanh(&h, device)?;
        self.linear_fc2.forward(&h, device)
    }
}

// ---------------------------------------------------------------------------
// Attention — full attention over all patches of one image, QKV packed,
// 2D vision RoPE.
// ---------------------------------------------------------------------------

/// Per-command-buffer score-matrix budget for the ViT full-attention pass, in
/// `query_rows * key_rows` elements (per head). The Qwen3-VL ViT attends over
/// every patch of one image (reference-faithful full attention — see the module
/// docstring), so the score matrix is `seq * seq` per head. A large image
/// (tens of thousands of patches) produces an O(seq²) score matrix whose single
/// `scaled_dot_product_attention` command buffer overruns the ~10 s Metal GPU
/// watchdog. We keep the math identical (each query still attends to all keys)
/// but split the query dimension into tiles so each tile's command buffer
/// covers `tile_rows * seq ≤ budget` score elements. The tile rows are derived
/// as `max(1, budget / seq)`, so the per-buffer work stays bounded regardless of
/// image size. The budget is chosen well below the size that timed out (a
/// ~5184-patch single buffer, `26.9M` elems/head) and well above one that
/// completes comfortably (a 784-patch buffer, `0.6M` elems/head).
const VIT_ATTN_SCORE_BUDGET: usize = 2_097_152;

#[allow(missing_debug_implementations)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

/// `rotate_half(x)` = `cat(-x[..., d/2:], x[..., :d/2])` over the last axis.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn rotate_half(x: &Array, device: Device) -> Result<Array> {
    let s = x.shape();
    let last = s.len() - 1;
    let d = s[last];
    let half = d / 2;
    let mut start = vec![0i32; s.len()];
    let mut stop = s.clone();
    let stride = vec![1i32; s.len()];
    stop[last] = half;
    let x1 = x.slice(&start, &stop, &stride, device)?;
    start[last] = half;
    stop[last] = d;
    let x2 = x.slice(&start, &stop, &stride, device)?;
    concatenate(&[&negative(&x2, device)?, &x1], last as i32, device)
}

/// `out = x*cos + rotate_half(x)*sin`. `x`: `[seq, num_heads, head_dim]`;
/// `cos`/`sin`: `[seq, head_dim]` (broadcast over the head axis).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn apply_rotary(x: &Array, cos: &Array, sin: &Array, device: Device) -> Result<Array> {
    let s = x.shape();
    let (seq, _heads, d) = (s[0], s[1], s[2]);
    let cos_b = cos.reshape(&[seq, 1, d], device)?;
    let sin_b = sin.reshape(&[seq, 1, d], device)?;
    let a = multiply(x, &cos_b, device)?;
    let b = multiply(&rotate_half(x, device)?, &sin_b, device)?;
    add(&a, &b, device)
}

impl Attention {
    /// `x`: `[seq, hidden]`. `cos`/`sin`: `[seq, head_dim]`. `mask`: optional
    /// additive `[1, seq, seq]` mask (None = full attention for one image).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let seq = x.shape()[0];
        let h = self.num_heads as i32;
        let d = self.head_dim as i32;

        // qkv -> [seq, 3, H, D] -> [3, seq, H, D]
        let qkv = self.qkv.forward(x, device)?;
        let qkv = qkv.reshape(&[seq, 3, h, d], device)?;
        let qkv = qkv.transpose(&[1, 0, 2, 3], device)?;
        let q = qkv
            .slice(&[0, 0, 0, 0], &[1, seq, h, d], &[1, 1, 1, 1], device)?
            .reshape(&[seq, h, d], device)?;
        let k = qkv
            .slice(&[1, 0, 0, 0], &[2, seq, h, d], &[1, 1, 1, 1], device)?
            .reshape(&[seq, h, d], device)?;
        let v = qkv
            .slice(&[2, 0, 0, 0], &[3, seq, h, d], &[1, 1, 1, 1], device)?
            .reshape(&[seq, h, d], device)?;

        let q = apply_rotary(&q, cos, sin, device)?;
        let k = apply_rotary(&k, cos, sin, device)?;

        // [seq, H, D] -> [1, H, seq, D]
        let to_bhsd = |a: &Array| -> Result<Array> {
            a.transpose(&[1, 0, 2], device)?
                .reshape(&[1, h, seq, d], device)
        };
        let q = to_bhsd(&q)?;
        let k = to_bhsd(&k)?;
        let v = to_bhsd(&v)?;

        let out = match mask {
            // Multi-image / per-frame masked path (not built by the single-image
            // serve path today): run as a single SDPA. A query tile would need
            // the mask sliced per tile; not wired because no caller produces a
            // mask here.
            Some(m) => {
                scaled_dot_product_attention(&q, &k, &v, self.scale, "array", Some(m), device)?
            }
            // Single-image full attention. Tile the query dimension so each
            // command buffer's score matrix (`tile_rows * seq`) stays under the
            // Metal GPU watchdog budget; mathematically identical to one SDPA
            // over all queries (each query attends to all keys). For a small
            // image (`seq * seq <= budget`) this collapses to the original
            // single SDPA, so small-image numerics are bit-identical.
            None => self.attend_tiled(&q, &k, &v, seq, h, d, VIT_ATTN_SCORE_BUDGET, device)?,
        };
        let out = out
            .reshape(&[h, seq, d], device)?
            .transpose(&[1, 0, 2], device)?
            .reshape(&[seq, h * d], device)?;
        self.proj.forward(&out, device)
    }

    /// Full attention with the query dimension tiled to bound the per-command-
    /// buffer score-matrix work. `q`/`k`/`v` are `[1, H, seq, D]`; returns
    /// `[1, H, seq, D]` (same as a single SDPA over all queries — each query
    /// attends to every key). Tiles only when `seq * seq` exceeds `budget`;
    /// otherwise a single SDPA (bit-identical to the pre-tiling path). `budget`
    /// is a parameter so a test can drive the tiling path at a small `seq`.
    #[allow(clippy::too_many_arguments)]
    fn attend_tiled(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        seq: i32,
        h: i32,
        d: i32,
        budget: usize,
        device: Device,
    ) -> Result<Array> {
        let seq_u = seq as usize;
        // One SDPA when the full score matrix fits the budget — preserves the
        // exact pre-tiling command buffer (and numerics) for small images.
        if seq_u.saturating_mul(seq_u) <= budget {
            return scaled_dot_product_attention(q, k, v, self.scale, "", None, device);
        }
        // Query rows per tile so `tile * seq <= budget`; at least 1.
        let tile = (budget / seq_u).max(1) as i32;
        debug!(
            seq,
            tile, "qwen3_vl_moe vision: tiling ViT attention query dim under watchdog budget"
        );
        let mut tiles: Vec<Array> = Vec::with_capacity((seq_u.div_ceil(tile as usize)).max(1));
        let mut qs = 0_i32;
        while qs < seq {
            let qe = (qs + tile).min(seq);
            // q[:, :, qs:qe, :] against full k/v — full attention for this tile.
            let q_tile = q.slice(&[0, 0, qs, 0], &[1, h, qe, d], &[1, 1, 1, 1], device)?;
            let o_tile = scaled_dot_product_attention(&q_tile, k, v, self.scale, "", None, device)?;
            // Flush this tile's command buffer so the watchdog never sees the
            // whole O(seq²) attention in one buffer.
            o_tile.eval()?;
            tiles.push(o_tile);
            qs = qe;
        }
        // Concatenate tile outputs back along the query (seq) axis -> [1,H,seq,D].
        let refs: Vec<&Array> = tiles.iter().collect();
        concatenate(&refs, 2, device)
    }
}

// ---------------------------------------------------------------------------
// Vision block — pre-norm residuals: LN1 -> attn -> +res ; LN2 -> mlp -> +res
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Attention,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let r = self
            .attn
            .forward(&self.norm1.forward(x, device)?, cos, sin, mask, device)?;
        let h = add(x, &r, device)?;
        let m = self.mlp.forward(&self.norm2.forward(&h, device)?, device)?;
        add(&h, &m, device)
    }
}

// ---------------------------------------------------------------------------
// PatchMerger: LayerNorm -> reshape(merge group) -> Linear -> GELU(tanh) ->
// Linear. `use_postshuffle_norm` controls whether the LayerNorm runs over the
// merged (post-shuffle) feature dim (`hidden*merge^2`) or the raw `hidden`.
// Deepstack mergers use postshuffle_norm=true; the final merger uses false.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct PatchMerger {
    norm: LayerNorm,
    linear_fc1: Linear,
    linear_fc2: Linear,
    /// `hidden * spatial_merge_size^2` — the merged feature dim.
    merged_dim: usize,
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    /// `x`: `[seq, hidden]`. Returns `[seq / merge^2, out_hidden]`.
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let normed = if self.use_postshuffle_norm {
            let merged = x.reshape(&[-1, self.merged_dim as i32], device)?;
            self.norm.forward(&merged, device)?
        } else {
            let n = self.norm.forward(x, device)?;
            n.reshape(&[-1, self.merged_dim as i32], device)?
        };
        let h = self.linear_fc1.forward(&normed, device)?;
        let h = gelu_tanh(&h, device)?;
        self.linear_fc2.forward(&h, device)
    }
}

// ---------------------------------------------------------------------------
// Vision tower
// ---------------------------------------------------------------------------

/// Qwen3-VL-MoE vision transformer. Output of [`forward`](Self::forward) is the
/// merged patch embeddings `[num_merged, out_hidden]` plus one
/// `deepstack_visual_embed` per `deepstack_visual_indexes` entry.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model struct — private weight fields; public API is forward(); adding a field requires updating load_weights and the qwen3_vl_moe vision loader"
)]
#[allow(missing_debug_implementations)]
pub struct Qwen3VlMoeVision {
    cfg: Qwen3VlMoeVisionConfig,
    patch_embed_w: Array, // [hidden, in_ch*tps*ps*ps]
    patch_embed_b: Array, // [hidden]
    pos_embed: Array,     // [num_position_embeddings, hidden]
    blocks: Vec<Block>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
    head_dim: usize,
    num_grid_per_side: usize,
}

/// Vision-tower output: the merged image embeddings + per-layer deepstack embeds.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — two fields are the complete vision-forward output contract; adding a field requires updating all Qwen3VlMoeVision::forward callers"
)]
#[allow(missing_debug_implementations)]
pub struct VisionOutput {
    /// `[num_merged, out_hidden]` — scattered at the image-token positions.
    pub image_embeds: Array,
    /// One `[num_merged, out_hidden]` per `deepstack_visual_indexes` entry.
    pub deepstack_embeds: Vec<Array>,
}

impl Qwen3VlMoeVision {
    /// Parsed vision sub-config this tower was built from.
    pub fn config(&self) -> &Qwen3VlMoeVisionConfig {
        &self.cfg
    }

    /// Run the ViT over one preprocessed image.
    ///
    /// `pixel_values`: row-major `[num_patches, in_ch*tps*ps*ps]` f32.
    /// `grid_thw`: `(t, h, w)` patch grid (h, w divisible by spatial_merge_size).
    /// Returns the merged image embeddings + deepstack embeds (all
    /// `[num_merged, out_hidden]`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward(
        &self,
        pixel_values: &[f32],
        grid_thw: (usize, usize, usize),
        device: Device,
    ) -> Result<VisionOutput> {
        let (gt, gh, gw) = grid_thw;
        let merge = self.cfg.spatial_merge_size;
        if !gh.is_multiple_of(merge) || !gw.is_multiple_of(merge) {
            return Err(Error::Model(format!(
                "qwen3_vl_moe vision: grid {gh}x{gw} not divisible by spatial_merge_size {merge}"
            )));
        }
        let num_patches = gt * gh * gw;
        let feat_len = self.patch_embed_w.shape()[1] as usize;
        if pixel_values.len() != num_patches * feat_len {
            return Err(Error::Model(format!(
                "qwen3_vl_moe vision: pixel_values len {} != num_patches {num_patches} * feat_len {feat_len}",
                pixel_values.len()
            )));
        }

        // ---- PatchEmbed (Conv3d kernel==stride -> reshape + matmul + bias) --
        let pv_arr = f32_arr(pixel_values, &[num_patches as i32, feat_len as i32])?;
        let mut h = matmul(
            &pv_arr,
            &self.patch_embed_w.transpose(&[1, 0], device)?,
            device,
        )?;
        h = add(&h, &self.patch_embed_b, device)?;

        // ---- learned pos-embed bilinear interpolation, added to patches -----
        let pos = self.fast_pos_embed_interpolate(gt, gh, gw, device)?;
        h = add(&h, &pos, device)?;

        // ---- 2D vision RoPE tables (host, pure geometry) --------------------
        let (cos_v, sin_v) = self.rot_pos_emb(gt, gh, gw);
        let cos = f32_arr(&cos_v, &[num_patches as i32, self.head_dim as i32])?;
        let sin = f32_arr(&sin_v, &[num_patches as i32, self.head_dim as i32])?;

        // Single image (t==1) -> full attention (no mask).
        let mask: Option<Array> = None;

        debug!(
            num_patches,
            grid_t = gt,
            grid_h = gh,
            grid_w = gw,
            blocks = self.blocks.len(),
            "qwen3_vl_moe vision: forward"
        );

        let mut deepstack_embeds = Vec::with_capacity(self.cfg.deepstack_visual_indexes.len());
        for (layer_num, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &cos, &sin, mask.as_ref(), device)?;
            // Flush each block's command buffer. The ViT runs full attention over
            // every patch (native tiling → tens of thousands of patches for a
            // large image); the in-block attention tiles its O(num_patches^2)
            // score matrix (see `attend_tiled`), and this per-block eval keeps the
            // surrounding qkv/proj/MLP matmuls from accumulating across all 27
            // blocks into one lazy graph that would overrun the Metal GPU
            // watchdog. No effect on small images (the eval is cheap once the
            // block is already computed).
            h.eval()?;
            if let Some(ds_idx) = self
                .cfg
                .deepstack_visual_indexes
                .iter()
                .position(|&x| x == layer_num)
            {
                let de = self.deepstack_mergers[ds_idx].forward(&h, device)?;
                // Materialize each deepstack merger output now so it is not
                // re-derived (pulling the full ViT graph) when injected into the
                // first text-prefill chunk.
                de.eval()?;
                deepstack_embeds.push(de);
            }
        }

        let image_embeds = self.merger.forward(&h, device)?;
        // Materialize the merged image embeds before returning so the downstream
        // scatter into the text sequence is its own command buffer, decoupled
        // from the prefill — and so the whole ViT (a tens-of-thousands-of-patches
        // full-attention graph) never folds into a single prefill buffer that
        // overruns the Metal watchdog.
        image_embeds.eval()?;
        Ok(VisionOutput {
            image_embeds,
            deepstack_embeds,
        })
    }

    /// Build the per-token 2D vision-RoPE `cos`/`sin` tables `[num_patches,
    /// head_dim]` (row-major). Faithful host port of `rot_pos_emb`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn rot_pos_emb(&self, gt: usize, gh: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
        let merge = self.cfg.spatial_merge_size;
        let head_dim = self.head_dim;
        let dim_rot = head_dim / 2;
        let n_freq = dim_rot / 2; // head_dim / 4
        let theta = 10_000.0_f64;
        let inv_freq: Vec<f64> = (0..n_freq)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / dim_rot as f64))
            .collect();

        let mut hpos = vec![0usize; gh * gw];
        let mut wpos = vec![0usize; gh * gw];
        {
            let hb = gh / merge;
            let wb = gw / merge;
            let mut idx = 0usize;
            for a in 0..hb {
                for b in 0..wb {
                    for i in 0..merge {
                        for j in 0..merge {
                            hpos[idx] = a * merge + i;
                            wpos[idx] = b * merge + j;
                            idx += 1;
                        }
                    }
                }
            }
        }

        let tokens_per_t = gh * gw;
        let seq = gt * tokens_per_t;
        let mut cos = vec![0.0_f32; seq * head_dim];
        let mut sin = vec![0.0_f32; seq * head_dim];
        for t in 0..gt {
            for p in 0..tokens_per_t {
                let tok = t * tokens_per_t + p;
                let hf = hpos[p] as f64;
                let wf = wpos[p] as f64;
                let base = tok * head_dim;
                for i in 0..n_freq {
                    let ah = (hf * inv_freq[i]) as f32;
                    let aw = (wf * inv_freq[i]) as f32;
                    cos[base + i] = ah.cos();
                    sin[base + i] = ah.sin();
                    cos[base + n_freq + i] = aw.cos();
                    sin[base + n_freq + i] = aw.sin();
                    cos[base + dim_rot + i] = ah.cos();
                    sin[base + dim_rot + i] = ah.sin();
                    cos[base + dim_rot + n_freq + i] = aw.cos();
                    sin[base + dim_rot + n_freq + i] = aw.sin();
                }
            }
        }
        (cos, sin)
    }

    /// Bilinearly interpolate the learned `pos_embed` (a
    /// `num_grid_per_side x num_grid_per_side` grid) to the image `h x w` grid,
    /// then reorder into the spatial-merge patch order. Returns
    /// `[num_patches, hidden]`. Faithful port of `fast_pos_embed_interpolate`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn fast_pos_embed_interpolate(
        &self,
        gt: usize,
        gh: usize,
        gw: usize,
        device: Device,
    ) -> Result<Array> {
        let ngps = self.num_grid_per_side;
        let merge = self.cfg.spatial_merge_size;
        let hidden = self.cfg.hidden_size;

        let linspace = |n: usize| -> Vec<f64> {
            if n == 1 {
                return vec![0.0];
            }
            let step = (ngps as f64 - 1.0) / (n as f64 - 1.0);
            (0..n).map(|i| i as f64 * step).collect()
        };
        let h_idxs = linspace(gh);
        let w_idxs = linspace(gw);

        let pe = self.pos_embed_host()?; // [ngps*ngps, hidden] f32

        let mut grid = vec![0.0_f32; gh * gw * hidden];
        for (r, &hy) in h_idxs.iter().enumerate() {
            let hf = hy.floor() as usize;
            let hc = (hf + 1).min(ngps - 1);
            let dh = hy - hf as f64;
            for (c, &wx) in w_idxs.iter().enumerate() {
                let wf = wx.floor() as usize;
                let wc = (wf + 1).min(ngps - 1);
                let dw = wx - wf as f64;

                let w00 = ((1.0 - dh) * (1.0 - dw)) as f32;
                let w01 = ((1.0 - dh) * dw) as f32;
                let w10 = (dh * (1.0 - dw)) as f32;
                let w11 = (dh * dw) as f32;

                let i00 = hf * ngps + wf;
                let i01 = hf * ngps + wc;
                let i10 = hc * ngps + wf;
                let i11 = hc * ngps + wc;

                let dst = (r * gw + c) * hidden;
                for k in 0..hidden {
                    let acc =
                        w00.mul_add(pe[i00 * hidden + k], w01.mul_add(pe[i01 * hidden + k], 0.0));
                    let acc = w10.mul_add(pe[i10 * hidden + k], acc);
                    grid[dst + k] = w11.mul_add(pe[i11 * hidden + k], acc);
                }
            }
        }

        // Reorder into spatial-merge patch order, tiled over t frames.
        let hb = gh / merge;
        let wb = gw / merge;
        let seq = gt * gh * gw;
        let mut out = vec![0.0_f32; seq * hidden];
        for t in 0..gt {
            let mut idx = t * gh * gw;
            for a in 0..hb {
                for b in 0..wb {
                    for i in 0..merge {
                        for j in 0..merge {
                            let r = a * merge + i;
                            let c = b * merge + j;
                            let src = (r * gw + c) * hidden;
                            let dst = idx * hidden;
                            out[dst..dst + hidden].copy_from_slice(&grid[src..src + hidden]);
                            idx += 1;
                        }
                    }
                }
            }
        }

        let arr = f32_arr(&out, &[seq as i32, hidden as i32])?;
        arr.astype(arr.dtype(), device)
    }

    /// Materialize `pos_embed` to a host f32 Vec `[ngps*ngps * hidden]`.
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn pos_embed_host(&self) -> Result<Vec<f32>> {
        let f = self.pos_embed.astype(Dtype::F32, Device::Cpu)?;
        Array::eval(&f)?;
        Ok(f.to_bytes()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

#[inline]
fn f32_arr(v: &[f32], shape: &[i32]) -> Result<Array> {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) };
    Array::from_bytes(bytes, shape, Dtype::F32)
}

// ---------------------------------------------------------------------------
// Loader (scans the main shards for `vision_tower.*`, upcasts to f32)
// ---------------------------------------------------------------------------

/// Load the Qwen3-VL-MoE vision tower (`vision_tower.*`) from `model_dir`'s
/// safetensors shards. All vision weights are BF16 on disk; the tower runs in
/// float32, so every weight is upcast at load.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn load_vision_tower(
    model_dir: &Path,
    cfg: &Qwen3VlMoeVisionConfig,
) -> Result<Qwen3VlMoeVision> {
    let shards = open_shards(model_dir)?;

    let load_array = |name: &str| -> Result<Array> {
        for h in &shards {
            let st = h.safetensors()?;
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
            "qwen3_vl_moe vision: tensor '{name}' not found in any shard"
        )))
    };
    let has_tensor = |name: &str| -> bool {
        shards
            .iter()
            .any(|h| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };
    let load_linear = |base: &str| -> Result<Linear> {
        let weight = load_array(&format!("{base}.weight"))?;
        let bias_name = format!("{base}.bias");
        let bias = if has_tensor(&bias_name) {
            Some(load_array(&bias_name)?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    };
    let load_ln = |base: &str| -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: load_array(&format!("{base}.weight"))?,
            bias: load_array(&format!("{base}.bias"))?,
            eps: cfg.layer_norm_eps,
        })
    };

    let head_dim = cfg.hidden_size / cfg.num_heads;
    let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let merged_dim = cfg.hidden_size * merge_unit;
    let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt().round() as usize;

    info!(
        depth = cfg.depth,
        hidden = cfg.hidden_size,
        intermediate = cfg.intermediate_size,
        num_heads = cfg.num_heads,
        head_dim,
        out_hidden = cfg.out_hidden_size,
        num_grid_per_side,
        deepstack = ?cfg.deepstack_visual_indexes,
        "qwen3_vl_moe: loading vision tower (f32, LayerNorm+bias, GELU-tanh MLP)"
    );

    // PatchEmbed Conv3d weight. On-disk (sanitized MLX layout) it is
    // `[hidden, tps, ps, ps, in_ch]` — the upstream `sanitize` transposes the
    // PyTorch `[out, in, kT, kH, kW]` to MLX conv order `[out, kT, kH, kW, in]`.
    // Our preprocessed `pixel_values` carry the feature axis in
    // `(in_ch, tps, ps, ps)` order (the processing transpose), so to do the
    // patch embed as a plain matmul we permute the weight to
    // `[hidden, in_ch, tps, ps, ps]` and flatten — aligning the contraction
    // axis order with the pixel features. (Upstream `PatchEmbed` instead
    // reshapes the pixels + Conv3d; the flat matmul is equivalent for
    // kernel==stride, but only if the two flattened orders match.)
    let pe = load_array("vision_tower.patch_embed.proj.weight")?;
    let pe_shape = pe.shape();
    let hidden = pe_shape[0];
    let pe = if pe_shape.len() == 5 {
        // [hidden, tps, ps, ps, in_ch] -> [hidden, in_ch, tps, ps, ps]
        pe.transpose(&[0, 4, 1, 2, 3], Device::Cpu)?
    } else {
        pe
    };
    let feat_len: i32 = pe.shape()[1..].iter().product();
    let patch_embed_w = pe.reshape(&[hidden, feat_len], Device::Cpu)?;
    let patch_embed_b = load_array("vision_tower.patch_embed.proj.bias")?;
    debug!(
        ?pe_shape,
        flat = ?[hidden, feat_len],
        "qwen3_vl_moe vision: patch_embed Conv3d -> flat matmul weight"
    );

    let pos_embed = load_array("vision_tower.pos_embed.weight")?;

    let mut blocks = Vec::with_capacity(cfg.depth);
    for i in 0..cfg.depth {
        let b = format!("vision_tower.blocks.{i}");
        blocks.push(Block {
            norm1: load_ln(&format!("{b}.norm1"))?,
            norm2: load_ln(&format!("{b}.norm2"))?,
            attn: Attention {
                qkv: load_linear(&format!("{b}.attn.qkv"))?,
                proj: load_linear(&format!("{b}.attn.proj"))?,
                num_heads: cfg.num_heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
            },
            mlp: Mlp {
                linear_fc1: load_linear(&format!("{b}.mlp.linear_fc1"))?,
                linear_fc2: load_linear(&format!("{b}.mlp.linear_fc2"))?,
            },
        });
    }

    let merger = PatchMerger {
        norm: load_ln("vision_tower.merger.norm")?,
        linear_fc1: load_linear("vision_tower.merger.linear_fc1")?,
        linear_fc2: load_linear("vision_tower.merger.linear_fc2")?,
        merged_dim,
        use_postshuffle_norm: false,
    };

    let mut deepstack_mergers = Vec::with_capacity(cfg.deepstack_visual_indexes.len());
    for k in 0..cfg.deepstack_visual_indexes.len() {
        let b = format!("vision_tower.deepstack_merger_list.{k}");
        deepstack_mergers.push(PatchMerger {
            norm: load_ln(&format!("{b}.norm"))?,
            linear_fc1: load_linear(&format!("{b}.linear_fc1"))?,
            linear_fc2: load_linear(&format!("{b}.linear_fc2"))?,
            merged_dim,
            use_postshuffle_norm: true,
        });
    }

    info!(
        total_blocks = cfg.depth,
        deepstack_mergers = deepstack_mergers.len(),
        "qwen3_vl_moe: vision tower loaded"
    );

    Ok(Qwen3VlMoeVision {
        cfg: cfg.clone(),
        patch_embed_w,
        patch_embed_b,
        pos_embed,
        blocks,
        merger,
        deepstack_mergers,
        head_dim,
        num_grid_per_side,
    })
}

/// Glob `model*.safetensors` (ignores the stale index — see the text loader).
fn open_shards(model_dir: &Path) -> Result<Vec<rmlx_loader::ShardHandle>> {
    let mut files: Vec<String> = std::fs::read_dir(model_dir)
        .map_err(|e| Error::Loader(format!("cannot read dir {}: {e}", model_dir.display())))?
        .filter_map(std::result::Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with("model") && n.ends_with(".safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::Loader(format!(
            "qwen3_vl_moe vision: no model*.safetensors found in {}",
            model_dir.display()
        )));
    }
    let mut handles = Vec::with_capacity(files.len());
    for f in &files {
        handles.push(rmlx_loader::ShardHandle::open(model_dir, f)?);
    }
    // ShardSet/index intentionally bypassed (stale index); reference to keep the
    // imports honest without forking the index path.
    let _ = (load_shard_index, ShardSet::open);
    Ok(handles)
}

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;
