// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Gemma4 SigLIP-style vision tower (ViT) + `MultimodalEmbedder`.
//!
//! Faithful port of `mlx_vlm/models/gemma4/vision.py` (`VisionModel`,
//! `VisionPatchEmbedder`, `VisionAttention`, `VisionTransformerBlock`,
//! `VisionPooler`, `apply_multidimensional_rope`, `ClippableLinear`) plus the
//! `MultimodalEmbedder` (RMSNormNoScale -> Linear) from `gemma4.py`.
//!
//! Pipeline (single image, batch=1):
//! `Gemma4PixelValues` (CHW f32) -> patchify `[num_patches, 3*p*p]`
//! -> `input_proj` (+ one-hot position embeddings)
//! -> encoder (16 blocks, full bidirectional attention, 2D RoPE)
//! -> `VisionPooler` (3x3 avg-pool to `num_soft_tokens`)
//! -> `MultimodalEmbedder` (`RMSNormNoScale` -> quantized `Linear`)
//! -> `[1, num_soft_tokens, text_hidden]` image-feature embeddings.
//!
//! ## Gemma4 deltas vs a stock ViT (CRITICAL — applied here)
//!
//! 1. **`ClippableLinear`.** Every encoder Linear is `q/k/v/o_proj` /
//!    `mlp.{gate,up,down}_proj` wrapped as `...{name}.linear.weight` plus four
//!    scalar clamp buffers (`input_min/max`, `output_min/max`). On the e4b
//!    snapshot these are *real finite* bounds (not the +-inf init), so the clamp
//!    is load-bearing. `use_clipped_linears` from config gates the buffers.
//!    `patch_embedder.input_proj` and `embed_vision.embedding_projection` are
//!    plain (un-clipped) Linears.
//! 2. **Patchify `2*(x-0.5)`.** Patches are flattened `[pH*pW, 3*p*p]` then
//!    rescaled to `[-1, 1]` before `input_proj`. CHW->patch reshape is
//!    `[C,pH,p,pW,p] -> [pH,pW,p,p,C] -> [pH*pW, p*p*C]`.
//! 3. **One-hot position embeddings.** A learned `[2, position_embedding_size,
//!    hidden]` table indexed by per-axis (x,y) patch positions; the two axes are
//!    summed. Done here as a host-side gather of two rows per patch.
//! 4. **2D multidimensional RoPE.** `head_dim` split into `ndim=2` partitions
//!    (`channels_per_dim = 2*(head_dim/(2*ndim)) = 32`); rotate_half applied
//!    *within each partition only* with a per-axis position. Precomputed on the
//!    host, applied with element-wise ops (jina_v4 idiom). Acts on q/k AFTER
//!    q_norm/k_norm. SDPA `scale=1.0`.
//! 5. **Per-block norms.** input_layernorm -> attn -> post_attention_layernorm
//!    + residual; pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm
//!    + residual. All `RMSNorm` with learned weight (no `+1`). q_norm/k_norm
//!      are `RMSNorm` over head_dim; v_norm is `RMSNormNoScale` (no scale).
//! 6. **Pooler.** Zero padding (none for single image), 3x3 average pool by
//!    patch position into `num_soft_tokens` cells, multiply by `sqrt(hidden)`.
//!
//! The vision tower runs in **float32** (every weight upcast at load), matching
//! the jina_v4 precedent: 16 blocks accumulate bf16 drift and there is no
//! decode loop so the f32 cost is negligible.

#![allow(clippy::items_after_statements, clippy::struct_field_names)]
use std::mem::size_of_val;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{
    add, clip, concatenate, gelu_tanh, matmul, multiply, negative, rms_norm,
    scaled_dot_product_attention, Array, Device, Dtype,
};
use tracing::{debug, info};

use super::config::Gemma4VisionConfig;
use super::preprocessor::Gemma4PixelValues;

// ---------------------------------------------------------------------------
// ClippableLinear — plain f32 Linear (no bias) + optional clamp buffers.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct ClippableLinear {
    /// `[out, in]` f32 weight (HF convention; transposed at matmul time).
    pub(super) weight: Array,
    /// Scalar clamp buffers. `None` when `use_clipped_linears = false`.
    pub(super) clip: Option<ClipBounds>,
}

#[allow(missing_debug_implementations)]
pub(super) struct ClipBounds {
    pub(super) input_min: Array,
    pub(super) input_max: Array,
    pub(super) output_min: Array,
    pub(super) output_max: Array,
}

impl ClippableLinear {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let x = match &self.clip {
            Some(c) => clip(x, &c.input_min, &c.input_max, device)?,
            None => x.try_clone()?,
        };
        let y = matmul(&x, &self.weight.transpose(&[1, 0], device)?, device)?;
        match &self.clip {
            Some(c) => clip(&y, &c.output_min, &c.output_max, device),
            None => Ok(y),
        }
    }
}

struct RmsNorm {
    weight: Option<Array>,
    eps: f32,
}

impl RmsNorm {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, self.weight.as_ref(), self.eps, device)
    }
}

// ---------------------------------------------------------------------------
// VisionMLP: gate/up/down ClippableLinear, gelu_approx(gate) * up -> down.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Mlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
}

impl Mlp {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = gelu_tanh(&self.gate_proj.forward(x, device)?, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// VisionAttention — full bidirectional attention, 2D multidimensional RoPE.
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Attention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: RmsNorm, // RMSNorm over head_dim
    k_norm: RmsNorm, // RMSNorm over head_dim
    v_norm_eps: f32, // RMSNormNoScale (weight=None) over head_dim
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    /// `x`: `[1, seq, hidden]`. `cos`/`sin`: `[seq, head_dim]`. `mask`:
    /// `[1, 1, seq, seq]` additive bidirectional padding mask.
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
        let seq = x.shape()[1];
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let d = self.head_dim as i32;

        let q = self
            .q_proj
            .forward(x, device)?
            .reshape(&[seq, nh, d], device)?;
        let k = self
            .k_proj
            .forward(x, device)?
            .reshape(&[seq, nkv, d], device)?;
        let v = self
            .v_proj
            .forward(x, device)?
            .reshape(&[seq, nkv, d], device)?;

        let q = self.q_norm.forward(&q, device)?;
        let k = self.k_norm.forward(&k, device)?;
        let v = rms_norm(&v, None, self.v_norm_eps, device)?;

        // 2D multidimensional RoPE on q and k (NOT v).
        let q = apply_multidimensional_rope(&q, cos, sin, device)?;
        let k = apply_multidimensional_rope(&k, cos, sin, device)?;

        // [seq, H, D] -> [1, H, seq, D] for SDPA (batch=1).
        let to_bhsd = |a: &Array, heads: i32| -> Result<Array> {
            a.transpose(&[1, 0, 2], device)?
                .reshape(&[1, heads, seq, d], device)
        };
        let q = to_bhsd(&q, nh)?;
        let k = to_bhsd(&k, nkv)?;
        let v = to_bhsd(&v, nkv)?;

        let out = scaled_dot_product_attention(&q, &k, &v, 1.0, "array", Some(mask), device)?;
        let out = out
            .reshape(&[nh, seq, d], device)?
            .transpose(&[1, 0, 2], device)?
            .reshape(&[seq, nh * d], device)?;
        self.o_proj.forward(&out, device)
    }
}

/// 2D multidimensional RoPE. `head_dim` split into `ndim=2` partitions of
/// `channels_per_dim`; rotate_half applied within each partition only.
///
/// `x`: `[seq, heads, head_dim]`. `cos`/`sin`: `[seq, head_dim]` precomputed
/// tables. Matches `apply_multidimensional_rope` in vision.py.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn apply_multidimensional_rope(
    x: &Array,
    cos: &Array,
    sin: &Array,
    device: Device,
) -> Result<Array> {
    let s = x.shape();
    let (seq, heads, head_dim) = (s[0], s[1], s[2]);
    let ndim = 2usize;
    let cpd = (2 * (head_dim as usize / (2 * ndim))) as i32; // 32 for head_dim=64
    let half = cpd / 2;

    let cos_b = cos.reshape(&[seq, 1, head_dim], device)?;
    let sin_b = sin.reshape(&[seq, 1, head_dim], device)?;

    // rotate_half WITHIN each partition: rot = concat(-x2, x1) per cpd slice.
    let mut rot_parts: Vec<Array> = Vec::with_capacity(ndim);
    for d in 0..ndim as i32 {
        let off = d * cpd;
        let x1 = x.slice(&[0, 0, off], &[seq, heads, off + half], &[1, 1, 1], device)?;
        let x2 = x.slice(
            &[0, 0, off + half],
            &[seq, heads, off + cpd],
            &[1, 1, 1],
            device,
        )?;
        rot_parts.push(concatenate(&[&negative(&x2, device)?, &x1], 2, device)?);
    }
    let rot_refs: Vec<&Array> = rot_parts.iter().collect();
    let rot = concatenate(&rot_refs, 2, device)?;

    let a = multiply(x, &cos_b, device)?;
    let b = multiply(&rot, &sin_b, device)?;
    add(&a, &b, device)
}

// ---------------------------------------------------------------------------
// VisionTransformerBlock
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Block {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
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
        let normed = self.input_layernorm.forward(x, device)?;
        let attn = self.attn.forward(&normed, cos, sin, mask, device)?;
        let attn = self.post_attention_layernorm.forward(&attn, device)?;
        let xs = x.shape();
        let attn = attn.reshape(&xs, device)?;
        let h = add(x, &attn, device)?;

        let normed_h = self.pre_feedforward_layernorm.forward(&h, device)?;
        let ffw = self.mlp.forward(&normed_h, device)?;
        let ffw = self.post_feedforward_layernorm.forward(&ffw, device)?;
        add(&h, &ffw, device)
    }
}

// ---------------------------------------------------------------------------
// MultimodalEmbedder: RMSNormNoScale -> Linear (quantized on the e4b snapshot)
// ---------------------------------------------------------------------------

/// Projects pooled vision soft tokens into the language-model hidden space.
/// `embedding_pre_projection_norm` is parameter-free (RMSNormNoScale);
/// `embedding_projection` is the (possibly quantized) `[text_hidden,
/// vision_hidden]` Linear loaded from `embed_vision.embedding_projection.*`.
#[allow(missing_debug_implementations)]
pub struct MultimodalEmbedder {
    projection: crate::layers::Linear,
    norm_eps: f32,
}

impl MultimodalEmbedder {
    /// `inputs_embeds`: `[1, num_soft_tokens, vision_hidden]`. Returns
    /// `[1, num_soft_tokens, text_hidden]`.
    pub fn forward(&self, inputs_embeds: &Array, device: Device) -> Result<Array> {
        let normed = rms_norm(inputs_embeds, None, self.norm_eps, device)?;
        self.projection.forward(&normed, device)
    }
}

// ---------------------------------------------------------------------------
// Vision tower
// ---------------------------------------------------------------------------

/// Gemma4 SigLIP-style vision encoder (`vision_tower.*`).
#[allow(missing_debug_implementations)]
pub struct VisionModel {
    cfg: Gemma4VisionConfig,
    /// `input_proj.weight` `[hidden, 3*p*p]` (plain, no clip).
    input_proj_w: Array,
    /// Position-embedding table `[2, position_embedding_size, hidden]`.
    position_embedding_table: Array,
    blocks: Vec<Block>,
    /// Optional `(std_bias, std_scale)` when `standardize = true`.
    standardize: Option<(Array, Array)>,
    head_dim: usize,
}

impl VisionModel {
    /// Parsed vision sub-config this tower was built from.
    pub fn config(&self) -> &Gemma4VisionConfig {
        &self.cfg
    }

    /// Run the ViT + pooler over one preprocessed image. Returns the pooled
    /// `[1, num_soft_tokens, hidden_size]` — feed into [`MultimodalEmbedder`].
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward(&self, pv: &Gemma4PixelValues, device: Device) -> Result<Array> {
        let p = self.cfg.patch_size;
        if !pv.height.is_multiple_of(p) || !pv.width.is_multiple_of(p) {
            return Err(Error::Model(format!(
                "gemma4 vision: image {}x{} not divisible by patch_size {p}",
                pv.height, pv.width
            )));
        }
        let p_h = pv.height / p;
        let p_w = pv.width / p;
        let num_patches = p_h * p_w;

        // ---- patchify on host: CHW f32 -> [num_patches, 3*p*p], 2*(x-0.5) ----
        let feat_len = 3 * p * p;
        let mut patches = vec![0.0_f32; num_patches * feat_len];
        let n_pixels = pv.height * pv.width;
        for ph in 0..p_h {
            for pw in 0..p_w {
                let dst = (ph * p_w + pw) * feat_len;
                for r in 0..p {
                    for c in 0..p {
                        let y = ph * p + r;
                        let x = pw * p + c;
                        for ch in 0..3 {
                            let src = ch * n_pixels + y * pv.width + x;
                            let off = (r * p + c) * 3 + ch;
                            patches[dst + off] = 2.0 * (pv.pixel_values[src] - 0.5);
                        }
                    }
                }
            }
        }
        let np = num_patches as i32;
        let fl = feat_len as i32;
        let patch_arr = Array::from_bytes(f32_bytes(&patches), &[np, fl], Dtype::F32)?;

        // input_proj: [hidden, feat_len] -> patches @ W^T -> [num_patches, hidden]
        let mut h = matmul(
            &patch_arr,
            &self.input_proj_w.transpose(&[1, 0], device)?,
            device,
        )?;

        // ---- one-hot position embeddings (host gather of two axis rows) ------
        let hidden = self.cfg.hidden_size as i32;
        let pos_size = self.cfg.position_embedding_size as i32;
        let mut x_idx = vec![0i32; num_patches];
        let mut y_idx = vec![0i32; num_patches];
        for ph in 0..p_h {
            for pw in 0..p_w {
                let i = ph * p_w + pw;
                x_idx[i] = pw as i32; // grid_x (column)
                y_idx[i] = ph as i32; // grid_y (row)
            }
        }
        let x_arr = Array::from_bytes(i32_bytes(&x_idx), &[np], Dtype::I32)?;
        let y_arr = Array::from_bytes(i32_bytes(&y_idx), &[np], Dtype::I32)?;
        let table_x = self
            .position_embedding_table
            .slice(&[0, 0, 0], &[1, pos_size, hidden], &[1, 1, 1], device)?
            .reshape(&[pos_size, hidden], device)?;
        let table_y = self
            .position_embedding_table
            .slice(&[1, 0, 0], &[2, pos_size, hidden], &[1, 1, 1], device)?
            .reshape(&[pos_size, hidden], device)?;
        let pe_x = table_x.take(&x_arr, 0, device)?; // [num_patches, hidden]
        let pe_y = table_y.take(&y_arr, 0, device)?;
        let pos_emb = add(&pe_x, &pe_y, device)?;
        h = add(&h, &pos_emb, device)?;

        // [num_patches, hidden] -> [1, num_patches, hidden]
        h = h.reshape(&[1, np, hidden], device)?;

        // ---- RoPE tables (host, pure geometry) -------------------------------
        let (cos, sin) = self.rope_tables(&x_idx, &y_idx)?;

        // ---- bidirectional attention (no padding: single image) -------------
        let mask = zero_mask(num_patches)?;

        debug!(
            num_patches,
            p_h,
            p_w,
            blocks = self.blocks.len(),
            "gemma4 vision: encoder forward"
        );
        for blk in &self.blocks {
            h = blk.forward(&h, &cos, &sin, &mask, device)?;
        }

        // ---- pooler: 3x3 avg-pool by patch position -> num_soft_tokens -------
        let pooled = self.pool(&h, &x_idx, &y_idx, p_h, p_w, device)?;

        match &self.standardize {
            Some((bias, scale)) => {
                let shifted = add(&pooled, &negative(bias, device)?, device)?;
                multiply(&shifted, scale, device)
            }
            None => Ok(pooled),
        }
    }

    /// Build per-patch 2D-RoPE cos/sin tables `[seq, head_dim]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn rope_tables(&self, x_idx: &[i32], y_idx: &[i32]) -> Result<(Array, Array)> {
        let head_dim = self.head_dim;
        let ndim = 2usize;
        let cpd = 2 * (head_dim / (2 * ndim)); // 32
        let half = cpd / 2; // 16
        let base = f64::from(self.cfg.rope_theta);
        let timescale: Vec<f64> = (0..half)
            .map(|i| base.powf((2.0 / cpd as f64) * i as f64))
            .collect();

        let seq = x_idx.len();
        let mut cos = vec![0.0_f32; seq * head_dim];
        let mut sin = vec![0.0_f32; seq * head_dim];
        for tok in 0..seq {
            let pos = [f64::from(x_idx[tok]), f64::from(y_idx[tok])];
            let row = tok * head_dim;
            for (d, &p) in pos.iter().enumerate().take(ndim) {
                let off = row + d * cpd;
                for i in 0..half {
                    let ang = p / timescale[i];
                    let (sn, cs) = (ang.sin() as f32, ang.cos() as f32);
                    cos[off + i] = cs;
                    cos[off + half + i] = cs;
                    sin[off + i] = sn;
                    sin[off + half + i] = sn;
                }
            }
        }
        let shape = [seq as i32, head_dim as i32];
        let cos_a = Array::from_bytes(f32_bytes(&cos), &shape, Dtype::F32)?;
        let sin_a = Array::from_bytes(f32_bytes(&sin), &shape, Dtype::F32)?;
        Ok((cos_a, sin_a))
    }

    /// `VisionPooler` 3x3 average pool by patch position into `num_soft_tokens`
    /// cells, then scale by `sqrt(hidden_size)`. Built as a `[num_soft, seq]`
    /// averaging matrix applied via matmul (equivalent to the one-hot einsum).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn pool(
        &self,
        h: &Array,
        x_idx: &[i32],
        y_idx: &[i32],
        p_h: usize,
        p_w: usize,
        device: Device,
    ) -> Result<Array> {
        let k = self.cfg.pooling_kernel_size;
        let k2 = (k * k) as f32;
        let out_h = p_h / k;
        let out_w = p_w / k;
        let num_soft = out_h * out_w;
        let seq = x_idx.len();

        let mut weights = vec![0.0_f32; num_soft * seq];
        for tok in 0..seq {
            let cx = (x_idx[tok] as usize) / k;
            let cy = (y_idx[tok] as usize) / k;
            let cell = cx + out_w * cy;
            weights[cell * seq + tok] = 1.0 / k2;
        }
        let w_arr = Array::from_bytes(
            f32_bytes(&weights),
            &[num_soft as i32, seq as i32],
            Dtype::F32,
        )?;
        let hidden = self.cfg.hidden_size as i32;
        let h2 = h.reshape(&[seq as i32, hidden], device)?;
        let pooled = matmul(&w_arr, &h2, device)?;
        let root = (self.cfg.hidden_size as f32).sqrt();
        let root_arr = rmlx_mlx::scalar_f32(root);
        let pooled = multiply(&pooled, &root_arr, device)?;
        pooled.reshape(&[1, num_soft as i32, hidden], device)
    }
}

/// Additive zero mask `[1, 1, seq, seq]` (bidirectional, no padding).
fn zero_mask(seq: usize) -> Result<Array> {
    let m = vec![0.0_f32; seq * seq];
    Array::from_bytes(f32_bytes(&m), &[1, 1, seq as i32, seq as i32], Dtype::F32)
}

#[inline]
pub(super) fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes; from_bytes copies immediately.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

#[inline]
pub(super) fn i32_bytes(v: &[i32]) -> &[u8] {
    // SAFETY: i32 is 4 bytes; from_bytes copies immediately.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// Loader — OPTIONAL. Loads vision_tower.* + embed_vision.* from the main
// safetensors shards. Text-only checkpoints skip this entirely.
// ---------------------------------------------------------------------------

/// Load the Gemma4 vision tower (`vision_tower.*`) + `MultimodalEmbedder`
/// (`embed_vision.*`) from a snapshot directory. Errors only if the caller
/// requested vision but the tensors are absent.
pub fn load_vision_tower(
    model_dir: &Path,
    cfg: &Gemma4VisionConfig,
) -> Result<(VisionModel, MultimodalEmbedder)> {
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
            "gemma4 vision: tensor '{name}' not found in any shard"
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
    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Some(load_f32(&shards, &format!("{name}.weight"))?),
            eps: cfg.rms_norm_eps,
        })
    };

    info!(
        layers = cfg.num_hidden_layers,
        hidden = cfg.hidden_size,
        heads = cfg.num_attention_heads,
        head_dim = cfg.head_dim,
        clipped = cfg.use_clipped_linears,
        "gemma4: loading vision tower (f32)"
    );

    let pe = "vision_tower.patch_embedder";
    let input_proj_w = load_f32(&shards, &format!("{pe}.input_proj.weight"))?;
    let position_embedding_table = load_f32(&shards, &format!("{pe}.position_embedding_table"))?;

    let head_dim = cfg.head_dim;
    let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let b = format!("vision_tower.encoder.layers.{i}");
        let sa = format!("{b}.self_attn");
        blocks.push(Block {
            input_layernorm: load_rms(&format!("{b}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{b}.post_attention_layernorm"))?,
            pre_feedforward_layernorm: load_rms(&format!("{b}.pre_feedforward_layernorm"))?,
            post_feedforward_layernorm: load_rms(&format!("{b}.post_feedforward_layernorm"))?,
            attn: Attention {
                q_proj: load_clip(&format!("{sa}.q_proj"))?,
                k_proj: load_clip(&format!("{sa}.k_proj"))?,
                v_proj: load_clip(&format!("{sa}.v_proj"))?,
                o_proj: load_clip(&format!("{sa}.o_proj"))?,
                q_norm: load_rms(&format!("{sa}.q_norm"))?,
                k_norm: load_rms(&format!("{sa}.k_norm"))?,
                v_norm_eps: cfg.rms_norm_eps,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim,
            },
            mlp: Mlp {
                gate_proj: load_clip(&format!("{b}.mlp.gate_proj"))?,
                up_proj: load_clip(&format!("{b}.mlp.up_proj"))?,
                down_proj: load_clip(&format!("{b}.mlp.down_proj"))?,
            },
        });
    }

    let standardize = if cfg.standardize {
        Some((
            load_f32(&shards, "vision_tower.std_bias")?,
            load_f32(&shards, "vision_tower.std_scale")?,
        ))
    } else {
        None
    };

    let vision = VisionModel {
        cfg: cfg.clone(),
        input_proj_w,
        position_embedding_table,
        blocks,
        standardize,
        head_dim,
    };

    // MultimodalEmbedder: embed_vision.embedding_projection (quantized if .scales).
    let embedder = load_multimodal_embedder(model_dir, "embed_vision", cfg.rms_norm_eps)?;

    info!(
        layers = cfg.num_hidden_layers,
        "gemma4: vision tower + multimodal embedder loaded"
    );
    Ok((vision, embedder))
}

/// Load a Gemma4 [`MultimodalEmbedder`] (`<base>.embedding_projection.*`) from a
/// snapshot directory. Shared by the vision (`embed_vision`) and audio
/// (`embed_audio`) towers — both are the identical `RMSNormNoScale -> Linear`
/// projection into the language-model hidden space, differing only in the
/// `embedding_dim` of the (possibly quantized) projection weight.
pub fn load_multimodal_embedder(
    model_dir: &Path,
    base: &str,
    norm_eps: f32,
) -> Result<MultimodalEmbedder> {
    use crate::layers::Linear as CoreLinear;

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
            "gemma4 embedder: tensor '{name}' not found in any shard"
        )))
    }
    let has = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };

    let proj_base = format!("{base}.embedding_projection");
    let projection = if has(&format!("{proj_base}.scales")) {
        let weight = load_raw(&shards, &format!("{proj_base}.weight"))?;
        let scales = load_raw(&shards, &format!("{proj_base}.scales"))?;
        let biases = if has(&format!("{proj_base}.biases")) {
            Some(load_raw(&shards, &format!("{proj_base}.biases"))?)
        } else {
            None
        };
        let (gs, bits, mode) = read_quant_params(model_dir)?;
        CoreLinear::Quantized {
            weight,
            scales,
            biases,
            group_size: gs,
            bits,
            mode,
        }
    } else {
        CoreLinear::Plain {
            weight: load_f32(&shards, &format!("{proj_base}.weight"))?,
        }
    };

    Ok(MultimodalEmbedder {
        projection,
        norm_eps,
    })
}

// ---------------------------------------------------------------------------
// build multimodal `inputs_embeds` (text + scattered vision features)
// ---------------------------------------------------------------------------

/// Gemma4 image soft-token id (`<image_soft_token>`). The chat template /
/// processor expands each image into a contiguous run of this id; the count
/// must equal the vision tower's `num_soft_tokens` for the scatter to align.
pub const IMAGE_TOKEN_ID: u32 = 258880;

/// Build the merged `inputs_embeds` for a Gemma4 image prompt.
///
/// Faithful host port of mlx-vlm `gemma4.py::Model.get_input_embeddings`:
/// 1. `inputs_embeds = embed_tokens(input_ids) * embed_scale` (scaled text).
/// 2. For each image: `embed_vision(vision_tower(pixels))` → `[1, n_soft,
///    hidden]` f32 → astype bf16 → scatter into `inputs_embeds` at that
///    image's contiguous run of `IMAGE_TOKEN_ID` positions.
/// 3. The per-layer-input ids mask the image positions to `0` (mlx-vlm
///    zeroes the soft-token ids so per-layer gating sees text-only ids at
///    the image positions).
///
/// Returns `(inputs_embeds [1, seq, hidden], masked_ids [seq])`. Both feed
/// [`super::model::Gemma4Text::forward_arr_embeds`].
///
/// Errors if the total `IMAGE_TOKEN_ID` count in `input_ids` does not equal
/// the sum of the images' `num_soft_tokens` (a misalignment would scatter
/// vision rows into the wrong positions and produce garbage output).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn build_inputs_embeds(
    model: &super::model::Gemma4Text,
    vision: &VisionModel,
    embedder: &MultimodalEmbedder,
    images: &[Gemma4PixelValues],
    input_ids: &[u32],
    device: Device,
    mm_cache: Option<&crate::multimodal_cache::MultimodalCache>,
) -> Result<(Array, Array)> {
    let hidden = model.cfg.hidden_size as i32;
    let seq = input_ids.len();

    // Locate the image-token positions (one contiguous run per image, in
    // order). The processor/template guarantees contiguity per image.
    let img_positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == IMAGE_TOKEN_ID)
        .map(|(i, _)| i)
        .collect();
    let expected: usize = images.iter().map(|pv| pv.num_soft_tokens).sum();
    if img_positions.len() != expected {
        return Err(Error::Model(format!(
            "gemma4 image: {} image-token ({IMAGE_TOKEN_ID}) positions in prompt != \
             {expected} vision soft tokens ({} image(s)) — scatter would misalign",
            img_positions.len(),
            images.len()
        )));
    }
    info!(
        image_tokens = img_positions.len(),
        images = images.len(),
        seq,
        "gemma4 image: building inputs_embeds (token count == soft tokens)"
    );

    // ---- scaled text embeddings: embed_tokens(ids) * sqrt(hidden) ----------
    // Image positions still embed their soft-token id here; they are
    // overwritten by the scatter below (mlx-vlm embeds then scatters).
    let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
    let ids_arr = Array::from_bytes(i32_bytes(&ids_i32), &[seq as i32], Dtype::I32)?;
    let h_raw = model.embed_tokens.forward(&ids_arr, device)?;
    let embed_scale = rmlx_mlx::scalar_f32((model.cfg.hidden_size as f32).sqrt());
    let mut embeds = multiply(&h_raw, &embed_scale, device)?;
    embeds = embeds.reshape(&[1, seq as i32, hidden], device)?;
    let embeds_dtype = embeds.dtype();

    // ---- per-image vision encode + scatter ---------------------------------
    let mut cursor = 0usize; // index into img_positions
    for (img_idx, pv) in images.iter().enumerate() {
        let n_soft = pv.num_soft_tokens;
        // Vision tower + multimodal embedder → [1, n_soft, hidden] f32.
        // Short-circuit on a cache hit; the cached entry already
        // holds the post-embedder output for this (preprocess) pixel buffer.
        let key_bytes = crate::multimodal_cache::pixel_f32_bytes(&pv.pixel_values);
        let key = crate::multimodal_cache::MmCacheKey::image_key(
            key_bytes,
            u16::try_from(pv.height).unwrap_or(u16::MAX),
            u16::try_from(pv.width).unwrap_or(u16::MAX),
            3,
            crate::multimodal_cache::MmDtype::F32,
        );
        let feats = crate::multimodal_cache::get_or_compute(mm_cache, key, || {
            embedder.forward(&vision.forward(pv, device)?, device)
        })?;
        let fs = feats.shape();
        if fs.first().copied() != Some(1)
            || fs.get(1).copied() != Some(n_soft as i32)
            || fs.get(2).copied() != Some(hidden)
        {
            return Err(Error::Model(format!(
                "gemma4 image: vision feature shape {fs:?} != [1, {n_soft}, {hidden}] \
                 for image {img_idx}"
            )));
        }
        // astype to the embedding dtype (bf16) before scatter.
        let feats = feats.astype(embeds_dtype, device)?;

        // This image's run of positions must be contiguous.
        let run = &img_positions[cursor..cursor + n_soft];
        let first = run[0];
        let contiguous = run.iter().enumerate().all(|(k, &p)| p == first + k);
        if !contiguous {
            return Err(Error::Model(format!(
                "gemma4 image: image-token positions for image {img_idx} are not \
                 contiguous (got {run:?})"
            )));
        }
        embeds = embeds.slice_update(
            &feats,
            &[0, first as i32, 0],
            &[1, (first + n_soft) as i32, hidden],
            &[1, 1, 1],
            device,
        )?;
        cursor += n_soft;
    }

    // ---- masked ids for per-layer-input gating -----------------------------
    // mlx-vlm zeroes image/audio token ids before get_per_layer_inputs so the
    // per-layer gating sees text-only ids at the multimodal positions.
    let mut masked: Vec<i32> = ids_i32;
    for &p in &img_positions {
        masked[p] = 0;
    }
    let masked_arr = Array::from_bytes(i32_bytes(&masked), &[seq as i32], Dtype::I32)?;

    Ok((embeds, masked_arr))
}

/// Load a packed quantized weight without dtype conversion (keep U32/U8/F16).
pub(super) fn load_raw(shards: &ShardSet, name: &str) -> Result<Array> {
    for (_, handle) in shards.iter() {
        let st = handle.safetensors()?;
        if let Ok(t) = st.tensor(name) {
            let tv = rmlx_loader::TensorView {
                name,
                dtype: t.dtype(),
                shape: t.shape().to_vec(),
                bytes: t.data(),
            };
            return Array::from_safetensor_view(&tv);
        }
    }
    Err(Error::Loader(format!(
        "gemma4 vision: tensor '{name}' not found in any shard"
    )))
}

/// Read top-level `quantization` (`group_size`, `bits`, `mode`) for the
/// `embed_vision.embedding_projection` quantized Linear.
pub(super) fn read_quant_params(model_dir: &Path) -> Result<(i32, i32, crate::layers::QuantMode)> {
    let v = crate::load_util::read_raw_config(model_dir)?;
    let q = v.get("quantization");
    let gs = q
        .and_then(|q| q.get("group_size"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32) as i32;
    let bits = q
        .and_then(|q| q.get("bits"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(8) as i32;
    let mode_str = q
        .and_then(|q| q.get("mode"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("mxfp8");
    Ok((gs, bits, crate::layers::QuantMode::from(mode_str)))
}

// ---------------------------------------------------------------------------
// Unified (encoder-free) vision embedder — `Gemma4UnifiedForConditionalGeneration`.
// ---------------------------------------------------------------------------

pub(crate) mod unified;

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
