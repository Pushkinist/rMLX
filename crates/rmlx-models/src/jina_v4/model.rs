// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! jina-embeddings-v4 text-tower hidden-state forward (bf16, no LoRA).
//!
//! jina-v4's text decoder is **plain Qwen2** (Qwen2.5-VL-3B language backbone):
//! plain RMSNorm (no `+1`), q/k/v additive bias, o_proj no bias, GQA 16/2,
//! head_dim 128, full RoPE θ from config, SwiGLU MLP, pre-norm residuals.
//! This structurally mirrors `crates/rmlx-models/src/qwen2.rs:314-583` (forward)
//! and `qwen2.rs:1098-1296` (single-scan safetensors loader), specialized to
//! the **pure-bf16, unquantized** jina snapshot — `Linear` is a plain bf16
//! matmul only, no dequant / quantized path, no KV cache, no decode loop.
//!
//! It returns the **post-final-norm hidden states** (`[1, seq, hidden]`)
//! *before* any `lm_head` / slice. jina-v4 has no `lm_head` (verified: not in
//! either shard header) — the embedding pipeline consumes this tensor directly
//! (ref `modeling_jina_embeddings_v4.py:171-204`, `get_last_hidden_states`).
//!
//! # Verified tensor naming (enumerated from the real safetensors headers)
//!
//! | Tensor | Key pattern |
//! |-------------------|------------------------------------------------------|
//! | embed_tokens | `model.embed_tokens.weight` (bf16, `[vocab, hidden]`)|
//! | layer norms | `model.layers.{N}.{input_layernorm,post_attention_layernorm}.weight` |
//! | q/k/v projections | `model.layers.{N}.self_attn.{q,k,v}_proj.{weight,bias}` |
//! | o_proj | `model.layers.{N}.self_attn.o_proj.weight` (no bias) |
//! | mlp | `model.layers.{N}.mlp.{gate,up,down}_proj.weight` |
//! | final norm | `model.norm.weight` (shard 2) |
//! | lm_head | absent — embeddings model, no causal head |
//!
//! Shards: `model-00001-of-00002.safetensors` (layers 0-19 + embed),
//! `model-00002-of-00002.safetensors` (layers 19-35 + `model.norm`).
//! Both shards are scanned for every tensor (the index may omit siblings) —
//! same single-scan pattern as `qwen2.rs`.
//!
//! # LoRA seam
//!
//! Every projection is a [`Linear`] carrying an `Option<LoraDelta>` slot that
//! is **always `None`** unless a LoRA adapter is loaded. [`Linear::forward`]
//! already routes the post-matmul activation through [`Linear::apply_lora`],
//! a no-op while the slot is empty. To activate LoRA: (a) parse
//! `adapters/adapter_model.safetensors`
//! (`base_model.model.model.language_model.layers.{N}.<proj>.lora_{A,B}.<task>.weight`),
//! and (b) populate each `Linear::lora` with a [`LoraDelta`] for the selected
//! task — the `y += scaling * (x @ A^T) @ B^T` math already lives in
//! [`LoraDelta::apply`], inert until a delta is constructed. No call-site
//! re-plumbing is required — the dispatch point is fixed here.

#![allow(clippy::items_after_statements, clippy::struct_field_names)]
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::{
    add, concatenate, multiply, negative, rms_norm, rope, scalar_f32, scaled_dot_product_attention,
    Array, Device, Dtype,
};
use tracing::{debug, info};

use super::config::JinaV4TextConfig;

// ---------------------------------------------------------------------------
// LoRA seam — inert until a LoRA adapter is loaded
// ---------------------------------------------------------------------------

/// Per-`Linear` additive LoRA delta for a single task adapter.
///
/// This type is the documented extension point for injecting a task-specific
/// LoRA delta without re-plumbing any call site. [`Linear::lora`] is `None`
/// until a LoRA adapter is loaded.
///
/// Semantics: `y += scaling * (x @ a^T) @ b^T`, where `a` is `[r, in]`,
/// `b` is `[out, r]`, `scaling = lora_alpha / r` (= 1.0 for jina: r = 32,
/// α = 32). Port reference: `custom_lora_module.py:64-112`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed LoRA delta struct — three fields (a/b/scaling) are the complete LoRA rank-factorization contract; adding a field requires updating all LoraDelta construction sites and apply()"
)]
#[allow(missing_debug_implementations)]
pub struct LoraDelta {
    /// LoRA `A` factor, `[rank, in_features]` (bf16).
    pub a: Array,
    /// LoRA `B` factor, `[out_features, rank]` (bf16).
    pub b: Array,
    /// `lora_alpha / rank` (1.0 for every jina-v4 task).
    pub scaling: f32,
}

impl LoraDelta {
    /// `out = base + scaling * (x @ a^T) @ b^T`.
    ///
    /// Not reached while `lora` is `None`. The body is the intended
    /// implementation — load a `LoraDelta` to activate it.
    fn apply(&self, base: &Array, x: &Array, device: Device) -> Result<Array> {
        let xa = rmlx_mlx::matmul(x, &self.a.transpose(&[1, 0], device)?, device)?;
        let xab = rmlx_mlx::matmul(&xa, &self.b.transpose(&[1, 0], device)?, device)?;
        let scaled = multiply(
            &xab,
            &scalar_f32(self.scaling).astype(xab.dtype(), device)?,
            device,
        )?;
        add(base, &scaled, device)
    }
}

/// The seven LoRA-adapted decoder projections, in a fixed canonical order.
///
/// jina-v4's adapter touches exactly these per decoder layer (verified from
/// the real `adapter_model.safetensors` header — see [`super::lora`]). Used as
/// the stable key the adapter loader maps task deltas onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::enum_variant_names)] // mirrors the on-disk `*_proj` key names
pub(super) enum ProjId {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl ProjId {
    /// All seven projections in canonical (load/assign) order.
    pub(super) const ALL: [ProjId; 7] = [
        ProjId::QProj,
        ProjId::KProj,
        ProjId::VProj,
        ProjId::OProj,
        ProjId::GateProj,
        ProjId::UpProj,
        ProjId::DownProj,
    ];

    /// The adapter-key proj segment, e.g. `self_attn.q_proj` / `mlp.gate_proj`.
    pub(super) fn key_segment(self) -> &'static str {
        match self {
            ProjId::QProj => "self_attn.q_proj",
            ProjId::KProj => "self_attn.k_proj",
            ProjId::VProj => "self_attn.v_proj",
            ProjId::OProj => "self_attn.o_proj",
            ProjId::GateProj => "mlp.gate_proj",
            ProjId::UpProj => "mlp.up_proj",
            ProjId::DownProj => "mlp.down_proj",
        }
    }
}

// ---------------------------------------------------------------------------
// Linear (plain bf16 only — jina-v4 is unquantized) + optional additive bias
// ---------------------------------------------------------------------------

/// A plain bf16 linear projection with an optional additive bias and a
/// LoRA-injection slot.
///
/// jina-v4 is pure bf16 (no `quantization` block in config) — there is
/// deliberately **no** `Quantized` variant here (contrast `qwen2.rs`, which
/// must also handle quantized snapshots). Single variant keeps the hot path a
/// straight `matmul`.
#[allow(missing_debug_implementations)]
pub(super) struct Linear {
    /// `[out_features, in_features]` bf16 weight (HF convention; transposed
    /// at matmul time, matching `qwen2.rs`).
    weight: Array,
    /// Additive bias `[out_features]` (q/k/v carry one; o_proj / mlp do not).
    bias: Option<Array>,
    /// LoRA seam — `None` until a LoRA adapter is loaded (see module docs).
    lora: Option<LoraDelta>,
}

impl Linear {
    fn new(weight: Array, bias: Option<Array>) -> Self {
        Self {
            weight,
            bias,
            lora: None,
        }
    }

    /// Install a task-specific LoRA delta into this projection's seam.
    ///
    /// Subtask 4: called by [`super::lora`] when a task is selected. Replaces
    /// any previously-active delta (task switch is a full replace, not a sum).
    pub(super) fn set_lora(&mut self, delta: LoraDelta) {
        self.lora = Some(delta);
    }

    /// Clear the LoRA seam back to the no-op base path (no task active).
    ///
    /// Test-only today (task switch is a clean replace via `set_lora`, so no
    /// production caller needs to go *back* to unadapted yet). `#[cfg(test)]`
    /// keeps it out of the dead-code clippy gate; ungate when a real
    /// no-adapter path exists.
    #[cfg(test)]
    pub(super) fn clear_lora(&mut self) {
        self.lora = None;
    }

    /// `y = x @ W^T (+ bias) (+ LoRA)`.
    ///
    /// LoRA is applied via [`Linear::apply_lora`] right after the base matmul
    /// (+ bias). This is the single fixed dispatch point — only `self.lora`
    /// changes when a LoRA adapter is loaded, never this method's callers.
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let y = rmlx_mlx::matmul(x, &self.weight.transpose(&[1, 0], device)?, device)?;
        let y = match &self.bias {
            Some(b) => add(&y, b, device)?,
            None => y,
        };
        self.apply_lora(&y, x, device)
    }

    /// LoRA dispatch point. No-op while `self.lora` is `None`.
    #[inline]
    fn apply_lora(&self, base: &Array, x: &Array, device: Device) -> Result<Array> {
        match &self.lora {
            None => base.try_clone(),
            Some(delta) => delta.apply(base, x, device),
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-vector projector (Linear 2048 -> 128, bias=true, + task LoRA)
// ---------------------------------------------------------------------------

/// jina-v4's `multi_vector_projector`: a plain bf16 `Linear` mapping the
/// pooled/per-token hidden (2048) to the 128-d multi-vector space, with an
/// additive `.bias` and the same per-task LoRA seam every decoder projection
/// carries. Base weights live in the **main** shards
/// (`multi_vector_projector.{weight,bias}`, shard 2 — enumerated from the real
/// safetensors header, not docs); the per-task LoRA factors come from the
/// adapter file (`base_model.model.multi_vector_projector.lora_{A,B}.<task>`,
/// single `model.` prefix). Port: `modeling_jina_embeddings_v4.py:212-214,262`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed layer struct — private Linear field; public API is forward() and set_lora(); adding a field requires updating load_weights"
)]
#[allow(missing_debug_implementations)]
pub struct MultiVectorProjector {
    proj: Linear,
}

impl MultiVectorProjector {
    /// `y = x @ W^T + bias (+ active-task LoRA)` over the last (hidden) axis.
    /// `x` is `[1, seq, hidden]`; the result is `[1, seq, projector_dim]`.
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        self.proj.forward(x, device)
    }

    /// Install the active task's projector LoRA delta (clean replace — task
    /// switch overwrites, never sums). Mirrors the decoder seam exactly.
    pub(super) fn set_lora(&mut self, delta: LoraDelta) {
        self.proj.set_lora(delta);
    }
}

/// Load the base `multi_vector_projector` (Linear 2048 -> projector_dim, with
/// `.bias`) from `model_dir`'s main safetensors shards. LoRA is attached
/// separately by the adapter bundle (see [`super::lora`]).
pub(super) fn load_multi_vector_projector(model_dir: &Path) -> Result<MultiVectorProjector> {
    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

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
                return Array::from_safetensor_view(&tv);
            }
        }
        Err(Error::Loader(format!(
            "jina-v4: tensor '{name}' not found in any shard"
        )))
    }

    let weight = load_array(&shards, "multi_vector_projector.weight")?;
    let bias = load_array(&shards, "multi_vector_projector.bias")?;
    info!(
        weight_shape = ?weight.shape(),
        "jina-v4: loaded multi_vector_projector base (bf16, bias=true)"
    );
    Ok(MultiVectorProjector {
        proj: Linear::new(weight, Some(bias)),
    })
}

// ---------------------------------------------------------------------------
// Embedding (plain bf16 — tied table, no lm_head)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Embedding {
    /// `[vocab, hidden]` bf16 lookup table (`model.embed_tokens.weight`).
    weight: Array,
}

impl Embedding {
    fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        self.weight.take(ids, 0, device)
    }
}

// ---------------------------------------------------------------------------
// RmsNorm (plain-gamma, no +1 shift — same as Qwen2)
// ---------------------------------------------------------------------------

struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, Some(&self.weight), self.eps, device)
    }
}

// ---------------------------------------------------------------------------
// Attention (GQA 16/2, head_dim 128, full RoPE, q/k/v additive bias)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_theta: f32,
}

impl Attention {
    /// Single full-sequence pass (offset = 0, no KV cache). Plain causal mask
    /// — jina single non-padded text — exactly `qwen2.rs:421-430` prefill with
    /// `pick_attn_mask_mode(0, seq) == "causal"`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let shape = x.shape(); // [batch, seq, hidden]
        let batch = shape[0];
        let seq = shape[1];

        // The q/k/v additive bias rides inside `Linear::forward` (jina q/k/v
        // carry `.bias`; o_proj does not).
        let q = self.q_proj.forward(x, device)?;
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]

        let k = self.k_proj.forward(x, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let k = k.transpose(&[0, 2, 1, 3], device)?;

        let v = self.v_proj.forward(x, device)?;
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // Full RoPE over the entire head_dim; offset 0 (single forward pass).
        let rope_dims = self.head_dim as i32;
        let q = rope(&q, rope_dims, false, self.rope_theta, 1.0, 0, device)?;
        let k = rope(&k, rope_dims, false, self.rope_theta, 1.0, 0, device)?;

        // GQA: expand K/V when n_heads > n_kv_heads.
        let repeat = self.n_heads / self.n_kv_heads;
        let (k, v) = if repeat > 1 {
            (
                repeat_kv(&k, repeat, device)?,
                repeat_kv(&v, repeat, device)?,
            )
        } else {
            (k, v)
        };

        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, "causal", None, device)?;
        let out = out.transpose(&[0, 2, 1, 3], device)?;
        let out = out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;
        self.o_proj.forward(&out, device)
    }

    /// Single full-sequence pass with **3D M-RoPE** applied via precomputed
    /// per-token `cos`/`sin` tables instead of the 1D MLX `rope` kernel.
    ///
    /// `cos`/`sin` are `[1, seq, head_dim]` (already collapsed across the
    /// temporal/height/width mrope sections — see [`super::image`]). The
    /// rotary application is `x*cos + rotate_half(x)*sin` with the NeoX
    /// `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])` convention, exactly
    /// matching `qwen2_5_vl.py::apply_multimodal_rotary_pos_emb`. The rest of
    /// attention (GQA expand, causal SDPA) is identical to [`Attention::forward`]
    /// — only the position encoding differs, so the committed 1D text path is
    /// untouched.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward_mrope(&self, x: &Array, cos: &Array, sin: &Array, device: Device) -> Result<Array> {
        let shape = x.shape(); // [batch(=1), seq, hidden]
        let batch = shape[0];
        let seq = shape[1];

        let q = self.q_proj.forward(x, device)?;
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]

        let k = self.k_proj.forward(x, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let k = k.transpose(&[0, 2, 1, 3], device)?;

        let v = self.v_proj.forward(x, device)?;
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // cos/sin: [1, seq, head_dim] -> [1, 1, seq, head_dim] to broadcast
        // over the head axis of [B, H, S, D] (unsqueeze_dim=1 in the ref).
        let cos_b = cos.reshape(&[1, 1, seq, self.head_dim as i32], device)?;
        let sin_b = sin.reshape(&[1, 1, seq, self.head_dim as i32], device)?;
        let q = apply_rotary_mrope(&q, &cos_b, &sin_b, device)?;
        let k = apply_rotary_mrope(&k, &cos_b, &sin_b, device)?;

        let repeat = self.n_heads / self.n_kv_heads;
        let (k, v) = if repeat > 1 {
            (
                repeat_kv(&k, repeat, device)?,
                repeat_kv(&v, repeat, device)?,
            )
        } else {
            (k, v)
        };

        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, "causal", None, device)?;
        let out = out.transpose(&[0, 2, 1, 3], device)?;
        let out = out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;
        self.o_proj.forward(&out, device)
    }
}

/// `x_embed = (x * cos) + (rotate_half(x) * sin)`, NeoX `rotate_half`.
///
/// `x`: `[B, H, S, D]`. `cos`/`sin`: `[1, 1, S, D]` (broadcast over B, H).
/// `rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])`. Matches
/// `qwen2_5_vl.py::rotate_half` + `apply_multimodal_rotary_pos_emb`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn apply_rotary_mrope(x: &Array, cos: &Array, sin: &Array, device: Device) -> Result<Array> {
    let s = x.shape();
    let (b, h, seq, d) = (s[0], s[1], s[2], s[3]);
    let half = d / 2;

    let x1 = x.slice(&[0, 0, 0, 0], &[b, h, seq, half], &[1, 1, 1, 1], device)?;
    let x2 = x.slice(&[0, 0, 0, half], &[b, h, seq, d], &[1, 1, 1, 1], device)?;
    let rot = concatenate(&[&negative(&x2, device)?, &x1], 3, device)?;

    let a = multiply(x, cos, device)?;
    let bterm = multiply(&rot, sin, device)?;
    add(&a, &bterm, device)
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    let s = x.shape();
    let (b, kv_h, seq, d) = (s[0], s[1], s[2], s[3]);
    let x5 = rmlx_mlx::expand_dims(x, 2, device)?;
    let bc = rmlx_mlx::broadcast_to(&x5, &[b, kv_h, repeat as i32, seq, d], device)?;
    bc.reshape(&[b, kv_h * repeat as i32, seq, d], device)
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = self.gate_proj.forward(x, device)?;
        let gate = rmlx_mlx::silu(&gate, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer (pre-norm residuals)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct DecoderLayer {
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl DecoderLayer {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let h = self.attn.forward(&h, device)?;
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let h = self.post_attn_norm.forward(&h, device)?;
        let h = self.mlp.forward(&h, device)?;
        add(&residual, &h, device)
    }

    /// Same pre-norm residual structure as [`DecoderLayer::forward`], but
    /// attention uses the 3D M-RoPE path (image case).
    fn forward_mrope(&self, x: &Array, cos: &Array, sin: &Array, device: Device) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let h = self.attn.forward_mrope(&h, cos, sin, device)?;
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let h = self.post_attn_norm.forward(&h, device)?;
        let h = self.mlp.forward(&h, device)?;
        add(&residual, &h, device)
    }
}

// ---------------------------------------------------------------------------
// Text tower
// ---------------------------------------------------------------------------

/// jina-v4 text tower: tied `embed_tokens` + 36 Qwen2 decoder layers +
/// final RMSNorm. Pure bf16, no KV cache, no `lm_head`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model struct — private weight fields; public API is forward_hidden(); adding a field requires updating load_weights and the jina-v4 loader"
)]
#[allow(missing_debug_implementations)]
pub struct JinaV4Text {
    cfg: JinaV4TextConfig,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
}

impl JinaV4Text {
    /// Full-sequence forward over `input_ids`, returning the post-final-norm
    /// hidden states `[1, seq, hidden]` (hidden = 2048).
    ///
    /// Single pass over the whole sequence — embed → 36 layers (plain causal
    /// mask) → final RMSNorm. No `lm_head`, no slice, no KV cache, no decode
    /// loop. This is the tensor jina's embedding pipeline pools over
    /// (`modeling_jina_embeddings_v4.py:171-204`).
    pub fn forward_hidden(&self, input_ids: &[i64], device: Device) -> Result<Array> {
        if input_ids.is_empty() {
            return Err(Error::Mlx(
                "jina-v4: forward_hidden called with empty input_ids".into(),
            ));
        }
        let seq = input_ids.len() as i32;

        // jina-v4 vocab is 151936 (< i32::MAX) — token ids fit i32. The
        // embedding gather expects an i32 index array.
        let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), ids_i32.len() * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq], Dtype::I32)?;

        let h = self.embed_tokens.forward(&ids_arr, device)?;
        let mut h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        for (i, layer) in self.layers.iter().enumerate() {
            debug!(layer = i, "jina-v4 forward layer");
            h = layer.forward(&h, device)?;
        }

        self.final_norm.forward(&h, device)
    }

    /// Embed `input_ids` to `[1, seq, hidden]` (the `embed_tokens` lookup
    /// only — no decoder layers). Used by the image path, which must scatter
    /// vision features into the embeddings *before* running the decoder.
    pub(super) fn embed_ids(&self, input_ids: &[i64], device: Device) -> Result<Array> {
        if input_ids.is_empty() {
            return Err(Error::Mlx(
                "jina-v4: embed_ids called with empty input_ids".into(),
            ));
        }
        let seq = input_ids.len() as i32;
        let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), ids_i32.len() * 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[seq], Dtype::I32)?;
        let h = self.embed_tokens.forward(&ids_arr, device)?;
        h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)
    }

    /// Run the decoder over precomputed input embeddings with **3D M-RoPE**.
    ///
    /// `inputs_embeds` is `[1, seq, hidden]` (text embeddings with the ViT
    /// image features already scattered at the `<|image_pad|>` positions —
    /// see [`super::image`]). `cos`/`sin` are the per-token M-RoPE tables
    /// `[1, seq, head_dim]` derived from the 3D position ids. Returns the
    /// post-final-norm hidden states `[1, seq, hidden]`. The active task's
    /// LoRA is live (same `Linear` seams as the text path). This path is only
    /// reached for image inputs; the committed text-only `forward_hidden`
    /// (1D RoPE) is byte-identical and untouched.
    pub(super) fn forward_hidden_from_embeds_mrope(
        &self,
        inputs_embeds: &Array,
        cos: &Array,
        sin: &Array,
        device: Device,
    ) -> Result<Array> {
        let mut h = inputs_embeds.try_clone()?;
        for (i, layer) in self.layers.iter().enumerate() {
            debug!(layer = i, "jina-v4 forward layer (m-rope)");
            h = layer.forward_mrope(&h, cos, sin, device)?;
        }
        self.final_norm.forward(&h, device)
    }

    /// Parsed text sub-config this tower was built from.
    pub fn config(&self) -> &JinaV4TextConfig {
        &self.cfg
    }

    /// Number of decoder layers (used by the adapter loader to size + validate
    /// the per-task delta map).
    pub(super) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Mutable handle to one decoder layer's projection [`Linear`].
    ///
    /// The (layer, [`ProjId`]) → `Linear` mapping is owned here because the
    /// `DecoderLayer` / `Attention` / `Mlp` field layout is private to this
    /// module — the adapter loader stays decoupled from the graph internals.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn linear_mut(&mut self, layer: usize, proj: ProjId) -> &mut Linear {
        let l = &mut self.layers[layer];
        match proj {
            ProjId::QProj => &mut l.attn.q_proj,
            ProjId::KProj => &mut l.attn.k_proj,
            ProjId::VProj => &mut l.attn.v_proj,
            ProjId::OProj => &mut l.attn.o_proj,
            ProjId::GateProj => &mut l.mlp.gate_proj,
            ProjId::UpProj => &mut l.mlp.up_proj,
            ProjId::DownProj => &mut l.mlp.down_proj,
        }
    }

    /// Install a task's full LoRA set, replacing whatever was active before.
    ///
    /// `take_delta` is called once per (layer, proj) cell in canonical order;
    /// it yields the [`LoraDelta`] for that cell. A task switch is a clean
    /// replace (each seam is overwritten), so no residue of the prior task
    /// remains live. Order is deterministic for reproducible forwards.
    pub(super) fn install_task_loras(
        &mut self,
        mut take_delta: impl FnMut(usize, ProjId) -> LoraDelta,
    ) {
        for layer in 0..self.num_layers() {
            for proj in ProjId::ALL {
                let delta = take_delta(layer, proj);
                self.linear_mut(layer, proj).set_lora(delta);
            }
        }
    }

    /// Drop every active LoRA delta (back to the unadapted bf16 base path).
    ///
    /// Test-only today — used by the differentiation test to prove the no-LoRA
    /// baseline differs from every task. `#[cfg(test)]` keeps the dead-code
    /// clippy gate green; ungate when a production no-adapter path lands.
    #[cfg(test)]
    pub(super) fn clear_all_loras(&mut self) {
        for layer in 0..self.num_layers() {
            for proj in ProjId::ALL {
                self.linear_mut(layer, proj).clear_lora();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loader (single-scan over both shards — mirrors qwen2.rs:1098-1296)
// ---------------------------------------------------------------------------

/// Load the jina-v4 text tower from `model_dir`'s safetensors shards.
///
/// Pure bf16; every tensor is scanned across all shards (the index may omit
/// siblings — same robustness pattern as `qwen2.rs`). The `cfg` is the
/// already-parsed `text_config`.
pub(super) fn load_text_tower(model_dir: &Path, cfg: &JinaV4TextConfig) -> Result<JinaV4Text> {
    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

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
                return Array::from_safetensor_view(&tv);
            }
        }
        Err(Error::Loader(format!(
            "jina-v4: tensor '{name}' not found in any shard"
        )))
    }

    fn has_tensor(shards: &ShardSet, name: &str) -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    }

    // Plain bf16 linear, optionally with an additive `.bias` sibling.
    let load_linear = |base: &str| -> Result<Linear> {
        let weight = load_array(&shards, &format!("{base}.weight"))?;
        let bias_name = format!("{base}.bias");
        let bias = if has_tensor(&shards, &bias_name) {
            Some(load_array(&shards, &bias_name)?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias))
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: load_array(&shards, &format!("{name}.weight"))?,
            eps: cfg.rms_norm_eps,
        })
    };

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        vocab_size = cfg.vocab_size,
        "jina-v4: loading text tower (bf16, no LoRA)"
    );

    let embed_tokens = Embedding {
        weight: load_array(&shards, "model.embed_tokens.weight")?,
    };

    let final_norm = load_rms("model.norm")?;

    let scale = (cfg.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let base = format!("model.layers.{i}");
        let a = format!("{base}.self_attn");

        let attn = Attention {
            q_proj: load_linear(&format!("{a}.q_proj"))?,
            k_proj: load_linear(&format!("{a}.k_proj"))?,
            v_proj: load_linear(&format!("{a}.v_proj"))?,
            o_proj: load_linear(&format!("{a}.o_proj"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale,
            rope_theta: cfg.rope_theta as f32,
        };

        let mlp = Mlp {
            gate_proj: load_linear(&format!("{base}.mlp.gate_proj"))?,
            up_proj: load_linear(&format!("{base}.mlp.up_proj"))?,
            down_proj: load_linear(&format!("{base}.mlp.down_proj"))?,
        };

        layers.push(DecoderLayer {
            input_norm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });
        debug!(layer = i, "jina-v4: loaded layer");
    }

    info!(
        total_layers = cfg.num_hidden_layers,
        "jina-v4: text tower loaded"
    );
    Ok(JinaV4Text {
        cfg: cfg.clone(),
        embed_tokens,
        layers,
        final_norm,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
