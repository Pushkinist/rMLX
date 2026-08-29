// LOC-exempt: full Qwen3 forward + loader + prompt-cache + decode driver in one
// arch module; shrinks across the model-agnostic refactor phases (decode loop
// already extracted) and lands under the soft cap as later phases continue.

//! Qwen3 text-only forward pass.
//!
//! Architecture: `Qwen3ForCausalLM` (dense, no MoE).
//!
//! Reference snapshots:
//! - `mlx-community__DR-Venus-4B-RL-mlx-8Bit` (36 layers, hidden=2560, heads=32/8, g64 b8)
//! - `prism-ml__Ternary-Bonsai-8B-mlx-2bit` (36 layers, hidden=4096, heads=32/8, g128 b2)
//!
//! # Key Qwen3 differences from Qwen2
//!
//! 1. **Per-head q/k RMSNorm before RoPE.**
//!    After reshaping to `[B, S, n_heads, head_dim]`, `q_norm` and `k_norm`
//!    are applied (plain-gamma RMSNorm, weight shape `[head_dim]`).
//!    Then the tensor is transposed to `[B, H, S, D]` and RoPE is applied.
//!    Reference: `qwen3.py` `Attention.__call__` lines 54-63.
//!
//! 2. **No additive bias** on q/k/v/o projections (`attention_bias=false`).
//!
//! 3. `head_dim` is an explicit field in config.json (not derived from
//!    `hidden_size / num_attention_heads`).
//!
//! 4. `rope_scaling` may specify a YaRN config (e.g. Ternary-Bonsai uses
//!    `rope_type=yarn`). For the probe_forward path we use plain RoPE with
//!    `rope_theta` only — YaRN frequency table computation is a follow-on.
//!
//! # Tensor naming convention (same prefix as Qwen2)
//!
//! | Tensor | Name pattern |
//! |-----------------|-------------------------------------------------------|
//! | embed_tokens | `model.embed_tokens.{weight,scales,biases}` |
//! | layer norm | `model.layers.N.input_layernorm.weight` |
//! | q/k_norm | `model.layers.N.self_attn.{q,k}_norm.weight` |
//! | q/k/v/o proj | `model.layers.N.self_attn.{q,k,v,o}_proj.{weight,...}` |
//! | mlp | `model.layers.N.mlp.{gate,up,down}_proj.{...}` |
//! | final norm | `model.norm.weight` |
//! | lm_head | `lm_head.weight` (top-level, or absent when tied) |

#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::cloned_instead_of_copied,
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::struct_field_names,
    clippy::too_many_lines
)]
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::compile::{compile_shapeless, Closure};
use rmlx_mlx::{
    add, concatenate, multiply, rms_norm, rope, rope_dynamic, rope_with_freqs, scalar_f32,
    scaled_dot_product_attention, silu, Array, Device, Dtype,
};
use tracing::{debug, info};

use crate::calibration_sink::CalibrationSink;
use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, reject_nan_prefill,
    DecodeCtx,
};
use crate::kv_cache::{
    kv_layer_quants, kv_max_seq_and_ceiling, warn_if_kv_codec_net_negative, KvLayerShape,
};
use crate::layers::{resolve_quant, QuantParams};
use crate::load_util::{bf16_param, bf16_scales, Weights};
use crate::prompt_cache::{
    chained_block_hashes_seeded, ArchPromptCache, Consumed, PromptCacheEntry, ReusePolicy,
    SsdHydrate,
};
use crate::sampler::TokenLogprobs;
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{HydratedBlock, SsdHydrator};

/// OpenAI `top_logprobs` ceiling (`0..=20`, enforced server-side). The
/// prompt-cache entry captures the first-token logprobs at this width so any
/// later exact-hit can replay them truncated to that request's `top_logprobs_k`
/// — independent of the storing request's `top_logprobs_k` (which may be 0).
const PROMPT_CACHE_LOGPROBS_K: usize = 20;

/// Truncate a stored [`TokenLogprobs`] to the first `k` `top` entries.
///
/// `top` is already sorted descending by logprob (see `compute_top_logprobs`),
/// so the first `k` are the request's top-`k` alternatives. `token_id` and
/// `token_logprob` are width-independent and pass through unchanged.
fn truncate_logprobs_top(mut lp: TokenLogprobs, k: usize) -> TokenLogprobs {
    if lp.top.len() > k {
        lp.top.truncate(k);
    }
    lp
}

// ---------------------------------------------------------------------------
// Prompt cache — Qwen3 entry type + global
// ---------------------------------------------------------------------------
//
// Qwen3 is a pure-attention arch (no GatedDeltaNet / linear-attention state),
// so the entry is simpler than Qwen3_5Moe: only Vec<KvCache> needs snapshotting.
//
// Only the Exact hit path is implemented (cached prompt token-for-token equal to
// the incoming prompt). Qwen3 has no SWA layers, so all caches are losslessly
// trimmable — but for simplicity and correctness we match the Qwen3_5Moe policy:
// exact hit only, partial hits fall back to full re-prefill. The dominant use
// case (CBB bench: same prompt run 2× for warm TTFT) is an Exact hit.

pub(crate) struct Qwen3Entry {
    pub(crate) prompt_token_ids: Vec<u32>,
    pub(crate) block_hashes: Vec<u64>,
    pub(crate) kv_caches: Vec<KvCache>,
    /// Argmax token from the first decode step after prefill.
    pub(crate) first_id: u32,
    /// Decoded piece for `first_id`.
    pub(crate) first_piece: String,
    /// Top-k logprobs for `first_id`, captured from the raw prefill
    /// logits at store time so an exact-hit replay can emit the same
    /// per-token logprob the Miss path produced (OpenAI requires exactly one
    /// `logprobs.content` entry per emitted token). Captured at the OpenAI
    /// `top_logprobs` ceiling (20) regardless of the storing request's
    /// `top_logprobs_k`, then truncated to the replaying request's `lp_k` on
    /// hit. `None` only on SSD-hydrated entries (no first decode token).
    pub(crate) first_logprobs: Option<TokenLogprobs>,
    /// Runtime `KvQuant` discriminant in effect when this snapshot was
    /// written (Plan §D8 / Task 11.5). See `Gemma4Entry::kv_quant`.
    pub(crate) kv_quant: Option<KvQuant>,
    /// True when this entry was reconstructed from the SSD tier and therefore
    /// stores only the block-aligned prefix KV — `first_id` / `first_piece` are
    /// placeholders, not a real decode token. The generate loop excludes such
    /// an entry from the Exact fast path so it falls through to a full
    /// re-prefill that recomputes the real first token.
    ///
    /// MUST be set only in `SsdHydrate::hydrate`; never by the RAM-cache push
    /// path. Do NOT use the `first_id == 0` heuristic as a substitute —
    /// `<bos>` token id is 0 for some models.
    pub(crate) is_ssd_hydrated: bool,
}

impl PromptCacheEntry for Qwen3Entry {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }

    fn deep_clone(&self) -> Result<Self> {
        let kv_caches: Result<Vec<_>> = self.kv_caches.iter().map(|c| c.try_deep_clone()).collect();
        Ok(Self {
            prompt_token_ids: self.prompt_token_ids.clone(),
            block_hashes: self.block_hashes.clone(),
            kv_caches: kv_caches?,
            first_id: self.first_id,
            first_piece: self.first_piece.clone(),
            first_logprobs: self.first_logprobs.clone(),
            kv_quant: self.kv_quant,
            is_ssd_hydrated: self.is_ssd_hydrated,
        })
    }

    fn kv_caches(&self) -> &[KvCache] {
        &self.kv_caches
    }

    fn kv_caches_mut(&mut self) -> &mut [KvCache] {
        &mut self.kv_caches
    }

    fn kv_quant(&self) -> Option<KvQuant> {
        self.kv_quant
    }

    fn is_ssd_hydrated(&self) -> bool {
        self.is_ssd_hydrated
    }

    // Pure-attention dense arch: no GDN linear state.
    fn lin_caches(&self) -> &[LinearAttnCache] {
        &[]
    }
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: trait defaults.
}

/// per-arch shell, `ExactOnly` policy. Qwen3 dense (Bonsai) has no
/// recurrent state, so the partial-prefix path would be technically safe; but
/// the dominant workload is the Exact-hit warm-TTFT (bench winner), and
/// keeping a single `Some(_) => Miss` arm is simpler than the SWA-style
/// partial path. The hard runtime gate enforces "no partial reuse" without a
/// comment.
pub(crate) static QWEN3_PROMPT_CACHE: ArchPromptCache<Qwen3Entry> = ArchPromptCache::new(
    "Qwen3ForCausalLM",
    ReusePolicy::ExactOnly,
    SHARES_KV_ACROSS_LAYERS,
);

/// Qwen3 (dense)'s decoder layers each project their own K/V — no cross-layer-KV
/// topology. This is the single producer of that fact for this arch: it is what
/// [`crate::kv_cache::kv_layer_quants`] resolves the boundary-layer codec
/// against, what the prompt-cache seed folds, and what
/// `Architecture::shares_kv_across_layers` reports. It is `false`, which is also
/// `KvCache`'s constructor default, but it is named rather than spelled at each
/// site because the value now selects a codec: a boundary layer of a `Mixed` /
/// `RotK` base is promoted in-family only on a stack that keeps no bf16 mirror,
/// so a flipped literal would change decoded output, not just residency.
pub(crate) const SHARES_KV_ACROSS_LAYERS: bool = false;

/// active SSD-tier `layout_key` for the qwen3 dense cache, or `0` when
/// the tier is OFF. `FNV_OFFSET ^ 0 == FNV_OFFSET` ⇒ legacy un-salted digests
/// when no SSD tier is attached, preserving byte-identical RAM-only behaviour.
fn qwen3_active_layout_key() -> u64 {
    QWEN3_PROMPT_CACHE.active_layout_key()
}

// ---------------------------------------------------------------------------
// SSD-spill sink — the blanket `impl SpillSink<E> for SsdSpiller` in
// `crate::prompt_cache` covers Qwen3 dense (pure-attention: `lin_caches()` is
// `&[]`, so the spill job carries `kv_caches` only).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SSD-hydrate source
// ---------------------------------------------------------------------------

/// Hydrate a `Qwen3Entry` from the SSD tier on a RAM-cache miss.
///
/// Pure-attention arch: the reconstructed block carries `kv_caches` only (no
/// GDN linear state — `lin_caches` from the block is discarded). The matched
/// block-aligned prefix token IDs become the entry's `prompt_token_ids`; block
/// hashes are recomputed and the runtime `kv_quant` recorded. `first_id` /
/// `first_piece` are sentinels (the SSD block stores no first decode token), so
/// the entry is flagged `is_ssd_hydrated = true`; the generate loop excludes it
/// from the Exact fast path and recomputes the real first token via re-prefill.
impl SsdHydrate<Qwen3Entry> for SsdHydrator {
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> Result<Option<Qwen3Entry>> {
        let Some((block, block_hashes)) = self.lookup_seeded(
            prompt_ids, seed, kv_quant, policy,
            // No cross-layer KV sharing on this stack: nothing reads a
            // Mixed/RotK bf16 mirror, so a hydrated cache builds none.
            false,
        )?
        else {
            return Ok(None);
        };
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches: _, // pure-attention arch has no GDN state
        } = block;
        Ok(Some(Qwen3Entry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            first_id: 0,
            first_piece: String::new(),
            // SSD block stores no first decode token, so no logprob to replay.
            first_logprobs: None,
            kv_quant: Some(kv_quant),
            // Block-aligned prefix only; the placeholder first_id must not be
            // replayed — the generate loop re-prefills to recompute it.
            is_ssd_hydrated: true,
        }))
    }
}

fn ensure_qwen3_prompt_cache(capacity: usize) {
    QWEN3_PROMPT_CACHE.ensure(capacity);
}

/// Read the current hit/miss/bytes stats for the Qwen3 prompt cache.
pub fn read_cache_stats() -> Option<crate::prompt_cache::CacheStats> {
    QWEN3_PROMPT_CACHE.read_cache_stats()
}

// ---------------------------------------------------------------------------
// Local Linear + Embedding -- carry optional biases for affine quant
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
enum Linear {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Linear {
    fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Linear::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Linear::Quantized {
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

// ---------------------------------------------------------------------------
// FusedQkvProjection — Q+K+V weights stacked into one quantized tensor
// ---------------------------------------------------------------------------
//
// L33: fuse the three separate quantized_matmul calls (q_proj, k_proj, v_proj)
// into a single `quantized_matmul` dispatch. At load time we concatenate:
//
// weight_fused : [q_out + k_out + v_out, in // pack_factor] (axis 0)
// scales_fused : [q_out + k_out + v_out, in // group_size ] (axis 0)
// biases_fused : [q_out + k_out + v_out, in // group_size ] (axis 0, if present)
//
// Per-row (per-channel) quant scales are preserved by row-concatenation — each
// output row retains its own independent scale vector along the input-group
// axis. No re-binding or scale adjustment is needed.
//
// Decode forward: one qmm -> [B, S, q_out + k_out + v_out], then three
// zero-copy slices to recover q, k, v. Saves 2 Metal kernel launches per layer
// per decode step (3 GEMVs -> 1 GEMV + 3 CPU-side view ops).
//
// Only activated when all three projections are quantized and share the same
// group_size + bits + mode (invariant for all supported Qwen3 snapshots).
// Plain (fp16/bf16) projections fall back to the three-call path unchanged.
//
// Bias note: quantized_matmul applies bias inside the kernel (per-row scale *
// quantized_weight + bias), so stacked bias [total_out, groups] is correct.
#[allow(missing_debug_implementations)]
struct FusedQkvProjection {
    /// Concatenated weight: [q_out + k_out + v_out, in // pack_factor].
    weight: Array,
    /// Concatenated scales: [q_out + k_out + v_out, in // group_size].
    scales: Array,
    /// Concatenated biases (if present): same shape as scales.
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: String,
    /// Number of output rows belonging to Q (= n_heads * head_dim).
    q_out: i32,
    /// Number of output rows belonging to each of K and V (= n_kv_heads * head_dim).
    kv_out: i32,
}

impl FusedQkvProjection {
    /// Try to build a fused projection from three separate `Linear` values.
    ///
    /// Returns `None` when any projection is plain (unquantized) — the caller
    /// falls back to three separate `Linear::forward` calls in that case.
    /// Returns `Err` only if the concatenation itself fails (MLX error).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn try_from_separate(
        q: &Linear,
        k: &Linear,
        v: &Linear,
        device: Device,
    ) -> Result<Option<Self>> {
        let (qw, qs, qb, group_size, bits, mode) = match q {
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => (weight, scales, biases, *group_size, *bits, mode.clone()),
            Linear::Plain { .. } => return Ok(None),
        };
        let (kw, ks, kb) = match k {
            Linear::Quantized {
                weight,
                scales,
                biases,
                ..
            } => (weight, scales, biases),
            Linear::Plain { .. } => return Ok(None),
        };
        let (vw, vs, vb) = match v {
            Linear::Quantized {
                weight,
                scales,
                biases,
                ..
            } => (weight, scales, biases),
            Linear::Plain { .. } => return Ok(None),
        };

        let q_out = qw.shape()[0]; // out rows of Q
        let kv_out = kw.shape()[0]; // out rows of K (== V)

        let weight = concatenate(&[qw, kw, vw], 0, device)?;
        let scales = concatenate(&[qs, ks, vs], 0, device)?;
        let biases = match (qb, kb, vb) {
            (Some(qb), Some(kb), Some(vb)) => Some(concatenate(&[qb, kb, vb], 0, device)?),
            (None, None, None) => None,
            // Mixed bias presence — should never happen in practice (whole-model
            // quant either has biases everywhere or nowhere), but fall back gracefully.
            _ => return Ok(None),
        };

        Ok(Some(FusedQkvProjection {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
            q_out,
            kv_out,
        }))
    }

    /// Single quantized_matmul dispatch, then slice into (q, k, v).
    ///
    /// Output slices are zero-copy views on the contiguous fused output tensor.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward(&self, x: &Array, device: Device) -> Result<(Array, Array, Array)> {
        let qkv = rmlx_mlx::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            self.group_size,
            self.bits,
            &self.mode,
            true,
            device,
        )?;

        // qkv shape: [B, S, q_out + 2*kv_out]
        let ndim = qkv.ndim() as i32;
        let total = self.q_out + 2 * self.kv_out;

        // Build start/stop/strides for each slice along the last axis.
        // All other axes are taken in full: start=0, stop=dim, stride=1.
        let shape = qkv.shape();
        let (b, s) = (shape[0], shape[1]);
        debug_assert_eq!(shape[2], total, "fused qkv output dim mismatch");
        let _ = ndim;

        let q_sl = qkv.slice(&[0, 0, 0], &[b, s, self.q_out], &[1, 1, 1], device)?;
        let k_sl = qkv.slice(
            &[0, 0, self.q_out],
            &[b, s, self.q_out + self.kv_out],
            &[1, 1, 1],
            device,
        )?;
        let v_sl = qkv.slice(
            &[0, 0, self.q_out + self.kv_out],
            &[b, s, total],
            &[1, 1, 1],
            device,
        )?;

        Ok((q_sl, k_sl, v_sl))
    }
}

#[allow(missing_debug_implementations)]
enum Embedding {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Embedding {
    fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => weight.take(ids, 0, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => qwen_embedding_lookup(
                ids,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                device,
            ),
        }
    }

    fn as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
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

/// On-device embedding lookup for quantized weights.
///
/// Mirrors `mlx_lm.nn.QuantizedEmbedding.__call__`:
/// `dequantize(weight[ids], scales[ids], biases[ids], …)`
///
/// Earlier versions ran this through `Device::Cpu` with an `eye(seq) @ w`
/// trick, forcing a GPU↔CPU round-trip on every decode step. That round-trip
/// blocks the `pending: Option<Array>` async pipeline and is the
/// dominant per-step cost on small dense Qwen3 (≈8 ms/step on Bonsai-2bit).
/// The on-device `take + dequantize` path keeps everything on `device`,
/// letting MLX fuse the lookup with subsequent layers.
fn qwen_embedding_lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    let weight_rows = weight.take(ids, 0, device)?;
    let scales_rows = scales.take(ids, 0, device)?;
    let biases_rows = biases.map(|b| b.take(ids, 0, device)).transpose()?;
    let dq = rmlx_mlx::dequantize(
        &weight_rows,
        &scales_rows,
        biases_rows.as_ref(),
        group_size,
        bits,
        mode,
        device,
    )?;
    // The downstream layers (RoPE, attention masks, RmsNorm) expect BF16
    // activations — the chunked prefill mask is hard-coded BF16
    // (`build_chunked_prefill_mask`). `dequantize` returns the scales' dtype,
    // which is FP16 for some snapshots (e.g. Bonsai). Force BF16 to keep
    // SDPA's mask-promotion happy.
    if dq.dtype() == Dtype::Bf16 {
        Ok(dq)
    } else {
        dq.astype(Dtype::Bf16, device)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
// YarnOverride
// ---------------------------------------------------------------------------

/// Runtime YARN RoPE override for Qwen3 models that lack `rope_scaling` in
/// their `config.json` but need context extension.
///
/// Passed through `ModelLoadConfig` → `load_from_path` →
/// `Qwen3Config::from_model_config`. When `Some`, it is tried as a fallback
/// after the JSON-parsed `rope_scaling` path; the JSON path always wins when
/// present.
///
/// Set via `--yarn-factor` / `--yarn-original-max` CLI flags
/// (env: `RMLX_YARN_FACTOR` / `RMLX_YARN_ORIGINAL_MAX`).
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::exhaustive_structs,
    reason = "two fields are the complete YARN-override contract; adding a field requires updating all construction sites and the CLI binding"
)]
pub struct YarnOverride {
    /// YARN scale factor (must be > 1.0 to activate).
    pub factor: f32,
    /// Original max position embeddings before YARN extension.
    /// When 0.0, the model's `max_position_embeddings` is used as the fallback.
    pub original_max: f32,
}

// ---------------------------------------------------------------------------

/// Subset of config.json fields for the Qwen3 forward pass.
///
/// Qwen3 stores all fields at the root of config.json (same as Qwen2).
/// Key addition: explicit `head_dim` field.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Qwen3 model contract; adding a field requires updating from_model_config and all Qwen3 layer constructors"
)]
#[derive(Debug, Clone)]
/// Parsed Qwen3ForCausalLM config (adds per-head q/k RMSNorm vs Qwen2).
pub struct Qwen3Config {
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Explicit in Qwen3 config.json (unlike Qwen2 where it is derived).
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether lm_head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string.
    pub quant_mode: String,
    /// From `max_position_embeddings`. Used to size the pre-allocated KV buffer.
    /// 0 if absent (caller falls back to `KV_MAX_SEQ_DEFAULT`).
    pub max_position_embeddings: u32,
    /// YARN RoPE config (parsed from `rope_scaling`). `Some` triggers
    /// the [`rmlx_mlx::rope_with_freqs`] path with the precomputed inv-freq
    /// table + `mscale`; `None` keeps the plain `rope_theta`-only path.
    pub yarn: Option<crate::rope::YarnConfig>,
}

impl Qwen3Config {
    /// Parse from a [`rmlx_loader::ModelConfig`] loaded from `config.json`.
    ///
    /// `yarn_override` provides a runtime YARN config for models whose
    /// `config.json` lacks `rope_scaling`. When `None`, only the JSON-parsed
    /// path is active. JSON-parsed YARN always takes precedence over the override.
    pub fn from_model_config(
        cfg: &rmlx_loader::ModelConfig,
        yarn_override: Option<&YarnOverride>,
    ) -> Result<Self> {
        let e = &cfg.extras;

        let hidden_size = e
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen3: missing hidden_size".into()))?
            as usize;
        let num_hidden_layers = e
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen3: missing num_hidden_layers".into()))?
            as usize;
        let intermediate_size = e
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen3: missing intermediate_size".into()))?
            as usize;
        let num_attention_heads = e
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen3: missing num_attention_heads".into()))?
            as usize;
        let num_key_value_heads = e
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen3: missing num_key_value_heads".into()))?
            as usize;
        let vocab_size = e
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen3: missing vocab_size".into()))?
            as usize;
        let rms_norm_eps = e
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        let rope_theta = e
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1_000_000.0) as f32;
        let tie_word_embeddings = e
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        // head_dim: explicit in Qwen3, fall back to hidden/heads if absent.
        let head_dim = e
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map_or(hidden_size / num_attention_heads, |v| v as usize);

        let (quant_group_size, quant_bits, quant_mode) = if let Some(q) = &cfg.quantization {
            (
                q.group_size as i32,
                i32::from(q.bits),
                q.mode_or_default().to_owned(),
            )
        } else {
            (64, 8, "affine".to_owned())
        };

        // max_position_embeddings may appear in extras or via TextConfig.
        // Try extras first (Qwen3 stores most fields at top level), then TextConfig.
        let max_position_embeddings = e
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
            .or_else(|| cfg.text_config.as_ref()?.max_position_embeddings)
            .unwrap_or(0);

        // YARN: parse `rope_scaling.rope_type == "yarn"` if present.
        // Bonsai ships `{ rope_type: "yarn", factor: 4.0,
        // original_max_position_embeddings: 16384 }` with
        // max_position_embeddings = 65536. Models without YARN keep
        // `yarn = None` and use the plain `rope_theta`-only path
        // (byte-identical to develop tip).
        //
        // Runtime override path: for Qwen3 models that lack `rope_scaling` in
        // config.json but want context extension, pass a `YarnOverride` via
        // `--yarn-factor` / `--yarn-original-max` CLI flags. JSON-parsed YARN
        // always takes precedence over the runtime override.
        let yarn = e
            .get("rope_scaling")
            .and_then(crate::rope::YarnConfig::from_extras)
            .or_else(|| {
                let ov = yarn_override?;
                let factor = ov.factor;
                let original = if ov.original_max > 0.0 {
                    ov.original_max
                } else {
                    max_position_embeddings as f32
                };
                if factor <= 1.0 || original <= 0.0 {
                    return None;
                }
                Some(crate::rope::YarnConfig::new(factor, original))
            });
        // Single info log covers both the
        // JSON-parsed and env-override paths; the inner or_else closure no
        // longer emits its own line.
        if let Some(y) = yarn {
            tracing::info!(
                factor = y.factor,
                original_max = y.original_max_position_embeddings,
                beta_fast = y.beta_fast,
                beta_slow = y.beta_slow,
                max_position_embeddings,
                "qwen3: YARN RoPE config detected"
            );
        }

        Ok(Qwen3Config {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            quant_group_size,
            quant_bits,
            quant_mode,
            max_position_embeddings,
            yarn,
        })
    }
}

// ---------------------------------------------------------------------------
// RmsNorm (plain-gamma, no +1 shift)
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
// Attention
// ---------------------------------------------------------------------------

/// Qwen3 attention with per-head q/k RMSNorm applied before RoPE.
///
/// Execution order (reference: qwen3.py Attention.__call__):
/// 1. q_proj(x) -> reshape [B, S, n_heads, head_dim]
/// 2. q_norm (RMSNorm over last dim, weight shape [head_dim])
/// 3. transpose -> [B, H, S, D]
/// 4. rope(q)
///
/// Same for k; v has no norm.
///
/// L33: when all three of q/k/v are quantized, `qkv_proj` holds the fused
/// stacked weight; `q_proj`/`k_proj`/`v_proj` are still stored for the
/// fallback Plain path but are not used in the hot decode path.
#[allow(missing_debug_implementations)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Fused Q+K+V projection (Some when all three are quantized).
    qkv_proj: Option<FusedQkvProjection>,
    // Per-head q/k RMSNorm (plain-gamma, weight shape [head_dim]).
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_theta: f32,
    /// When YARN is active for this model, the precomputed
    /// `[head_dim/2]` f32 inv-freq table for [`rope_with_freqs`]. `None` keeps
    /// the plain `rope`-on-`rope_theta` path (byte-identical to develop).
    yarn_freqs: Option<Array>,
    /// Scalar Array wrapping `mscale = 0.1*ln(factor)+1.0`,
    /// precomputed at load time when mscale != 1.0. Avoids 2 × `scalar_f32`
    /// allocs per layer per decode step (72 allocs/step on Bonsai-36-layer).
    /// `None` when YARN is not active or mscale is exactly 1.0 (no scaling).
    ///
    /// Stored at bf16 (cast at load because the activation stream is fixed bf16).
    /// A strong-f32 scalar here would be defense-in-depth against future dtype
    /// changes, not the observed f32-KV leak; the actual cause was fp16 norm
    /// weights and fp16 quant scales/biases promoting the projection output to f32
    /// before this site. Precomputing as bf16 avoids a per-layer per-step cast.
    yarn_mscale_arr: Option<Array>,
}

impl Attention {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn forward_with_sink(
        &self,
        x: &Array,
        offset: i32,
        mut cache: Option<&mut KvCache>,
        mut sink: Option<&mut dyn CalibrationSink>,
        layer_idx: usize,
        device: Device,
    ) -> Result<Array> {
        let shape = x.shape(); // [batch, seq, hidden]
        let batch = shape[0];
        let seq = shape[1];

        // Q + K + V projections.
        //
        // L33: when a fused QKV projection is available (all three weights are
        // quantized), dispatch one quantized_matmul on the stacked weight
        // [q_out+k_out+v_out, in] and slice the output into q/k/v parts.
        // Saves 2 Metal kernel launches per layer per decode step vs. 3
        // separate calls. Falls back to three-call path for plain weights.
        //
        // A wider `qk_norm_rope_fused` variant extending the closure to include
        // transpose + RoPE via `mlx_fast_rope_dynamic` was implemented but on
        // Bonsai-2bit k8v4 measured -0.65% vs the `qk_norm_fused` baseline
        // (104.14 vs 104.82 TPS, 4 runs).
        //
        // Re-tested under the queue-depth model on both Bonsai k8v4 and
        // Qwen3.6-35B k8v8, hypothesising 5-10us queue-stall savings per saved
        // Metal launch × 5 launches/layer × 36-60 layers. Measured: Bonsai
        // +0.22% (best, mean ~0%), Qwen35B +0.87% best / +0.40% mean. Both
        // below the +1 TPS DoD floor — reverted again. Implication: saved-launch
        // value is ~0.3us per launch on these decode steps, not 5us, because
        // the GPU is the bottleneck and host-side queue refill is already
        // shadowed by kernel work. See docs/reports/ for full measurements.
        // `qk_norm_rope_fused` retained as #[allow(dead_code)] scaffolding
        // for future experiments (prefill-only paths, ParaQ kernel batches).
        let (q, k, v) = if let Some(ref fused) = self.qkv_proj {
            // Fused path (L33): 1 qmm + 3 CPU-side slice view ops.
            let (q_flat, k_flat, v_flat) = fused.forward(x, device)?;
            let q = q_flat.reshape(
                &[batch, seq, self.n_heads as i32, self.head_dim as i32],
                device,
            )?;
            let k = k_flat.reshape(
                &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
                device,
            )?;
            let v = v_flat.reshape(
                &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
                device,
            )?;
            (q, k, v)
        } else {
            // Fallback: three separate matmuls (plain / mixed-quant layers).
            let q = self.q_proj.forward(x, device)?.reshape(
                &[batch, seq, self.n_heads as i32, self.head_dim as i32],
                device,
            )?;
            let k = self.k_proj.forward(x, device)?.reshape(
                &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
                device,
            )?;
            let v = self.v_proj.forward(x, device)?.reshape(
                &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
                device,
            )?;
            (q, k, v)
        };

        let (q, k) = qk_norm_fused(
            &q,
            &k,
            &self.q_norm.weight,
            &self.k_norm.weight,
            self.q_norm.eps,
            device,
        )?;
        // YARN branch. When `yarn_freqs.is_some()` we transpose both
        // q/k up-front (needed before mscale multiply), then route through
        // `rope_with_freqs` using the precomputed inv-freq table after
        // multiplying q/k by the YARN `mscale` correction. Mirrors
        // `mlx_lm.YarnRoPE.__call__` and matches the DFlash drafter wiring
        // (`speculative::dflash::apply_rope`) so both Qwen3 paths converge on
        // the same numerics.
        //
        // The non-YARN arm preserves the original operator order —
        // transpose(q) → rope(q) → transpose(k) → rope(k) — so the MLX graph
        // hash and op-fusion behaviour remain byte-identical for models that do
        // not use YARN (Qwen3.5-MoE, plain Qwen3, etc.).
        #[allow(
            clippy::single_match_else,
            reason = "match form reads more clearly than `if let ... else` for a two-armed Some/None branch with non-trivial bodies"
        )]
        let (q, k) = match &self.yarn_freqs {
            Some(freqs) => {
                // YARN path: hoist both transposes before mscale scaling.
                let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]
                let k = k.transpose(&[0, 2, 1, 3], device)?;
                // Reuse the precomputed scalar Array (already bf16 at load)
                // rather than allocating a fresh one per layer per step.
                // The scalar was cast to bf16 at load because the activation
                // stream is fixed bf16; a residual strong-f32 mscale here
                // would be defense-in-depth against future dtype changes, not
                // the observed leak (q/k arrive already promoted to f32 by
                // fp16 model params before this site, when uncast). No runtime
                // astype needed.
                let q_scaled = if let Some(m) = &self.yarn_mscale_arr {
                    multiply(&q, m, device)?
                } else {
                    q.try_clone()?
                };
                let k_scaled = if let Some(m) = &self.yarn_mscale_arr {
                    multiply(&k, m, device)?
                } else {
                    k.try_clone()?
                };
                let q = rope_with_freqs(
                    &q_scaled,
                    self.head_dim as i32,
                    false,
                    1.0,
                    offset,
                    freqs,
                    device,
                )?;
                let k = rope_with_freqs(
                    &k_scaled,
                    self.head_dim as i32,
                    false,
                    1.0,
                    offset,
                    freqs,
                    device,
                )?;
                (q, k)
            }
            None => {
                // Non-YARN path: preserve develop's interleaved transpose→rope
                // ordering so the MLX graph remains byte-identical.
                let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]
                let q = rope(
                    &q,
                    self.head_dim as i32,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                    device,
                )?;
                let k = k.transpose(&[0, 2, 1, 3], device)?;
                let k = rope(
                    &k,
                    self.head_dim as i32,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                    device,
                )?;
                (q, k)
            }
        };

        // V: no per-head norm; already reshaped above.
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // GQA: MLX's fast SDPA kernel handles head broadcasting natively when
        // `n_q_heads % n_kv_heads == 0`. Skip the manual `repeat_kv` expand to
        // avoid two broadcast+reshape ops that materialise the repeated cache
        // (72 ops per decode step on Bonsai-2bit, 36 layers × 2).
        let _ = repeat_kv;

        // Universal dispatch (Task 5): a single `KvCache::update_and_sdpa` call
        // covers every cache variant. The wrapper internally selects:
        // - `KvQuant::Mixed` → `update_and_sdpa_mixed` (quantized SDPA)
        // - K8V4 TurboFlash → `sdpa_dispatch` (split-K kernel when eligible)
        // - everything else → legacy `update` + `scaled_dot_product_attention`
        //
        // Mask discipline: the explicit additive mask is built only for
        // `mask_mode == "array"` (chunked prefill). For `"causal"`, MLX builds
        // the causal mask internally — passing `mask_arr` alongside
        // `mask_mode="causal"` is a hard MLX error. The Mixed in-prefill branch
        // accepts `additive_mask=None` for "causal" prefill and falls back to
        // MLX's internal causal mask, matching mlx-lm-turboquant semantics.
        let mask_mode = crate::layers::pick_attn_mask_mode(offset, seq);
        let q_dtype = q.dtype();
        let chunked_mask_owned: Option<Array> = if mask_mode == "array" {
            Some(crate::layers::build_chunked_prefill_mask(
                offset, seq, device,
            )?)
        } else {
            None
        };
        let cast_mask_owned: Option<Array> = match &chunked_mask_owned {
            Some(m) if m.dtype() != q_dtype => Some(m.astype(q_dtype, device)?),
            _ => None,
        };
        let additive_mask: Option<&Array> =
            cast_mask_owned.as_ref().or(chunked_mask_owned.as_ref());

        // Calibration sink (post-RoPE, pre-SDPA). Production callers
        // pass `sink=None` and this branch is dead-code-eliminated; only the
        // calibration runtime supplies `Some`. We materialise the last query
        // row before SDPA so that the captured (q_last, k_full) pair reflects
        // post-RoPE / post-RMSNorm tensors as seen by attention.
        if let Some(s) = sink.as_mut() {
            // q shape is [B, n_q_heads, seq, head_dim]; slice the last row.
            let q_last = q.slice(
                &[0, 0, seq - 1, 0],
                &[batch, self.n_heads as i32, seq, self.head_dim as i32],
                &[1, 1, 1, 1],
                device,
            )?;
            // k_full reflects the post-RoPE current-chunk K only; for the
            // calibration use case the prompt is prefilled in one go, so this
            // IS the full accumulated K. (cache writes happen inside the SDPA
            // step below; calibrating sinks must therefore not depend on
            // KvCache state.)
            tracing::debug!(
                target: "rmlx_models::calibration_sink",
                layer_idx,
                q_shape = ?q_last.shape(),
                k_shape = ?k.shape(),
                "calibration sink capture"
            );
            s.record(layer_idx, &q_last, &k)?;
        }

        let out = if let Some(c) = cache.as_mut() {
            c.update_and_sdpa(&q, &k, &v, self.scale, mask_mode, additive_mask, device)?
        } else {
            // No cache — run SDPA directly on the pre-RoPE K/V.
            scaled_dot_product_attention(&q, &k, &v, self.scale, mask_mode, additive_mask, device)?
        };
        let out = out.transpose(&[0, 2, 1, 3], device)?;
        let out = out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;
        self.o_proj.forward(&out, device)
    }
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
        let up = self.up_proj.forward(x, device)?;
        // mx.compile-fused `silu(gate) * up` — 1 dispatch instead of 3
        // (sigmoid + multiply + multiply). Mirrors mlx-lm-turboquant's
        // `@partial(mx.compile, shapeless=True) def swiglu(gate, x):`
        // (`mlx-lm-turboquant/mlx_lm/models/activations.py:9-11`).
        let gated = swiglu_fused(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// swiglu_fused — compile_shapeless fusion of `silu(gate) * up`
// ---------------------------------------------------------------------------
//
// Reference: mlx-lm-turboquant/mlx_lm/models/activations.py:9-11
// @partial(mx.compile, shapeless=True)
// def swiglu(gate, x): return nn.silu(gate) * x
//
// `silu(x)` in rMLX is `x * sigmoid(x)` (2 ops); together with the `* up`
// multiply that's 3 kernel launches per layer's MLP, x 36 layers x every
// decode step on Qwen3 dense (Bonsai). mx.compile fuses these 3 ops into a
// single Metal program, dropping launch count and CPU-side IR rebuild time
// per step.
//
// Cache: keyed by (in_dtype_tag, device_tag). Closures are shape-agnostic
// (compile_shapeless) so a single compiled Closure handles every
// (batch, seq, hidden) shape we see in decode (1x1xH) AND prefill (1xTxH).
// Pattern lifted from `gemma4/layers.rs::geglu_fused`.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SwigluKey {
    in_dtype_tag: u8,
    device_tag: u8,
}

fn swiglu_dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::F16 => 1,
        Dtype::F32 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I32 => 5,
    }
}

fn swiglu_device_tag(d: Device) -> u8 {
    match d {
        Device::Gpu => 0,
        Device::Cpu => 1,
    }
}

static SWIGLU_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<SwigluKey, std::sync::Arc<Closure>>>> =
    OnceLock::new();

fn swiglu_compile_cache() -> &'static Mutex<FxHashMap<SwigluKey, std::sync::Arc<Closure>>> {
    SWIGLU_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn swiglu_get_or_compile(key: SwigluKey, device: Device) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = swiglu_compile_cache()
            .lock()
            .expect("swiglu cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    // Build outside the lock (compile is the slow path).
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 2 {
            return Err(Error::Mlx(format!(
                "swiglu_fused closure: expected 2 inputs (gate, up), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let gate = iter.next().expect("gate");
        let up = iter.next().expect("up");
        let g = silu(&gate, device)?;
        let out = multiply(&g, &up, device)?;
        Ok(vec![out])
    });
    let compiled =
        compile_shapeless(raw).map_err(|e| Error::Mlx(format!("swiglu compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = swiglu_compile_cache()
        .lock()
        .expect("swiglu cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute `silu(gate) * up` via an mx.compile-fused closure.
///
/// Identical math to the unfused `silu(&gate)?` then `multiply(&g, &up)?`
/// pair, but executes as a single compiled Metal program, dropping the 3
/// per-layer pointwise kernel launches to 1.
fn swiglu_fused(gate: &Array, up: &Array, device: Device) -> Result<Array> {
    let key = SwigluKey {
        in_dtype_tag: swiglu_dtype_tag(gate.dtype()),
        device_tag: swiglu_device_tag(device),
    };
    let compiled = swiglu_get_or_compile(key, device)?;
    let mut outs = compiled.apply(&[gate, up])?;
    outs.pop()
        .ok_or_else(|| Error::Mlx("swiglu_fused: closure returned no outputs".to_owned()))
}

// ---------------------------------------------------------------------------
// qk_norm_fused — compile_shapeless fusion of (q rms_norm, k rms_norm)
// ---------------------------------------------------------------------------
//
// QK-norm pre-SDPA compile experiment.
//
// Hypothesis: wrap the per-head Q/K RMSNorms in one compile_shapeless closure
// so MLX can fuse the two `rms_norm` invocations into a single Metal program
// per layer per step (instead of two separate fused-rms_norm dispatches).
// RoPE stays outside the compile boundary because its `offset: i32` would
// force per-step retracing under closure capture — adding `mlx_fast_rope_dynamic`
// to the FFI surface is out of scope for this experiment.
//
// Cache: keyed by (in_dtype_tag, device_tag, eps_bits). eps_bits is the f32
// bit-pattern; in practice all Qwen3 layers share the same `rms_norm_eps`,
// so a single compiled closure handles every layer's (B, S, H, D) shape under
// compile_shapeless. Pattern lifted from `swiglu_fused` above.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QkNormKey {
    in_dtype_tag: u8,
    device_tag: u8,
    eps_bits: u32,
}

static QK_NORM_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<QkNormKey, std::sync::Arc<Closure>>>> =
    OnceLock::new();

fn qk_norm_compile_cache() -> &'static Mutex<FxHashMap<QkNormKey, std::sync::Arc<Closure>>> {
    QK_NORM_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_get_or_compile(
    key: QkNormKey,
    eps: f32,
    device: Device,
) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = qk_norm_compile_cache()
            .lock()
            .expect("qk_norm cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 4 {
            return Err(Error::Mlx(format!(
                "qk_norm_fused closure: expected 4 inputs (q, k, q_w, k_w), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let q_w = iter.next().expect("q_w");
        let k_w = iter.next().expect("k_w");
        let qn = rms_norm(&q, Some(&q_w), eps, device)?;
        let kn = rms_norm(&k, Some(&k_w), eps, device)?;
        Ok(vec![qn, kn])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("qk_norm compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = qk_norm_compile_cache()
        .lock()
        .expect("qk_norm cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute (rms_norm(q, q_w, eps), rms_norm(k, k_w, eps)) via one compiled
/// closure — fuses the two RMSNorm dispatches per layer per step into a single
/// compiled Metal program. Math identical to two separate `rms_norm` calls.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_fused(
    q: &Array,
    k: &Array,
    q_w: &Array,
    k_w: &Array,
    eps: f32,
    device: Device,
) -> Result<(Array, Array)> {
    let key = QkNormKey {
        in_dtype_tag: swiglu_dtype_tag(q.dtype()),
        device_tag: swiglu_device_tag(device),
        eps_bits: eps.to_bits(),
    };
    let compiled = qk_norm_get_or_compile(key, eps, device)?;
    let mut outs = compiled.apply(&[q, k, q_w, k_w])?;
    if outs.len() != 2 {
        return Err(Error::Mlx(format!(
            "qk_norm_fused: expected 2 outputs, got {}",
            outs.len()
        )));
    }
    let kn = outs.pop().expect("kn");
    let qn = outs.pop().expect("qn");
    Ok((qn, kn))
}

// ---------------------------------------------------------------------------
// qk_norm_rope_fused — extends qk_norm_fused to include transpose + RoPE
// ---------------------------------------------------------------------------
//
// Experiment: extend the qk_norm_fused closure to include the
// transpose-to-[B,H,S,D] reshape and the RoPE rotation. Saves 4 FFI dispatches
// per layer per decode step (Q-transpose, Q-rope, K-transpose, K-rope) by
// folding them into the single compiled Metal program produced by mx.compile.
//
// Why this is now possible: `mlx_fast_rope_dynamic` accepts a 0-D i32 `offset`
// **as an mlx_array** rather than a captured C-int. The compile_shapeless
// closure can take the offset as a runtime input — the compiled program is
// the same across all decode steps, only the offset value changes.
//
// Cache key: in addition to (dtype, device, eps_bits) we add (head_dim,
// rope_theta_bits) since these are baked into the closure body as constants.
//
// Closure inputs (in order): [q, k, q_w, k_w, offset]
// q, k: [B, S, H, D], pre-norm
// q_w, k_w: [head_dim] gamma weights
// offset: 0-D i32 scalar (current decode position)
//
// Closure outputs: [q_out, k_out], both [B, H, S, D] post-norm-transpose-rope.
//
// Pattern lifted from `qk_norm_fused` above.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QkNormRopeKey {
    in_dtype_tag: u8,
    device_tag: u8,
    eps_bits: u32,
    head_dim: i32,
    rope_theta_bits: u32,
}

static QK_NORM_ROPE_COMPILE_CACHE: OnceLock<
    Mutex<FxHashMap<QkNormRopeKey, std::sync::Arc<Closure>>>,
> = OnceLock::new();

fn qk_norm_rope_compile_cache() -> &'static Mutex<FxHashMap<QkNormRopeKey, std::sync::Arc<Closure>>>
{
    QK_NORM_ROPE_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_rope_get_or_compile(
    key: QkNormRopeKey,
    eps: f32,
    head_dim: i32,
    rope_theta: f32,
    device: Device,
) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = qk_norm_rope_compile_cache()
            .lock()
            .expect("qk_norm_rope cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 5 {
            return Err(Error::Mlx(format!(
                "qk_norm_rope_fused closure: expected 5 inputs (q, k, q_w, k_w, offset), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let q_w = iter.next().expect("q_w");
        let k_w = iter.next().expect("k_w");
        let offset = iter.next().expect("offset");
        // Q: norm -> transpose [B,S,H,D]->[B,H,S,D] -> rope_dynamic.
        let qn = rms_norm(&q, Some(&q_w), eps, device)?;
        let qt = qn.transpose(&[0, 2, 1, 3], device)?;
        let qr = rope_dynamic(&qt, head_dim, false, rope_theta, 1.0, &offset, device)?;
        // K: identical pipeline.
        let kn = rms_norm(&k, Some(&k_w), eps, device)?;
        let kt = kn.transpose(&[0, 2, 1, 3], device)?;
        let kr = rope_dynamic(&kt, head_dim, false, rope_theta, 1.0, &offset, device)?;
        Ok(vec![qr, kr])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("qk_norm_rope compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = qk_norm_rope_compile_cache()
        .lock()
        .expect("qk_norm_rope cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute (rope(transpose(rms_norm(q,q_w,eps))), rope(transpose(rms_norm(k,k_w,eps)))) via
/// one compiled closure — fuses 6 dispatches per layer (2 norms + 2 transposes
/// + 2 ropes) into one compiled Metal program.
///
/// Inputs: q,k of shape `[B, S, H, D]` (post-projection, pre-norm).
/// Outputs: q,k of shape `[B, H, S, D]` (post-norm, post-rope).
///
/// `offset` is the current decode position; internally constructed as a 0-D
/// i32 scalar Array passed to `mlx_fast_rope_dynamic`.
///
/// **Negative result**: on Bonsai-2bit k8v4 (Qwen3 dense, fastest
/// Qwen-arch cell at 104 TPS) this fused closure measured -0.65% vs
/// `qk_norm_fused` (104.14 vs 104.82 TPS, 4 runs). Closure-pack overhead
/// (extra offset input + 5-input pack) cancels the 4 saved dispatches at
/// step times <10 ms. Retained as scaffolding for future experiments
/// (e.g. prefill-only paths, ParaQ kernel batches) where the closure
/// overhead amortises better.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_rope_fused(
    q: &Array,
    k: &Array,
    q_w: &Array,
    k_w: &Array,
    eps: f32,
    head_dim: i32,
    rope_theta: f32,
    offset: i32,
    device: Device,
) -> Result<(Array, Array)> {
    let key = QkNormRopeKey {
        in_dtype_tag: swiglu_dtype_tag(q.dtype()),
        device_tag: swiglu_device_tag(device),
        eps_bits: eps.to_bits(),
        head_dim,
        rope_theta_bits: rope_theta.to_bits(),
    };
    let compiled = qk_norm_rope_get_or_compile(key, eps, head_dim, rope_theta, device)?;
    // Build the offset Array once per call (0-D i32). Lifetime is fine — apply()
    // dispatches the compiled program; outputs hold their own MLX-internal
    // refs to inputs through the lazy graph.
    let off_bytes = offset.to_le_bytes();
    let off_arr = Array::from_bytes(&off_bytes, &[], Dtype::I32)?;
    let mut outs = compiled.apply(&[q, k, q_w, k_w, &off_arr])?;
    if outs.len() != 2 {
        return Err(Error::Mlx(format!(
            "qk_norm_rope_fused: expected 2 outputs, got {}",
            outs.len()
        )));
    }
    let kr = outs.pop().expect("kr");
    let qr = outs.pop().expect("qr");
    Ok((qr, kr))
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
struct DecoderLayer {
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl DecoderLayer {
    fn forward_with_sink(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        sink: Option<&mut dyn CalibrationSink>,
        layer_idx: usize,
        device: Device,
    ) -> Result<Array> {
        // Attention sub-layer: pre-norm → attn → residual add. No `try_clone`
        // needed — `x` is `&Array`, and `h_attn_res` outlives every use of it.
        let h = self.input_norm.forward(x, device)?;
        let h = self
            .attn
            .forward_with_sink(&h, offset, cache, sink, layer_idx, device)?;
        let h_attn_res = add(x, &h, device)?;

        // MLP sub-layer: pre-norm → mlp → residual add. Same structure;
        // `h_attn_res` is borrowed by both `post_attn_norm` and the final `add`.
        let h = self.post_attn_norm.forward(&h_attn_res, device)?;
        let h = self.mlp.forward(&h, device)?;
        add(&h_attn_res, &h, device)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model struct — private weight fields; public API is forward_seq() and forward_multimodal(); adding a field requires updating load_weights and the Qwen3 loader"
)]
/// Qwen3ForCausalLM model weights + forward pass.
#[allow(missing_debug_implementations)]
pub struct Qwen3Text {
    /// Parsed model configuration.
    pub cfg: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    /// `None` when `tie_word_embeddings = true`.
    lm_head: Option<Linear>,
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

impl Qwen3Text {
    /// Full-sequence forward pass (no KV cache).
    ///
    /// Returns logits for the last position, shape `[1, 1, vocab_size]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        self.forward_seq_with_cache(ids, None, device)
    }

    /// Forward pass with optional KV cache.
    /// When `caches` is `Some`, each entry corresponds to one decoder layer.
    /// When `None`, behaves exactly as `forward_seq`.
    pub fn forward_seq_with_cache(
        &self,
        ids: &[u32],
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_arr = Array::from_i32_slice(&ids_i32, &[seq as i32])?;
        self.forward_arr(&ids_arr, seq as i32, caches, device)
    }

    /// Forward pass with a calibration sink installed. The sink
    /// receives one `(q_last, k_full)` capture per layer at the post-RoPE,
    /// pre-SDPA boundary. Used only by `rmlx kv-calibrate --recipe softmax_mass`;
    /// production `forward_seq_with_cache` is unchanged (no per-call overhead).
    pub fn forward_seq_with_cache_calibrated(
        &self,
        ids: &[u32],
        caches: Option<&mut [KvCache]>,
        sink: &mut dyn CalibrationSink,
        device: Device,
    ) -> Result<Array> {
        let seq = ids.len();
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_arr = Array::from_i32_slice(&ids_i32, &[seq as i32])?;
        self.forward_arr_with_sink(&ids_arr, seq as i32, caches, Some(sink), device)
    }

    /// Forward pass with token IDs already in an MLX `Array`.
    ///
    /// Used by the async-pipelined decode loop so the next forward
    /// can chain on top of the prior step's `argmax` Array without forcing a
    /// CPU sync via `to_bytes()`. Mirrors
    /// `qwen3_5_moe::Qwen3_5MoeText::forward_arr`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_arr(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        device: Device,
    ) -> Result<Array> {
        self.forward_arr_with_sink(ids_arr, seq, caches, None, device)
    }

    /// `forward_arr` with optional per-layer calibration sink.
    /// `sink=None` is the production path (no behavioural change vs
    /// `forward_arr`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn forward_arr_with_sink(
        &self,
        ids_arr: &Array,
        seq: i32,
        caches: Option<&mut [KvCache]>,
        mut sink: Option<&mut dyn CalibrationSink>,
        device: Device,
    ) -> Result<Array> {
        let base_offset = caches
            .as_ref()
            .and_then(|cs| cs.first())
            .map_or(0, |c| c.offset());

        let h = self.embed_tokens.forward(ids_arr, device)?;
        let h = h.reshape(&[1, seq, self.cfg.hidden_size as i32], device)?;

        let mut h = h;
        match caches {
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "qwen3 forward layer");
                    // Reborrow `sink` each iteration: `as_deref_mut` borrows
                    // the Option through the outer mut binding, so a fresh
                    // reborrow per iteration is required to avoid the
                    // "borrowed in previous iteration" diagnostic.
                    let sink_iter: Option<&mut dyn CalibrationSink> = match &mut sink {
                        Some(s) => Some(&mut **s),
                        None => None,
                    };
                    h = layer.forward_with_sink(&h, base_offset, None, sink_iter, i, device)?;
                }
            }
            Some(cs) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    debug!(layer = i, "qwen3 forward layer (cached)");
                    let sink_iter: Option<&mut dyn CalibrationSink> = match &mut sink {
                        Some(s) => Some(&mut **s),
                        None => None,
                    };
                    h = layer.forward_with_sink(
                        &h,
                        base_offset,
                        Some(&mut cs[i]),
                        sink_iter,
                        i,
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

    /// full-sequence forward returning logits at **every** position,
    /// not just the last. Used by the offline PPL scorer to read the
    /// per-position log-likelihood of the actual prompt token.
    ///
    /// Returns an `Array` of shape `[1, seq, vocab_size]`. No KV cache — the
    /// scorer slides a fresh window each call; reusing the decode-loop cache
    /// would constrain absolute-position embeddings.
    ///
    /// Surgical to `Qwen3Text` (Bonsai is the smoke target). Adding the
    /// equivalent path for Gemma4 / Qwen3.5MoE / Qwen3VL is follow-up work; the
    /// CLI subcommand rejects other archs explicitly.
    pub fn forward_seq_logits_all(&self, ids: &[u32], device: Device) -> Result<Array> {
        let seq = ids.len();
        if seq == 0 {
            return Err(Error::Other(
                "forward_seq_logits_all: empty prompt".to_owned(),
            ));
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids_arr = Array::from_i32_slice(&ids_i32, &[seq as i32])?;
        let h = self.embed_tokens.forward(&ids_arr, device)?;
        let mut h = h.reshape(&[1, seq as i32, self.cfg.hidden_size as i32], device)?;
        for (i, layer) in self.layers.iter().enumerate() {
            debug!(layer = i, "qwen3 forward_seq_logits_all layer");
            h = layer.forward_with_sink(&h, 0, None, None, i, device)?;
        }
        let h = self.final_norm.forward(&h, device)?;
        match &self.lm_head {
            Some(lm) => lm.forward(&h, device),
            None => self.embed_tokens.as_linear(&h, device),
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke probe — generate_greedy
// ---------------------------------------------------------------------------

/// Count NaN values in a byte buffer of floats (F32 or Bf16).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn count_nan_in_bytes(bytes: &[u8], dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .filter(|c| f32::from_le_bytes((*c).try_into().unwrap()).is_nan())
            .count(),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .filter(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16).is_nan()
            })
            .count(),
        _ => 0,
    }
}

/// Compute max(|logit|) from a byte buffer. Returns 0.0 on empty or unknown dtype.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn max_abs_from_bytes(bytes: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes((*c).try_into().unwrap()).abs())
            .fold(0.0_f32, f32::max),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16).abs()
            })
            .fold(0.0_f32, f32::max),
        _ => 0.0,
    }
}

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// Returns `Vec<ProbeStep>` — same shape as `gemma4::generate_greedy`.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Recommended: 4. Only the Exact hit path is active (identical-prompt repeat
/// skips re-prefill entirely, same contract as Qwen3_5Moe).
/// Pass 0 to disable the cache: nothing is stored, so every request prefills.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub fn generate_greedy<'a>(
    model: &Qwen3Text,
    tokenizer: &'a tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &'a [u32],
    step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    // A6.2: optional sampler constraint. See gemma4::generate_greedy.
    // The shared `DecodeCtx` bundles every per-request borrow under one
    // lifetime, so these references share `'a` (a `&mut dyn` trait-object
    // reborrow is invariant and cannot be re-unified once split).
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. See gemma3::generate_greedy.
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    tracing::info!(
        arch = "Qwen3ForCausalLM",
        ?kv_quant,
        ?max_ctx_override,
        prompt_cache_slots,
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

    ensure_qwen3_prompt_cache(prompt_cache_slots);

    // ------------------------------------------------------------------
    // Prompt cache lookup via the shared consume() engine. Qwen3 dense is
    // pure-attention with no GDN recurrent state and uses ReusePolicy::ExactOnly
    // (it overrides none of the reuse hooks), so the only reachable outcomes are
    // Exact (identical-prompt repeat skips re-prefill) and Miss (full
    // re-prefill). The engine owns the find → SSD-hydrate retry → quant-mismatch
    // guard → SSD-hydrated exclusion → Exact decision and traces every degrade
    // branch. The ExactOnly policy tripwire lives here at the call site — not in
    // the generic engine — because the engine is policy-agnostic and shared across
    // architectures that may use different policies.
    // ------------------------------------------------------------------
    assert_eq!(
        QWEN3_PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "Qwen3 prompt cache must be ExactOnly — pure-attention with no GDN recurrent state \
         cannot safely reuse a partial prefix whose residual state is not stored",
    );
    let consumed = QWEN3_PROMPT_CACHE.consume(
        prompt_ids,
        kv_quant,
        model.cfg.num_hidden_layers,
        false,
        model.model_sig,
    );

    // Path A: exact cache hit — skip re-prefill, jump straight to decode.
    if let Consumed::Exact(cloned) = consumed {
        let Qwen3Entry {
            mut kv_caches,
            first_id: last_id,
            first_piece: piece,
            first_logprobs,
            ..
        } = cloned;
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = last_id,
            "qwen3 generate_greedy: prompt cache EXACT HIT"
        );
        // The cached first token was produced by a prior request's
        // prefill, but its raw-logit top-k logprobs were captured and stored
        // alongside `first_id` at store time. Replay the same true logprob the
        // Miss path emits, truncated to this request's `top_logprobs_k`. The
        // gate keeps the disabled path (`lp_k == 0`) emitting `None` so the
        // zero-overhead decode stays byte-identical. This guarantees exactly
        // one `logprobs.content` entry per emitted token on both cache paths.
        let lp_k = sampler_cfg.top_logprobs_k as usize;
        let first_lp = if lp_k > 0 {
            first_logprobs.map(|lp| truncate_logprobs_top(lp, lp_k))
        } else {
            None
        };
        steps.push(ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: first_lp,
        });
        step_fn(steps.last().unwrap());
        token_history.push(last_id);

        if eos_ids.contains(&last_id) {
            return Ok(steps);
        }

        // Shared pipelined decode loop. The exact-hit funnel is just
        // "caches + last_id → decode"; the loop is the call site.
        let (stats, post) = {
            let mut ctx = DecodeCtx {
                tokenizer,
                vocab,
                n_tokens,
                device,
                eos_ids,
                step_fn,
                constraint: constraint.take(),
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
                arch: "Qwen3ForCausalLM",
                resolve_pieces: true,
            };
            pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
                model.forward_arr(y, 1, Some(&mut kv_caches), device)
            })?
        };

        let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
        let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
        let step_ms = (stats.step_total_ns as f64) / 1.0e6;
        let n = f64::from(stats.decode_steps.max(1));
        tracing::info!(
            target: "decode_profile",
            arch = "Qwen3ForCausalLM",
            cache_path = "exact_hit",
            n_steps = stats.decode_steps,
            forward_total_ms = forward_ms,
            eval_total_ms = eval_ms,
            step_total_ms = step_ms,
            forward_per_step_ms = forward_ms / n,
            eval_per_step_ms = eval_ms / n,
            "decode_profile"
        );
        // Store KV-cache bytes on exact-hit path. Once per generate call, after
        // decode — same lifecycle point as the Miss path store below.
        {
            let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum();
            model.kv_bytes.store(kv_bytes, post);
        }
        return Ok(steps);
    }

    // Path B (Miss): full re-prefill from scratch.
    let prefill_t0 = Instant::now();

    // Derive initial ring size + virtual ceiling (issue #25): `--max-ctx` is a
    // ceiling the ring grows lazily up to, not an eager allocation.
    // `initial_max_seq` is the small lazy start; `max_seq_ceiling` caps growth
    // and rejects over-long prompts.
    let (initial_max_seq, max_seq_ceiling) =
        kv_max_seq_and_ceiling(max_ctx_override, model.cfg.max_position_embeddings as i32);

    // Allocate one KvCache per decoder layer using the selected quant mode.
    // Force K8V8 for boundary layers (first head_n + last tail_n).
    let n_layers = model.cfg.num_hidden_layers;
    let mut caches: Vec<KvCache> = kv_layer_quants(n_layers, kv_quant, SHARES_KV_ACROSS_LAYERS)
        .into_iter()
        .enumerate()
        .map(|(i, q)| {
            KvCache::with_quant_max_seq(q, initial_max_seq)
                .with_max_seq_ceiling(max_seq_ceiling)
                .with_layer_idx(i)
        })
        .collect();

    // Advise once if the resolved codec is estimated to increase resident KV vs
    // bf16 on this layer mix. Keyed on geometry + codec only, exactly as the
    // other arches call it — a codec that costs more memory than bf16 does so
    // because of its store layout, which no architecture can change, so the
    // operator has to hear it on every arch and not only where it happened to
    // be wired first. Every qwen3 layer is full-attention (no sliding window).
    {
        let layer_shapes: Vec<KvLayerShape> = (0..n_layers)
            .map(|_| KvLayerShape {
                head_dim: model.cfg.head_dim as u64,
                kv_heads: model.cfg.num_key_value_heads as u64,
                window: None,
            })
            .collect();
        let eff_seq = (max_seq_ceiling.max(0) as u64).max(prompt_ids.len() as u64);
        warn_if_kv_codec_net_negative(kv_quant, &layer_shapes, eff_seq, false);
    }

    // Prefill: encode the prompt in fixed-size chunks via the shared Fresh
    // chunked-prefill helper. It brackets the loop with enter_prefill() /
    // exit_prefill() so K/V are stored as raw BF16 during prefill instead of
    // being quantize-dequantized on every chunk, evals only the cache state
    // (not the logits) on non-final chunks, and returns None on rejection.
    //
    // Chunk size is per-arch; default 256 for qwen3, override via
    // `RMLX_PREFILL_CHUNK` (global) or `RMLX_PREFILL_CHUNK_QWEN3` (per-arch).
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3");
    let prefill_logits = chunked_prefill(
        &mut caches,
        prompt_ids,
        prefill_chunk,
        device,
        "Qwen3ForCausalLM",
        |chunk, caches| model.forward_seq_with_cache(chunk, Some(caches), device),
    )?;

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
    reject_nan_prefill(
        "Qwen3ForCausalLM",
        logits_flat.dtype(),
        nan_count,
        max_abs_logit,
        prompt_ids.len(),
    )?;

    // top-k logprob capture (0 = disabled, hot-loop zero-overhead).
    let lp_k = sampler_cfg.top_logprobs_k as usize;

    // Build the shared decode context ONCE for the prefill-tail selection AND
    // the Miss decode loop, so the per-request state is borrowed a single time
    // (a `&mut dyn` trait-object reborrow is invariant in its lifetime — two
    // separate `DecodeCtx` over the same params would not compile). Intervening
    // prefill-tail ops route through `ctx.constraint` / `ctx.token_history` /
    // `ctx.step_fn`.
    let mut ctx = DecodeCtx {
        tokenizer,
        vocab,
        n_tokens,
        device,
        eos_ids,
        step_fn,
        constraint: constraint.take(),
        sampler_cfg,
        rng,
        penalty_cfg,
        token_history,
        arch: "Qwen3ForCausalLM",
        resolve_pieces: true,
    };

    // Prefill-tail token selection via the shared sampling fork. Gate the mask
    // ONCE here, before the post-selection `advance()` below — matching the
    // old prefill-tail timing (wants_mask can flip on engagement).
    let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
    let top = choose_token(&mut ctx, &logits_flat, mask_active)?;
    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    // A6.3: advance constraint regardless of mask state (warm-up scans).
    if let Some(c) = ctx.constraint.as_mut() {
        c.advance(last_id);
    }
    // A7.3: push prefill token into history.
    ctx.token_history.push(last_id);
    let prefill_total_ns = prefill_t0.elapsed().as_nanos();

    let piece = tokenizer
        .id_to_token(last_id)
        .unwrap_or_else(|| format!("<unk:{last_id}>"));

    tracing::debug!(
        step = 0,
        token_id = last_id,
        piece = %piece,
        max_abs_logit,
        nan_count,
        prompt_len = prompt_ids.len(),
        "qwen3 generate_greedy prefill"
    );

    // prefill is non-pipelined (last_id already materialised), so the
    // prefill token's logprobs come straight from this step's logits.
    let prefill_logprobs = if lp_k > 0 {
        capture_logprobs(&logits_flat, &top, lp_k)
    } else {
        None
    };
    steps.push(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: prefill_logprobs,
    });
    (ctx.step_fn)(steps.last().unwrap());

    // Push this prefill snapshot to the prompt cache (Miss → store).
    // We clone the post-prefill KV caches (refcount bump, no data copy) before
    // the decode loop starts writing new decode-step K/V into them.
    //
    // The `kv_cache_bytes` metric is NOT sampled here: this is the prefill
    // snapshot, before the decode loop allocates its ring, so a sample here
    // would omit the ring on ring-backed codecs. It is recorded post-decode
    // below, gated by the `PostDecode` witness.
    {
        let cloned_caches: Result<Vec<KvCache>> =
            caches.iter().map(|c| c.try_deep_clone()).collect();
        if let Ok(kv_snapshot) = cloned_caches {
            // Materialize GPU arrays on the current inference thread before
            // storing in the prompt cache.  Each spawn_blocking request runs
            // on its own tokio thread with its own Metal GPU stream.  If these
            // lazy arrays are evicted on a *different* inference thread later,
            // that thread's eval_for_spill would fail with "There is no
            // Stream(gpu, N) in current thread".  Pre-eval here makes eval()
            // a no-op from any future thread.
            // The drain thread's spill() refcount-clones these arrays (no new graph) and evals
            // the clone, which shares the same buffers — so materializing here makes that eval a no-op.
            match kv_snapshot.iter().try_for_each(|c| c.eval_for_spill()) {
                Ok(()) => {
                    // salt chained walk with the active layout_key + KV codec
                    // (issue #26). When tier is OFF and the codec is constant,
                    // this is the legacy stream for that codec.
                    let lk = qwen3_active_layout_key();
                    let block_hashes = chained_block_hashes_seeded(
                        prompt_ids,
                        crate::prompt_cache::request_cache_seed(
                            lk,
                            kv_quant,
                            model.cfg.num_hidden_layers,
                            SHARES_KV_ACROSS_LAYERS,
                            model.model_sig,
                        ),
                    );
                    // Capture the first-token logprobs at the OpenAI ceiling so
                    // a later exact-hit replays a true logprob (truncated to its own
                    // `top_logprobs_k`) regardless of whether THIS request asked for
                    // logprobs. `logits_flat` is already host-evaluated above; this is
                    // one extra host log-softmax + top-k per cache store (per unique
                    // prompt), not per token.
                    let first_logprobs =
                        capture_logprobs(&logits_flat, &top, PROMPT_CACHE_LOGPROBS_K);
                    let entry = Qwen3Entry {
                        prompt_token_ids: prompt_ids.to_vec(),
                        block_hashes,
                        kv_caches: kv_snapshot,
                        first_id: last_id,
                        first_piece: tokenizer
                            .id_to_token(last_id)
                            .unwrap_or_else(|| format!("<unk:{last_id}>")),
                        first_logprobs,
                        kv_quant: Some(kv_quant),
                        is_ssd_hydrated: false,
                    };
                    QWEN3_PROMPT_CACHE.with_inner_mut(|guard| {
                        if let Some(cache) = guard.as_mut() {
                            if cache.push(entry).is_some() {
                                let stats = cache.stats();
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    cache_hits = stats.hits,
                                    cache_misses = stats.misses,
                                    cache_bytes = stats.bytes,
                                    "qwen3 generate_greedy: pushed snapshot to prompt cache (miss path)"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "qwen3 generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
            }
        }
    }

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    // Decode: shared pipelined async loop, reusing the prefill-tail `ctx`. The
    // Miss funnel is just "caches + last_id → decode"; the loop is the call site.
    // The pipeline ordering (choose_token → async_eval → drain previous pending →
    // feed) overlaps host sampling with the in-flight GPU forward; see
    // decode_loop.rs.
    let (stats, post) = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
        model.forward_arr(y, 1, Some(&mut caches), device)
    })?;

    // Store KV-cache bytes post-decode: the decode ring is resident now, so the
    // sample includes it on ring-backed codecs. Same lifecycle point as the
    // exact-hit path above.
    {
        let kv_bytes: u64 = caches.iter().map(|c| c.resident_bytes()).sum();
        model.kv_bytes.store(kv_bytes, post);
    }

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Qwen3ForCausalLM",
        n_steps = stats.decode_steps,
        prefill_ms,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        "decode_profile"
    );

    Ok(steps)
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a Qwen3 model from a snapshot directory.
///
/// `yarn_override` provides a runtime YARN config for models whose
/// `config.json` lacks `rope_scaling`. See [`YarnOverride`].
pub fn load_from_path(model_dir: &Path, yarn_override: Option<&YarnOverride>) -> Result<Qwen3Text> {
    let cfg_raw = load_config(model_dir)?;
    let cfg = Qwen3Config::from_model_config(&cfg_raw, yarn_override)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        head_dim = cfg.head_dim,
        vocab_size = cfg.vocab_size,
        quant_bits = cfg.quant_bits,
        quant_group_size = cfg.quant_group_size,
        "Qwen3: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    // qwen3/bonsai ships an honest index → index-first fetch with header-scan
    // fallback via the shared `Weights` helper. (`ShardSet::open`, not
    // `open_dir`, since the index is trustworthy.)
    let shards = ShardSet::open(model_dir, &idx)?;
    let w = Weights::new(&shards, &idx);

    // Global quant params; qwen3 has no per-tensor `config.json` overrides, so
    // the resolver's `.biases`-sibling affine rule is the only thing that can
    // change the mode (for affine checkpoints like Bonsai it is a no-op).
    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);
    let overrides = std::collections::HashMap::new();

    // Thin adapter: calls the shared seam, then converts the shared
    // `crate::layers::Linear` (mode: QuantMode) to the arch-local
    // `Linear` (mode: String). OOM classification is inside `w.linear`
    // — no `.map_err` needed here. `Paro` is unreachable from `w.linear`
    // (it only builds Plain/Quantized) but the match is exhaustive.
    let lin = |base: &str| -> Result<Linear> {
        use crate::layers::Linear as SharedLinear;
        match w.linear(base, |hb| resolve_quant(base, hb, &defaults, &overrides))? {
            SharedLinear::Plain { weight } => Ok(Linear::Plain { weight }),
            SharedLinear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Ok(Linear::Quantized {
                weight,
                // Quant scales/biases are loaded at their on-disk dtype, FP16 for
                // some snapshots (e.g. Bonsai). `quantized_matmul` promotes its
                // BF16 activation against an FP16 scale to an F32 result, which
                // then carries F32 through Q/K/V, attention, and the `--kv-quant
                // none` KV cache — doubling residency. Forcing the scale/bias to
                // BF16 keeps the projection output BF16. Only float scales are
                // cast: mxfp8/mxfp4 ship uint8 E8M0 scales the dequant kernel
                // requires verbatim (`bf16_scales` gates on dtype).
                //
                // This is one-dtype uniformity, NOT reference parity: on an fp16
                // checkpoint mlx-lm unifies on fp16, so bf16 here is coarser than
                // both the weights and the reference. See `bf16_param`.
                scales: bf16_scales(scales)?,
                biases: biases.map(bf16_param).transpose()?,
                group_size,
                bits,
                mode: mode.as_str().to_owned(),
            }),
            SharedLinear::Paro { .. } => Err(Error::Loader(format!(
                "{base}: unexpected Paro variant from w.linear"
            ))),
        }
    };

    // Norm weight adopts the BF16 activation dtype (see `bf16_param`): `rms_norm`
    // of a BF16 activation against an FP16 weight promotes to F32 and leaks into
    // the KV cache.
    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: bf16_param(w.array(&format!("{name}.weight"))?)?,
            eps: cfg.rms_norm_eps,
        })
    };

    let pfx = "model";

    // Embedding table. Thin adapter mirrors `lin`: converts shared Embedding
    // (mode: QuantMode) to the arch-local Embedding (mode: String).
    let embed_tokens = {
        use crate::layers::Embedding as SharedEmbedding;
        let base = format!("{pfx}.embed_tokens");
        match w.embedding(&base, |hb| resolve_quant(&base, hb, &defaults, &overrides))? {
            SharedEmbedding::Plain { weight } => Embedding::Plain { weight },
            SharedEmbedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Embedding::Quantized {
                weight,
                // The tied-lm_head path (`as_linear`) calls `quantized_matmul`
                // with these scales/biases against a bf16 hidden state. On
                // snapshots that ship embedding scales at fp16 (e.g. Bonsai),
                // that produces f32 logits; cast to bf16 at load to keep the
                // output bf16 — consistent with `lin`'s treatment and
                // mlx-lm's uniform model-dtype discipline. uint8 E8M0
                // (mxfp8/mxfp4) scales pass through (`bf16_scales`).
                scales: bf16_scales(scales)?,
                biases: biases.map(bf16_param).transpose()?,
                group_size,
                bits,
                mode: mode.as_str().to_owned(),
            },
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    // lm_head: separate when not weight-tied.
    // Ternary-Bonsai (tie=false) stores lm_head at the top level: `lm_head.weight`.
    let lm_head = if cfg.tie_word_embeddings {
        info!("Qwen3: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let base = if w.has("lm_head.weight")? {
            "lm_head"
        } else {
            "model.lm_head"
        };
        info!(%base, "Qwen3: loading separate lm_head");
        // `lin` falls through to `Linear::Plain` when no `.scales` sibling is present.
        Some(lin(base)?)
    };

    // Decoder layers.
    let scale = (cfg.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    // Precompute the YARN inv-freq table once, then alias-clone per
    // layer. `compute_yarn_freqs` is pure CPU + a small (`head_dim/2 = 64` for
    // Bonsai) `Array::from_f32_slice`; `try_clone` increments the MLX
    // refcount, so all 36 layers share the same backing buffer.
    let (yarn_freqs_proto, yarn_mscale): (Option<Array>, f32) = match cfg.yarn {
        Some(yc) => {
            let (freqs, mscale) =
                crate::rope::compute_yarn_freqs(cfg.head_dim, cfg.rope_theta, yc)?;
            info!(
                factor = yc.factor,
                original = yc.original_max_position_embeddings,
                mscale,
                "qwen3: YARN RoPE freqs precomputed"
            );
            (Some(freqs), mscale)
        }
        None => (None, 1.0),
    };

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let a = format!("{base}.self_attn");

        let q_norm = load_rms(&format!("{a}.q_norm"))?;
        let k_norm = load_rms(&format!("{a}.k_norm"))?;

        let q_proj = lin(&format!("{a}.q_proj"))?;
        let k_proj = lin(&format!("{a}.k_proj"))?;
        let v_proj = lin(&format!("{a}.v_proj"))?;

        // L33: attempt to fuse Q+K+V into a single stacked weight at load time.
        // FusedQkvProjection::try_from_separate returns None for plain weights;
        // the separate q/k/v projections are kept for the fallback path.
        // Concatenation happens on Device::Cpu at load time (weights are host
        // tensors), then evaluated lazily by MLX on first use.
        let qkv_proj =
            FusedQkvProjection::try_from_separate(&q_proj, &k_proj, &v_proj, Device::Cpu)?;
        if qkv_proj.is_some() {
            debug!(layer = i, "qwen3: fused QKV projection built (L33)");
        }

        let yarn_freqs = yarn_freqs_proto
            .as_ref()
            .map(Array::try_clone)
            .transpose()?;
        let attn = Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj: lin(&format!("{a}.o_proj"))?,
            qkv_proj,
            q_norm,
            k_norm,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale,
            rope_theta: cfg.rope_theta,
            yarn_freqs,
            // Precompute once; all layers share the same mscale value, and
            // scalar arrays carry no layer-specific state, so no try_clone
            // is needed.
            // Cast to bf16 at load; the activation stream is fixed bf16, so the
            // scalar dtype is known here. The per-step multiply then stays bf16
            // without any runtime astype call.
            yarn_mscale_arr: if (yarn_mscale - 1.0).abs() > 1e-6 {
                Some(scalar_f32(yarn_mscale).astype(Dtype::Bf16, Device::Cpu)?)
            } else {
                None
            },
        };

        let mlp = Mlp {
            gate_proj: lin(&format!("{base}.mlp.gate_proj"))?,
            up_proj: lin(&format!("{base}.mlp.up_proj"))?,
            down_proj: lin(&format!("{base}.mlp.down_proj"))?,
        };

        layers.push(DecoderLayer {
            input_norm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attn_norm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });

        debug!(layer = i, "qwen3: loaded layer");
    }

    info!(
        total_layers = cfg.num_hidden_layers,
        "Qwen3: all layers loaded"
    );
    Ok(Qwen3Text {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        kv_bytes: crate::kv_bytes::KvBytesCounter::default(),
        model_sig: crate::prompt_cache::model_cache_sig(model_dir),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "qwen3_tests.rs"]
mod qwen3_tests;
