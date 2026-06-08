//! KV cache primitive for incremental decoding.
//!
//! `KvCache` holds the accumulated K and V tensors for one attention layer.
//! Layout: `[B, kv_heads, S, head_dim]` (B=1 in all current use-cases).
//!
//! Usage:
//! - Pass `cache: Option<&mut KvCache>` into `Attention::forward`.
//! - When `None`: Attention recomputes K/V from scratch (existing behaviour).
//! - When `Some`: K/V are written into a pre-allocated buffer on every call and
//!   the filled slice is returned for SDPA. `cache.offset()` tells callers what
//!   position offset to pass for RoPE.
//!
//! The generator allocates one `Vec<KvCache>` (one entry per layer) and resets
//! all on every new request via `KvCache::reset`.
//!
//! # Quantization modes (S2.4)
//!
//! Default KV caches are quantized (K8V4, K8V8, Planar). The unquantised
//! `KvQuant::None` path (removed then restored)
//! is opt-in only via `--kv-quant none` (alias `bf16`) for apples-to-apples
//! comparison against mlx-lm's bf16-KV champion. It pre-allocates a
//! `[B, kv_h, max_seq, D]` bf16 buffer per layer for both K and V — at 4096
//! ctx this is fine; at 128k ctx it would be ~64 GB. Auto-resolver default
//! is unchanged (still K8V8).
//!
//! `KvQuant::K8V4` is a CLAUDE.md-mandated baseline for Qwen MoE PPL recovery:
//! - K uses affine q8_0 (symmetric 8-bit, `group_size=128`).
//! - V uses TurboQuant 4-bit Lloyd-Max N(0,1) codebook.
//! - The split is per-axis (K vs V), NOT per layer-index — the Python fork
//!   uses a fake `8,4` flag where both K and V are the same width, split
//!   only by layer index. rMLX implements real asymmetric K/V.
//!
//! `KvQuant::K8V8` stores both K and V with affine q8_0. This is the
//! per-arch default — verified coherent on all 11 Open Models with the SWA
//! per-layer mask dispatch fix.
//!
//! `KvQuant::Planar` is K = q8_0, V = PlanarQuant 4-bit (S3.4 — 2026-05).
//!
//! # Qwen MoE catastrophe (CLAUDE.md mandate)
//!
//! Symmetric TurboQuant on Qwen2.5/Qwen3 + Q4_K_M weights causes PPL
//! degradation from 218 to 8641 (catastrophic). The 7:1 GQA ratio amplifies
//! K-head quantization error through softmax. K8V8 (or K8V4) is required;
//! never run Qwen MoE with a 4-bit-K cache.

#![allow(clippy::match_same_arms, clippy::trivially_copy_pass_by_ref)]
pub mod attention_dispatch;
pub mod cache_type;

use rmlx_kv_quant::KvQuant;

// The `pub use rmlx_kv_quant::*` and `pub use rmlx_kv_ssd::*` shim
// re-exports were dropped. All callers import the codec layer
// (`KvCache`, `LinearAttnCache`, `KvQuant`, `storage`, `paged`, `mixed_quant`,
// `rot_k`, `rotating`, `q8`, etc.) directly from `rmlx_kv_quant::*`, and the
// SSD-tier layer (`KvBlockReader`, `KvBlockWriter`, `SsdKvIndex`, `SsdSpiller`,
// `SsdHydrator`, `SpillJob`, `HydratedBlock`, `write_caches`, `set_ssd_*_hook`,
// `call_ssd_*_hook`) directly from `rmlx_kv_ssd::*`.

#[cfg(test)]
mod tests;

// ── Public constant ───────────────────────────────────────────────────────────

// `KV_MAX_SEQ_DEFAULT` lives in `rmlx_kv_quant::quant`.
// The re-export shim was dropped; import directly from
// `rmlx_kv_quant::KV_MAX_SEQ_DEFAULT`.

/// Default number of tail layers forced to `KvQuant::K8V8` by
/// [`kv_quant_for_layer`]. Matches the N70 reference experiment.
pub const LAYER_ADAPTIVE_TAIL_N: usize = 8;

/// Default number of head layers forced to `KvQuant::K8V8` by
/// [`kv_quant_for_layer`] when `ctx >= 32K`.
///
/// Bench findings show first-layer KV vectors carry the highest absolute magnitudes
/// (embedding residual is large before deep normalisation) and that forcing
/// q8_0 on the first 2 layers recovers 37–91% of turbo2 quality loss.
pub const LAYER_ADAPTIVE_HEAD_N: usize = 2;

/// Layer-adaptive KV quantization.
///
/// Returns `KvQuant::K8V8` for:
/// - the **first** `head_n` layers (by absolute layer index); and
/// - the **last** `tail_n` layers (by absolute layer index).
///
/// Returns `base_quant` for all other layers.
///
/// Rationale:
/// - **Tail**: last-layer K/V vectors carry the highest per-token information.
///   Forcing 8-bit for the tail recovers PPL quality lost to aggressive
///   V-quant (turbo3/planar).
/// - **Head**: first-layer K/V vectors carry large absolute magnitudes
///   (embedding residual before deep normalisation). q8_0 on the first 2
///   layers recovers 37–91% of turbo2 quality degradation at ≥32K ctx.
///
/// # Usage
///
/// Call during cache-vector construction inside each arch's `generate_greedy`:
///
/// ```ignore
/// let caches: Vec<KvCache> = (0..n_layers)
/// .map(|i| {
/// let q = kv_quant_for_layer(
/// i, n_layers, kv_quant,
/// LAYER_ADAPTIVE_TAIL_N, LAYER_ADAPTIVE_HEAD_N,
/// );
/// KvCache::with_quant_max_seq(q, max_seq)
/// })
/// .collect();
/// ```
///
/// General-purpose: works for any arch with any `KvQuant` base mode. The
/// override is by layer index, not by model name or KV mode — no hardcoded
/// model branches.
///
/// When `base_quant` is already `KvQuant::K8V8`, the override is a no-op.
/// The only observable change is on `KvQuant::Planar`, `KvQuant::K8V4` (V
/// side), and `KvQuant::Mixed` boundary layers, which are promoted to K8V8.
///
/// When `head_n == 0` and `tail_n == 0`, `base_quant` is always returned.
pub fn kv_quant_for_layer(
    layer_idx: usize,
    n_layers: usize,
    base_quant: KvQuant,
    tail_n: usize,
    head_n: usize,
) -> KvQuant {
    let is_tail = tail_n > 0 && layer_idx >= n_layers.saturating_sub(tail_n);
    let is_head = head_n > 0 && layer_idx < head_n;
    if is_tail || is_head {
        KvQuant::K8V8
    } else {
        base_quant
    }
}

/// Per-layer attributes for the resolve-time net-benefit check.
///
/// Model-agnostic: every field is a layer geometry attribute the arch parser
/// already knows. No arch name is carried — the check keys off geometry +
/// codec only, so any architecture interleaving windowed + global attention is
/// covered identically.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct KvLayerShape {
    /// Per-KV-head dimension (e.g. 256 for Gemma4, 128 for Bonsai/Qwen).
    pub head_dim: u64,
    /// Number of KV heads (GQA group count).
    pub kv_heads: u64,
    /// Sliding-window size in tokens for windowed layers, or `None` for a
    /// global (full-attention) layer.
    pub window: Option<u64>,
}

/// Emit one structured `warn!` when the resolved KV codec is estimated to
/// **increase** resident KV versus plain bf16 on the active layer mix
/// (issue #34).
///
/// Why this can happen, generally: a quantized codec keeps a warm-TTFT bf16
/// decode seed (`decode_fp16_k/v`) alongside its packed codes + per-group
/// scales on every **global** layer (see
/// [`rmlx_kv_quant::KvQuant::feeds_bf16_k_at_decode`]). At small effective
/// context the codes + scales are pure overhead on top of a buffer the same
/// size as bf16, so the codec is net-negative. **Windowed layers always run
/// the bf16 rotating ring regardless of the flag**
/// (`RotatingKVCache.to_quantized` raises `NotImplementedError` in mlx-lm;
/// rMLX matches it), so they are a no-op for the codec and contribute zero to
/// the delta — the net-negative is a property of the global layers only.
///
/// This is advisory: the codec is **not** changed (keeping it is the operator's
/// explicit choice, and forcing bf16 globally would change numerics). The warn
/// gives the operator the byte math so they can pick `--kv-quant none` when the
/// codec buys nothing at their context size.
///
/// Keyed entirely on `(KvLayerShape, KvQuant, eff_seq)` — no arch branch. Call
/// once per request from the arch `generate` path after the codec is resolved
/// and the layer mix is known.
///
/// `layers` is the per-layer shape vector (one entry per decoder layer);
/// `eff_seq` is the effective prompt+generate length the global layers will
/// hold (the resolved `--max-ctx` ceiling or the prompt length — either is a
/// fine estimate for the sign of the saving).
pub fn warn_if_kv_codec_net_negative(quant: KvQuant, layers: &[KvLayerShape], eff_seq: u64) {
    let (total_saving, n_global, n_windowed) = kv_codec_net_saving_total(quant, layers, eff_seq);
    if total_saving < 0 {
        tracing::warn!(
            kv_quant = %quant,
            eff_seq,
            n_global,
            n_windowed,
            est_extra_bytes = -total_saving,
            "KV codec increases resident KV vs bf16 on this layer mix — the per-global-layer warm-TTFT bf16 seed plus codec scales exceed the bytes saved at this context; windowed layers already run bf16 and are unaffected. Consider --kv-quant none if memory is the goal."
        );
    }
}

/// Pure decision behind [`warn_if_kv_codec_net_negative`]: total estimated
/// net byte saving across the layer mix (negative = codec costs more than
/// bf16), plus the global / windowed layer counts.
///
/// Split out so the sign decision is unit-testable without a tracing
/// subscriber. `KvQuant::None` (or an empty layer list) returns `(0, _, _)` —
/// bf16 is never net-negative against itself.
///
/// Keyed entirely on `(KvLayerShape, KvQuant, eff_seq)`; model-agnostic.
#[must_use]
pub fn kv_codec_net_saving_total(
    quant: KvQuant,
    layers: &[KvLayerShape],
    eff_seq: u64,
) -> (i64, usize, usize) {
    if matches!(quant, KvQuant::None) || layers.is_empty() {
        return (0, 0, 0);
    }
    let mut total_saving: i64 = 0;
    let mut n_global: usize = 0;
    let mut n_windowed: usize = 0;
    for l in layers {
        let is_windowed = l.window.is_some();
        if is_windowed {
            n_windowed += 1;
        } else {
            n_global += 1;
        }
        let seq = match l.window {
            Some(w) => eff_seq.min(w),
            None => eff_seq,
        };
        total_saving = total_saving.saturating_add(quant.estimated_net_saving_per_layer(
            seq,
            l.head_dim,
            l.kv_heads,
            is_windowed,
        ));
    }
    (total_saving, n_global, n_windowed)
}

/// Resolve `(initial_max_seq, ceiling)` for a request from `--max-ctx`.
///
/// This is the shared policy that makes `--max-ctx` a **virtual ceiling**
/// rather than an eager allocation, fixing the short-prompt decode penalty
/// (issue #25): a server started with a large `--max-ctx` must serve short
/// requests at full speed by growing the KV ring lazily up to the ceiling,
/// not by allocating the whole ceiling up front.
///
/// - `initial_max_seq` is what the per-layer `KvCache` is *first* sized to.
///   It is always the small lazy default ([`rmlx_kv_quant::KV_MAX_SEQ_DEFAULT`]),
///   capped by the ceiling so a sub-default ceiling is honoured. The codec's
///   power-of-two grow path ([`KvCache::ensure_prefill_capacity`]) takes it
///   from there up to the ceiling as the prompt fills.
/// - `ceiling` is the resolved bound: `min(max_ctx_override, mpe)` when an
///   override is given, else `min(mpe, KV_MAX_SEQ_DEFAULT)` — the same chain
///   the server's `effective_max_ctx` uses, so the per-cache ceiling and the
///   per-request prompt-length guard agree. `mpe <= 0` means the arch does not
///   expose `max_position_embeddings`; treat it as "unknown" and ignore it.
/// - `max_ctx_override = Some(n)` where `n <= 0` is treated as "unset" (falls
///   through to the arch-default path) and emits a `warn!` event so the caller
///   can diagnose unexpected zero/negative values from config parsing.
///
/// Wire it in an arch `generate` as:
/// ```ignore
/// let (initial_max_seq, ceiling) = kv_max_seq_and_ceiling(max_ctx_override, mpe);
/// KvCache::with_quant_max_seq(q, initial_max_seq).with_max_seq_ceiling(ceiling)
/// ```
#[must_use]
pub fn kv_max_seq_and_ceiling(max_ctx_override: Option<i32>, mpe: i32) -> (i32, i32) {
    let default = rmlx_kv_quant::KV_MAX_SEQ_DEFAULT;
    let ceiling = match max_ctx_override {
        Some(n) if n > 0 => {
            if mpe > 0 {
                n.min(mpe)
            } else {
                n
            }
        }
        Some(n) => {
            // n <= 0: treat as unset; log so the caller can diagnose
            // unexpected zero/negative values from CLI or config parsing.
            tracing::warn!(
                max_ctx_override = n,
                "max_ctx_override <= 0 treated as unset; using arch default"
            );
            if mpe > 0 {
                mpe.min(default)
            } else {
                default
            }
        }
        None => {
            if mpe > 0 {
                mpe.min(default)
            } else {
                default
            }
        }
    };
    // Start the ring at the lazy default, but never above the ceiling (a
    // sub-default ceiling must not be pre-grown past).
    let initial_max_seq = default.min(ceiling);
    (initial_max_seq, ceiling)
}

// ── KvCacheBuilder ────────────────────────────────────────────────────────────

/// Returns the appropriate `KvQuant` default for a named architecture.
///
/// Called by the generator/server to select the right mode per CLAUDE.md
/// mandate without hard-coding arch strings at every call site.
///
/// # Default
///
/// All architectures default to `KvQuant::K8V8`. The `KvQuant::None`
/// (unquantised) path was removed: it pre-allocated full-precision bf16 at
/// `max_seq` (64 GB at 128k ctx) and every Stage-1/2 arch was already
/// coherent at K8V8 per the 4k-bench (11/11 Open Models).
///
/// # Refinement
///
/// The single-arg `for_arch_default(arch_class)` is preserved as the safe
/// fallback when no `ModelConfig` is available. Auto-resolution paths now
/// prefer [`KvCacheBuilder::resolve_default`], which also consults the
/// checkpoint config (hidden_size, MoE flag, paroquant, bits) and picks per
/// the PPL × TPS frontier table:
///
/// | Arch class | Signal | Best KV |
/// |-----------------------------------------------|-----------------------|------------------|
/// | `Qwen3_5MoeForConditionalGeneration` | (any) | K8V8 |
/// | `Qwen3_5ForConditionalGeneration` (dense PARO)| `is_paroquant` | K8V4 |
/// | `Qwen3ForCausalLM` | bits=2 | Mixed{k8,v4,g64} |
/// | `Qwen3ForCausalLM` | bits=8 / other | K8V8 |
/// | `Qwen2ForCausalLM`, `LagunaForCausalLM` | (any) | K8V8 |
/// | `Gemma3ForConditionalGeneration` | (any) | Planar |
/// | `Gemma4ForConditionalGeneration` | `has_moe` | K8V8 |
/// | `Gemma4ForConditionalGeneration` | hidden_size ≤ 2560, non-MoE, non-paroquant | K8V8 (composite audit; was K8VTurbo3) |
/// | `Gemma4ForConditionalGeneration` | hidden_size ≤ 2560, non-MoE, is_paroquant  | K8V4      |
/// | `Gemma4ForConditionalGeneration` | hidden_size ≥ 5376 | Planar |
///
/// Unknown arch / inconclusive signals fall through to K8V8 (safe default —
/// matches `for_arch_default` behaviour, no surprise regressions).
///
/// Source: `docs/reports/ppl-tps-frontier.md` § "Per-arch
/// class best quant recommendation".
///
/// ## Layer-name fuzzy-match helper
///
/// **No in-tree caller yet. Consumed by future codec encode paths.**
/// Returns `None` if no calibration is attached to the builder.
///
/// Looks up `layer_key` in a `KvCalibration` layers map using case-insensitive
/// comparison on the first three dot-separated components (the "dotted prefix").
///
/// # Examples
///
/// ```text
/// query: "model.layers.0.self_attn"  →  matches "MODEL.LAYERS.0.SELF_ATTN"
/// query: "model.layers.0"            →  matches key "model.layers.0.self_attn"
///                                        via 3-prefix (component count = 3, passes guard)
/// query: "model.layers.0.self_attn.k_proj" → 3-prefix matches
///        "model.layers.0.self_attn" key (returns that entry)
/// ```
///
/// Returns `None` when no fuzzy match is found.
pub fn lookup_layer_calibration<'c>(
    calib: &'c rmlx_loader::KvCalibration,
    layer_key: &str,
) -> Option<&'c rmlx_loader::LayerCalib> {
    // Fast path: exact match first.
    if let Some(entry) = calib.layers.get(layer_key) {
        return Some(entry);
    }

    // Fuzzy path: case-insensitive 3-dotted-component prefix comparison.
    // Take the first 3 dot-components of the query key (e.g. "model.layers.0")
    // and compare them case-insensitively against the first 3 components of
    // each map key.
    let query_prefix: Vec<&str> = layer_key.splitn(4, '.').take(3).collect();
    if query_prefix.len() < 3 {
        return None;
    }
    for (k, v) in &calib.layers {
        let key_prefix: Vec<&str> = k.splitn(4, '.').take(3).collect();
        if key_prefix.len() >= 3
            && key_prefix
                .iter()
                .zip(query_prefix.iter())
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            return Some(v);
        }
    }
    None
}

/// Builder for KV-cache construction with optional calibration attachment.
///
/// Gains a `calibration` field so the per-arch generate path can pass
/// per-layer high-precision indices to codec storage during cache construction.
/// `with_calibration()` stores the calibration; the existing static methods
/// (`for_arch_default`, `resolve_default`) are unchanged.
///
/// **No in-tree caller yet.** The surface is wired end-to-end
/// (loader → `ModelLoadConfig` → `KvCacheBuilder`) but per-arch construction
/// code that calls `with_calibration` is deferred.
/// Codec behavior is unchanged — indices are stored but not yet consumed.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed builder — field set is the complete calibration-attach contract; adding a field requires a review of all call sites"
)]
#[derive(Debug, Default)]
pub struct KvCacheBuilder {
    /// Optional KV calibration to forward to codec storage.
    ///
    /// `None` (the default) = no calibration; codec behavior unchanged.
    /// `Some(calib)` = calibration present; per-layer lookup via
    /// [`lookup_layer_calibration`] during cache construction.
    pub calibration: Option<rmlx_loader::KvCalibration>,
}

/// Inputs to [`KvCacheBuilder::resolve_default`].
///
/// Each field is optional/defaultable so callers without full config metadata
/// can pass partial signals; missing fields fall through to the K8V8 default.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub struct ResolverSignals {
    /// `text_config.hidden_size` (Gemma4 small ≤ 2560, dense ≥ 5376).
    pub hidden_size: Option<u32>,
    /// `text_config.enable_moe_block` (Gemma4 26B has MoE; e2b/e4b/31b do not).
    pub has_moe: bool,
    /// `quantization_config.quant_method == "paroquant"` (z-lab PARO checkpoints).
    pub is_paroquant: bool,
    /// `quantization.bits` (e.g. 2 for Bonsai ternary, 8 for affine, mxfp8 for Gemma4).
    pub weight_bits: Option<u8>,
}

impl ResolverSignals {
    /// Extract the resolver-relevant fields from a parsed `ModelConfig`.
    ///
    /// Reads `text_config.hidden_size`, `text_config.enable_moe_block`,
    /// `is_paroquant()`, and `quantization.bits`. Missing fields stay `None`.
    pub fn from_config(cfg: &rmlx_loader::ModelConfig) -> Self {
        let (hidden_size, has_moe) = match &cfg.text_config {
            Some(tc) => {
                let hs = tc.hidden_size;
                let moe = tc
                    .extras
                    .get("enable_moe_block")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                (hs, moe)
            }
            None => (None, false),
        };
        let weight_bits = cfg.quantization.as_ref().map(|q| q.bits);
        Self {
            hidden_size,
            has_moe,
            is_paroquant: cfg.is_paroquant(),
            weight_bits,
        }
    }
}

impl KvCacheBuilder {
    /// Attach an optional [`KvCalibration`] to this builder.
    ///
    /// Stores the calibration for forwarding to codec storage
    /// during per-layer cache construction. Returns `self` for chaining.
    ///
    /// Codec behavior is not changed — the indices are stored for future
    /// consumption by encode/decode paths.
    ///
    /// [`KvCalibration`]: rmlx_loader::KvCalibration
    #[must_use]
    pub fn with_calibration(mut self, calib: Option<rmlx_loader::KvCalibration>) -> Self {
        self.calibration = calib;
        self
    }

    /// Select the KV quantization mode for the given architecture name.
    ///
    /// Returns `KvQuant::K8V8` for every known arch. Unknown arch names
    /// also fall through to `K8V8` (safe default — never the unquantised
    /// path that was removed).
    ///
    /// Prefer [`KvCacheBuilder::resolve_default`] when a `ModelConfig` is
    /// available — it picks per the PPL × TPS frontier table.
    ///
    /// # Deprecation note
    ///
    /// This function was a no-op (`_arch_name` was unused) and has been
    /// superseded by [`KvCacheBuilder::resolve_default`], which consults
    /// checkpoint signals (hidden_size, MoE flag, paroquant, bits) and
    /// selects per the composite-score audit. All auto-resolution
    /// paths already call `resolve_default`; this function is retained only
    /// for external callers that cannot yet migrate.
    #[deprecated(
        since = "0.1.0",
        note = "use KvCacheBuilder::resolve_default with ResolverSignals extracted \
                from ModelConfig; this function ignores arch_name and always returns K8V8"
    )]
    pub fn for_arch_default(_arch_name: &str) -> KvQuant {
        KvQuant::K8V8
    }

    /// Resolve the per-arch best `KvQuant` using arch class + checkpoint signals.
    ///
    /// Implements the PPL × TPS frontier table (see type-level docs on
    /// [`KvCacheBuilder`]), updated by the composite-score
    /// audit. Falls back to `K8V8` for unknown arch classes or inconclusive
    /// signals — same safe default as [`KvCacheBuilder::for_arch_default`].
    ///
    /// # Composite-score audit
    ///
    /// 3-term degraded composite (2026-05-31):
    ///
    /// ```text
    /// score = 0.571 × decode_tps_norm + 0.286 × cosine_norm + 0.143 × mem_norm
    /// where:
    ///   decode_tps_norm = decode_tps / max_decode_tps_per_model
    ///   cosine_norm     = clamp((cosine − 0.94) / 0.06, 0.0, 1.0)
    ///   mem_norm        = (1/mem_bits) / max(1/mem_bits) across candidates
    /// ```
    ///
    /// Per-arch outcomes (see `docs/KV_QUANT.md` § "Per-arch defaults"):
    ///
    /// - **Gemma4 small** (hidden ≤ 2560, non-MoE, non-paroquant): K8V8 wins
    ///   (score 0.942) over K8VTurbo3 (0.887). Reverts the earlier K8VTurbo3 promotion.
    ///   K8V8 +1.4% TPS and +0.0183 cosine vs K8VTurbo3 — both exceed the
    ///   conservatism thresholds (±1% TPS, ±0.002 cosine).
    /// - **Gemma4 MoE**, **Qwen3.6-MoE**, **Qwen3ForCausalLM 8-bit**, unknown:
    ///   K8V8 already wins — no flip.
    /// - **Bonsai / Qwen3ForCausalLM 2-bit**: Mixed{k8g64,v4g64} (score 0.946)
    ///   over turbo3_tcq (0.819), turbo2_tcq (0.714) — no flip.
    /// - **Gemma4 dense** (hidden ≥ 5376): Planar — no cell data, no flip.
    ///
    /// Symmetric candidates (iso3_sym, rotor3_sym, tsym3) excluded
    /// from direct comparison: measured at 2-token short-prompt shape which
    /// inflates TPS relative to the 4096-token canary baseline used for all
    /// other candidates.
    pub fn resolve_default(arch_class: &str, signals: ResolverSignals) -> KvQuant {
        match arch_class {
            // Qwen3-VL-MoE — 4-bit affine weights + image-conditioned
            // attention. K8V8 measurably degrades decode quality on this
            // checkpoint (incoherent text + image output); unquantised bf16 KV
            // reproduces the mlx-vlm reference exactly. Default to bf16.
            "Qwen3VLMoeForConditionalGeneration" => KvQuant::None,
            // Qwen3.5 MoE (Qwen3.6-35B-A3B-8bit etc.) — affine 8b weights, GQA-safe at K8V8.
            //
            // Routing FullAttention layers via Mixed { K=8, V=8 }
            // (byte-for-byte port of mlx-lm-tq's MixedQuantKVCache path used by Bonsai)
            // regressed decode by -11.9% (88.88 vs 100.90 TPS on Qwen3.6-35B-A3B-8bit k8v8).
            // The hybrid GDN+FA arch only has 25% FA layers, so the per-decode-step
            // `mx.quantize` + 2× quantized_matmul overhead amortises poorly: the rMLX
            // K8V8 fused dequantize + bf16 fast SDPA is faster on FA-light archs than on
            // dense Bonsai (36/36 FA layers, +24%). Reverted to K8V8 default.
            "Qwen3_5MoeForConditionalGeneration" => KvQuant::K8V8,
            // Qwen3.5 dense PARO (z-lab Qwen3.6-27B-PARO) — K8V4 wins on memory + TPS,
            // bit-exact decode vs paroquant ref.
            //
            // Same Mixed K8V4 routing test on the FA layers
            // regressed decode by -28% (20.08 vs 27.97 TPS on Qwen3.6-27B-PARO). Reverted.
            "Qwen3_5ForConditionalGeneration" => {
                if signals.is_paroquant {
                    KvQuant::K8V4
                } else {
                    KvQuant::K8V8
                }
            }
            // Qwen3 dense (Bonsai 2-bit ternary, also affine 8b).
            //
            // For Bonsai (bits=2) route to the mlx-lm-tq Mixed { K=8, V=4 } path
            // which feeds the quantized 3-tuple directly into two
            // `mx.quantized_matmul` calls inside SDPA — skipping the
            // per-decode-step full dequantize that dominated the prior k8v4 path
            // (138.98 vs 104.82 TPS). Other Qwen3 dense (e.g. affine 8b) continue on K8V8.
            "Qwen3ForCausalLM" => match signals.weight_bits {
                Some(2) => KvQuant::Mixed {
                    k_bits: 8,
                    v_bits: 4,
                    k_group_size: 64,
                    v_group_size: 64,
                },
                _ => KvQuant::K8V8,
            },
            // Qwen2 dense + Laguna MoE — kept on the safe K8V8 default.
            "Qwen2ForCausalLM" | "LagunaForCausalLM" => KvQuant::K8V8,
            // Gemma3 (medgemma) — planar wins on TPS, sim divergence is chat-template not kernel.
            "Gemma3ForConditionalGeneration" => KvQuant::Planar,
            // Gemma4 family: dispatch by MoE flag + hidden_size + paroquant.
            "Gemma4ForConditionalGeneration" => {
                if signals.has_moe {
                    // Gemma4 MoE (26B): k8v8 ties planar on TPS, prefer the universally-validated path.
                    KvQuant::K8V8
                } else if matches!(signals.hidden_size, Some(h) if h <= 2560) {
                    // Gemma4 small (e2b hidden=1536, e4b hidden=2560).
                    if signals.is_paroquant {
                        // Theoretical small PARO (27B-class hidden) — k8v4 per PPL×TPS table.
                        KvQuant::K8V4
                    } else {
                        // Composite-score audit: K8V8 wins composite score for Gemma4
                        // small (score 0.942 vs K8VTurbo3 0.887). K8V8 is +1.4% TPS and
                        // +0.0183 cosine — both exceed the ±1% TPS / ±0.002 cosine conservatism
                        // thresholds. Reverts the earlier K8VTurbo3 promotion.
                        //
                        // K8VTurbo3 remains available via --kv-quant k8vturbo3 for operators
                        // who prefer the smaller 11-bit memory footprint over quality.
                        KvQuant::K8V8
                    }
                } else if matches!(signals.hidden_size, Some(h) if h >= 5376) {
                    // Gemma4 dense (31b mxfp8, 31B PARO) — planar wins TPS at low GQA-tradeoff.
                    KvQuant::Planar
                } else {
                    // Hidden size in (2560, 5376) without MoE flag — unknown territory.
                    // Fall back to K8V8 (safe default).
                    KvQuant::K8V8
                }
            }
            // Unknown arch — safe fallback.
            _ => KvQuant::K8V8,
        }
    }
}

/// Auto-KV-by-ctx server policy.
///
/// Selects the best `KvQuant` for a given prompt length when the server is in
/// auto mode (i.e. the user did not pass an explicit `--kv-quant` flag).
///
/// ## Policy
///
/// | prompt_len tokens | selected quant |
/// |-------------------|----------------|
/// | ≤ 8 192 | K8V4 |
/// | ≤ 16 384 | None (bf16) |
/// | ≤ 32 768 | K8V8 |
/// | > 32 768 | Planar |
///
/// Rationale:
/// - Short ctx (≤8K): K8V4 is most memory-efficient with acceptable PPL.
/// - Mid ctx (8K–16K): bf16 (unquantized) gives best quality while KV is still
///   small enough to fit.
/// - Long ctx (16K–32K): K8V8 balances memory and quality.
/// - Very long ctx (>32K, incl. ≥64K): Planar wins TPS outright (71.53 TPS at
///   64K Qwen3.6-35B-A3B vs K8V8 65.2 TPS in the bench).
///
/// This function is only called when no explicit `--kv-quant` flag was given.
/// Explicit user overrides always take precedence and bypass this function.
///
/// ## Note on Qwen MoE safety
///
/// For Qwen MoE architectures, `K8V4` is safe (asymmetric; K stays 8-bit).
/// `None` (bf16) is also safe — just large. `K8V8` and `Planar` are both
/// validated on Qwen3.6-35B-A3B-8bit. No arch-specific special-casing is
/// needed here because `kv_quant_for_layer` still forces the last 8
/// layers to K8V8 regardless of the base mode returned by this function.
pub fn kv_quant_for_ctx(prompt_len: usize) -> KvQuant {
    if prompt_len <= 8_192 {
        KvQuant::K8V4
    } else if prompt_len <= 16_384 {
        KvQuant::None
    } else if prompt_len <= 32_768 {
        KvQuant::K8V8
    } else {
        KvQuant::Planar
    }
}

// The SSD-tier process-global event recorder + 5 Prometheus observation hook
// globals (`set_ssd_event_recorder`,
// `set_ssd_{spill_prom,hydrate_prom,bytes_used,evict_total}_hook`, plus the
// internal `call_*` accessors) live in `rmlx_kv_ssd::hooks`. The re-export
// shim was dropped; cross-crate callers in `rmlx-server` / `rmlx-cli` import
// directly from `rmlx_kv_ssd::set_ssd_*_hook`.

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use cache_type::{
    parse as parse_cache_type_str, resolve as resolve_cache_type,
    validate_resolved as validate_resolved_kv_quant, CacheType, CacheTypeSpec,
    ParseError as CacheTypeParseError, ResolveError, ResolverContext,
};
// `KvCache` and `LinearAttnCache` re-exports were dropped — every
// caller imports them directly from `rmlx_kv_quant::*`.
// `ResolverSignals`, `kv_quant_for_layer`, `kv_quant_for_ctx`,
// `LAYER_ADAPTIVE_TAIL_N` are defined inline above.
// No re-export needed — already top-level items in this module.
