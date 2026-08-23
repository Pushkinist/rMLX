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
//! `KvQuant::None` path (removed then restored) is opt-in via `--kv-quant
//! none` (alias `bf16`), as the closest available comparison against mlx-lm's
//! bf16-KV champion. It holds a `[B, kv_h, S, D]` bf16 K and V buffer per
//! layer. On the arches that adopted the lazy ring — gemma4, qwen3,
//! qwen3_5_moe, qwen3_vl_moe, the ones that call `kv_max_seq_and_ceiling` and
//! `with_max_seq_ceiling` — that buffer starts small and grows toward the
//! `--max-ctx` ceiling instead of being allocated at it. laguna, gemma3,
//! qwen2 and bitnet still construct at the resolved `--max-ctx`
//! (`max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT)`, no ceiling set), so a
//! large `--max-ctx` is an eager allocation there. Auto-resolver default is
//! unchanged (still K8V8).
//!
//! `none` is a pure-bf16 control on every arch: [`kv_quant_for_layer`]'s
//! boundary promotion applies only to a base mode that quantizes, so no layer
//! of a `none` run holds a packed store. This was not always true — the
//! promotion used to fire under `None` too, which made `none` a bf16/K8V8
//! mixture measuring up to 1.16× true bf16 (gemma-4-26b at 32k). Historical
//! "vs `none`" numbers recorded before that fix carry a per-arch correction
//! factor; see `docs/KV_QUANT.md` §Layer-adaptive overrides.
//!
//! `KvQuant::K8V4` is the recorded baseline for Qwen MoE PPL recovery
//! (`docs/KV_QUANT.md` §"Qwen MoE catastrophe"). It is opt-in, never automatic:
//! - K uses affine q8_0 (symmetric 8-bit, `group_size=128`).
//! - V uses TurboQuant 4-bit Lloyd-Max N(0,1) codebook.
//! - The split is per-axis (K vs V), NOT per layer-index — the Python fork
//!   uses a fake `8,4` flag where both K and V are the same width, split
//!   only by layer index. rMLX implements real asymmetric K/V.
//!
//! `KvQuant::K8V8` stores both K and V with affine q8_0 — verified coherent on
//! all 11 Open Models with the SWA per-layer mask dispatch fix. It is opt-in:
//! `auto` resolves to [`DEFAULT_KV_QUANT`] (bf16) on every arch.
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
/// [`kv_quant_for_layer`], at every context length.
///
/// Bench findings show first-layer KV vectors carry the highest absolute magnitudes
/// (embedding residual is large before deep normalisation) and that forcing
/// q8_0 on the first 2 layers recovers 37–91% of turbo2 quality loss at ≥32K
/// context. That sweep is the evidence for the constant, not a gate on it:
/// [`kv_quant_for_layer`] is never handed a context length.
pub const LAYER_ADAPTIVE_HEAD_N: usize = 2;

/// Layer-adaptive KV quantization.
///
/// Returns `KvQuant::K8V8` for a **quantizing** `base_quant` on:
/// - the **first** `head_n` layers (by absolute layer index); and
/// - the **last** `tail_n` layers (by absolute layer index).
///
/// Returns `base_quant` for all other layers, and for every layer when
/// `base_quant` quantizes neither side.
///
/// Rationale:
/// - **Tail**: last-layer K/V vectors carry the highest per-token information.
///   Forcing 8-bit for the tail recovers PPL quality lost to aggressive
///   V-quant (turbo3/planar).
/// - **Head**: first-layer K/V vectors carry large absolute magnitudes
///   (embedding residual before deep normalisation). q8_0 on the first 2
///   layers was measured to recover 37–91% of turbo2 quality degradation at
///   ≥32K ctx; the promotion itself is unconditional.
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
/// The promotion is a **quality floor for a codec that quantizes**: it buys
/// back loss the base codec introduced on the layers where that loss costs
/// most. A base mode that quantizes *neither* axis has no such loss, so the
/// promotion is skipped for it — see [`base_is_unquantized`]. That leaves it a
/// no-op for `KvQuant::K8V8` (already the target) and for `KvQuant::None`
/// (bf16 both sides), and in force for every quantizing base mode, including
/// the K-only families whose V side is bf16 but whose K side is below the
/// floor.
///
/// Two per-arch filters can cancel the promotion independently of this
/// function: a windowed layer runs the bf16 rotating ring regardless of the
/// flag, and a shared-KV layer (Gemma4 `num_kv_shared_layers`) owns no cache
/// to promote. Per-arch counts and the measured byte ratios are in
/// `docs/KV_QUANT.md` §Layer-adaptive overrides.
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
    if (is_tail || is_head) && !base_is_unquantized(base_quant) {
        KvQuant::K8V8
    } else {
        base_quant
    }
}

/// The **nominal** per-layer codec vector for a model of `n_layers` layers at
/// base codec `base` — one entry per decoder layer, `kv_quant_for_layer` at the
/// standard [`LAYER_ADAPTIVE_TAIL_N`] / [`LAYER_ADAPTIVE_HEAD_N`] constants.
///
/// This is the **one** producer of that vector. Every consumer calls it rather
/// than re-running the loop: each arch's cache-construction loop
/// (`caches[i]` is built at `quants[i]`), the SSD attach that folds the vector
/// into the layout key, and the per-request prompt-cache seed. Two of those
/// three describe what the third builds, so a second copy of the loop is not a
/// duplication of style — it is a way for the description to stop matching the
/// thing described, silently, the next time the constants or the rule move.
/// `scripts/check_kv_layer_quants.sh` (in `make ci`) keeps it that way by
/// failing on a direct [`kv_quant_for_layer`] call outside this module.
///
/// **Nominal, not effective.** Two per-arch filters can make an entry a no-op
/// on the built cache and are deliberately *not* folded in here, because they
/// are properties of the layer's geometry rather than of the codec policy: a
/// windowed layer runs the bf16 rotating ring whatever codec it is handed, and
/// a shared-KV layer (Gemma4 `num_kv_shared_layers`) owns no cache at all. A
/// consumer that needs the effective codec of a *built* cache must read the
/// cache, not this vector.
#[must_use]
pub fn kv_layer_quants(n_layers: usize, base: KvQuant) -> Vec<KvQuant> {
    (0..n_layers)
        .map(|i| {
            kv_quant_for_layer(
                i,
                n_layers,
                base,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
            )
        })
        .collect()
}

/// Code width [`KvQuant::approx_code_bits`] reports for a side that is kept at
/// model dtype (bf16) instead of quantized.
const MODEL_DTYPE_CODE_BITS: u32 = 16;

/// Does `base_quant` keep **both** K and V at model dtype?
///
/// Keyed off the codec's own code widths, never off a codec name or an arch:
/// any mode that quantizes nothing reports `MODEL_DTYPE_CODE_BITS` on both
/// sides ([`KvQuant::None`] today) and any mode that quantizes at least one
/// side reports that side below it — including the K-only families
/// (`PlanarK`, `IsoKOnly*`, `RotorKOnly*`), which keep a bf16 V but a 3-/4-bit
/// K and therefore still have K-side loss for the boundary promotion to
/// recover.
///
/// Used by [`kv_quant_for_layer`] to decide whether the boundary promotion has
/// anything to buy. Promoting an unquantized base allocates a packed q8_0 K+V
/// store on top of the model-dtype buffers the layer already holds and can
/// only *lower* its precision — the inverse of what the override exists for.
///
/// The test is deliberately "quantizes nothing", not "at least as wide as the
/// K8V8 floor on both axes": the second would also divert equal-width bases of
/// a different family (`mixed_k8g64_v8g64`, `rot_k_v8g64`) out of the
/// promotion, which is a separate question about codecs that genuinely read
/// their packed store and is not decided here.
fn base_is_unquantized(base_quant: KvQuant) -> bool {
    let (k_bits, v_bits) = base_quant.approx_code_bits();
    k_bits >= MODEL_DTYPE_CODE_BITS && v_bits >= MODEL_DTYPE_CODE_BITS
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
/// **increase** resident KV versus plain bf16 on the active layer mix.
///
/// Why this can happen, generally: a quantized codec can keep a warm-TTFT bf16
/// decode seed (`decode_fp16_k/v`) alongside its packed codes and per-group
/// scales on every **global** layer (see
/// [`rmlx_kv_quant::KvQuant::feeds_bf16_k_at_decode`]). Where it does, those
/// codes and scales are pure overhead on top of a buffer the same size as bf16
/// and the codec is net-negative. A codec whose decode reads only the mirror
/// builds no store at all
/// ([`rmlx_kv_quant::KvQuant::materialises_packed_store`]), so it holds exactly
/// the bf16 bytes and never reaches this warn.
/// **Windowed layers always run the bf16 rotating ring regardless of the flag**
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
///
/// The emitted byte count is an estimate on both sides of the true figure, so
/// read the sign and not the magnitude. `KvQuant::estimated_resident_bytes_per_layer`
/// over-charges the affine and q8_0 sidebands (it models one f32 per 32 values
/// for stores whose cadence is one per 64 or 128), and under-reports an iso
/// codec over the window between `exit_prefill`, which bulk-encodes into CPU
/// blocks 2.97x the GPU ring, and the first fused decode step that drops them.
/// The sign is what the warning is for; a reader must not size a buffer from
/// the number.
pub fn warn_if_kv_codec_net_negative(quant: KvQuant, layers: &[KvLayerShape], eff_seq: u64) {
    let (total_saving, n_global, n_windowed) = kv_codec_net_saving_total(quant, layers, eff_seq);
    if total_saving < 0 {
        tracing::warn!(
            kv_quant = %quant,
            eff_seq,
            n_global,
            n_windowed,
            est_extra_bytes = -total_saving,
            "KV codec increases resident KV vs bf16 on this layer mix — the per-global-layer warm-TTFT bf16 seed plus codec scales exceed the bytes saved at this context; windowed layers already run bf16 and are unaffected. Read the sign, not the magnitude: the estimator over-charges the affine and q8_0 per-group sidebands, and under-reports an iso codec until its fused decode path drops the CPU blocks the prefill encode built. Consider --kv-quant none if memory is the goal."
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

// ── The auto default ──────────────────────────────────────────────────────────

/// The KV codec `--kv-quant auto` resolves to — for every architecture, every
/// checkpoint and every prompt length.
///
/// **Unquantised bf16.** This is the one producer of the auto default; a caller
/// that needs "whatever auto picks" reads this constant and nothing else. There
/// is no per-arch table and no per-context re-selection behind it, so an
/// operator who passes no flag gets the same cache the CLI, the server, the
/// image branch and every speculative drafter build.
///
/// # Why bf16 and not a quantised codec
///
/// Measured on the current tree, not inherited:
///
/// * **The bf16-mirror codecs cost bytes they do not save.** `K8V8`, `K8V4`,
///   `Planar`, `Planar3`, `PlanarK`, the turbo and iso/rotor asymmetric
///   families all decode off the bf16 mirror `exit_prefill` materialises and
///   never read their packed store, so that store is no longer built
///   ([`rmlx_kv_quant::KvQuant::materialises_packed_store`]). Their resident KV
///   is therefore *equal* to bf16's — byte-identical, measured — and so is
///   their output at temp=0. They are the same cache under another name.
/// * **The one store-reading codec a default ever picked loses on both axes.**
///   Ten variants read their packed store at decode
///   ([`rmlx_kv_quant::KvQuant::decode_reads_packed_store`]); `Mixed` is the
///   only one the retired per-arch table selected, on 2-bit Qwen3 dense. It
///   holds that store *beside* a bf16 mirror, so it is bf16 plus a store:
///   measured resident KV ratio `none`/`Mixed` of 0.777 / 0.775 / 0.774 at
///   4k / 8k / 32k on Ternary-Bonsai-8B, with lossy output (token ids part from
///   bf16 at id 57 / 56 / 35 of 100) and no throughput to show for it — `none`
///   is +3.00% and +2.58% SEPARATED at 4k and 8k and INCONCLUSIVE at 32k.
///
/// So the codec axis has no cell that beats bf16 on memory, and none that beats
/// it on decode. A default that picks one is charging the operator for a label.
///
/// # What this does not say
///
/// It does not say quantised KV cannot pay. It says these implementations do
/// not, on this hardware, today: the byte saving inside a packed path converts
/// to time at an efficiency far below 1, which is what cancels it. The measured
/// break-even condition and the epsilon values behind that live in
/// `docs/KV_QUANT.md` §"Fused flash-decode over a quant store"; no single
/// summary percentage is quoted here because none is recorded.
/// Every codec stays selectable with an explicit `--kv-quant` / `--cache-type-*`
/// / `--kv-preset`; only what `auto` resolves to is fixed here. When the
/// fused-decode-over-quantised-store work lands, this constant is the one place
/// that has to change — and it must change on a fresh measurement, not by
/// restoring a table.
///
/// See `docs/KV_QUANT.md` § "The auto default".
pub const DEFAULT_KV_QUANT: KvQuant = KvQuant::None;

/// Layer-name fuzzy-match helper.
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

// ── KvCacheBuilder ────────────────────────────────────────────────────────────

/// Builder for KV-cache construction with optional calibration attachment.
///
/// Gains a `calibration` field so the per-arch generate path can pass
/// per-layer high-precision indices to codec storage during cache construction.
/// `with_calibration()` stores the calibration.
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
// `DEFAULT_KV_QUANT`, `kv_quant_for_layer` and `LAYER_ADAPTIVE_TAIL_N` are
// defined inline above.
// No re-export needed — already top-level items in this module.
