// promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill) can still reach them across the crate
// boundary. Doc/visibility warnings on the promoted surface are silenced; the
// API is otherwise unchanged.
#![allow(missing_docs, missing_debug_implementations, unreachable_pub)]
//! `KvCache` struct definition + constructors + accessors.
// unsafe_code: inherited from parent module (f32 reinterpret, mlx-rs zero-copy)
#![allow(unsafe_code)]
#![allow(clippy::too_many_lines)]

use rmlx_mlx::Array;

use crate::kvcache::fused_qk_shadow::FusedQkShadow;
use crate::rotating::RotatingState;
use crate::storage::KvStorage;
use crate::KvQuant;
use crate::KV_MAX_SEQ_DEFAULT;

/// Per-layer KV cache. Append-only; `reset` clears for the next request.
///
/// # Rotating (SWA) mode
///
/// When constructed via [`KvCache::with_quant_max_seq_window`] with a non-`None`
/// `sliding_window`, this cache becomes a byte-for-byte port of mlx-lm's
/// `RotatingKVCache` regardless of the caller's KV quant flag: the K/V buffer
/// is pre-allocated bf16 to `[B, kv_h, sliding_window, D]` and writes rotate
/// modulo the window. After the buffer fills, attention reads at most
/// `sliding_window` tokens, eliminating the per-step SWA mask and the 8×
/// excess matmul work.
///
/// **Why bf16 even when `--kv-quant` requests a quantized codec**: mlx-lm's
/// `RotatingKVCache.to_quantized` raises `NotImplementedError`
/// (`mlx_lm/models/cache.py:551-552`). The reference behaviour is that SWA
/// layers stay bf16 even when full-attention layers are quantized. Rotating
/// activation applies to all `--kv-quant` modes; full-attention
/// layers continue to quantize per the caller's flag while SWA layers always
/// rotate bf16. This matches mlx-lm semantics exactly.
#[allow(missing_debug_implementations)]
pub struct KvCache {
    pub storage: KvStorage,
    pub(super) offset: i32,
    pub(super) quant: KvQuant,
    /// Model-side layer index (0-based). Set at construction by the arch
    /// builder via [`KvCache::with_layer_idx`]; defaults to `0`.
    ///
    /// Used by the rotor3/rotor4 codec families to seed distinct per-layer
    /// rotor tables via [`crate::clifford::rotor_seed`]
    /// `(layer_idx, head_idx, group_idx)`. Distinct seeds are the mechanism
    /// that provides cross-layer decorrelation — all other quantised codecs
    /// ignore this field.
    ///
    /// The field is immutable after construction; it is NOT reset by
    /// [`KvCache::reset`] (the layer identity does not change between
    /// requests).
    pub(super) layer_idx: usize,
    pub(super) prefill_raw_k: Option<Array>,
    pub(super) prefill_raw_v: Option<Array>,
    pub(super) in_prefill: bool,
    pub(super) decode_fp16_k: Option<Array>,
    pub(super) decode_fp16_v: Option<Array>,
    /// If `Some`, this cache uses the SWA ring-buffer code path
    /// (RotatingKvCache port). Activated by SWA layers in
    /// `KvQuant::None` mode only.
    pub(super) rotating: Option<RotatingState>,
    // ── Head-major persistent K8V4 storage ──
    //
    // The legacy `KvStorage::K8V4` GPU buffers are chunk-appended (rows packed
    // by call-order) — fine for "dequantise once, run SDPA on bf16" but useless
    // for a per-token decode kernel that wants `[B, kv_h, max_seq, D/.]`
    // head-major reads. P1.A.1 worked around this by re-quantising the entire
    // active prefix from `decode_fp16_k/v` on every TurboFlash dispatch, which
    // is O(prefix) per decode token and erased the kernel speedup.
    //
    // P2.A.1 (this commit) carves out a parallel set of K8V4 GPU buffers held
    // directly on `KvCache`, allocated head-major to `[B, kv_h, max_seq, D/.]`,
    // seeded once from the prefill bf16 prefix on the first TurboFlash
    // dispatch, then per-decode-token appended via 4-D `slice_update` at
    // `[:, :, prev_offset:prev_offset+1, :]`. Total per-decode write traffic:
    // `B * kv_h * D / 4` u32 codes + `B * kv_h * D / 128` f32 scales (for K)
    // and `B * kv_h * D / 8` u32 codes + `B * kv_h * D / 32` f32 scales
    // (for V) — kilobytes, not megabytes.
    //
    // Layout details (Qwen35B example, B=1, kv_h=8, max_seq=128K, head_dim=128):
    // flash_k_codes: u32 [B, kv_h, max_seq, head_dim/4] — q8_0 codes
    // flash_k_scales: f32 [B, kv_h, max_seq, head_dim/Q8_GROUP]
    // flash_v_codes: u32 [B, kv_h, max_seq, head_dim/8] — turbo4 codes
    // flash_v_scales: f32 [B, kv_h, max_seq, head_dim/TQ4_GROUP]
    //
    // The kernel (`turbo_flash_sdpa`) reads these as flat 1-D buffers but
    // computes per-token offsets using `t_stride = max_seq` (this layout) and
    // iterates `t < t_active = current_seq`. Bytes between `current_seq` and
    // `max_seq` are zero (allocated via `zeros()`) and never read.
    pub(super) flash_k_codes: Option<Array>,
    pub(super) flash_k_scales: Option<Array>,
    pub(super) flash_v_codes: Option<Array>,
    pub(super) flash_v_scales: Option<Array>,
    /// `max_seq` the head-major flash buffers were allocated for. Used as the
    /// per-token row stride passed to the kernel.
    pub(super) flash_max_seq: i32,
    /// Number of K/V tokens currently materialised in the flash buffers. This
    /// MAY lag `self.offset` between layers when only some layers have flushed
    /// (single-decoder is sequential so usually equals `self.offset`).
    pub(super) flash_filled: i32,
    // ── Head-major persistent K storage for fused-QK kernels ──
    //
    // Generalises the TurboFlash `flash_*` field set above to all five
    // fused-QK codec families (q8, TurboSym3, TurboSym4, Iso3/4Sym,
    // Rotor3/4Sym). Allocated lazily on the first fused-QK decode
    // dispatch from the bf16 prefill prefix; appended head-major on every
    // subsequent decode token. See
    // `docs/research/fused-qk-storage-design.md`.
    pub(super) fused_qk_shadow: Option<FusedQkShadow>,
}

impl KvCache {
    /// Create a new `KvCache` with the given quantization and the default max sequence length.
    pub fn with_quant(quant: KvQuant) -> Self {
        Self::with_quant_max_seq(quant, KV_MAX_SEQ_DEFAULT)
    }

    /// Borrow the internal [`KvStorage`] for serialization.
    ///
    /// Used by `block_io::write_caches` to bridge a `&[KvCache]` (what the
    /// prompt-cache entries hold) into the `&[KvStorage]`-oriented
    /// `KvBlockWriter` without copying tensor data. Behavior-neutral read.
    pub fn storage(&self) -> &KvStorage {
        &self.storage
    }

    /// Set the model-side layer index on this cache.
    ///
    /// Call this in the arch builder loop immediately after `with_quant_max_seq`
    /// or `with_quant_max_seq_window` for every layer that may use a
    /// rotor3/rotor4 KV codec. The layer index seeds distinct per-layer rotor
    /// tables via [`crate::clifford::rotor_seed`]; without it every layer uses
    /// the same table and cross-layer decorrelation is lost.
    ///
    /// Non-rotor codecs ignore this field — setting it is always safe but only
    /// meaningful for `KvQuant::Rotor*` variants.
    ///
    #[must_use]
    pub fn with_layer_idx(mut self, idx: usize) -> Self {
        self.layer_idx = idx;
        self
    }

    /// Model-side layer index that seeds rotor3/rotor4 codec tables.
    ///
    /// Read-only accessor for the field set by [`Self::with_layer_idx`] or by
    /// [`Self::from_storage`] (SSD hydrate). Used by the SSD round-trip
    /// test in `rmlx-kv-ssd` to verify the `write_caches` positional contract.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Rebuild a `KvCache` from a reconstructed [`KvStorage`] (SSD
    /// hydrate). Inverse of [`KvCache::storage`] + `block_io::write_caches`:
    /// the SSD reader reconstructs the per-layer `KvStorage` from a `.kvb`
    /// file, and this wraps each one as a decode-ready `KvCache` whose
    /// `offset`/`quant` match the spilled snapshot.
    ///
    /// `offset` is the filled sequence length recorded in the block header
    /// (`seq_len`). SWA / rotating layers are not spilled (their bf16 ring
    /// lives off-storage and `KvBlockWriter` records them as geometry-only
    /// `None`); a hydrated cache therefore never carries a `RotatingState`.
    /// All other flash / fp16-seed scratch is left `None` — it is lazily
    /// re-seeded on the first decode dispatch exactly as after a cold prefill.
    ///
    /// `layer_idx` is the 0-based model-side layer index. Pass the same index
    /// used when the cache was originally constructed so that any re-quantize
    /// path that fires after hydration uses the correct rotor table seed.
    pub fn from_storage(storage: KvStorage, quant: KvQuant, offset: i32, layer_idx: usize) -> Self {
        Self {
            storage,
            offset,
            quant,
            layer_idx,
            prefill_raw_k: None,
            prefill_raw_v: None,
            in_prefill: false,
            decode_fp16_k: None,
            decode_fp16_v: None,
            rotating: None,
            flash_k_codes: None,
            flash_k_scales: None,
            flash_v_codes: None,
            flash_v_scales: None,
            flash_max_seq: 0,
            flash_filled: 0,
            fused_qk_shadow: None,
        }
    }

    /// Create a new `KvCache` with the given quantization and an explicit max sequence length.
    pub fn with_quant_max_seq(quant: KvQuant, max_seq: i32) -> Self {
        Self {
            storage: KvStorage::new(quant, max_seq),
            offset: 0,
            quant,
            layer_idx: 0,
            prefill_raw_k: None,
            prefill_raw_v: None,
            in_prefill: false,
            decode_fp16_k: None,
            decode_fp16_v: None,
            rotating: None,
            flash_k_codes: None,
            flash_k_scales: None,
            flash_v_codes: None,
            flash_v_scales: None,
            flash_max_seq: 0,
            flash_filled: 0,
            fused_qk_shadow: None,
        }
    }

    /// Construct a KV cache, optionally as a rotating ring buffer for SWA layers.
    ///
    /// When `sliding_window.is_some()` (and window > 0), the returned cache
    /// uses the RotatingKvCache code path (byte-for-byte port of
    /// mlx-lm `RotatingKVCache`) **regardless of the `quant` flag**. mlx-lm's
    /// `RotatingKVCache.to_quantized` raises `NotImplementedError`, so SWA
    /// layers always stay bf16 even when full-attention layers are quantized.
    /// `quant` is recorded on the cache (so an external observer sees the
    /// requested codec) but not used by the rotating update path.
    ///
    /// `sliding_window` is the window size in tokens (e.g. 512 for Gemma4
    /// e2b/e4b, 1024 for medgemma / Gemma4-31B). `max_seq` is ignored on the
    /// rotating path — buffer is sized to `sliding_window`.
    pub fn with_quant_max_seq_window(
        quant: KvQuant,
        max_seq: i32,
        sliding_window: Option<i32>,
    ) -> Self {
        let mut cache = Self::with_quant_max_seq(quant, max_seq);
        if let Some(window) = sliding_window {
            if window > 0 {
                cache.rotating = Some(RotatingState::new(window));
            }
        }
        cache
    }

    /// True if this cache is using the rotating ring-buffer path.
    pub fn is_rotating(&self) -> bool {
        self.rotating.is_some()
    }

    /// True if this cache can be losslessly rolled back via [`truncate_to`].
    ///
    /// Byte-for-byte port of mlx-lm `can_trim_prompt_cache`'s per-cache
    /// predicate (`cache.py`): plain `KVCache` / `QuantizedKVCache` are
    /// always trimmable (`is_trimmable() -> True`); a `RotatingKVCache` is
    /// trimmable only while it has NOT wrapped (`offset < max_size`,
    /// `cache.py:542-543`). Once an SWA ring buffer wraps, the pre-wrap K/V
    /// have been overwritten and `truncate_to` silently no-ops on it
    /// (`rotating::trim_lossless` returns 0) — truncating the other layers
    /// while this one stays full would desync the caches and corrupt the
    /// re-prefilled tail. C1's gemma4 partial-prefix path uses this to gate
    /// the block-truncate fast path: it only fires when EVERY layer cache
    /// reports `is_trimmable()` (mlx-lm `all(c.is_trimmable())`).
    pub fn is_trimmable(&self) -> bool {
        match &self.rotating {
            Some(rot) => rot.offset < rot.max_size,
            None => true,
        }
    }

    /// Current fill offset in tokens (number of tokens appended so far).
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Total allocated sequence length of the KV buffer.
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Returns `true` when this cache holds actual K/V data that reflects the
    /// true sequence position for RoPE base-offset determination.
    ///
    /// Returns `false` for `KvStorage::None` caches — these arise from SWA
    /// (Sliding Window Attention) layers that were serialised to the SSD tier
    /// as tag `"none"` (the rotating ring buffer cannot be spilled). They
    /// carry the block's `seq_len` as `self.offset` for RoPE correctness but
    /// hold no actual quantised K/V payload, so they must be skipped when
    /// choosing which cache to read `base_offset` from — the first
    /// full-attention layer's cache is authoritative.
    pub fn has_persistent_cache(&self) -> bool {
        !matches!(self.storage, KvStorage::None { .. })
    }

    /// Returns `true` when this cache uses the mixed-precision
    /// quantized path. Callers (qwen3 Attention forward) dispatch through
    /// [`KvCache::update_and_sdpa_mixed`] instead of [`KvCache::update`] +
    /// `mx.fast.scaled_dot_product_attention`.
    pub fn is_mixed(&self) -> bool {
        // RotK rides the same Mixed mx.quantize 3-tuple + quantized-SDPA
        // path, so attention forwards must dispatch it through the Mixed route.
        self.quant.uses_mixed_path()
    }

    /// Test-only accessor: borrow the internal `decode_fp16_k` Option.
    ///
    /// Used by regression tests to verify the bf16 K seed is
    /// populated for the warm-TTFT shortcut codecs at `exit_prefill`, and is
    /// **absent** for the K-only codecs (IsoKOnly3/4, RotorKOnly3/4) since
    /// the K-only path gates materialisation on `KvQuant::feeds_bf16_k_at_decode()`
    /// (dead memory otherwise).
    #[cfg(test)]
    pub fn decode_fp16_k_for_test(&self) -> Option<&Array> {
        self.decode_fp16_k.as_ref()
    }

    /// Test-only accessor: borrow the internal `decode_fp16_v` Option.
    ///
    /// Used by tests to assert the bf16 V seed stays live even when the K seed
    /// is dropped for the K-only family (the K-only decode path reads V via
    /// `update_decode_fp16_v_only`).
    #[cfg(test)]
    pub fn decode_fp16_v_for_test(&self) -> Option<&Array> {
        self.decode_fp16_v.as_ref()
    }

    /// Offline-calibration K accessor.
    ///
    /// Returns the bf16 K buffer accumulated during prefill, in the layout
    /// `[1, n_kv_heads, S, head_dim]`. Populated only when the cache ran
    /// through the bf16 path (`KvQuant::None`) or when the Mixed/RotK paths
    /// surfaced the bf16 accumulator via `update_and_sdpa_returning_kv`.
    ///
    /// Intended exclusively for the offline `rmlx kv-calibrate
    /// --recipe head_budget` pass, which needs a host-side view of the
    /// accumulated K tensor to measure per-(layer, head) attention-mass
    /// distribution. Never call this from a hot decode path — the buffer
    /// is not the canonical K representation for the quantised storage
    /// variants and reading it for live SDPA would dequant twice.
    pub fn calibration_k_bf16(&self) -> Option<&Array> {
        self.prefill_raw_k.as_ref().or(self.decode_fp16_k.as_ref())
    }
}
