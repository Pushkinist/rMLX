//! `KvQuant` — closed enum tagging the active KV codec.
//!
//! Moved from `rmlx-models::kv_cache::mod`. The enum, its
//! `Display` / `FromStr` impls, and the `KV_MAX_SEQ_DEFAULT` constant live
//! here because every codec module in this crate (`storage`, `kvcache`,
//! `mixed_quant`) carries `KvQuant` as a tag.
//!
//! Policy wrappers (`KvCacheBuilder`, `kv_quant_for_layer`,
//! `DEFAULT_KV_QUANT`) remain in `rmlx-models::kv_cache`
//! and are re-exported there alongside `pub use rmlx_kv_quant::KvQuant`.

/// Default maximum sequence length for the pre-allocated KV buffer.
///
/// Arch loaders should pass the model's `max_position_embeddings` via
/// `KvCache::with_quant_max_seq`. If absent or >4096, this cap applies.
/// Stage-1/2 smoke prompts are ≤1024 tokens, so 4096 comfortably covers
/// all current development runs.
pub const KV_MAX_SEQ_DEFAULT: i32 = 4096;

/// Quantization mode for the KV cache.
///
/// See [`rmlx_models::kv_cache`] module docs for the Qwen MoE PPL disaster
/// rationale.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed registry enum — KV codec variants; adding a variant requires updating KvCacheBuilder, all dispatch match arms, and CLAUDE.md asymmetric-K/V invariants"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuant {
    /// Asymmetric: K = affine q8_0 (group_size=128), V = TurboQuant 4-bit.
    ///
    /// The recorded safe choice for Qwen MoE when a quantised cache is wanted —
    /// symmetric Q4 on K is the PPL-218->8641 disaster. Opt-in, not automatic.
    /// Per-axis split is real (CLAUDE.md hard rule 5) — K and V are quantized
    /// independently, not by layer index like the Python fork's fake `8,4` flag.
    K8V4,
    /// K = q8_0, V = q8_0 (symmetric 8-bit). Opt-in; `auto` is bf16.
    K8V8,
    /// K = affine q8_0 (group_size=128), V = PlanarQuant 4-bit.
    ///
    /// PlanarQuant applies a Givens-rotation (16-entry codebook) per pair of V
    /// values before 4-bit quantization, using per-pair scales. This reduces
    /// per-element error ~2-3× compared to TurboQuant V4 on Gaussian-distributed
    /// KV vectors. rMLX-first on Apple Silicon (S3.4 — 2026-05).
    ///
    /// Opt-in via `--kv-quant planar`. Not the default for any arch until
    /// baseline comparison is complete.
    Planar,
    /// Unquantised KV cache (bf16 / model dtype, full max_seq buffer).
    ///
    /// Selected by `--kv-quant none` (alias `bf16`) — and by `auto`, which
    /// resolves here on every architecture and every context length
    /// (`DEFAULT_KV_QUANT` in `rmlx-models::kv_cache`). It is also the
    /// baseline the codecs are measured against: no codec in the tree holds
    /// less resident KV than this one.
    ///
    /// Memory cost: one full `[B, kv_h, max_seq, head_dim]` bf16 buffer per
    /// layer for both K and V. ~64 GB at 128k context — reserve for short-ctx
    /// benches only. The 4096-ctx default is fine.
    None,
    /// Mixed-precision KV cache: K and V stored as `mx.quantize`
    /// 3-tuples at independent bit widths and group sizes. Default is
    /// K=8, V=4, group=64 — exact byte-for-byte port of
    /// `mlx_lm.models.mixed_quant_cache.MixedQuantKVCache`.
    ///
    /// Decode SDPA goes through two `mx.quantized_matmul` calls instead of
    /// the bf16-dequantize-then-fast-SDPA pipeline used by `K8V4`/`K8V8`.
    /// Eliminates the per-step full dequantize that dominates the rMLX k8v4
    /// hot path (decode-step audit).
    ///
    /// The codec is arch-agnostic. It is opt-in on every arch — `auto` resolves
    /// to bf16 and never selects it. It was the auto default for
    /// `Qwen3ForCausalLM` (Bonsai-2bit) until the per-arch table was retired.
    Mixed {
        /// K quantization bit-width.
        k_bits: u8,
        /// V quantization bit-width.
        v_bits: u8,
        /// K affine group size.
        k_group_size: u16,
        /// V affine group size.
        v_group_size: u16,
    },
    /// K-side rotation codec: K is affine-quantized at 8-bit/group=64 in
    /// a **rotated basis** (Hadamard `R`), V is affine `v_bits`/`v_group_size`.
    ///
    /// Storage and SDPA reuse the [`Mixed`](KvQuant::Mixed) machinery: the only
    /// additions are (1) rotate K by `R` before `mx.quantize` and (2) pre-rotate
    /// Q by the same `R` before the score matmul, so the rotations cancel
    /// (`(Q Rᵀ)(K Rᵀ)ᵀ = Q Kᵀ`) and K is never inverse-rotated. See
    /// [`crate::rot_k`] for the math and `docs/KV_CACHE.md` §5.3.
    ///
    /// K stays 8-bit because rotation is a PPL win *over* affine-K quantization,
    /// not a way to drop K below 8-bit (rotation does not rescue 2-bit K — see
    /// hard rule 6). Opt-in **only** via `--ctk rot_k`; never an auto default.
    RotK {
        /// V quantization bit-width (K is always 8-bit rotated affine).
        /// Validated by [`validate_mixed_side`] — the V slot is `Mixed`'s.
        v_bits: u8,
        /// V affine group size. Validated by [`validate_mixed_side`].
        v_group_size: u16,
    },
    /// K = affine q8_0 (group_size=128),
    /// V = TurboQuant **3-bit** Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Opt-in on every arch — `auto` resolves to bf16 and never selects it. It
    /// was the auto default for Gemma4 small (non-MoE, `hidden_size` ≤ 2560,
    /// non-paroquant) until the per-arch table was retired.
    /// Validated on Gemma4-e4b: −0.40% vs K8V4 at canary shape (within the
    /// <1% promote gate). Cosine gate ≥ 0.9807 passes.
    ///
    /// Implementation scope: **CPU dequant only** — no MSL kernel for 3-bit
    /// TurboQuant (deferred). The GPU path falls through to dequant-then-SDPA,
    /// same as K8V4 without TurboFlash.
    ///
    /// Also available via `--kv-quant k8vturbo3` on any arch.
    K8VTurbo3,
    /// Symmetric TurboQuant 3-bit — K = `QuantKTurbo3`,
    /// V = `QuantV` (bits=3). Both axes use the **Lloyd-Max N(0,1) 3-bit
    /// codebook** (axis-agnostic CPU + same MSL kernel as the V-side
    /// `K8VTurbo3` path). Maps to the `speed` mtq preset.
    ///
    /// **Arch guard (Contract A.y — mandatory)**: must NEVER run on Qwen MoE
    /// (`Qwen3_5MoeForConditionalGeneration`). Symmetric 3-bit K is the
    /// PPL-disaster path on Qwen MoE (7:1 GQA amplifies K-head error through
    /// softmax; see the 218->8641 baseline in `docs/KV_QUANT.md`). The auto default
    /// never returns `TurboSym3` for Qwen MoE; explicit `--kv-quant tsym3` on
    /// Qwen MoE is rejected at resolve-time via `QwenMoeTurboKRejected`.
    ///
    /// **V-side GPU dispatch**: V side forced CPU path (same as asymmetric
    /// `K8VTurbo3` precedent — GPU V-side dispatch regressed −2% TPS gate on
    /// K8VTurbo3). K side uses the GPU turbo3 MSL kernel when `Device::Gpu`.
    ///
    /// Opt-in via `--kv-quant tsym3`. Also auto-selected by `--kv-preset speed`.
    TurboSym3,
    /// Symmetric TurboQuant 4-bit: K = `QuantKTurbo4`,
    /// V = `QuantV` (bits=4). Both axes use the **Lloyd-Max N(0,1) 4-bit
    /// codebook** (axis-agnostic CPU + MSL kernel from the existing V-side
    /// `K8V4` path). Closes the asymmetric K8V4 gap for the `quality` /
    /// `agents_*` mtq presets.
    ///
    /// **Arch guard (CLAUDE.md mandate)**: must NEVER run on Qwen MoE
    /// (`Qwen3_5MoeForConditionalGeneration`). Symmetric 4-bit K is the
    /// PPL-218→8641 disaster path. `--kv-quant tsym4` on Qwen MoE is rejected
    /// at resolve-time by `rmlx_models::kv_cache::validate_resolved`; this
    /// variant is **never** an auto baseline.
    ///
    /// Opt-in only via `--kv-quant tsym4` (or `--kv-preset quality`).
    TurboSym4,
    /// K = affine q8_0 (group_size=128), V = PlanarQuant **3-bit**.
    ///
    /// 3.25-bit V codec extending the existing `Planar` (4.25-bit) codec downward.
    /// Same Givens-rotation + per-pair scale algorithm; 3-bit Lloyd-Max N(0,1) codebook
    /// (8 centroids). Pack format: 10 vals/u32 (3 × 10 = 30 bits, 2 wasted per u32) —
    /// ForgeAttention-compatible, same u32 word count as 4-bit (ceil(32/10) = 4 = 4).
    ///
    /// Storage: routes to `KvStorage::Planar { bits: 3 }` — no new `KvStorage` variant.
    /// CPU path: scalar `planar_quantize(bits=3)`. GPU: `planar_quantize_v3_gpu`.
    /// Cosine gate: measured on LCG fixture (see `planarquant_tests.rs`).
    ///
    /// Opt-in via `--kv-quant planar3`. Never an auto default.
    Planar3,
    /// K-axis PlanarQuant 4-bit (mtq `k_only_planar` preset).
    ///
    /// Opposite of [`Planar`](KvQuant::Planar): Givens-rotation 4-bit codec on
    /// the **K** axis; V stays unquantised (bf16, lives on
    /// `KvCache::decode_fp16_v`).
    ///
    /// **Arch guard (Contract A.y — mandatory)**: K-side 4-bit on Qwen MoE is
    /// the PPL-disaster (218→8641; 7:1 GQA amplifies K-head error through
    /// softmax). The auto default never returns `PlanarK` for
    /// `Qwen3_5MoeForConditionalGeneration` / `Qwen3VLMoeForConditionalGeneration`,
    /// and `cache_type::validate_resolved` rejects it. Opt-in only via
    /// `--kv-quant planar_k`. Requires `head_dim % 32 == 0`. MSL kernel is
    /// shared with `Planar` (PlanarQuant is axis-agnostic). See `docs/KV_QUANT.md`.
    PlanarK,
    /// K = affine q8_0 (group_size=128), V = TurboQuant **2-bit**
    /// Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Native 2.25-bit V codec mirroring multi-turboquant's `turbo2`
    /// (~7× compression). rMLX's only sub-3-bit native V codec — the existing
    /// `Mixed{v_bits:2}` path is MLX affine 2-bit (different algorithm).
    ///
    /// Ships **naïve** Lloyd-Max 2-bit (no outlier-mask). The
    /// outlier-mask machinery is non-trivial; deferred pending the calibration
    /// loader. Expect a cosine drop vs the published mtq number due to the
    /// missing outlier handling.
    ///
    /// Implementation scope: structurally identical to [`K8VTurbo3`](KvQuant::K8VTurbo3)
    /// but with `bits=2` in the [`QuantV`](crate::storage::QuantV) slot. CPU
    /// dequant only — the MSL kernel (`turbo2_v_msl.rs`) is wired as a
    /// future-reference hook (parity-tested CPU↔GPU but not dispatched on the
    /// hot update path).
    ///
    /// Opt-in **only** via `--kv-quant k8vturbo2`; never an auto default.
    K8VTurbo2,
    /// K = affine q8_0 (group_size=128), V = IsoQuant 3-bit
    /// (quaternion SO(4) rotation + Lloyd-Max codebook).
    ///
    /// V-side rotation codec from the IsoQuant family. The fixed golden-ratio
    /// quaternion (`FIXED_QUAT`) is applied per group of 4 elements before 3-bit
    /// Lloyd-Max quantization — see [`crate::isoquant`] for the algorithm.
    ///
    /// **Implementation scope**: CPU codec only. SDPA falls through to the
    /// dequant-then-SDPA legacy fallback path. Opt-in only via
    /// `--kv-quant iso3`. Requires `head_dim % 4 == 0` (quaternion block
    /// alignment).
    Iso3,
    /// K = affine q8_0 (group_size=128), V = IsoQuant 4-bit
    /// (quaternion SO(4) rotation + Lloyd-Max 4-bit codebook).
    ///
    /// 4.25-bit V codec — same quaternion rotation as [`Iso3`](KvQuant::Iso3)
    /// with the 16-centroid Lloyd-Max N(0,1) codebook and dense 8-vals-per-u32
    /// pack (vs iso3's 10 vals/u32). Higher fidelity than iso3 at the cost of
    /// one extra bit per value.
    ///
    /// **Implementation scope**: CPU codec only. SDPA falls through to the
    /// dequant-then-SDPA legacy fallback path. The existing iso3 MSL kernel is
    /// hard-coded for `bits=3`; an iso4 MSL variant is deferred. Opt-in only
    /// via `--kv-quant iso4`. Requires `head_dim % 4 == 0` (quaternion block
    /// alignment).
    Iso4,
    /// Symmetric IsoQuant 3-bit — K = IsoQuant 3-bit, V = IsoQuant 3-bit
    /// (axis-agnostic quaternion SO(4) + 3-bit Lloyd-Max codebook).
    ///
    /// Mirrors the existing V-only [`Iso3`](KvQuant::Iso3) on both axes.
    ///
    /// **Arch guard (Contract A.y — mandatory)**: K-side ≤4-bit on Qwen MoE
    /// is the PPL-disaster zone (218→8641 on Q4_K_M baseline; 7:1 GQA
    /// amplifies K-head error through softmax). The auto default
    /// never returns `Iso3Sym` for Qwen MoE; explicit `--kv-quant iso3_sym`
    /// on Qwen MoE is rejected at resolve-time by
    /// `rmlx_models::kv_cache::validate_resolved`. Opt-in only via
    /// `--kv-quant iso3_sym`. Requires `head_dim % 4 == 0`.
    Iso3Sym,
    /// Symmetric IsoQuant 4-bit — same arch guard rationale as
    /// [`Iso3Sym`](KvQuant::Iso3Sym); K-side 4-bit on Qwen MoE is the
    /// PPL-disaster path. Opt-in only via `--kv-quant iso4_sym`.
    Iso4Sym,
    /// K-only IsoQuant 3-bit — K = IsoQuant 3-bit; V stays bf16
    /// (lives on `KvCache::decode_fp16_v`, same machinery as
    /// [`PlanarK`](KvQuant::PlanarK)).
    ///
    /// **Arch guard (Contract A.y — mandatory)**: rejected on Qwen MoE for
    /// the same reason as [`Iso3Sym`](KvQuant::Iso3Sym). Opt-in only via
    /// `--kv-quant k_iso3`. Requires `head_dim % 4 == 0`.
    IsoKOnly3,
    /// K-only IsoQuant 4-bit — same shape as
    /// [`IsoKOnly3`](KvQuant::IsoKOnly3) with the dense 4-bit codebook on K.
    /// V is bf16. Opt-in only via `--kv-quant k_iso4`. Arch-guarded against
    /// Qwen MoE.
    IsoKOnly4,
    /// K = affine q8_0 (group_size=128), V = rotor3 (Cl(3,0)
    /// Clifford rotor sandwich + 3-bit Lloyd-Max codebook).
    ///
    /// First rotation-V codec from the Clifford algebra family. Each
    /// V-vector is embedded into Cl(3,0) (8-element multivector) in groups
    /// of 3, sandwiched by a **static per-(layer, head)** rotor table
    /// (loaded once, not per token), and 3-bit-quantised against the
    /// Lloyd-Max N(0,1) codebook. Pack format: 10 vals/u32 (planar3 / iso3
    /// convention).
    ///
    /// **Implementation scope**: CPU codec only. SDPA falls through to the
    /// dequant-then-SDPA legacy fallback path. No MSL kernel (deferred).
    /// Single-codebook simplification: 8 multivector components share one
    /// codebook (Python reference splits into vector vs trivector grades; rMLX
    /// defers grade-aware). No QJL residual (K-only stage, out of scope for V
    /// codec).
    ///
    /// Opt-in only via `--kv-quant rotor3` (alias `rotor_v_3`). Requires
    /// `head_dim > 0`; tail-padded for `head_dim % 3 != 0`.
    Rotor3,
    /// K = affine q8_0 (group_size=128), V = rotor4 (Cl(3,0)
    /// Clifford rotor sandwich + 4-bit Lloyd-Max codebook).
    ///
    /// 4.25-bit V codec — same Clifford Cl(3,0) sandwich as rotor3 with the
    /// 16-centroid Lloyd-Max N(0,1) codebook and dense 8-vals-per-u32 pack
    /// (iso4 convention: 8 codes × 4 bits = 32 bits = 1 u32 per group).
    /// ~10.7 bpe in codes at bits=4 (single-codebook simplification; grade-aware
    /// split deferred per spec) — but the per-group `f32` scale sits beside the
    /// `u32` code word, so **21.75 bits per value reach the store**, above
    /// bf16's 16.0. Not a memory win; see `crate::rotorquant` § "Effective bpe".
    ///
    /// **Implementation scope**: CPU codec only. SDPA falls through to the
    /// dequant-then-SDPA legacy fallback path. No MSL kernel (deferred). Single-
    /// codebook: all 8 multivector components share the 16-centroid codebook
    /// (grade-aware deferred). No QJL residual.
    ///
    /// Opt-in only via `--kv-quant rotor4` (alias `rotor_v_4`). Requires
    /// `head_dim > 0`; tail-padded for `head_dim % 3 != 0`.
    Rotor4,
    /// K = affine q8_0 (group_size=128), V = TurboQuant **3-bit**
    /// with **Viterbi trellis (TCQ) assignment** over the same Lloyd-Max N(0,1)
    /// codebook as [`K8VTurbo3`](KvQuant::K8VTurbo3).
    ///
    /// 3.25-bit V codec. Same on-disk layout and packing as `K8VTurbo3` — only
    /// the encode-time assignment differs (Viterbi-optimal path through a
    /// 4-state trellis instead of nearest-centroid). The decoder is bit-for-bit
    /// identical to plain `turbo_dequantize`, so spill/hydrate, SSD layout, and
    /// the warm-TTFT fp16 seed all share the K8VTurbo3 machinery — only the
    /// `KvQuant` discriminator and SSD layout-key tag differ.
    ///
    /// Cosine target ≥ 0.9807 (mtq turbo3_tcq row 0.9817 − 0.001 empirical
    /// floor). Pairs with the `turbo3_tcq` calibration recipe (`recipe_to_internal`
    /// → `turboquant35` — same `high_precision_indices` as `turbo3`; no
    /// codebook override since TCQ reuses the standard Lloyd-Max codebook).
    ///
    /// **Implementation scope**: CPU Viterbi encode + CPU dequant on the hot
    /// path; the MSL Viterbi kernel ([`crate::tcq_v_msl`](crate::tcq_v_msl))
    /// ships as a future-reference hook (K8VTurbo3 / K8VTurbo2 MSL hooks both
    /// regressed the −2 % TPS gate when dispatched on the hot path).
    ///
    /// Opt-in **only** via `--kv-quant k8vturbo3tcq`; never an auto default.
    K8VTurbo3Tcq,
    /// Symmetric rotor3 — K = rotor3, V = rotor3 (Cl(3,0)
    /// Clifford rotor sandwich + 3-bit Lloyd-Max codebook on both axes).
    /// K-side carries an optional 1-bit QJL residual sideband when
    /// [`crate::rotor_qjl::rotor_qjl_enabled`] is `true` at first append —
    /// **off by default**, because the sideband has no MSL kernel and forces the
    /// whole rotor K path onto the CPU (see [`crate::rotor_qjl`]).
    ///
    /// **Arch guard (Contract A.y — mandatory)**: K-side ≤4-bit on Qwen MoE
    /// is the PPL-disaster zone (218→8641 on Q4_K_M baseline; 7:1 GQA
    /// amplifies K-head error through softmax). The auto default
    /// never returns `Rotor3Sym` for Qwen MoE; explicit `--kv-quant rotor3_sym`
    /// on Qwen MoE is rejected at resolve-time. Opt-in only via
    /// `--kv-quant rotor3_sym`. Requires `head_dim > 0`.
    Rotor3Sym,
    /// Symmetric rotor4 — same arch guard rationale as
    /// [`Rotor3Sym`](KvQuant::Rotor3Sym) with the 16-centroid 4-bit Lloyd-Max
    /// codebook on both axes. Opt-in only via `--kv-quant rotor4_sym`.
    Rotor4Sym,
    /// K-only rotor3 — K is rotor3 (Cl(3,0) Clifford rotor
    /// sandwich + 3-bit Lloyd-Max codebook + optional 1-bit QJL residual);
    /// V stays bf16 (lives on `KvCache::decode_fp16_v`, same machinery as
    /// [`PlanarK`](KvQuant::PlanarK) / [`IsoKOnly3`](KvQuant::IsoKOnly3)).
    ///
    /// **Arch guard (Contract A.y — mandatory)**: rejected on Qwen MoE for
    /// the same reason as [`Rotor3Sym`](KvQuant::Rotor3Sym). Opt-in only via
    /// `--kv-quant k_rotor3`. Requires `head_dim > 0`.
    RotorKOnly3,
    /// K-only rotor4 — same shape as
    /// [`RotorKOnly3`](KvQuant::RotorKOnly3) with the dense 4-bit codebook
    /// on K. Opt-in only via `--kv-quant k_rotor4`. Arch-guarded against
    /// Qwen MoE.
    RotorKOnly4,
    /// Asymmetric rotor3 K + affine V — K is rotor3 (Cl(3,0)
    /// Clifford rotor sandwich + 3-bit Lloyd-Max codebook, optional QJL
    /// residual); V is MLX-affine `v_bits` / `v_group_size`.
    ///
    /// Closes the gap between [`Rotor3Sym`](KvQuant::Rotor3Sym) (rotor V) and
    /// [`RotorKOnly3`](KvQuant::RotorKOnly3) (bf16 V): any of the standard MLX
    /// affine V codecs (Q8G128, Q8G64, Q4G128, Q4G64, Q3G64, Q2G64) can be
    /// paired with rotor3 K. Storage routes to
    /// [`KvStorage::RotorKAsym3`](crate::storage::KvStorage::RotorKAsym3); SDPA
    /// reuses the [`RotorKOnly3`](KvQuant::RotorKOnly3) dequant-then-SDPA path
    /// for K and the existing affine V encode/decode for V.
    ///
    /// **Arch guard (Contract A.y — mandatory)**: rejected on Qwen MoE for
    /// the same reason as [`Rotor3Sym`](KvQuant::Rotor3Sym). Opt-in only via
    /// the compose-form `--ctk rotor3 --ctv q{v_bits}_g{v_group_size}` (e.g.
    /// `--ctv q8_g128`). Display form: `rotor_k_3_asym_v{v_bits}_g{v_group_size}`.
    RotorK3Asym {
        /// V quantization bit-width (affine).
        v_bits: u8,
        /// V affine group size.
        v_group_size: u16,
    },
    /// Asymmetric rotor4 K + affine V — same shape as
    /// [`RotorK3Asym`](KvQuant::RotorK3Asym) with the 16-centroid 4-bit rotor
    /// codebook on K. Opt-in only via `--ctk rotor4 --ctv q{v_bits}_g{v_group_size}`.
    /// Arch-guarded against Qwen MoE.
    RotorK4Asym {
        /// V quantization bit-width (affine).
        v_bits: u8,
        /// V affine group size.
        v_group_size: u16,
    },
    /// K = affine q8_0 (group_size=128), V = TurboQuant **2-bit**
    /// with **Viterbi trellis (TCQ) assignment** over the same Lloyd-Max N(0,1)
    /// 2-bit codebook as [`K8VTurbo2`](KvQuant::K8VTurbo2) (4 centroids).
    ///
    /// 2.25-bit V codec. Same on-disk layout and packing as `K8VTurbo2` — only
    /// the encode-time assignment differs (Viterbi-optimal path through a
    /// 4-state trellis instead of nearest-centroid). The decoder is bit-for-bit
    /// identical to plain `turbo_dequantize` at 2-bit. The trellis state count
    /// stays at `TCQ_NUM_STATES = 4` (independent of bit-width).
    ///
    /// Cosine gate is empirical (measured − 0.001). Pairs with the `turbo2_tcq`
    /// calibration recipe (`recipe_to_internal` → `turboquant25` — same
    /// `high_precision_indices` as `turbo2`; no codebook override needed since
    /// TCQ reuses the standard Lloyd-Max codebook). Maps to the `max_compression`
    /// preset in `multi-turboquant/presets.py`.
    ///
    /// **Outlier-mask decision**: `high_precision_indices` attachment is
    /// deferred. This port ships naïve (mirrors the plain turbo2 path with no
    /// outlier masking). The calibration surface (`QuantV::high_precision_indices`)
    /// is already present in `QuantV`; wiring it for 2-bit TCQ is a follow-up.
    ///
    /// **Implementation scope**: CPU Viterbi encode + CPU dequant on the hot
    /// path. A GPU Viterbi kernel is not currently wired — the previous
    /// parked hook had no production dispatch caller and was removed.
    ///
    /// Opt-in **only** via `--kv-quant k8vturbo2tcq`; never an auto default.
    K8VTurbo2Tcq,
}

/// Every [`KvQuant`] variant, once each, with representative parameters for the
/// four that carry fields.
///
/// It lives here, beside the enum, on purpose: a test that needs to sweep the
/// codec surface can only be exhaustive if the list it sweeps breaks in the
/// same file the variant was added to. A list kept in a test crate silently
/// stops covering the newest codec, which is the shape of gate this repo has
/// shipped before.
///
/// `variant_index_has_one_arm_per_listed_codec` pins this list against
/// [`KvQuant::variant_index`], a `match` the compiler checks for exhaustiveness,
/// so a variant added to the enum and not added here fails there. It counts
/// that match's arms out of this file's source, because a variant missing from
/// this list can be constructed nowhere in the crate and so is invisible to
/// every test that sweeps it.
pub const ALL_KV_QUANTS: &[KvQuant] = &[
    KvQuant::None,
    KvQuant::K8V4,
    KvQuant::K8V8,
    KvQuant::Planar,
    KvQuant::Planar3,
    KvQuant::PlanarK,
    KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    },
    KvQuant::RotK {
        v_bits: 8,
        v_group_size: 64,
    },
    KvQuant::K8VTurbo3,
    KvQuant::K8VTurbo3Tcq,
    KvQuant::K8VTurbo2,
    KvQuant::K8VTurbo2Tcq,
    KvQuant::TurboSym3,
    KvQuant::TurboSym4,
    KvQuant::Iso3,
    KvQuant::Iso4,
    KvQuant::Iso3Sym,
    KvQuant::Iso4Sym,
    KvQuant::IsoKOnly3,
    KvQuant::IsoKOnly4,
    KvQuant::Rotor3,
    KvQuant::Rotor4,
    KvQuant::Rotor3Sym,
    KvQuant::Rotor4Sym,
    KvQuant::RotorKOnly3,
    KvQuant::RotorKOnly4,
    // `validate_rotor_k_asym_v` accepts (4, 128|64|32) and (3|2, 64) only.
    KvQuant::RotorK3Asym {
        v_bits: 4,
        v_group_size: 64,
    },
    KvQuant::RotorK4Asym {
        v_bits: 4,
        v_group_size: 64,
    },
];

/// The packed-store layout one axis of a codec writes.
///
/// This is what [`KvQuant::estimated_resident_bytes_per_layer`] sizes a side
/// from — the store's own group geometry, never the codebook width reported by
/// [`KvQuant::approx_code_bits`]. Two members of the same family at different
/// bit widths can (and for planar, iso and rotor do) occupy byte-identical
/// storage, so the width is not what sets the rate.
///
/// Every variant models its store **byte for byte**;
/// `every_codec_byte_model_matches_the_store_it_writes` asserts the equality
/// against bytes that store's own encoder produced. There is no rounding term
/// and no deliberate over-charge to remember.
///
/// **Only three of the seven are reachable from the estimator.** It sizes a
/// side only when [`KvQuant::materialises_packed_store`] holds, which is true
/// for exactly ten codecs (`Mixed`, `RotK`, `IsoKOnly3/4`, `Iso3Sym/4Sym`,
/// `RotorKOnly3/4`, `Rotor3Sym/4Sym`), and their sides name only [`Self::Affine`],
/// [`Self::IsoRing`] and [`Self::Rotor`]. [`Self::Q8`], [`Self::Turbo`],
/// [`Self::Planar`] and [`Self::IsoBlocks`] are **latent**: they are the true
/// layout of the side that names them, but every codec that names one decodes
/// off the bf16 mirror and so builds no store for the estimator to size. The
/// cadence test is the only caller that reaches them, which is why
/// [`packed_side_bytes`] is a free fn rather than inlined into the estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SideStore {
    /// q8_0 K (`crate::storage::QuantK`): one `u8` code per value plus one
    /// `f32` scale per [`crate::q8::Q8_GROUP_SIZE`]-element group — 8.25 bits
    /// per value. Latent (see the type doc): every codec with a q8_0 side
    /// decodes off the mirror.
    Q8,
    /// TurboQuant / TCQ V (`crate::storage::QuantV`): `bits` code bits per
    /// value plus one `f32` scale per [`crate::turboquant::GROUP_SIZE`]-element
    /// group — `bits + 1.0` bits per value. The one family whose stored width
    /// and codebook width agree. Latent.
    Turbo,
    /// MLX affine 3-tuple (`Mixed` / `RotK`, `crate::mixed_quant::MixedTuple`):
    /// `bits` code bits per value plus a scale **and** a bias per `group`, each
    /// at the KV stream's dtype.
    ///
    /// **The sideband is 32 bits per group, measured.** `mx.quantize(mode =
    /// "affine")` emits `scales` and `biases` at the input dtype, and the KV
    /// stream reaching the store is bf16: `cast_store_bf16` floors the prefill
    /// buffer that `exit_prefill` bulk-quantizes (`kvcache/update.rs`), so a
    /// 2-byte scalar each is what the store holds on every shipped model.
    /// `affine_sideband_is_thirty_two_bits_per_group` reads the figure off a
    /// real `MixedTuple` rather than restating it.
    ///
    /// So this is `bits + 32/group` bits per value: 8.5 at group 64, 9.0 at
    /// group 32, 8.25 at group 128 for an 8-bit side. The one arm the estimator
    /// evaluates in production.
    Affine {
        /// Affine group size of this side, as the codec carries it. The store's
        /// own parameter — `mx.quantize` emits one scale and one bias per group
        /// of this many values.
        group: u32,
    },
    /// PlanarQuant K or V (`QuantPlanarK` / `QuantPlanarV`): 4 `u32` code words
    /// per 32-element group, one `f32` scale **per pair**, and 2 `u32` rotation
    /// words per 32-element group.
    ///
    /// **22.00 bits per value at every `head_dim` and at both bit widths** —
    /// above bf16's 16.0. The per-pair scale alone is 16 bits per value, a whole
    /// bf16 value's worth of sideband before a single code bit is spent, so the
    /// 3-bit and 4-bit members occupy byte-identical storage (the 3-bit pack is
    /// 10 vals/`u32`, `ceil(32/10) = 4` words — the same word count as 4-bit's
    /// 8 vals/`u32`). Reading it as codes plus one `f32` per 32 values would
    /// give 5.0 bits for the 4-bit members and 4.0 for `Planar3` — an
    /// understatement of 4.4× and 5.5× respectively, in the codec's favour.
    /// Latent.
    Planar,
    /// IsoQuant on the GPU ring (`QuantKGpuRing`): one `u32` code word + one
    /// `f32` scale per 4-element quaternion group, plus one `f32` norm per
    /// token. `16 + 32/head_dim` bits per value — 16.25 at `head_dim = 128`,
    /// approaching bf16's 16.0 from above and never reaching it.
    ///
    /// The rotation is the compile-time `crate::isoquant::FIXED_QUAT` and is not
    /// stored. This is the resident form for every iso codec that materialises a
    /// store: the fused-decode append seeds the ring from the prefill CPU blocks
    /// and then drops them, so the ring is the sole resident copy from the first
    /// fused decode step onward.
    ///
    /// Two shapes hold the blocks instead and cost 2.97× this, so the estimate
    /// runs low for them. Both are observable:
    ///
    /// * **transient** — the window between `exit_prefill`, which bulk-encodes
    ///   on the CPU, and the first fused decode step, which drops the blocks;
    /// * **permanent** — a layer whose shape the flash dispatcher's gate
    ///   rejects (batch > 1, or a `head_dim` that is not a power of two at most
    ///   512 — `head_dim = 80` is enough). The fused append never runs, the ring
    ///   is never allocated, and the block-dropping step is a no-op, so the
    ///   blocks are what that layer holds for the whole request. This one is a
    ///   property of the model's geometry, not a startup window: it does not
    ///   end.
    IsoRing,
    /// IsoQuant in CPU `IsoBlocks`: [`SideStore::IsoRing`] plus a 4×`f32`
    /// quaternion per group — 48.25 bits per value at `head_dim = 128`, 2.97×
    /// the ring.
    ///
    /// The quaternion is the constant `FIXED_QUAT` replicated per group, not
    /// data. This form is what a codec with no ring path would hold — which is
    /// what `Iso3` / `Iso4` name, and why they are latent: their decode
    /// early-returns to the bf16 mirror, so no store is built at all.
    IsoBlocks,
    /// RotorQuant: one `u32` code word + one `f32` scale per 3-element group,
    /// plus one `f32` norm per token — `(64 * ceil(head_dim/3) + 32) / head_dim`
    /// bits per value, 21.75 at `head_dim = 128`.
    ///
    /// The ring and the CPU blocks carry the same payload here (rotor has no
    /// quaternion analogue), so one layout covers both. The static
    /// per-(layer, head) rotor table is not per-token and is omitted — estimate,
    /// not census.
    Rotor,
}

/// Bytes the MLX affine 3-tuple spends per group on one side: a scale and a
/// bias, each at the KV stream's dtype.
///
/// bf16 on every shipped model — the store boundary floors it there
/// (`cast_store_bf16`). An f32 KV stream would double this, and at `group == 32`
/// (which `validate_mixed_side` accepts) that is the one configuration where
/// this figure would under-count the store rather than match it; the floor is
/// what rules it out.
const AFFINE_SIDEBAND_BYTES_PER_GROUP: u64 = 4;

/// Bytes one side's packed store holds for `elems` values laid out as
/// `n_tokens` rows of `head_dim`.
///
/// Split out of [`KvQuant::estimated_resident_bytes_per_layer`] so the cadence
/// of a store no live codec materialises yet is still reachable from a test:
/// four of the seven layouts are latent (see [`SideStore`]), and a cadence only
/// the estimator could call would be a gate that cannot fail.
fn packed_side_bytes(store: SideStore, bits: u32, elems: u64, head_dim: u64, n_tokens: u64) -> u64 {
    let codes = elems.saturating_mul(u64::from(bits)) / 8;
    match store {
        SideStore::Q8 => {
            let scales = (elems / crate::q8::Q8_GROUP_SIZE as u64).saturating_mul(4);
            codes.saturating_add(scales)
        }
        SideStore::Turbo => {
            let scales = (elems / crate::turboquant::GROUP_SIZE as u64).saturating_mul(4);
            codes.saturating_add(scales)
        }
        SideStore::Affine { group } => {
            let sideband =
                (elems / u64::from(group)).saturating_mul(AFFINE_SIDEBAND_BYTES_PER_GROUP);
            codes.saturating_add(sideband)
        }
        SideStore::Planar => {
            // 4 u32 codes + 2 u32 rotation words per 32-element group, and one
            // f32 scale per pair. Independent of `bits` — see `SideStore::Planar`.
            let groups = elems / 32;
            let planar_codes = groups.saturating_mul(4 * 4);
            let rotations = groups.saturating_mul(2 * 4);
            let scales = (elems / 2).saturating_mul(4);
            planar_codes
                .saturating_add(scales)
                .saturating_add(rotations)
        }
        SideStore::IsoRing => {
            // code u32 + scale f32; the rotation is FIXED_QUAT.
            let groups = elems / 4;
            groups
                .saturating_mul(4 + 4)
                .saturating_add(n_tokens.saturating_mul(4))
        }
        SideStore::IsoBlocks => {
            // The ring's code u32 + scale f32, plus the 4x f32 quaternion the
            // CPU blocks replicate per group.
            let groups = elems / 4;
            groups
                .saturating_mul(4 + 4 + 16)
                .saturating_add(n_tokens.saturating_mul(4))
        }
        SideStore::Rotor => {
            // group size 3: per-token head_dim.div_ceil(3), NOT elems/3.
            let groups = head_dim.div_ceil(3).saturating_mul(n_tokens);
            groups
                .saturating_mul(4 + 4)
                .saturating_add(n_tokens.saturating_mul(4))
        }
    }
}

impl KvQuant {
    /// Discriminant index, used only to prove [`ALL_KV_QUANTS`] names every
    /// variant. The `match` is exhaustive, so a new variant fails to compile
    /// here, and the distinct indices make the list's coverage checkable.
    #[must_use]
    pub fn variant_index(&self) -> usize {
        match self {
            KvQuant::None => 0,
            KvQuant::K8V4 => 1,
            KvQuant::K8V8 => 2,
            KvQuant::Planar => 3,
            KvQuant::Planar3 => 4,
            KvQuant::PlanarK => 5,
            KvQuant::Mixed { .. } => 6,
            KvQuant::RotK { .. } => 7,
            KvQuant::K8VTurbo3 => 8,
            KvQuant::K8VTurbo3Tcq => 9,
            KvQuant::K8VTurbo2 => 10,
            KvQuant::K8VTurbo2Tcq => 11,
            KvQuant::TurboSym3 => 12,
            KvQuant::TurboSym4 => 13,
            KvQuant::Iso3 => 14,
            KvQuant::Iso4 => 15,
            KvQuant::Iso3Sym => 16,
            KvQuant::Iso4Sym => 17,
            KvQuant::IsoKOnly3 => 18,
            KvQuant::IsoKOnly4 => 19,
            KvQuant::Rotor3 => 20,
            KvQuant::Rotor4 => 21,
            KvQuant::Rotor3Sym => 22,
            KvQuant::Rotor4Sym => 23,
            KvQuant::RotorKOnly3 => 24,
            KvQuant::RotorKOnly4 => 25,
            KvQuant::RotorK3Asym { .. } => 26,
            KvQuant::RotorK4Asym { .. } => 27,
        }
    }
}

impl KvQuant {
    /// Stable per-codec salt for namespacing the in-RAM prompt/prefix cache
    /// key by KV codec.
    ///
    /// A single resident model can serve requests under different KV codecs
    /// (hot-swap, no weight reload). The cached K/V bytes are codec-specific —
    /// a prefix cached under `None` (bf16) must NOT serve a `K8V4` request. The
    /// prompt-cache block-hash chain is salted with this value (XOR'd into the
    /// FNV seed, the same mixing the SSD `layout_key` uses), so two requests
    /// with the same tokens but different codecs produce disjoint digest
    /// streams and occupy distinct cache slots — no cross-codec serve.
    ///
    /// The salt is a deterministic FNV-1a-64 hash over the codec's canonical
    /// [`Display`](std::fmt::Display) string (`"none"`, `"k8v4"`,
    /// `"mixed_k8g64_v4g64"`, …), so it is stable across runs and covers every
    /// variant — including payload-bearing ones (`Mixed`, `RotK`, `RotorK*Asym`)
    /// whose payload is part of the Display form. Two codecs render to the same
    /// string iff they are byte-identical, so the salt is collision-free across
    /// the enum.
    #[must_use]
    pub fn cache_key_salt(&self) -> u64 {
        // FNV-1a-64 over the canonical Display bytes. Constants are the
        // standard offset basis / prime — self-contained here because
        // `rmlx-kv-quant` does not (and must not) depend on `rmlx-kv-ssd`,
        // where the prompt-cache FNV constants live. The *values* match by
        // construction (same standard FNV-1a-64 constants).
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let label = self.to_string();
        let mut h = FNV_OFFSET;
        for byte in label.as_bytes() {
            h ^= u64::from(*byte);
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    /// True for the Mixed-machinery hot path (Mixed + RotK), which dispatches
    /// through `mx.quantize` 3-tuples + `mixed_quantized_sdpa`.
    pub fn uses_mixed_path(&self) -> bool {
        matches!(self, KvQuant::Mixed { .. } | KvQuant::RotK { .. })
    }

    /// True for KV codecs that store K below 8 bits. Sub-8-bit K is
    /// catastrophic on high-GQA architectures (measured: Qwen MoE PPL
    /// 218 → 8641). `validate_resolved` in `rmlx-models` decides which
    /// archs reject these codecs — this predicate only names the codec
    /// property. `K8V4` / `K8V8` / `Planar` / `K8VTurbo3` keep K at
    /// 8-bit and are NOT included.
    ///
    /// `PlanarK` is guarded separately (dedicated resolve error) before
    /// this check runs.
    pub fn k_below_8bit(&self) -> bool {
        matches!(
            self,
            KvQuant::TurboSym3
                | KvQuant::TurboSym4
                | KvQuant::Iso3Sym
                | KvQuant::Iso4Sym
                | KvQuant::IsoKOnly3
                | KvQuant::IsoKOnly4
                | KvQuant::Rotor3Sym
                | KvQuant::Rotor4Sym
                | KvQuant::RotorKOnly3
                | KvQuant::RotorKOnly4
                | KvQuant::RotorK3Asym { .. }
                | KvQuant::RotorK4Asym { .. }
        )
    }

    /// True when the decode path reads the bf16 `decode_fp16_k` seed that
    /// `exit_prefill` materialises (the warm-TTFT shortcut codecs: K8V*,
    /// Planar*, Turbo*, Iso* with a quantised-or-bf16 K mirror that decode
    /// consults).
    ///
    /// **False for the K-only family** (`IsoKOnly3/4`, `RotorKOnly3/4`): these
    /// re-quantise K at every decode step (`ks.append`) and route V through
    /// `update_decode_fp16_v_only`, so they never read `decode_fp16_k`. For
    /// those variants `exit_prefill` skips populating the bf16 K seed — it was
    /// dead memory. The bf16 **V** seed (`decode_fp16_v`) is still populated for
    /// them; only the K seed is gated.
    ///
    /// **False for the fused rotor symmetric codecs** (`Rotor3Sym`,
    /// `Rotor4Sym`): their decode is a flash kernel over both packed rings, so
    /// neither axis reads a mirror. See [`Self::feeds_bf16_v_at_decode`].
    ///
    /// The match is **exhaustive on purpose** (no wildcard `_`): adding a new
    /// `KvQuant` variant will produce a compile error until it is classified
    /// here, preventing a new K-only-style variant from silently defaulting to
    /// `true` and reintroducing the dead-seed leak.
    pub fn feeds_bf16_k_at_decode(&self) -> bool {
        match self {
            // Two families never read a bf16 K at decode, for two reasons:
            //
            // * K-only (IsoKOnly*, RotorKOnly*) — K is re-quantised at every
            //   decode step; V routes through `update_decode_fp16_v_only`.
            // * Fused symmetric (Iso{3,4}Sym, Rotor{3,4}Sym) — decode runs a
            //   flash kernel straight off the packed K and V rings, so neither
            //   axis reads a mirror (see `feeds_bf16_v_at_decode`).
            KvQuant::IsoKOnly3
            | KvQuant::IsoKOnly4
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4
            | KvQuant::Iso3Sym
            | KvQuant::Iso4Sym
            | KvQuant::Rotor3Sym
            | KvQuant::Rotor4Sym => false,
            // All other variants: decode reads the bf16 K seed materialised by
            // exit_prefill (the warm-TTFT shortcut codecs and bf16 KV).
            KvQuant::None
            | KvQuant::K8V4
            | KvQuant::K8V8
            | KvQuant::Planar
            | KvQuant::Planar3
            | KvQuant::PlanarK
            | KvQuant::Mixed { .. }
            | KvQuant::RotK { .. }
            | KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo2Tcq
            | KvQuant::TurboSym3
            | KvQuant::TurboSym4
            | KvQuant::Iso3
            | KvQuant::Iso4
            | KvQuant::Rotor3
            | KvQuant::Rotor4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. } => true,
        }
    }

    /// True when the decode path reads the bf16 `decode_fp16_v` seed that
    /// `exit_prefill` materialises.
    ///
    /// Sibling of [`Self::feeds_bf16_k_at_decode`] for the V axis. Almost every
    /// codec is `true`: even the K-only re-quantise family routes V through
    /// `update_decode_fp16_v_only`, and `KvQuant::None` stores its bf16 V here
    /// outright.
    ///
    /// **False only for the fused symmetric codecs** (`Iso3Sym`, `Iso4Sym`,
    /// `Rotor3Sym`, `Rotor4Sym`): the quant-V flash kernel unpacks V straight out
    /// of the packed ring inside the SV loop, so a bf16 V is never read. Keeping
    /// one is not a small waste — a full `seq * head_dim * 2` bytes per layer of
    /// V (plus the same for K) is the *dominant* term in these codecs' residency
    /// and is what made a ~3-bits-per-axis codec cost more than plain bf16.
    ///
    /// Exhaustive on purpose, same reasoning as the K-side predicate: a new
    /// variant must be classified rather than silently inherit a mirror.
    pub fn feeds_bf16_v_at_decode(&self) -> bool {
        match self {
            // Fused symmetric: V is unpacked from the quant store inside the
            // flash kernel's SV loop; no bf16 V exists.
            KvQuant::Iso3Sym | KvQuant::Iso4Sym | KvQuant::Rotor3Sym | KvQuant::Rotor4Sym => false,
            // Everything else reads the bf16 V seed at decode — including the
            // K-only family (via `update_decode_fp16_v_only`) and `None`, whose
            // bf16 V *is* this buffer.
            KvQuant::None
            | KvQuant::K8V4
            | KvQuant::K8V8
            | KvQuant::Planar
            | KvQuant::Planar3
            | KvQuant::PlanarK
            | KvQuant::Mixed { .. }
            | KvQuant::RotK { .. }
            | KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo2Tcq
            | KvQuant::TurboSym3
            | KvQuant::TurboSym4
            | KvQuant::Iso3
            | KvQuant::Iso4
            | KvQuant::IsoKOnly3
            | KvQuant::IsoKOnly4
            | KvQuant::Rotor3
            | KvQuant::Rotor4
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. } => true,
        }
    }

    /// True when some decode-time read path of this codec consults the packed
    /// [`crate::storage::KvStorage`] payload.
    ///
    /// This is the complement of the two `feeds_bf16_*` predicates, and the
    /// three together classify what `exit_prefill` has to materialise: a codec
    /// that feeds both axes from the bf16 mirror **and** never reads its packed
    /// store at decode gets no packed store at all, because it would be written
    /// once and then held, unread, for the whole decode window — `O(context)`
    /// per layer of pure overhead on top of a mirror that is already the same
    /// size as plain bf16.
    ///
    /// `false` does **not** mean "the store is useless": it is still the
    /// authority for a cache that has no mirror — one reconstructed by
    /// [`crate::KvCache::from_storage`] (SSD hydrate), or one that never
    /// bracketed a prefill. It means only that a *seeded* cache never reads it,
    /// which is exactly the condition under which allocating it is waste.
    ///
    /// Which codecs read their store, and where:
    ///
    /// * `Mixed` / `RotK` — `update_and_sdpa_mixed` appends to and reads the
    ///   MLX affine 3-tuples every decode step (`mixed_quantized_sdpa`).
    /// * The K-only re-quantise family (`IsoKOnly3/4`, `RotorKOnly3/4`) —
    ///   K is appended to the packed store per step and the flash-decode arm
    ///   reads it back.
    /// * The fused symmetric family (`Iso3Sym`, `Iso4Sym`, `Rotor3Sym`,
    ///   `Rotor4Sym`) — decode is a flash kernel over both packed rings.
    ///
    /// Everything else decodes off the bf16 mirror alone. That includes the
    /// codecs whose *other* GPU fast paths look like store reads but are not:
    /// TurboFlash (`K8V4`) and fused-QK (`K8V4`/`K8V8`/`TurboSym3/4`/
    /// `RotorK{3,4}Asym`) each maintain their **own** head-major buffer
    /// re-encoded from the mirror, and `PlanarK`'s fused-QK / flash-decode arm
    /// is gated on the mirror being *absent*. None of them touches the packed
    /// store, so no `DispatchPolicy` gate can turn a `false` here into a
    /// `true` — the classification is a property of the codec alone.
    ///
    /// A codec that grows a decode kernel over its own packed store must flip
    /// its arm here in the same change, or `exit_prefill` will not have built
    /// the buffer the kernel wants to read.
    ///
    /// Exhaustive on purpose, same reasoning as the two `feeds_bf16_*`
    /// predicates: a new variant must be classified rather than silently
    /// inherit a value.
    #[must_use]
    pub fn decode_reads_packed_store(&self) -> bool {
        match self {
            // Quantized-SDPA over the affine 3-tuples, appended per step.
            KvQuant::Mixed { .. }
            | KvQuant::RotK { .. }
            // K re-quantised into the packed store every decode step.
            | KvQuant::IsoKOnly3
            | KvQuant::IsoKOnly4
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4
            // Flash decode straight off both packed rings.
            | KvQuant::Iso3Sym
            | KvQuant::Iso4Sym
            | KvQuant::Rotor3Sym
            | KvQuant::Rotor4Sym => true,
            // `None` has no packed store to read; the rest are the bf16-mirror
            // family, whose store is written once at `exit_prefill` and never
            // read again on a seeded cache.
            KvQuant::None
            | KvQuant::K8V4
            | KvQuant::K8V8
            | KvQuant::Planar
            | KvQuant::Planar3
            | KvQuant::PlanarK
            | KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo2Tcq
            | KvQuant::TurboSym3
            | KvQuant::TurboSym4
            | KvQuant::Iso3
            | KvQuant::Iso4
            | KvQuant::Rotor3
            | KvQuant::Rotor4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. } => false,
        }
    }

    /// True when `exit_prefill` materialises this codec's packed store.
    ///
    /// The store is built when decode reads it ([`Self::decode_reads_packed_store`])
    /// **or** when either axis decodes without a bf16 mirror to fall back on —
    /// the second disjunct is what keeps a half-mirrored codec (K quantised, V
    /// bf16, or the reverse) from losing the side that has no mirror.
    ///
    /// This is the predicate `exit_prefill` gates the allocation on, and the one
    /// the byte estimate reads. The **spill path does not read it** — it asks
    /// [`crate::storage::KvStorage::geometry_only_max_seq`], a predicate on the
    /// other enum, and nothing in the type system couples the two. A codec
    /// classified `false` here whose storage variant lands in that function's
    /// "payload is not an `Option`" arm would make the writer stamp a codec
    /// geometry with no tensors behind it. `a_storeless_codec_always_has_a_geometry_only_storage`
    /// is the only place that pairing is enforced.
    #[must_use]
    pub fn materialises_packed_store(&self) -> bool {
        self.decode_reads_packed_store()
            || !self.feeds_bf16_k_at_decode()
            || !self.feeds_bf16_v_at_decode()
    }

    /// True when this codec dispatches at least one custom Metal
    /// (MSL) kernel on its production hot path, so its shaders pay a one-time
    /// cold-compile on first dispatch.
    ///
    /// Used by [`crate::precompile::precompile_kv_codec_msl`] to decide whether
    /// the load-time MSL warm should run (and to keep `none` off the warm path),
    /// and by the resolve-time readiness logic to widen the first-serve window
    /// only for shader-heavy codecs.
    ///
    /// **Important:** "carries MSL" is *not* the same as "runs entirely on
    /// Metal". The iso / rotor families also return `true` here (their K-side is
    /// q8_0 MSL and they ship V/K GPU encoders), yet their production V encode +
    /// dequant run on **CPU** — see [`cpu_hot_path_reason`](Self::cpu_hot_path_reason).
    /// The only codec with no MSL at all is `None` (raw bf16, `slice_update`).
    ///
    /// Exhaustive on purpose (no wildcard) so a new variant must be classified.
    pub fn carries_msl(&self) -> bool {
        match self {
            // Raw bf16 KV: no quantization kernel, just slice_update on a bf16
            // buffer. Nothing to cold-compile.
            KvQuant::None => false,
            // Everything else quantizes K with the q8_0 MSL kernel (or, for the
            // Mixed/RotK family, MLX-native affine `mx.quantize`, itself a
            // compiled Metal op) and therefore carries at least one shader.
            KvQuant::K8V4
            | KvQuant::K8V8
            | KvQuant::Planar
            | KvQuant::Planar3
            | KvQuant::PlanarK
            | KvQuant::Mixed { .. }
            | KvQuant::RotK { .. }
            | KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo2Tcq
            | KvQuant::TurboSym3
            | KvQuant::TurboSym4
            | KvQuant::Iso3
            | KvQuant::Iso4
            | KvQuant::Iso3Sym
            | KvQuant::Iso4Sym
            | KvQuant::IsoKOnly3
            | KvQuant::IsoKOnly4
            | KvQuant::Rotor3
            | KvQuant::Rotor4
            | KvQuant::Rotor3Sym
            | KvQuant::Rotor4Sym
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. } => true,
        }
    }

    /// `Some(reason)` when this codec runs its V (and, for the K-only
    /// / symmetric variants, K) **encode + dequant on the CPU** on the default
    /// production hot path — i.e. it falls through to the dequant-then-SDPA
    /// legacy path with a host-side scalar codec rather than a Metal kernel.
    ///
    /// This is the honest Metal-vs-CPU verdict (CLAUDE.md hard rule 7),
    /// grounded in the actual decode/prefill dispatch in
    /// [`crate::kvcache`]'s `update_*` functions — not in assumptions:
    ///
    /// - **V-only iso / rotor** (`Iso3/4(/Sym)`, `Rotor3/4(/Sym)`,
    ///   `RotorK{3,4}Asym`): `update_iso3*` / `update_rotor3*` early-return to
    ///   the warm-TTFT bf16 decode seed (`decode_fp16_k.is_some()`) at decode,
    ///   so the GPU iso/rotor branch is shadowed and the codec encode that does
    ///   run (at prefill) is CPU → `Some(reason)`.
    /// - **K-only iso** (`IsoKOnly3/4`): NO bf16 early-return — the iso K MSL
    ///   kernel dispatches every decode step on GPU → `None` (Metal, and
    ///   GPU-resident end to end: the flash-decode kernel reads the packed ring
    ///   in place, so the growing prefix never crosses to the host).
    /// - **K-only rotor** (`RotorKOnly3/4`): NO bf16 early-return; the runtime
    ///   dispatcher (`update_rotor_k_only_{3,4}` and the sdpa fast path) gates
    ///   the GPU K encode on the store's sticky `use_qjl()` flag. QJL off
    ///   (default) → Metal (`None`); QJL on (opt-in `--rotor-qjl on`) → CPU
    ///   (`Some`). This classifier reads the process-global
    ///   [`crate::rotor_qjl::rotor_qjl_enabled`], which equals the store flag
    ///   only before the store is built (it is the value the store takes at
    ///   first append), so the verdict tracks the dispatcher on any live cache.
    ///
    /// The `Some(reason)` cases are the source of the 30–60× first-forward
    /// slowdown and the monotonic decode decay as KV grows. A `None` here MUST
    /// mean a Metal kernel demonstrably dispatches on the hot path — never an
    /// assumption.
    ///
    /// Returns `None` for codecs whose hot path is genuinely Metal (q8_0 K +
    /// tq4/planar/affine V, the Turbo/Mixed/RotK families, plus the K-only iso /
    /// QJL-off rotor families above).
    ///
    /// Exhaustive on purpose (no wildcard) so a new variant must be classified.
    #[allow(
        clippy::match_same_arms,
        reason = "the K-only iso arm returns None like the Metal arm but is kept separate to document the per-codec Metal-vs-CPU verdict this fn exists for: the K-only iso codec reaches Metal by its own encode + flash-decode kernels rather than the shared q8/turbo path, and unlike the rotor arm it carries no QJL gate that could flip the verdict. Merging the arms would erase that per-codec record."
    )]
    pub fn cpu_hot_path_reason(&self) -> Option<&'static str> {
        match self {
            // V-only iso variants: the K side is GPU affine q8_0 and a GPU iso
            // V encode/dequant branch EXISTS, but at the decode hot path
            // `update_iso3*` early-returns to the warm-TTFT bf16 decode seed
            // (`decode_fp16_k.is_some()`), so the GPU iso branch is shadowed;
            // the iso V-encode that does run (at prefill) is CPU.
            KvQuant::Iso3 | KvQuant::Iso4 => Some(
                "IsoQuant (quaternion SO(4)) V-only: a GPU iso encode/dequant branch \
                 exists but is shadowed by the bf16 decode seed; prefill V-encode runs \
                 on CPU",
            ),
            // Symmetric iso variants: NO bf16 decode-seed early-return — decode
            // is the quant-V flash kernel over both packed iso rings. Iso carries
            // no QJL sideband, so there is no CPU-fallback gate; the hot path is
            // Metal.
            KvQuant::Iso3Sym | KvQuant::Iso4Sym => None,
            // K-only iso variants: NO bf16 decode-seed early-return — the iso K
            // codec fires every decode step. On GPU, `update_iso_k_only_{3,4}`
            // dispatches the real iso{3,4} MSL encode kernel; IsoKOnly3 also runs
            // the iso3 MSL dequant kernel. Decode reads the packed ring through
            // the iso flash-decode kernel, so the growing prefix stays on device;
            // the only host readback (`packed_view_cpu`) is reached from
            // `dequant()` / `dequant_gpu()` at a block-rebuild or SSD-spill
            // boundary, never from a decode step. Metal hot path, no host stage.
            KvQuant::IsoKOnly3 | KvQuant::IsoKOnly4 => None,
            // V-only rotor variants and the rotor-K-asym variants early-return to
            // the bf16 decode seed at decode (`decode_fp16_k.is_some()`), so the
            // rotor K codec only fires at prefill on CPU; the GPU fused-QK encoder
            // is opt-in (`--fused-qk`) and does not fire on the standard flow.
            KvQuant::Rotor3
            | KvQuant::Rotor4
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. } => Some(
                "RotorQuant (Clifford Cl(3,0)) encode + dequant run on CPU on the \
                 default hot path (the bf16 decode seed shadows the GPU branch); the \
                 GPU fused-QK encoder is opt-in (--fused-qk)",
            ),
            // Symmetric rotor variants: NO bf16 decode-seed early-return — decode
            // is the quant-V flash kernel over both packed rings. Same QJL gate as
            // the K-only family, and for the same reason: the QJL residual cannot
            // be reproduced in the flash inner loop, so a QJL-carrying store keeps
            // the CPU dequant path on BOTH axes.
            KvQuant::Rotor3Sym | KvQuant::Rotor4Sym => {
                if crate::rotor_qjl::rotor_qjl_enabled() {
                    Some(
                        "RotorQuant (Clifford Cl(3,0)) symmetric with QJL enabled \
                         (rotor_qjl_enabled): the QJL residual forces K and V onto the \
                         CPU encode + dequant path every decode step; disable QJL \
                         (--rotor-qjl off) to route both axes through the Metal \
                         flash-decode kernel",
                    )
                } else {
                    None
                }
            }
            // K-only rotor variants: NO bf16 decode-seed early-return — the rotor
            // K codec fires every decode step. `update_rotor_k_only_{3,4}` gates
            // the GPU K encode on the store's sticky QJL flag (`use_qjl()`, fixed
            // at first append), matching the sdpa fast path:
            //   - QJL on (opt-in `--rotor-qjl on`): K append runs on CPU → CPU hot path.
            //   - QJL off (default): `rotor{3,4}_gpu_append_into_k_blocks`
            //     dispatches the per-codec rotor MSL encode kernel, and decode
            //     reads the packed ring through the rotor flash-decode kernel →
            //     Metal hot path, GPU-resident end to end, no host stage.
            KvQuant::RotorKOnly3 | KvQuant::RotorKOnly4 => {
                if crate::rotor_qjl::rotor_qjl_enabled() {
                    Some(
                        "RotorQuant (Clifford Cl(3,0)) K-only with QJL enabled \
                         (rotor_qjl_enabled): the QJL residual forces the K append onto \
                         CPU every decode step; disable QJL (--rotor-qjl off) to route \
                         the rotor K encode through the Metal kernel",
                    )
                } else {
                    None
                }
            }
            // Genuinely Metal on the hot path (or no-op bf16): not a CPU codec.
            KvQuant::None
            | KvQuant::K8V4
            | KvQuant::K8V8
            | KvQuant::Planar
            | KvQuant::Planar3
            | KvQuant::PlanarK
            | KvQuant::Mixed { .. }
            | KvQuant::RotK { .. }
            | KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo2Tcq
            | KvQuant::TurboSym3
            | KvQuant::TurboSym4 => None,
        }
    }

    /// `true` for the K-only iso / rotor codecs whose K side dispatches the
    /// iso/rotor MSL kernel (NOT the shared q8_0 K kernel) on the GPU hot path:
    /// `IsoKOnly3/4`, `RotorKOnly3/4`. These return `None` from
    /// [`cpu_hot_path_reason`](Self::cpu_hot_path_reason) (they are Metal on the
    /// hot path, modulo the rotor-QJL gate) but are *not* q8-K codecs, so the
    /// load-time q8 precompile must skip them — their K kernel compiles lazily on
    /// first prefill. See [`crate::precompile::precompile_kv_codec_msl`].
    #[must_use]
    pub fn is_k_only_iso_rotor(&self) -> bool {
        matches!(
            self,
            KvQuant::IsoKOnly3 | KvQuant::IsoKOnly4 | KvQuant::RotorKOnly3 | KvQuant::RotorKOnly4
        )
    }

    /// The `(k_bits, v_bits, k_group_size, v_group_size)` the Mixed state should
    /// be built with for this quant, or `None` for non-Mixed-path variants.
    ///
    /// RotK fixes K at 8-bit/group=64 (rotation is a PPL win over affine-K, not
    /// a sub-8-bit-K enabler) and carries V's bits/group from the tag.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn mixed_params(&self) -> Option<(i32, i32, i32, i32)> {
        match self {
            KvQuant::Mixed {
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
            } => Some((
                i32::from(*k_bits),
                i32::from(*v_bits),
                i32::from(*k_group_size),
                i32::from(*v_group_size),
            )),
            KvQuant::RotK {
                v_bits,
                v_group_size,
            } => Some((8, i32::from(*v_bits), 64, i32::from(*v_group_size))),
            _ => None,
        }
    }

    /// Per-side `(k_bits, v_bits)` **codebook** width an estimator can use to
    /// model this codec, model-agnostically.
    ///
    /// A codebook width, not a delivered density and not a quality claim.
    /// **A side kept at model dtype instead of quantized reports 16**, which is
    /// the one property callers other than the byte estimator may key off:
    /// `None` (bf16) reports 16/16 and is the only codec that quantizes
    /// neither side, while the K-only families (`PlanarK`, `IsoKOnly*`,
    /// `RotorKOnly*`) report a quantized K and a 16-bit V.
    ///
    /// **Three families do not store at their codebook width.** The iso and
    /// rotor stores spend one whole `u32` code word *and* one `f32` scale per
    /// group — 4 head-dim slots for iso, 3 for rotor — and the planar store
    /// spends one `f32` scale per *pair*. A 3-bit and a 4-bit member of any of
    /// the three therefore occupy byte-identical storage: 16.25 bits per value
    /// for iso (on the ring), 21.75 for rotor and 22.00 for planar at
    /// `head_dim = 128`, all against bf16's 16.0. The width below is what the
    /// codebook quantizes to, not what reaches memory.
    /// [`Self::estimated_resident_bytes_per_layer`] does not size those three
    /// families from this number at all — it sizes every side from
    /// [`SideStore`], the store's own group geometry.
    ///
    /// What this number is still read for outside the byte estimate: **a side
    /// kept at model dtype reports 16**, and [`Self::side_stores`] reports
    /// `None` for the same side. `side_stores_agree_with_approx_code_bits` pins
    /// the two together.
    #[must_use]
    #[allow(
        clippy::match_same_arms,
        reason = "explicit per-variant bit widths read clearer than collapsing arms"
    )]
    pub fn approx_code_bits(&self) -> (u32, u32) {
        match self {
            KvQuant::None => (16, 16),
            KvQuant::K8V8 => (8, 8),
            KvQuant::K8V4 => (8, 4),
            KvQuant::Planar => (8, 4),
            KvQuant::Planar3 => (8, 3),
            KvQuant::PlanarK => (4, 16),
            KvQuant::Mixed { k_bits, v_bits, .. } => (u32::from(*k_bits), u32::from(*v_bits)),
            KvQuant::RotK { v_bits, .. } => (8, u32::from(*v_bits)),
            KvQuant::K8VTurbo3 | KvQuant::K8VTurbo3Tcq => (8, 3),
            KvQuant::K8VTurbo2 | KvQuant::K8VTurbo2Tcq => (8, 2),
            KvQuant::TurboSym3 => (3, 3),
            KvQuant::TurboSym4 => (4, 4),
            KvQuant::Iso3 => (8, 3),
            KvQuant::Iso4 => (8, 4),
            KvQuant::Iso3Sym => (3, 3),
            KvQuant::Iso4Sym => (4, 4),
            KvQuant::IsoKOnly3 => (3, 16),
            KvQuant::IsoKOnly4 => (4, 16),
            KvQuant::Rotor3 => (8, 3),
            KvQuant::Rotor4 => (8, 4),
            KvQuant::Rotor3Sym => (3, 3),
            KvQuant::Rotor4Sym => (4, 4),
            KvQuant::RotorKOnly3 => (3, 16),
            KvQuant::RotorKOnly4 => (4, 16),
            KvQuant::RotorK3Asym { v_bits, .. } => (3, u32::from(*v_bits)),
            KvQuant::RotorK4Asym { v_bits, .. } => (4, u32::from(*v_bits)),
        }
    }

    /// The packed-store layout each axis of this codec writes, as
    /// `(K, V)`. `None` on a side means that axis is a plain buffer at model
    /// dtype and has no packed store at all.
    ///
    /// Sole authority for which layout [`Self::estimated_resident_bytes_per_layer`]
    /// sizes a side from. It agrees with [`Self::approx_code_bits`] on which
    /// sides are unquantised — a side reporting 16 bits there is `None` here and
    /// vice versa — and `side_stores_agree_with_approx_code_bits` pins that.
    ///
    /// Three pairs are easy to get wrong and are called out:
    ///
    /// * **The two iso layouts.** `IsoKOnly3/4` and `Iso3Sym/4Sym` decode from
    ///   the GPU ring and their fused append drops the CPU blocks once it is
    ///   live, so they are [`SideStore::IsoRing`]. `Iso3` / `Iso4` have no ring
    ///   path — their decode early-returns to the bf16 mirror — so the store
    ///   they would hold is [`SideStore::IsoBlocks`], 2.97× larger. Collapsing
    ///   the two would mis-size one family or the other by that factor.
    /// * **`RotorK{3,4}Asym`'s V is not affine.** Its name and its storage
    ///   field say `QuantV`, and `QuantV::new_affine_decode` is a misnomer: the
    ///   codec behind it is the TurboQuant N(0,1) Lloyd-Max one at a fixed
    ///   32-element group ([`validate_rotor_k_asym_v`]). `v_group_size` is a
    ///   layout-key tag the encoder never reads, so it must not reach a store
    ///   parameter here.
    /// * **`RotK`'s K group is fixed at 64**, by `MixedKvState::new_rotated` —
    ///   the codec carries no `k_group_size` field to read it from.
    ///
    /// Exhaustive on purpose, same reasoning as the decode predicates: a new
    /// variant must state where its bytes go rather than inherit a layout.
    #[allow(
        clippy::match_same_arms,
        reason = "the families that share a layout are kept in separate arms so the \
                  match reads as a per-family record of where each codec's bytes go; merging \
                  them would collapse q8/turbo/affine into one unlabelled arm"
    )]
    fn side_stores(self) -> (Option<SideStore>, Option<SideStore>) {
        match self {
            KvQuant::None => (None, None),
            KvQuant::K8V4 => (Some(SideStore::Q8), Some(SideStore::Turbo)),
            KvQuant::K8V8 => (Some(SideStore::Q8), Some(SideStore::Q8)),
            KvQuant::Planar | KvQuant::Planar3 => (Some(SideStore::Q8), Some(SideStore::Planar)),
            KvQuant::PlanarK => (Some(SideStore::Planar), None),
            KvQuant::Mixed {
                k_group_size,
                v_group_size,
                ..
            } => (
                Some(SideStore::Affine {
                    group: u32::from(k_group_size),
                }),
                Some(SideStore::Affine {
                    group: u32::from(v_group_size),
                }),
            ),
            KvQuant::RotK { v_group_size, .. } => (
                Some(SideStore::Affine { group: 64 }),
                Some(SideStore::Affine {
                    group: u32::from(v_group_size),
                }),
            ),
            KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo2Tcq => (Some(SideStore::Q8), Some(SideStore::Turbo)),
            KvQuant::TurboSym3 | KvQuant::TurboSym4 => {
                (Some(SideStore::Turbo), Some(SideStore::Turbo))
            }
            // V-only iso: K is affine q8_0, V is the CPU-block iso form.
            KvQuant::Iso3 | KvQuant::Iso4 => (Some(SideStore::Q8), Some(SideStore::IsoBlocks)),
            KvQuant::Iso3Sym | KvQuant::Iso4Sym => {
                (Some(SideStore::IsoRing), Some(SideStore::IsoRing))
            }
            KvQuant::IsoKOnly3 | KvQuant::IsoKOnly4 => (Some(SideStore::IsoRing), None),
            KvQuant::Rotor3 | KvQuant::Rotor4 => (Some(SideStore::Q8), Some(SideStore::Rotor)),
            KvQuant::Rotor3Sym | KvQuant::Rotor4Sym => {
                (Some(SideStore::Rotor), Some(SideStore::Rotor))
            }
            KvQuant::RotorKOnly3 | KvQuant::RotorKOnly4 => (Some(SideStore::Rotor), None),
            KvQuant::RotorK3Asym { .. } | KvQuant::RotorK4Asym { .. } => {
                (Some(SideStore::Rotor), Some(SideStore::Turbo))
            }
        }
    }

    /// Estimate the resident KV bytes per layer this codec holds for a
    /// **global (full-attention)** layer of `seq` tokens.
    ///
    /// Model-agnostic: the estimate is derived purely from layer attributes
    /// (`seq`, `head_dim`, `kv_heads`) and codec attributes (the per-side store
    /// layout from [`Self::side_stores`], the codebook width from
    /// [`Self::approx_code_bits`], and whether the codec retains a bf16 decode
    /// seed via [`Self::feeds_bf16_k_at_decode`]) — never from an arch name.
    ///
    /// Per side, the store layout sets the cadence; see [`SideStore`] for each
    /// one's group geometry. Two properties matter to a reader of the result:
    ///
    /// * **Only the side that carries a family codec uses its formula.** The
    ///   V-only variants (`Iso3`/`Iso4`/`Rotor3`/`Rotor4`) keep an 8-bit q8_0
    ///   K, so their K side is [`SideStore::Q8`]; the K-only variants
    ///   (`PlanarK`, `IsoKOnly*`, `RotorKOnly*`) keep a bf16 V.
    /// * **Each side is byte-exact against its store, with one exception.**
    ///   Every cadence in [`SideStore`] is measured against the store's own
    ///   encoder, so there is no rounding term to remember. The exception is
    ///   iso, whose side is sized from the GPU ring: a cache holding the CPU
    ///   blocks `exit_prefill` built holds 2.97× that on the iso axis. That is
    ///   a transient window on a layer the fused decode path serves, and a
    ///   permanent under-report on a layer whose shape that path's gate rejects
    ///   (batch > 1, or a `head_dim` that is not a power of two at most 512).
    ///   See [`SideStore::IsoRing`].
    ///
    /// Two terms sit on top of the per-side store:
    ///
    /// - **bf16 decode seed**: a codec keeps a full `seq * head_dim * 2` bf16
    ///   mirror of each axis whose decode reads it —
    ///   [`Self::feeds_bf16_k_at_decode`] for K, [`Self::feeds_bf16_v_at_decode`]
    ///   for V. This is the warm-TTFT shortcut buffer. Codecs with a fused
    ///   decode over both packed axes keep neither.
    /// - **no packed store**: a codec that mirrors both axes and has no decode
    ///   path over its store ([`Self::materialises_packed_store`] is `false`)
    ///   allocates no codes and no scales at all, so its estimate is exactly
    ///   the two bf16 mirrors — the same bytes as `None`. The store term applies
    ///   only to codecs that keep a store something reads.
    ///
    /// `None` (bf16) returns just the two bf16 buffers and no seed.
    ///
    /// This is an estimate (page-rounding, GPU/CPU residual coexistence, and
    /// rotation/bias buffers are not modelled) used only for the resolve-time
    /// net-benefit `warn!` — the authoritative number is
    /// [`crate::KvCache::resident_bytes`] read after the first measurement.
    #[must_use]
    pub fn estimated_resident_bytes_per_layer(
        &self,
        seq: u64,
        head_dim: u64,
        kv_heads: u64,
    ) -> u64 {
        let elems = seq.saturating_mul(head_dim).saturating_mul(kv_heads);
        if matches!(self, KvQuant::None) {
            // Two bf16 buffers (K + V), no codes, no seed.
            return elems.saturating_mul(2).saturating_mul(2);
        }
        let (k_bits, v_bits) = self.approx_code_bits();
        let (k_store, v_store) = self.side_stores();
        let n_tokens = seq.saturating_mul(kv_heads);

        // A codec whose decode never reads its packed store does not allocate
        // one (`exit_prefill` skips the bulk encode), so the store term is zero
        // for it and its resident cost is exactly the mirror.
        let packs_a_store = self.materialises_packed_store();
        let side_bytes = |bits: u32, retains_seed: bool, store: Option<SideStore>| -> u64 {
            let Some(store) = store else {
                // Unquantised axis: one buffer at model dtype, counted once.
                return elems.saturating_mul(2);
            };
            if !packs_a_store {
                // `!materialises_packed_store()` is defined to imply both
                // `feeds_bf16_*`, which is what `retains_seed` is at both call
                // sites — so this side is a bf16 mirror and nothing else. The
                // implication is asserted over every variant by
                // `a_storeless_codec_mirrors_both_axes`, not by a
                // `debug_assert` here that could never fire and that
                // `release-perf` compiles out regardless.
                let _ = retains_seed;
                return elems.saturating_mul(2);
            }
            let stored = packed_side_bytes(store, bits, elems, head_dim, n_tokens);
            let seed = if retains_seed {
                elems.saturating_mul(2)
            } else {
                0
            };
            stored.saturating_add(seed)
        };
        // Each seed is retained only when this codec's decode actually reads it
        // — the same two predicates `exit_prefill` gates the real allocation on,
        // so the estimate cannot drift from what is materialised. The K-only
        // re-quantize family drops the K seed; the fused rotor symmetric codecs
        // drop both.
        let k_bytes = side_bytes(k_bits, self.feeds_bf16_k_at_decode(), k_store);
        let v_bytes = side_bytes(v_bits, self.feeds_bf16_v_at_decode(), v_store);
        k_bytes.saturating_add(v_bytes)
    }

    /// Estimated **net byte saving** of running this codec versus plain bf16 on
    /// a single layer, given its attributes. Positive = the codec saves memory;
    /// **negative = the codec costs more memory than bf16** (the net-negative
    /// condition).
    ///
    /// `is_windowed` layers always run the bf16 rotating ring regardless of the
    /// codec flag (mlx-lm `RotatingKVCache.to_quantized` raises
    /// `NotImplementedError`; rMLX matches it), so the codec is a no-op there
    /// and the net saving is exactly `0`. For global layers the saving is
    /// `bf16_bytes(seq) - codec_bytes(seq)`.
    ///
    /// Callers pass the effective per-layer `seq` (for a windowed layer, clamp
    /// it to the window before calling — a windowed layer never holds more than
    /// `window` tokens, though the result is `0` for windowed regardless).
    ///
    /// Keyed entirely on layer + codec attributes; no arch name is consulted.
    #[must_use]
    pub fn estimated_net_saving_per_layer(
        &self,
        seq: u64,
        head_dim: u64,
        kv_heads: u64,
        is_windowed: bool,
    ) -> i64 {
        if is_windowed {
            // Windowed layer always runs the bf16 ring → codec is a no-op,
            // identical bytes either way.
            return 0;
        }
        let bf16 = KvQuant::None.estimated_resident_bytes_per_layer(seq, head_dim, kv_heads);
        let codec = self.estimated_resident_bytes_per_layer(seq, head_dim, kv_heads);
        // saving = bf16 - codec; negative when the codec is bigger.
        i64::try_from(bf16).unwrap_or(i64::MAX) - i64::try_from(codec).unwrap_or(i64::MAX)
    }
}

// ── KvQuant Display / FromStr ────────────────────────────────────────────────

/// Error returned by `<KvQuant as FromStr>::from_str`.
#[non_exhaustive]
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvQuantParseError {
    /// The whole input does not match any canonical KvQuant form.
    #[error(
        "unknown KvQuant '{0}' — valid: none, k8v4, k8v8, planar, planar3, planar_k, k8vturbo3, k8vturbo3tcq, k8vturbo2tcq, tsym3, tsym4, k8vturbo2, iso3, iso4, iso3_sym, iso4_sym, k_iso3, k_iso4, rotor3, rotor4, rotor3_sym, rotor4_sym, k_rotor3, k_rotor4, rotor_k_3_asym_v<vb>_g<vg>, rotor_k_4_asym_v<vb>_g<vg>, rot_k_v<vb>g<vg>, mixed_k<kb>g<kg>_v<vb>g<vg>"
    )]
    Unknown(String),
    /// The `mixed_*` shape matched but a numeric component failed to parse.
    #[error("invalid Mixed KvQuant '{input}': {reason}")]
    InvalidMixed {
        /// The raw input string that triggered the parse attempt.
        input: String,
        /// Why the numeric component failed to parse.
        reason: String,
    },
    /// The name is a codec that was withdrawn from the enum. Carries the
    /// codec that supersedes it so the operator has one edit to make, not a
    /// search. A retired name must keep failing loudly: silently aliasing it
    /// to its replacement would let a recorded bench cell, a spilled SSD
    /// block tag or a saved CLI line keep naming a codec that no longer runs.
    #[error("KvQuant '{input}' was retired — use '{replacement}' instead")]
    Retired {
        /// The retired name the caller passed.
        input: String,
        /// The canonical name of the codec that supersedes it.
        replacement: &'static str,
    },
    /// The `rotor_k_{3,4}_asym_v*_g*` shape matched but the numeric component
    /// failed to parse or the (`v_bits`, `v_group_size`) tuple is not one of
    /// the accepted MLX-affine V codecs.
    #[error("invalid RotorK*Asym KvQuant '{input}': {reason}")]
    InvalidRotorKAsym {
        /// The raw input string that triggered the parse attempt.
        input: String,
        /// Why the V codec spec failed to validate.
        reason: String,
    },
    /// The `rot_k_v*g*` shape matched but a numeric component failed to parse
    /// or the (`v_bits`, `v_group_size`) tuple is not one the MLX affine
    /// quantizer implements.
    #[error("invalid RotK KvQuant '{input}': {reason}")]
    InvalidRotK {
        /// The raw input string that triggered the parse attempt.
        input: String,
        /// Why the V codec spec failed to validate.
        reason: String,
    },
}

/// Validate that `(v_bits, v_group_size)` matches one of the supported V codecs
/// the asymmetric rotor-K variants accept.
///
/// The V slot routes through the existing [`crate::storage::QuantV`] path
/// (the same TurboQuant N(0,1) Lloyd-Max codec used by `K8V4` / `K8VTurbo3` /
/// `K8VTurbo2` for the V side). `QuantV` is hard-coded to internal `GROUP_SIZE`
/// = 32 and supports `bits ∈ {2, 3, 4}` via [`crate::turboquant::lloyd_gaussian_codebook`].
///
/// Accepted tuples — the `v_group_size` field is carried through to the
/// layout key for deterministic SSD round-trip, but the underlying TurboQuant
/// codec keeps its fixed 32-element group internally:
/// - `(4, 128)`, `(4, 64)`, `(4, 32)` — TurboQuant V 4-bit (same as K8V4 V).
/// - `(3, 64)` — TurboQuant V 3-bit (same as K8VTurbo3 V).
/// - `(2, 64)` — TurboQuant V 2-bit (same as K8VTurbo2 V).
///
/// `(8, *)` tuples are rejected because TurboQuant has no 8-bit path; pair
/// `rotor_k_3` / `rotor_k_4` with `--ctv bf16` for the K-only RotorKOnly* path
/// or with `--ctv rotor_v_3` / `--ctv rotor_v_4` for the symmetric Rotor*Sym
/// path instead.
pub fn validate_rotor_k_asym_v(v_bits: u8, v_group_size: u16) -> Result<(), String> {
    match (v_bits, v_group_size) {
        (4, 128 | 64 | 32) | (3 | 2, 64) => Ok(()),
        _ => Err(format!(
            "unsupported (v_bits={v_bits}, v_group_size={v_group_size}) for RotorK*Asym; \
             valid V codecs: q4_g128, q4_g64, q4_g32, q3_g64, q2_g64 (TurboQuant V) \
             (v_group_size is layout-tag-only — TurboQuant V uses GROUP_SIZE=32 regardless). \
             For bf16 V use --kv-quant k_rotor3 / k_rotor4; for rotor V use --kv-quant rotor3_sym / rotor4_sym."
        )),
    }
}

/// Validate one side of a `mixed_k<kb>g<kg>_v<vb>g<vg>` spec.
///
/// The Mixed path hands `(bits, group_size)` straight to MLX's affine
/// `quantize`, which implements a fixed set of widths. Without this check the
/// parser accepted any `u8`: `mixed_k16g64_v16g64` parsed, reported
/// `approx_code_bits() == (16, 16)` — the value that means "kept at model
/// dtype" — and so read as a codec that quantizes nothing, which silently
/// opted it out of every check keyed on that property (the boundary-layer
/// quality floor among them) while still packing 16-bit affine codes at
/// runtime. A width the codec cannot store is a parse error, not a mode.
///
/// [`KvQuant::RotK`] shares this validator, not a second table: its store *is*
/// `Mixed` storage (`MixedKvState::new_rotated`), so its V slot reaches the same
/// quantizer and must accept the same set. The other parametric family has its
/// own codec and so its own check ([`validate_rotor_k_asym_v`]).
pub fn validate_mixed_side(side: char, bits: u8, group_size: u16) -> Result<(), String> {
    match (bits, group_size) {
        (2 | 3 | 4 | 5 | 6 | 8, 32 | 64 | 128) => Ok(()),
        _ => Err(format!(
            "unsupported ({side}_bits={bits}, {side}_group_size={group_size}) for Mixed; \
             valid bits: 2, 3, 4, 5, 6, 8 (MLX affine quantize), \
             valid group sizes: 32, 64, 128. \
             For an unquantized side use --kv-quant none (bf16 K and V)."
        )),
    }
}

impl std::fmt::Display for KvQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvQuant::None => f.write_str("none"),
            KvQuant::K8V4 => f.write_str("k8v4"),
            KvQuant::K8V8 => f.write_str("k8v8"),
            KvQuant::Planar => f.write_str("planar"),
            KvQuant::Mixed {
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
            } => write!(f, "mixed_k{k_bits}g{k_group_size}_v{v_bits}g{v_group_size}"),
            KvQuant::RotK {
                v_bits,
                v_group_size,
            } => write!(f, "rot_k_v{v_bits}g{v_group_size}"),
            KvQuant::K8VTurbo3 => f.write_str("k8vturbo3"),
            KvQuant::TurboSym3 => f.write_str("tsym3"),
            KvQuant::TurboSym4 => f.write_str("tsym4"),
            KvQuant::Planar3 => f.write_str("planar3"),
            KvQuant::PlanarK => f.write_str("planar_k"),
            KvQuant::K8VTurbo2 => f.write_str("k8vturbo2"),
            KvQuant::Iso3 => f.write_str("iso3"),
            KvQuant::Iso4 => f.write_str("iso4"),
            KvQuant::Rotor3 => f.write_str("rotor3"),
            KvQuant::Rotor4 => f.write_str("rotor4"),
            KvQuant::K8VTurbo3Tcq => f.write_str("k8vturbo3tcq"),
            KvQuant::K8VTurbo2Tcq => f.write_str("k8vturbo2tcq"),
            KvQuant::Iso3Sym => f.write_str("iso3_sym"),
            KvQuant::Iso4Sym => f.write_str("iso4_sym"),
            KvQuant::IsoKOnly3 => f.write_str("k_iso3"),
            KvQuant::IsoKOnly4 => f.write_str("k_iso4"),
            KvQuant::Rotor3Sym => f.write_str("rotor3_sym"),
            KvQuant::Rotor4Sym => f.write_str("rotor4_sym"),
            KvQuant::RotorKOnly3 => f.write_str("k_rotor3"),
            KvQuant::RotorKOnly4 => f.write_str("k_rotor4"),
            // Payload-bearing asymmetric rotor-K variants.
            KvQuant::RotorK3Asym {
                v_bits,
                v_group_size,
            } => write!(f, "rotor_k_3_asym_v{v_bits}_g{v_group_size}"),
            KvQuant::RotorK4Asym {
                v_bits,
                v_group_size,
            } => write!(f, "rotor_k_4_asym_v{v_bits}_g{v_group_size}"),
        }
    }
}

impl std::str::FromStr for KvQuant {
    type Err = KvQuantParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" | "bf16" | "f16" => return Ok(KvQuant::None),
            "k8v4" => return Ok(KvQuant::K8V4),
            "k8v8" => return Ok(KvQuant::K8V8),
            "planar" => return Ok(KvQuant::Planar),
            "planar3" => return Ok(KvQuant::Planar3),
            "k8vturbo3" => return Ok(KvQuant::K8VTurbo3),
            "tsym3" => return Ok(KvQuant::TurboSym3),
            "tsym4" => return Ok(KvQuant::TurboSym4),
            "planar_k" => return Ok(KvQuant::PlanarK),
            "k8vturbo2" => return Ok(KvQuant::K8VTurbo2),
            "iso3" => return Ok(KvQuant::Iso3),
            "iso4" => return Ok(KvQuant::Iso4),
            "rotor3" | "rotor_v_3" => return Ok(KvQuant::Rotor3),
            "rotor4" | "rotor_v_4" => return Ok(KvQuant::Rotor4),
            "k8vturbo3tcq" => return Ok(KvQuant::K8VTurbo3Tcq),
            "k8vturbo2tcq" => return Ok(KvQuant::K8VTurbo2Tcq),
            // Symmetric / K-only iso variants.
            "iso3_sym" => return Ok(KvQuant::Iso3Sym),
            "iso4_sym" => return Ok(KvQuant::Iso4Sym),
            "k_iso3" => return Ok(KvQuant::IsoKOnly3),
            "k_iso4" => return Ok(KvQuant::IsoKOnly4),
            // Symmetric / K-only rotor variants.
            "rotor3_sym" => return Ok(KvQuant::Rotor3Sym),
            "rotor4_sym" => return Ok(KvQuant::Rotor4Sym),
            "k_rotor3" => return Ok(KvQuant::RotorKOnly3),
            "k_rotor4" => return Ok(KvQuant::RotorKOnly4),
            _ => {}
        }

        // Withdrawn codecs: reject by name, and name the successor. Not an
        // alias — a retired name has to keep failing, or a recorded bench cell
        // or a saved CLI line would keep running under a codec it does not
        // name. `rot_k_tq4v` (rotated affine-8 K + TurboQuant-4 V) rebuilt a
        // full bf16 K *and* V from its packed store on every decode step and
        // then ran ordinary bf16 SDPA; `rot_k_v4g64` is the same rotated 8-bit
        // K with an MLX-affine 4-bit V that `mixed_quantized_sdpa` consumes
        // without materialising either axis.
        if s == "rot_k_tq4v" {
            return Err(KvQuantParseError::Retired {
                input: s.to_string(),
                replacement: "rot_k_v4g64",
            });
        }

        // "rot_k_v<vb>g<vg>" — RotK Display form round-trip.
        if let Some(rest) = s.strip_prefix("rot_k_v") {
            // The shape already matched, so a malformed numeric component is a
            // bad `rot_k_*` spelling and not an unknown codec: reporting it as
            // `Unknown` printed the whole codec list and never said which part
            // of the tag failed.
            let mk_err = |reason: String| KvQuantParseError::InvalidRotK {
                input: s.to_string(),
                reason,
            };
            let (v_bits, v_group_size) = rest
                .split_once('g')
                .ok_or_else(|| mk_err(format!("missing 'g' separator in 'v{rest}'")))
                .and_then(|(bits_str, group_str)| {
                    let v_bits: u8 = bits_str
                        .parse()
                        .map_err(|e| mk_err(format!("bad v_bits in 'v{rest}': {e}")))?;
                    let v_group_size: u16 = group_str
                        .parse()
                        .map_err(|e| mk_err(format!("bad v_group_size in 'v{rest}': {e}")))?;
                    Ok((v_bits, v_group_size))
                })?;
            // RotK's V slot *is* Mixed's V slot: `KvStorage::new` builds it
            // with `MixedKvState::new_rotated`, which hands (bits, group_size)
            // to the same MLX affine quantizer. Validating it with the same
            // function keeps the two from accepting different sets — the arm
            // used to accept every `u8` / `u16` pair, so `rot_k_v99g7` parsed
            // into a codec whose first encode would ask for a 99-bit quantize.
            validate_mixed_side('v', v_bits, v_group_size).map_err(mk_err)?;
            return Ok(KvQuant::RotK {
                v_bits,
                v_group_size,
            });
        }

        // "rotor_k_3_asym_v<vb>_g<vg>" / "rotor_k_4_asym_v<vb>_g<vg>".
        if let Some(rest) = s.strip_prefix("rotor_k_3_asym_") {
            let (v_bits, v_group_size) = parse_rotor_k_asym_v_suffix(rest, s)?;
            return Ok(KvQuant::RotorK3Asym {
                v_bits,
                v_group_size,
            });
        }
        if let Some(rest) = s.strip_prefix("rotor_k_4_asym_") {
            let (v_bits, v_group_size) = parse_rotor_k_asym_v_suffix(rest, s)?;
            return Ok(KvQuant::RotorK4Asym {
                v_bits,
                v_group_size,
            });
        }

        // Mixed shape: "mixed_k<kb>g<kg>_v<vb>g<vg>".
        if let Some(rest) = s.strip_prefix("mixed_") {
            // Split on the single '_' between the K- and V-side specs.
            let (k_part, v_part) =
                rest.split_once('_')
                    .ok_or_else(|| KvQuantParseError::InvalidMixed {
                        input: s.to_string(),
                        reason: "missing '_' between K-side and V-side".to_string(),
                    })?;

            let (k_bits, k_group_size) =
                parse_kv_side(k_part, 'k').map_err(|reason| KvQuantParseError::InvalidMixed {
                    input: s.to_string(),
                    reason,
                })?;
            let (v_bits, v_group_size) =
                parse_kv_side(v_part, 'v').map_err(|reason| KvQuantParseError::InvalidMixed {
                    input: s.to_string(),
                    reason,
                })?;
            validate_mixed_side('k', k_bits, k_group_size).map_err(|reason| {
                KvQuantParseError::InvalidMixed {
                    input: s.to_string(),
                    reason,
                }
            })?;
            validate_mixed_side('v', v_bits, v_group_size).map_err(|reason| {
                KvQuantParseError::InvalidMixed {
                    input: s.to_string(),
                    reason,
                }
            })?;

            return Ok(KvQuant::Mixed {
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
            });
        }

        Err(KvQuantParseError::Unknown(s.to_string()))
    }
}

/// Parse the `v<vb>_g<vg>` suffix of a `rotor_k_*_asym_*` tag and validate
/// the resulting tuple against [`validate_rotor_k_asym_v`].
///
/// The Display form uses an underscore between the bits and group-size
/// components (e.g. `v4_g128`), unlike the `mixed_*` Display form
/// (`v4g128`). This dedicated parser handles the underscore-separated
/// shape; [`parse_kv_side`] is preserved for the legacy `mixed_*` syntax.
fn parse_rotor_k_asym_v_suffix(
    rest: &str,
    full_input: &str,
) -> Result<(u8, u16), KvQuantParseError> {
    let mk_err = |reason: String| KvQuantParseError::InvalidRotorKAsym {
        input: full_input.to_string(),
        reason,
    };
    let v_part = rest
        .strip_prefix('v')
        .ok_or_else(|| mk_err(format!("expected 'v' prefix in '{rest}'")))?;
    let (bits_str, group_part) = v_part
        .split_once("_g")
        .ok_or_else(|| mk_err(format!("missing '_g' separator in '{rest}'")))?;
    let v_bits: u8 = bits_str
        .parse()
        .map_err(|e| mk_err(format!("bad v_bits in '{rest}': {e}")))?;
    let v_group_size: u16 = group_part
        .parse()
        .map_err(|e| mk_err(format!("bad v_group_size in '{rest}': {e}")))?;
    validate_rotor_k_asym_v(v_bits, v_group_size).map_err(mk_err)?;
    Ok((v_bits, v_group_size))
}

/// Parse a side-spec like `"k8g128"` or `"v4g64"` into `(bits, group_size)`.
fn parse_kv_side(spec: &str, expected_prefix: char) -> Result<(u8, u16), String> {
    let rest = spec
        .strip_prefix(expected_prefix)
        .ok_or_else(|| format!("expected '{expected_prefix}' prefix in '{spec}'"))?;
    let (bits_str, group_str) = rest
        .split_once('g')
        .ok_or_else(|| format!("missing 'g' separator in '{spec}'"))?;
    let bits: u8 = bits_str
        .parse()
        .map_err(|e| format!("bad bits in '{spec}': {e}"))?;
    let group_size: u16 = group_str
        .parse()
        .map_err(|e| format!("bad group_size in '{spec}': {e}"))?;
    Ok((bits, group_size))
}

#[cfg(test)]
#[path = "quant_tests.rs"]
mod quant_tests;
