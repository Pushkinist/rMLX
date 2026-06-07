//! `KvQuant` — closed enum tagging the active KV codec.
//!
//! Moved from `rmlx-models::kv_cache::mod`. The enum, its
//! `Display` / `FromStr` impls, and the `KV_MAX_SEQ_DEFAULT` constant live
//! here because every codec module in this crate (`storage`, `kvcache`,
//! `mixed_quant`) carries `KvQuant` as a tag.
//!
//! Policy wrappers (`KvCacheBuilder`, `kv_quant_for_layer`,
//! `kv_quant_for_ctx`, `ResolverSignals`) remain in `rmlx-models::kv_cache`
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
    /// CLAUDE.md mandates this for Qwen MoE — symmetric Q4 on K is the
    /// PPL-218→8641 disaster. Per-axis split is real — K and V are quantized
    /// independently, not by layer index like the Python fork's fake `8,4` flag.
    K8V4,
    /// K = q8_0, V = q8_0 (symmetric 8-bit). Per-arch default.
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
    /// Opt-in only via `--kv-quant none`
    /// (alias `bf16`). Auto-resolver default is unchanged (still K8V8 per
    /// this variant exists for apples-to-apples comparison against
    /// mlx-lm's bf16-KV champion.
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
    /// Currently wired for `Qwen3ForCausalLM` only (Bonsai-2bit) via
    /// `KvCacheBuilder::resolve_default`; other archs continue to use their
    /// existing per-arch defaults.
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
        v_bits: u8,
        /// V affine group size.
        v_group_size: u16,
    },
    /// K-side rotation + TurboQuant V: K is the same rotated-affine 8-bit
    /// codec as [`RotK`](KvQuant::RotK), but V is stored as **TurboQuant 4-bit**
    /// (Lloyd-Max N(0,1) codebook, group=32) instead of MLX affine.
    ///
    /// Storage: K-side uses [`MixedKvState`](crate::mixed_quant::MixedKvState)
    /// with `rotate_k=true` (same as RotK). V-side uses
    /// [`QuantV`](crate::storage::QuantV) (same as K8V4).
    ///
    /// SDPA: dequantize K from its MLX affine 3-tuple to bf16, pre-rotate Q,
    /// dequantize V from TurboFlash to bf16, run standard scaled_dot_product_attention.
    /// This is a dequant-then-SDPA path (not fused `quantized_matmul`) but gives the
    /// full memory savings of tq4 V (4-bit vs 16-bit).
    ///
    /// Requires power-of-two `head_dim` (Hadamard) and `head_dim ∈ {128, 256}` (tq4).
    /// Opt-in only via `--ctk rot_k --ctv tq4`; never an auto default.
    RotKTq4V,
    /// K = affine q8_0 (group_size=128),
    /// V = TurboQuant **3-bit** Lloyd-Max N(0,1) codebook (group=32).
    ///
    /// Auto default for Gemma4 small (non-MoE, hidden_size ≤ 2560, non-paroquant).
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
    /// softmax; see 218→8641 baseline in CLAUDE.md). `KvCacheBuilder::resolve_default`
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
    /// softmax). `KvCacheBuilder::resolve_default` never returns `PlanarK` for
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
    /// amplifies K-head error through softmax). `KvCacheBuilder::resolve_default`
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
    /// ~10.7 bpe pre-scale at bits=4 (single-codebook simplification; grade-aware
    /// split deferred per spec).
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
    /// [`crate::rotor_qjl::rotor_qjl_enabled`] is `true` at first append
    /// (default ON).
    ///
    /// **Arch guard (Contract A.y — mandatory)**: K-side ≤4-bit on Qwen MoE
    /// is the PPL-disaster zone (218→8641 on Q4_K_M baseline; 7:1 GQA
    /// amplifies K-head error through softmax). `KvCacheBuilder::resolve_default`
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
    /// path; the MSL Viterbi kernel ships as a future-reference hook in
    /// `tcq_v2_msl.rs`.
    ///
    /// Opt-in **only** via `--kv-quant k8vturbo2tcq`; never an auto default.
    K8VTurbo2Tcq,
}

impl KvQuant {
    /// Issue #26: stable per-codec salt for namespacing the in-RAM
    /// prompt/prefix cache key by KV codec.
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

    /// True for the RotKTq4V hybrid path: rotated K (affine 8-bit) + tq4 V.
    /// Dispatches through `rot_k_tq4v_sdpa` (dequant-then-SDPA, not fused matmul).
    pub fn uses_rot_k_tq4v_path(&self) -> bool {
        matches!(self, KvQuant::RotKTq4V)
    }

    /// True for KV codecs that put K below 8-bit, which is catastrophic on
    /// Qwen MoE (PPL 218 → 8641 per CLAUDE.md). Used by `validate_resolved` to
    /// hard-reject explicit `--kv-quant tsym4` on a Qwen MoE arch. `K8V4` /
    /// `K8V8` / `Planar` / `K8VTurbo3` keep K at 8-bit and are NOT included.
    ///
    /// `PlanarK` is guarded separately via a dedicated `QwenMoePlanarKRejected`
    /// error (Contract A.y) before this check runs.
    pub fn refuses_qwen_moe(&self) -> bool {
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
    /// `exit_prefill` materialises (the warm-TTFT shortcut codecs: K8V*, *Sym,
    /// Planar*, Turbo*, Iso*, Rotor* with a quantised-or-bf16 K mirror that
    /// decode consults).
    ///
    /// **False for the K-only family** (`IsoKOnly3/4`, `RotorKOnly3/4`): these
    /// re-quantise K at every decode step (`ks.append`) and route V through
    /// `update_decode_fp16_v_only`, so they never read `decode_fp16_k`. For
    /// those variants `exit_prefill` skips populating the bf16 K seed — it was
    /// dead memory. The bf16 **V** seed (`decode_fp16_v`) is still populated for
    /// them; only the K seed is gated.
    ///
    /// The match is **exhaustive on purpose** (no wildcard `_`): adding a new
    /// `KvQuant` variant will produce a compile error until it is classified
    /// here, preventing a new K-only-style variant from silently defaulting to
    /// `true` and reintroducing the F2 dead-seed leak.
    pub fn feeds_bf16_k_at_decode(&self) -> bool {
        match self {
            // K-only family: K is re-quantised at every decode step; decode
            // never reads decode_fp16_k. exit_prefill skips the K seed for
            // these variants.
            KvQuant::IsoKOnly3
            | KvQuant::IsoKOnly4
            | KvQuant::RotorKOnly3
            | KvQuant::RotorKOnly4 => false,
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
            | KvQuant::RotKTq4V
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
            | KvQuant::Rotor3
            | KvQuant::Rotor4
            | KvQuant::Rotor3Sym
            | KvQuant::Rotor4Sym
            | KvQuant::RotorK3Asym { .. }
            | KvQuant::RotorK4Asym { .. } => true,
        }
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
}

// ── KvQuant Display / FromStr ────────────────────────────────────────────────

/// Error returned by `<KvQuant as FromStr>::from_str`.
#[non_exhaustive]
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvQuantParseError {
    /// The whole input does not match any canonical KvQuant form.
    #[error(
        "unknown KvQuant '{0}' — valid: none, k8v4, k8v8, planar, planar3, planar_k, k8vturbo3, k8vturbo3tcq, k8vturbo2tcq, tsym3, tsym4, k8vturbo2, iso3, iso4, iso3_sym, iso4_sym, k_iso3, k_iso4, rotor3, rotor4, rotor3_sym, rotor4_sym, k_rotor3, k_rotor4, rotor_k_3_asym_v<vb>_g<vg>, rotor_k_4_asym_v<vb>_g<vg>, rot_k_tq4v, rot_k_v<vb>g<vg>, mixed_k<kb>g<kg>_v<vb>g<vg>"
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
            KvQuant::RotKTq4V => f.write_str("rot_k_tq4v"),
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

        if s == "rot_k_tq4v" {
            return Ok(KvQuant::RotKTq4V);
        }

        // "rot_k_v<vb>g<vg>" — RotK Display form round-trip.
        if let Some(rest) = s.strip_prefix("rot_k_v") {
            let (v_bits, v_group_size) = rest
                .split_once('g')
                .ok_or_else(|| KvQuantParseError::Unknown(s.to_string()))
                .and_then(|(bits_str, group_str)| {
                    let v_bits: u8 = bits_str
                        .parse()
                        .map_err(|_| KvQuantParseError::Unknown(s.to_string()))?;
                    let v_group_size: u16 = group_str
                        .parse()
                        .map_err(|_| KvQuantParseError::Unknown(s.to_string()))?;
                    Ok((v_bits, v_group_size))
                })?;
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
