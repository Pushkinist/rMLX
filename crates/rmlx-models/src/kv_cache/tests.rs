// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret in test helpers
#![cfg_attr(test, allow(unsafe_code))]

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::{
        kv_codec_net_saving_total, kv_layer_quants, kv_max_seq_and_ceiling, kv_quant_for_layer,
        lookup_layer_calibration, KvCacheBuilder, KvLayerShape, LAYER_ADAPTIVE_HEAD_N,
        LAYER_ADAPTIVE_TAIL_N,
    };
    use rmlx_kv_quant::kvcache::KvCache;
    use rmlx_kv_quant::storage::KvStorage;
    use rmlx_kv_quant::{KvQuant, SharedKv, KV_MAX_SEQ_DEFAULT};
    use rmlx_loader::KvCalibration;
    use rmlx_mlx::{Array, Device, Dtype};

    /// Unwrap a producer's share into `(out, K, V)`.
    ///
    /// Every cache in this module runs on `Device::Cpu`, on a rotating ring, or
    /// on the Mixed path — none of which can reach a fused-over-store arm — so
    /// the share is always bf16 here. A `Store` share would mean a dispatch gate
    /// regressed, hence the panic rather than a silent dequant.
    #[allow(
        clippy::panic,
        reason = "test helper: a Store share on a CPU/rotating/Mixed cache is a gate regression, and must fail the test loudly"
    )]
    fn split_bf16_share(pair: (Array, SharedKv)) -> (Array, Array, Array) {
        let (out, share) = pair;
        match share {
            SharedKv::Bf16(k, v) => (out, k, v),
            SharedKv::Store { .. } => {
                panic!("expected a bf16 share — no fused-over-store arm is eligible here")
            }
        }
    }

    // Helper: make a [B, kv_h, S, D] F32 array with deterministic LCG data in [-1, 1].
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn make_lcg_array(shape: &[i32], seed: u64) -> (Array, Vec<f32>) {
        let n: usize = shape.iter().map(|&x| x as usize).product();
        let mut state = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let frac = ((state >> 33) as f32) / (u32::MAX as f32);
                frac * 2.0 - 1.0
            })
            .collect();
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        let arr = Array::from_bytes(bytes, shape, Dtype::F32).expect("make_lcg_array failed");
        (arr, data)
    }

    // Helper: extract f32 values from an Array (must be F32 dtype, already materialised).
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn array_to_vec(a: &Array) -> Vec<f32> {
        a.eval().unwrap();
        let bytes = a.to_bytes().unwrap();
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn kv_cache_quant_k8v4_roundtrip_within_tolerance() {
        let device = Device::Cpu;
        let shape: &[i32] = &[1, 2, 3, 128];

        let (new_k, k_data) = make_lcg_array(shape, 0xCAFE_BABE_u64);
        let (new_v, v_data) = make_lcg_array(shape, 0xDEAD_BEEF_u64);

        let mut cache = KvCache::with_quant(KvQuant::K8V4);
        let (k_full, v_full) = cache
            .update(&new_k, &new_v, device)
            .expect("K8V4 update failed");

        assert_eq!(k_full.shape(), vec![1, 2, 3, 128], "K shape mismatch");
        assert_eq!(v_full.shape(), vec![1, 2, 3, 128], "V shape mismatch");

        let k_recon = array_to_vec(&k_full);
        let v_recon = array_to_vec(&v_full);

        let k_max_err = k_data
            .iter()
            .zip(&k_recon)
            .map(|(&o, &r)| (o - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            k_max_err < 0.05,
            "K max abs error {k_max_err:.6} exceeds tolerance 0.05 for q8_0"
        );

        let v_max_err = v_data
            .iter()
            .zip(&v_recon)
            .map(|(&o, &r)| (o - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            v_max_err < 0.15,
            "V max abs error {v_max_err:.6} exceeds tolerance 0.15 for TurboQuant V4"
        );

        assert!(
            k_recon.iter().all(|v| v.is_finite()),
            "K contains non-finite"
        );
        assert!(
            v_recon.iter().all(|v| v.is_finite()),
            "V contains non-finite"
        );
    }

    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn kv_cache_quant_planar_roundtrip_within_tolerance() {
        let device = Device::Cpu;
        let shape: &[i32] = &[1, 2, 3, 128];

        let (new_k, k_data) = make_lcg_array(shape, 0xCAFE_BABE_u64);
        let (new_v, v_data) = make_lcg_array(shape, 0xDEAD_BEEF_u64);

        let mut cache = KvCache::with_quant(KvQuant::Planar);
        let (k_full, v_full) = cache
            .update(&new_k, &new_v, device)
            .expect("Planar update failed");

        assert_eq!(k_full.shape(), vec![1, 2, 3, 128], "K shape mismatch");
        assert_eq!(v_full.shape(), vec![1, 2, 3, 128], "V shape mismatch");

        let k_recon = array_to_vec(&k_full);
        let v_recon = array_to_vec(&v_full);

        let k_max_err = k_data
            .iter()
            .zip(&k_recon)
            .map(|(&o, &r)| (o - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            k_max_err < 0.05,
            "Planar K max abs error {k_max_err:.6} exceeds 0.05 for q8_0"
        );

        let v_max_err = v_data
            .iter()
            .zip(&v_recon)
            .map(|(&o, &r)| (o - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            v_max_err < 0.10,
            "Planar V max abs error {v_max_err:.6} exceeds 0.10 for PlanarQuant V4"
        );

        assert!(
            k_recon.iter().all(|v| v.is_finite()),
            "K contains non-finite"
        );
        assert!(
            v_recon.iter().all(|v| v.is_finite()),
            "V contains non-finite"
        );
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_cache_k8v4_two_updates_shape() {
        let device = Device::Cpu;

        let (k1, _) = make_lcg_array(&[1, 2, 3, 128], 1);
        let (v1, _) = make_lcg_array(&[1, 2, 3, 128], 2);

        let mut cache = KvCache::with_quant(KvQuant::K8V4);
        let (kf1, vf1) = cache.update(&k1, &v1, device).unwrap();
        assert_eq!(kf1.shape(), vec![1, 2, 3, 128]);
        assert_eq!(vf1.shape(), vec![1, 2, 3, 128]);
        assert_eq!(cache.offset(), 3);

        let (k2, _) = make_lcg_array(&[1, 2, 1, 128], 3);
        let (v2, _) = make_lcg_array(&[1, 2, 1, 128], 4);
        let (kf2, vf2) = cache.update(&k2, &v2, device).unwrap();
        assert_eq!(kf2.shape(), vec![1, 2, 4, 128], "accumulated K shape wrong");
        assert_eq!(vf2.shape(), vec![1, 2, 4, 128], "accumulated V shape wrong");
        assert_eq!(cache.offset(), 4);
    }

    #[test]
    fn linear_attn_cache_default_empty() {
        use rmlx_kv_quant::LinearAttnCache;
        let c = LinearAttnCache::new();
        assert!(c.conv_state.is_none(), "fresh cache must have no conv tail");
        assert!(
            c.delta_state.is_none(),
            "fresh cache must have no delta state"
        );
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn linear_attn_cache_reset_clears_both_states() {
        use rmlx_kv_quant::LinearAttnCache;
        let mut c = LinearAttnCache::new();
        let arr = Array::from_bytes(&[0u8; 16], &[1, 4], Dtype::F32).unwrap();
        c.conv_state = Some(arr.try_clone().unwrap());
        c.delta_state = Some(arr);
        assert!(c.conv_state.is_some());
        assert!(c.delta_state.is_some());

        c.reset();
        assert!(c.conv_state.is_none(), "reset must drop conv tail");
        assert!(c.delta_state.is_none(), "reset must drop delta state");
    }

    #[test]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    fn with_quant_max_seq_stores_correct_capacity() {
        let c_default = KvCache::with_quant(KvQuant::K8V4);
        let default_max = match &c_default.storage {
            KvStorage::K8V4 { max_seq, .. } => *max_seq,
            _ => panic!("expected K8V4 storage"),
        };
        assert_eq!(
            default_max, KV_MAX_SEQ_DEFAULT,
            "with_quant(K8V4) must cap at KV_MAX_SEQ_DEFAULT={KV_MAX_SEQ_DEFAULT}"
        );

        let c_long = KvCache::with_quant_max_seq(KvQuant::K8V4, 8192);
        let long_max = match &c_long.storage {
            KvStorage::K8V4 { max_seq, .. } => *max_seq,
            _ => panic!("expected K8V4 storage"),
        };
        assert_eq!(
            long_max, 8192,
            "with_quant_max_seq(K8V4, 8192) must store max_seq=8192, not KV_MAX_SEQ_DEFAULT"
        );
    }

    // ── resident_bytes unit tests ─────────────────────────────────────────────
    //
    // These assert against the buffers each cache really allocated, never
    // against a per-codec bits-per-element figure. A nominal bit-width is not a
    // cache's memory: q8_0 also carries per-group scales, and the quantized
    // paths carry warm-TTFT bf16 seeds on top of the codes. Restating a formula
    // here would only re-derive the accounting from itself and would go blind
    // the moment a store grows a buffer.

    /// Zero offset → zero bytes regardless of quant mode.
    #[test]
    fn resident_bytes_empty_cache_is_zero() {
        for quant in [KvQuant::K8V8, KvQuant::K8V4, KvQuant::None, KvQuant::Planar] {
            let cache = KvCache::with_quant(quant);
            assert_eq!(
                cache.resident_bytes(),
                0,
                "empty {quant:?} cache must be 0 bytes"
            );
        }
    }

    /// `KvQuant::None` holds nothing but the two bf16 mirrors, so its residency
    /// is exactly their filled prefix — and its storage contributes nothing.
    ///
    /// Shape: B=1, kv_h=4, seq=256, head_dim=128 → 256 × 4 × 128 × 2 B per
    /// mirror, two mirrors.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn resident_bytes_none_quant_is_the_two_bf16_mirrors() {
        let device = Device::Cpu;
        // Use a large max_seq so the cache isn't truncated.
        let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 4096);
        // enter_prefill + update + exit_prefill so decode_fp16 buffers are set.
        cache.enter_prefill();
        let k = make_lcg_array(&[1, 4, 256, 128], 0xABCD).0;
        let v = make_lcg_array(&[1, 4, 256, 128], 0x1234).0;
        cache.update(&k, &v, device).expect("update must not fail");
        cache
            .exit_prefill(device)
            .expect("exit_prefill must not fail");

        assert_eq!(
            cache.storage().resident_bytes(),
            0,
            "KvStorage::None is geometry-only — the bf16 K/V live on the cache's mirrors"
        );
        // Two bf16 mirrors at the filled length: 256 × 4 × 128 × 2 B each.
        let one_mirror: u64 = 256 * 4 * 128 * 2;
        assert_eq!(
            cache.resident_bytes(),
            2 * one_mirror,
            "None (bf16) cache must report exactly its two bf16 mirrors"
        );
    }

    /// `KvQuant::K8V8` residency after a prefill is **exactly** the two
    /// warm-TTFT bf16 mirrors — the same bytes a `KvQuant::None` cache of the
    /// same shape holds — because its decode reads those mirrors and nothing
    /// reads a packed store, so `exit_prefill` builds none.
    ///
    /// The `None` cache is the oracle rather than a byte formula: it shares no
    /// arithmetic with the accounting under test, and a store built anyway
    /// (codes *and* per-group scales) shows up immediately as the two totals
    /// diverging.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn resident_bytes_k8v8_is_the_two_bf16_mirrors_and_no_store() {
        let device = Device::Cpu;
        let shape = [1, 4, 256, 128];
        let prefilled = |quant| {
            let mut cache = KvCache::with_quant_max_seq(quant, 4096);
            cache.enter_prefill();
            let k = make_lcg_array(&shape, 0xBEEF).0;
            let v = make_lcg_array(&shape, 0xCAFE).0;
            cache.update(&k, &v, device).expect("update must not fail");
            cache
                .exit_prefill(device)
                .expect("exit_prefill must not fail");
            cache
        };
        let quantized = prefilled(KvQuant::K8V8);
        let bf16 = prefilled(KvQuant::None);

        assert_eq!(
            quantized.storage().resident_bytes(),
            0,
            "K8V8 must hold no packed store after prefill — nothing reads it at decode"
        );
        assert_eq!(
            quantized.resident_bytes(),
            bf16.resident_bytes(),
            "K8V8 residency must equal plain bf16 at the same shape: both hold two \
             bf16 mirrors and nothing else"
        );
        // And the figure is the mirrors, not zero — a cache that reported
        // nothing at all would pass the equality above vacuously.
        let seq: u64 = 256;
        let bhd: u64 = 4 * 128; // kv_h=4, head_dim=128 (B=1 implicit)
        assert_eq!(
            quantized.resident_bytes(),
            seq * bhd * 2 * 2,
            "the reported bytes must be the two bf16 mirrors at the filled length"
        );
    }

    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    fn kv_cache_truncate_k8v8_path() {
        let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V8, 32);

        let device = Device::Cpu;
        let k = make_lcg_array(&[1, 1, 8, 128], 10).0;
        let v = make_lcg_array(&[1, 1, 8, 128], 11).0;
        cache.update(&k, &v, device).expect("update must not fail");
        assert_eq!(cache.offset(), 8);

        cache.truncate_to(3);
        assert_eq!(cache.offset(), 3, "offset must equal truncation target");
        match &cache.storage {
            KvStorage::K8V8 { k, v, .. } => {
                if let Some(qk) = k {
                    assert_eq!(
                        qk.shape[2], 3,
                        "QuantK shape[2] must equal truncation target"
                    );
                }
                if let Some(qv) = v {
                    assert_eq!(
                        qv.shape[2], 3,
                        "QuantV shape[2] must equal truncation target"
                    );
                }
            }
            _ => panic!("expected K8V8 storage"),
        }
    }

    /// Falsifies #284 at the real production entry point: `KvCache::update` →
    /// dispatch → `QuantIsoV3::append`, then `KvCache::truncate_to` → dispatch
    /// → `QuantIsoV3::truncate_to`, at `kv_h = 4` (`kv_h == 1` is the masked
    /// case that hides the bug — see `kv_cache_truncate_k8v8_path` above,
    /// which stays at `kv_h == 1` on purpose as the pre-existing regression
    /// baseline).
    ///
    /// One token appended per `update` call so every block is exactly one
    /// sequence position (block boundaries align with truncate targets).
    /// This is the same production path SWA-context-slide, speculative-decode
    /// rollback, and prompt-cache partial-prefix trim all drive — see
    /// `docs/KV_QUANT.md` for the reachability audit per arch (Bonsai-8B
    /// currently has no live serve trigger; Gemma4's prompt-cache `Partial`
    /// reuse policy does).
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "shape[2] index bounded by the codec's fixed 4-element [B, kv_h, S, D] shape, established by construction"
    )]
    fn kv_cache_truncate_iso3_kv_h_gt_1_path() {
        let mut cache = KvCache::with_quant_max_seq(KvQuant::Iso3, 32);

        let device = Device::Cpu;
        let kv_h = 4_i32;
        let head_dim = 32_i32; // kv_h * head_dim = 128 = Q8_GROUP_SIZE (K-side q8_0)
        let total_tokens = 8;
        for tok in 0..total_tokens {
            let k = make_lcg_array(&[1, kv_h, 1, head_dim], 10 + tok as u64).0;
            let v = make_lcg_array(&[1, kv_h, 1, head_dim], 11 + tok as u64).0;
            cache.update(&k, &v, device).expect("update must not fail");
        }
        assert_eq!(cache.offset(), total_tokens);

        let keep = 3;
        cache.truncate_to(keep);
        assert_eq!(cache.offset(), keep, "offset must equal truncation target");
        match &cache.storage {
            KvStorage::IsoV3 { v, .. } => {
                let vs = v
                    .as_ref()
                    .expect("V codec must be populated after 8 appends");
                assert_eq!(
                    vs.shape[2], keep,
                    "QuantIsoV3 shape[2] must equal truncation target"
                );
                assert_eq!(
                    vs.blocks.len(),
                    keep as usize,
                    "must keep exactly `keep` blocks, not floor(keep / kv_h) (#284)"
                );
                let kept_rows: usize = vs.blocks.iter().map(|b| b.n_tokens).sum();
                assert_eq!(
                    kept_rows,
                    keep as usize * kv_h as usize,
                    "kept rows must equal keep * kv_h, not keep (#284)"
                );
                vs.dequant()
                    .expect("dequant must succeed after truncate at kv_h>1 (#284)");
            }
            _ => panic!("expected IsoV3 storage"),
        }
    }

    // ── kv_quant_for_layer unit tests ────────────────────────────────────────

    #[test]
    fn kv_quant_for_layer_returns_base_for_middle_layers() {
        // With tail_n=8, head_n=2, n_layers=40, layers 2..32 (exclusive)
        // should return the base quant unchanged.
        let n = 40;
        let base = KvQuant::K8V4;
        for i in 2..32 {
            let q = kv_quant_for_layer(i, n, base, 8, 2, false);
            assert_eq!(q, base, "middle layer {i} should return base quant");
        }
    }

    #[test]
    fn kv_quant_for_layer_overrides_tail_to_k8v8() {
        // With tail_n=8 and n_layers=40, layers 32..40 should be K8V8.
        let n = 40;
        let base = KvQuant::K8V4;
        for i in 32..40 {
            let q = kv_quant_for_layer(i, n, base, 8, 0, false);
            assert_eq!(q, KvQuant::K8V8, "tail layer {i} should be K8V8");
        }
    }

    #[test]
    fn kv_quant_for_layer_overrides_head_to_k8v8() {
        // With head_n=2 and n_layers=40, layers 0..2 should be K8V8.
        let n = 40;
        let base = KvQuant::K8V4;
        for i in 0..2 {
            let q = kv_quant_for_layer(i, n, base, 0, 2, false);
            assert_eq!(q, KvQuant::K8V8, "head layer {i} should be K8V8");
        }
        // Layer 2 is not a head layer.
        assert_eq!(
            kv_quant_for_layer(2, n, base, 0, 2, false),
            base,
            "layer 2 should not be overridden"
        );
    }

    #[test]
    fn kv_quant_for_layer_zero_tail_and_head_is_noop() {
        // tail_n=0 and head_n=0 should never override — returns base for all.
        let n = 10;
        let base = KvQuant::Planar;
        for i in 0..n {
            let q = kv_quant_for_layer(i, n, base, 0, 0, false);
            assert_eq!(q, base, "zero tail+head should not override layer {i}");
        }
    }

    #[test]
    fn kv_quant_for_layer_k8v8_base_is_noop() {
        // If base is already K8V8, any override is the same — no regression.
        let n = 10;
        let base = KvQuant::K8V8;
        for i in 0..n {
            let q = kv_quant_for_layer(
                i,
                n,
                base,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
                false,
            );
            assert_eq!(q, KvQuant::K8V8, "K8V8 base should stay K8V8 at layer {i}");
        }
    }

    /// `KvQuant::None` is exempt from the boundary promotion on every layer.
    ///
    /// The promotion recovers quantization loss; an unquantized base has none,
    /// so promoting it would allocate a packed q8_0 K+V store on top of the
    /// bf16 buffers the layer already holds and could only lower its
    /// precision. Pinned at Ternary-Bonsai-8B's shape (36 layers, all
    /// full-attention), the arch where the promotion used to bite hardest —
    /// +14.3% resident KV under `--kv-quant none` for zero numerical effect.
    #[test]
    fn kv_quant_for_layer_leaves_none_base_alone() {
        let n = 36;
        let base = KvQuant::None;
        let promoted: Vec<usize> = (0..n)
            .filter(|&i| {
                kv_quant_for_layer(
                    i,
                    n,
                    base,
                    LAYER_ADAPTIVE_TAIL_N,
                    LAYER_ADAPTIVE_HEAD_N,
                    false,
                ) != base
            })
            .collect();
        assert!(
            promoted.is_empty(),
            "`none` must be passed through on every layer, promoted on {promoted:?}"
        );
    }

    /// The exemption is keyed on "quantizes neither side", not on a codec name.
    ///
    /// Every quantizing base — including the K-only families, whose V side is
    /// already bf16, and the equal-width-but-different-family parametric bases
    /// (`mixed_k8g64_v8g64`, `rot_k_v8g64`) — must still be promoted at the
    /// boundaries. Only a base whose `approx_code_bits` is model-dtype on both
    /// sides is passed through.
    #[test]
    fn kv_quant_for_layer_promotes_every_quantizing_base() {
        let quantizing: &[KvQuant] = &[
            KvQuant::K8V4,
            KvQuant::K8V8,
            KvQuant::Planar,
            KvQuant::Planar3,
            KvQuant::PlanarK,
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
            KvQuant::Mixed {
                k_bits: 8,
                v_bits: 4,
                k_group_size: 64,
                v_group_size: 64,
            },
            KvQuant::Mixed {
                k_bits: 8,
                v_bits: 8,
                k_group_size: 64,
                v_group_size: 64,
            },
            KvQuant::RotK {
                v_bits: 4,
                v_group_size: 64,
            },
            KvQuant::RotK {
                v_bits: 8,
                v_group_size: 64,
            },
            KvQuant::RotorK3Asym {
                v_bits: 4,
                v_group_size: 64,
            },
            KvQuant::RotorK4Asym {
                v_bits: 4,
                v_group_size: 64,
            },
        ];
        // The width a side actually decodes at. `approx_code_bits` reports what
        // the codebook quantizes to, which for a side whose decode reads the
        // bf16 mirror is not what reaches the kernel — `planar_k` codes K at 4
        // bits and decodes it at model dtype. Reading the mirror predicates
        // first is what keeps this test from calling a bf16 side "8-bit".
        let effective = |q: KvQuant, shares_kv: bool| -> (u32, u32) {
            let (k, v) = q.approx_code_bits();
            (
                if q.feeds_bf16_k_at_decode(shares_kv) {
                    16
                } else {
                    k
                },
                if q.feeds_bf16_v_at_decode(shares_kv) {
                    16
                } else {
                    v
                },
            )
        };
        let n = 36;
        for shares_kv in [false, true] {
            for &base in quantizing {
                let (base_k, base_v) = effective(base, shares_kv);
                for i in [0, 1, 28, 35] {
                    let promoted = kv_quant_for_layer(
                        i,
                        n,
                        base,
                        LAYER_ADAPTIVE_TAIL_N,
                        LAYER_ADAPTIVE_HEAD_N,
                        shares_kv,
                    );
                    let (k_bits, v_bits) = effective(promoted, shares_kv);
                    // "At or above the floor", not "quantized to 8": a side
                    // that decodes bf16 reports 16 and satisfies this — an
                    // `IsoKOnly4` boundary layer does exactly that on V. Which
                    // target delivers the floor is asserted per family by
                    // `kv_quant_for_layer_promotion_target_per_family`, and its
                    // byte consequence by
                    // `store_bearing_boundary_promotion_never_costs_more`.
                    assert!(
                        k_bits >= 8 && v_bits >= 8,
                        "boundary layer {i} of quantizing base {base} \
                         (shares_kv={shares_kv}) must decode at or above the 8-bit floor, \
                         got {promoted} ({k_bits}, {v_bits})"
                    );
                    // A floor may only raise a side. Distinct from the check
                    // above, which a target that raised K to 8 while dropping a
                    // bf16 V to 8 would still pass.
                    assert!(
                        k_bits >= base_k && v_bits >= base_v,
                        "boundary layer {i} of {base} (shares_kv={shares_kv}) promoted to \
                         {promoted}, decoding at ({k_bits}, {v_bits}) against the base's \
                         ({base_k}, {base_v}) — a floor must not lower a side"
                    );
                }
                assert_eq!(
                    kv_quant_for_layer(
                        10,
                        n,
                        base,
                        LAYER_ADAPTIVE_TAIL_N,
                        LAYER_ADAPTIVE_HEAD_N,
                        shares_kv,
                    ),
                    base,
                    "middle layer of {base} (shares_kv={shares_kv}) must stay on the base codec"
                );
            }
        }
    }

    /// The promotion target, pinned per family and per topology.
    ///
    /// Off a stack that keeps no bf16 mirror, a base whose widths are
    /// parameters keeps its family and raises both axes to 8 bits; everything
    /// else switches to `K8V8`. On a stack that keeps the mirror there is no
    /// in-family target — the store would be charged on top of two bf16
    /// buffers — so every base takes `K8V8`.
    ///
    /// Companion to the property assertion above, which only checks the width.
    /// This is also where the non-trivial parametric rewrites live:
    /// `ALL_KV_QUANTS` carries a single `RotK`, already at `v_bits: 8`, so the
    /// sweep in `store_bearing_boundary_promotion_never_costs_more` exercises
    /// that arm only as a no-op. The `rot_k_v4g64` and `mixed_k4g32_v2g128`
    /// cases below are the ones that actually move a width.
    #[test]
    fn kv_quant_for_layer_promotion_target_per_family() {
        let n = 36;
        let cases: &[(KvQuant, KvQuant)] = &[
            (
                KvQuant::Mixed {
                    k_bits: 8,
                    v_bits: 4,
                    k_group_size: 64,
                    v_group_size: 64,
                },
                KvQuant::Mixed {
                    k_bits: 8,
                    v_bits: 8,
                    k_group_size: 64,
                    v_group_size: 64,
                },
            ),
            // Group geometry is the base's, not a constant: a 32/128 base is
            // promoted to a 32/128 target.
            (
                KvQuant::Mixed {
                    k_bits: 4,
                    v_bits: 2,
                    k_group_size: 32,
                    v_group_size: 128,
                },
                KvQuant::Mixed {
                    k_bits: 8,
                    v_bits: 8,
                    k_group_size: 32,
                    v_group_size: 128,
                },
            ),
            // Already at the floor on both axes → the promotion is a no-op.
            (
                KvQuant::Mixed {
                    k_bits: 8,
                    v_bits: 8,
                    k_group_size: 64,
                    v_group_size: 64,
                },
                KvQuant::Mixed {
                    k_bits: 8,
                    v_bits: 8,
                    k_group_size: 64,
                    v_group_size: 64,
                },
            ),
            (
                KvQuant::RotK {
                    v_bits: 4,
                    v_group_size: 64,
                },
                KvQuant::RotK {
                    v_bits: 8,
                    v_group_size: 64,
                },
            ),
            // Non-parametric widths have no 8-bit form of their own family.
            (KvQuant::K8V4, KvQuant::K8V8),
            (KvQuant::Planar3, KvQuant::K8V8),
            (KvQuant::TurboSym3, KvQuant::K8V8),
            (KvQuant::IsoKOnly4, KvQuant::K8V8),
        ];
        for &(base, want) in cases {
            for i in [0, 1, 28, 35] {
                assert_eq!(
                    kv_quant_for_layer(
                        i,
                        n,
                        base,
                        LAYER_ADAPTIVE_TAIL_N,
                        LAYER_ADAPTIVE_HEAD_N,
                        false,
                    ),
                    want,
                    "boundary layer {i} of {base} on a stack that keeps no mirror"
                );
                assert_eq!(
                    kv_quant_for_layer(
                        i,
                        n,
                        base,
                        LAYER_ADAPTIVE_TAIL_N,
                        LAYER_ADAPTIVE_HEAD_N,
                        true,
                    ),
                    KvQuant::K8V8,
                    "boundary layer {i} of {base} on a stack that keeps the Mixed/RotK \
                     mirror must take the fallback: the in-family store would be charged \
                     on top of two bf16 buffers"
                );
            }
        }
    }

    /// A boundary layer must never cost more than the `K8V8` it replaces.
    ///
    /// This is the property the promotion target has to have and the one it
    /// lost twice over. `K8V8` materialises no packed store, so a layer
    /// promoted to it holds two full bf16 mirrors — byte-identical to `none`,
    /// 16 bits per value — and promoting a codec that stores 6.50 there was a
    /// 2.46x increase, not a floor. Promoting it in-family on a stack that
    /// *keeps* the mirror is the inverse error: the store is charged on top of
    /// the two mirrors, 24.50 bits per value, 1.53x the fallback.
    ///
    /// Swept over every base in `ALL_KV_QUANTS` that materialises a packed
    /// store — the population for which the promotion is a byte question at all
    /// — at both cross-layer-KV topologies, using each codec's own byte model
    /// at a full-attention layer shape.
    ///
    /// Driven off `materialises_packed_store` rather than a variant list, so a
    /// new store-bearing codec that lands in `boundary_floor`'s fallback arm
    /// fails here instead of inheriting the exemption. The eight that take that
    /// arm today are named below and nowhere else.
    #[test]
    fn store_bearing_boundary_promotion_never_costs_more() {
        /// Store-bearing bases with no 8-bit form of their own family: an
        /// SO(4)-rotated or rotor 3-/4-bit ring cannot be widened to 8 without
        /// leaving the family, so their floor is bought at the fallback rather
        /// than delivered from bytes they already spend. `SideStore::IsoRing`
        /// is 12.125 bits per value, so the iso four pay a 1.32x byte
        /// regression for it; `SideStore::Rotor` is 16.25, above bf16, so the
        /// rotor four are neutral-to-favourable. Recorded in
        /// `docs/KV_QUANT.md` §Layer-adaptive overrides.
        const FALLBACK_BY_DESIGN: &[KvQuant] = &[
            KvQuant::Iso3Sym,
            KvQuant::Iso4Sym,
            KvQuant::IsoKOnly3,
            KvQuant::IsoKOnly4,
            KvQuant::Rotor3Sym,
            KvQuant::Rotor4Sym,
            KvQuant::RotorKOnly3,
            KvQuant::RotorKOnly4,
        ];
        let (seq, head_dim, kv_heads) = (4096_u64, 128_u64, 8_u64);
        let n = 36;
        let mut store_bearing = 0_usize;
        let mut named_hit = vec![false; FALLBACK_BY_DESIGN.len()];
        for shares_kv in [false, true] {
            // `K8V8` is two bf16 mirrors and nothing else, so this is both the
            // fallback's cost and plain bf16's.
            let fallback = KvQuant::K8V8
                .estimated_resident_bytes_per_layer(seq, head_dim, kv_heads, shares_kv);
            let mut in_family = 0_usize;
            for &base in rmlx_kv_quant::ALL_KV_QUANTS {
                if !base.materialises_packed_store() {
                    continue;
                }
                store_bearing += 1;
                let promoted = kv_quant_for_layer(
                    0,
                    n,
                    base,
                    LAYER_ADAPTIVE_TAIL_N,
                    LAYER_ADAPTIVE_HEAD_N,
                    shares_kv,
                );
                let got =
                    promoted.estimated_resident_bytes_per_layer(seq, head_dim, kv_heads, shares_kv);
                assert!(
                    got <= fallback,
                    "{base} (shares_kv={shares_kv}) promotes to {promoted} at {got} B/layer, \
                     above the {fallback} B of the K8V8 fallback it replaces"
                );
                if promoted == KvQuant::K8V8 {
                    // Landing on the storeless fallback is a decision, so it has
                    // to be one of the two recorded ones: a family with no 8-bit
                    // form, or a target whose bf16 mirror survives the promotion
                    // and already decodes above the floor.
                    let mirrors_both = promoted.feeds_bf16_k_at_decode(shares_kv)
                        && promoted.feeds_bf16_v_at_decode(shares_kv)
                        && base.feeds_bf16_k_at_decode(shares_kv)
                        && base.feeds_bf16_v_at_decode(shares_kv);
                    match FALLBACK_BY_DESIGN
                        .iter()
                        .position(|&q| q == base)
                        .and_then(|i| named_hit.get_mut(i))
                    {
                        Some(hit) => *hit = true,
                        None => assert!(
                            mirrors_both,
                            "{base} (shares_kv={shares_kv}) materialises a packed store but \
                             promotes to the storeless K8V8, and is neither named as a family \
                             without an 8-bit form nor mirroring both axes at this topology — \
                             it would hold two bf16 mirrors instead of a floor"
                        ),
                    }
                } else {
                    in_family += 1;
                    assert!(
                        promoted.materialises_packed_store(),
                        "{base} (shares_kv={shares_kv}) promotes to {promoted}, which builds \
                         no packed store"
                    );
                    assert!(
                        got < fallback,
                        "{base} (shares_kv={shares_kv}) promotes in-family to {promoted} at \
                         {got} B/layer, not under the fallback's {fallback} B — an in-family \
                         target only pays off when the layer keeps no mirror"
                    );
                }
            }
            // The topology is what decides whether an in-family target is
            // reachable: with the mirror retained there is no such target for
            // any base, and every store-bearing one takes the fallback.
            if shares_kv {
                assert_eq!(
                    in_family, 0,
                    "a stack that keeps the Mixed/RotK mirror has no in-family target to \
                     promote to, but {in_family} base(s) were promoted in-family"
                );
            } else {
                assert!(
                    in_family >= 2,
                    "expected the parametric bases to promote in-family off a mirror-less \
                     stack, promoted {in_family}"
                );
            }
        }
        assert_eq!(
            store_bearing,
            2 * 10,
            "expected ten store-bearing codecs in ALL_KV_QUANTS at each topology, swept \
             {store_bearing}"
        );
        for (named, hit) in FALLBACK_BY_DESIGN.iter().zip(&named_hit) {
            assert!(
                *hit,
                "{named} is named as a deliberate fallback but never took it — drop it from \
                 the list"
            );
        }
    }

    /// Whole-stack byte model for `mixed_k8g64_v4g64` at Qwen3-8B's geometry.
    ///
    /// The per-layer codec vector and the per-codec byte model are the two
    /// halves of the cache's delivered density, and neither one alone shows it.
    /// Pinned as bits per stored value so the figure is directly comparable to
    /// a codebook width: this codec's codebook is 6.50 bits per value (K 8 +
    /// 32/64 sideband, V 4 + 32/64), and what the stack delivers is that width
    /// diluted by the boundary layers.
    #[test]
    fn mixed_layer_stack_delivered_bits_per_value() {
        // Qwen3-8B: 36 full-attention layers, 8 KV heads, head_dim 128.
        let (n_layers, seq, head_dim, kv_heads) = (36_usize, 4096_u64, 128_u64, 8_u64);
        let base = KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        };
        let total: u64 = kv_layer_quants(n_layers, base, false)
            .iter()
            .map(|q| q.estimated_resident_bytes_per_layer(seq, head_dim, kv_heads, false))
            .sum();
        // Two axes (K and V) per stored position.
        let values = seq * head_dim * kv_heads * 2 * n_layers as u64;
        let bits_per_value = (total * 8) as f64 / values as f64;
        // The boundary layers at the 8.50 floor, the rest at the 6.50 codebook
        // width. Derived from the constants rather than restated as 10 and 26,
        // so moving either one moves the expectation with it.
        let boundary = (LAYER_ADAPTIVE_HEAD_N + LAYER_ADAPTIVE_TAIL_N) as f64;
        let interior = n_layers as f64 - boundary;
        let want = (interior * 6.50 + boundary * 8.50) / n_layers as f64;
        assert!(
            (bits_per_value - want).abs() < 1e-9,
            "delivered {bits_per_value:.4} bits/value, expected {want:.4}"
        );
        // The competitors' symmetric 8-bit KV cache, which this cell has to
        // undercut to be worth its asymmetry.
        assert!(
            bits_per_value < 8.50,
            "delivered {bits_per_value:.4} bits/value, not below the 8.50 both \
             llama.cpp q8_0/q8_0 and mlx-lm q8 ship"
        );
    }

    /// The byte consequence of the exemption, measured on a real cache vector.
    ///
    /// A `--kv-quant none` layer stack must hold **only** its two bf16 mirrors
    /// on every layer. A promoted boundary layer would add a packed q8_0 K+V
    /// store on top of those same mirrors (decode reads the mirrors either
    /// way), so the total is the sharp discriminator: the bf16 identity, or
    /// the identity plus 10 stores.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn none_layer_stack_resident_bytes_is_the_bf16_identity() {
        let device = Device::Cpu;
        // Ternary-Bonsai-8B's layer count at a small test shape.
        let n_layers = 36_usize;
        let (seq, kv_h, head_dim) = (256_u64, 4_u64, 128_u64);

        let mut total: u64 = 0;
        for i in 0..n_layers {
            let q = kv_quant_for_layer(
                i,
                n_layers,
                KvQuant::None,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
                false,
            );
            let mut cache = KvCache::with_quant_max_seq(q, 4096);
            cache.enter_prefill();
            let shape = [1, kv_h as i32, seq as i32, head_dim as i32];
            let k = make_lcg_array(&shape, 0x5EED + i as u64).0;
            let v = make_lcg_array(&shape, 0xD00D + i as u64).0;
            cache.update(&k, &v, device).expect("update must not fail");
            cache
                .exit_prefill(device)
                .expect("exit_prefill must not fail");
            total += cache.resident_bytes();
        }

        let bf16_identity =
            u64::try_from(n_layers).expect("layer count fits u64") * 2 * seq * kv_h * head_dim * 2;
        assert_eq!(
            total,
            bf16_identity,
            "`none` must hold the bf16 identity and nothing else — \
             an excess of {} B is a packed store on the boundary layers",
            total.saturating_sub(bf16_identity)
        );
    }

    #[test]
    fn kv_quant_for_layer_planar_tail_overridden() {
        // Planar base: last 8 of 40 layers become K8V8; head 2 also become K8V8.
        let n = 40;
        let base = KvQuant::Planar;
        // Head layers become K8V8.
        assert_eq!(
            kv_quant_for_layer(0, n, base, 8, 2, false),
            KvQuant::K8V8,
            "head layer 0"
        );
        assert_eq!(
            kv_quant_for_layer(1, n, base, 8, 2, false),
            KvQuant::K8V8,
            "head layer 1"
        );
        // Middle layers stay Planar.
        assert_eq!(
            kv_quant_for_layer(2, n, base, 8, 2, false),
            KvQuant::Planar,
            "middle layer 2"
        );
        assert_eq!(
            kv_quant_for_layer(31, n, base, 8, 2, false),
            KvQuant::Planar,
            "middle layer 31"
        );
        // Tail layers become K8V8.
        assert_eq!(
            kv_quant_for_layer(32, n, base, 8, 2, false),
            KvQuant::K8V8,
            "tail layer 32"
        );
        assert_eq!(
            kv_quant_for_layer(39, n, base, 8, 2, false),
            KvQuant::K8V8,
            "tail layer 39"
        );
    }

    /// Parity test: the new universal `KvCache::update_and_sdpa` wrapper must
    /// produce byte-equivalent output to the legacy `update + SDPA` pattern on
    /// the K8V8 path (the most-used non-Mixed, non-K8V4-flash code path).
    ///
    /// Setup: two K8V8 caches seeded with an identical 16-token prefill, then
    /// one decode step run through the legacy pattern (cache A) and the new
    /// wrapper (cache B). `max(|out_a - out_b|)` must be below 1e-3.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn update_and_sdpa_matches_legacy_k8v8_path() {
        use rmlx_mlx::scaled_dot_product_attention;

        let device = Device::Cpu;
        let n_kv_heads: i32 = 4;
        let head_dim: i32 = 128;
        let prefill_seq: i32 = 16;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        // Identical prefill K/V for both caches.
        let prefill_shape: &[i32] = &[1, n_kv_heads, prefill_seq, head_dim];
        let (k_pref, _) = make_lcg_array(prefill_shape, 0xA1A1_A1A1_u64);
        let (v_pref, _) = make_lcg_array(prefill_shape, 0xB2B2_B2B2_u64);

        // Decode-step queries / new K / new V (seq=1).
        let step_shape: &[i32] = &[1, n_kv_heads, 1, head_dim];
        let (queries, _) = make_lcg_array(step_shape, 0xC3C3_C3C3_u64);
        let (new_k, _) = make_lcg_array(step_shape, 0xD4D4_D4D4_u64);
        let (new_v, _) = make_lcg_array(step_shape, 0xE5E5_E5E5_u64);

        // Cache A — legacy `update` + `scaled_dot_product_attention`.
        let mut cache_a = KvCache::with_quant(KvQuant::K8V8);
        cache_a
            .update(&k_pref, &v_pref, device)
            .expect("K8V8 prefill failed on cache A");
        let (k_full_a, v_full_a) = cache_a
            .update(&new_k, &new_v, device)
            .expect("K8V8 decode update failed on cache A");
        let out_a =
            scaled_dot_product_attention(&queries, &k_full_a, &v_full_a, scale, "", None, device)
                .expect("legacy SDPA failed");

        // Cache B — new universal wrapper.
        let mut cache_b = KvCache::with_quant(KvQuant::K8V8);
        cache_b
            .update(&k_pref, &v_pref, device)
            .expect("K8V8 prefill failed on cache B");
        let out_b = cache_b
            .update_and_sdpa(&queries, &new_k, &new_v, scale, "", None, device)
            .expect("update_and_sdpa failed");

        // max(|out_a - out_b|) < 1e-3
        assert_eq!(out_a.shape(), out_b.shape(), "output shape mismatch");
        let a = array_to_vec(&out_a);
        let b = array_to_vec(&out_b);
        let max_abs_diff = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-3,
            "update_and_sdpa diverges from legacy path: max |Δ| = {max_abs_diff:.6}"
        );
    }

    /// Smoke test for the cross-layer-KV-sharing sibling wrapper on the K8V8
    /// path: build a cache, seed it with 16 tokens, run one decode step via
    /// `update_and_sdpa_shared_source`, assert the call returns three arrays
    /// with the expected shapes.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn update_and_sdpa_shared_source_k8v8_smoke() {
        let device = Device::Cpu;
        let n_kv_heads: i32 = 4;
        let head_dim: i32 = 128;
        let prefill_seq: i32 = 16;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let prefill_shape: &[i32] = &[1, n_kv_heads, prefill_seq, head_dim];
        let (k_pref, _) = make_lcg_array(prefill_shape, 0xA1A1_A1A1_u64);
        let (v_pref, _) = make_lcg_array(prefill_shape, 0xB2B2_B2B2_u64);

        let step_shape: &[i32] = &[1, n_kv_heads, 1, head_dim];
        let (queries, _) = make_lcg_array(step_shape, 0xC3C3_C3C3_u64);
        let (new_k, _) = make_lcg_array(step_shape, 0xD4D4_D4D4_u64);
        let (new_v, _) = make_lcg_array(step_shape, 0xE5E5_E5E5_u64);

        let mut cache = KvCache::with_quant(KvQuant::K8V8);
        cache
            .update(&k_pref, &v_pref, device)
            .expect("K8V8 prefill failed");
        let (out, k_full, v_full) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(&queries, &new_k, &new_v, scale, "", None, device)
                .expect("update_and_sdpa_shared_source K8V8 failed"),
        );

        // After one decode step on top of the 16-token prefill: total = 17.
        let total_kv: i32 = prefill_seq + 1;
        assert_eq!(
            out.shape(),
            vec![1, n_kv_heads, 1, head_dim],
            "SDPA output shape"
        );
        assert_eq!(
            k_full.shape(),
            vec![1, n_kv_heads, total_kv, head_dim],
            "accumulated K shape"
        );
        assert_eq!(
            v_full.shape(),
            vec![1, n_kv_heads, total_kv, head_dim],
            "accumulated V shape"
        );
    }

    /// K8V4 also exposes accumulated `(K, V)` via `update()` (the codec
    /// dequantises on read), so the sibling wrapper must accept it. Only Mixed
    /// — whose fused quantized SDPA hides K/V inside the storage — is rejected.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn update_and_sdpa_shared_source_k8v4_smoke() {
        let device = Device::Cpu;
        let n_kv_heads: i32 = 4;
        let head_dim: i32 = 128;
        let prefill_seq: i32 = 16;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let prefill_shape: &[i32] = &[1, n_kv_heads, prefill_seq, head_dim];
        let (k_pref, _) = make_lcg_array(prefill_shape, 0xA1A1_A1A1_u64);
        let (v_pref, _) = make_lcg_array(prefill_shape, 0xB2B2_B2B2_u64);

        let step_shape: &[i32] = &[1, n_kv_heads, 1, head_dim];
        let (queries, _) = make_lcg_array(step_shape, 0xC3C3_C3C3_u64);
        let (new_k, _) = make_lcg_array(step_shape, 0xD4D4_D4D4_u64);
        let (new_v, _) = make_lcg_array(step_shape, 0xE5E5_E5E5_u64);

        let mut cache = KvCache::with_quant(KvQuant::K8V4);
        cache
            .update(&k_pref, &v_pref, device)
            .expect("K8V4 prefill failed");
        let (out, k_full, v_full) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(&queries, &new_k, &new_v, scale, "", None, device)
                .expect("update_and_sdpa_shared_source K8V4 failed"),
        );

        let total_kv: i32 = prefill_seq + 1;
        assert_eq!(out.shape(), vec![1, n_kv_heads, 1, head_dim]);
        assert_eq!(k_full.shape(), vec![1, n_kv_heads, total_kv, head_dim]);
        assert_eq!(v_full.shape(), vec![1, n_kv_heads, total_kv, head_dim]);
    }

    /// `update_and_sdpa_shared_source` now SUPPORTS Mixed caches via
    /// dequant-before-share. The fused quantized SDPA stores K/V as quant
    /// 3-tuples, but the wrapper surfaces the accumulated bf16 K/V (prefill-raw
    /// during prefill, maintained `decode_fp16_k/v` during decode) so a
    /// cross-layer-KV consumer (Gemma4) gets the full prefix every step.
    ///
    /// Mirrors the Gemma4 cache-holding full-attention layer flow: enter_prefill
    /// → wrapper (multi-token chunk) → exit_prefill → wrapper (decode step). The
    /// call must SUCCEED (no longer error) for a Mixed cache, return three arrays
    /// with the expected shapes (prefill = prefill_seq tokens, decode extends by
    /// exactly one), and produce finite K/V/output.
    ///
    /// NOTE: runs on `Device::Cpu`, where the MLX-C `mlx_slice_update` backend
    /// drops non-leading kv heads on any axis-2 sub-slice write into a larger
    /// pre-allocated buffer (a pre-existing primitive quirk, unrelated to —
    /// it corrupts K8V8/K8V4/None `decode_fp16` and `prefill_raw` accumulators on
    /// CPU identically, which is why the existing `resident_bytes_*` CPU tests only
    /// assert byte counts, never values). Therefore this test asserts shape,
    /// finiteness and offset progression on CPU; value-level coherence of the
    /// dequant-before-share K/V is validated by the Gemma4 e4b/26b Mixed baseline
    /// runs on GPU.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_and_sdpa_shared_source_mixed_shared_kv() {
        let device = Device::Cpu;
        let n_kv_heads: i32 = 4;
        let head_dim: i32 = 128;
        let prefill_seq: i32 = 16;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let prefill_shape: &[i32] = &[1, n_kv_heads, prefill_seq, head_dim];
        let (k_pref, _) = make_lcg_array(prefill_shape, 0xA1A1_A1A1_u64);
        let (v_pref, _) = make_lcg_array(prefill_shape, 0xB2B2_B2B2_u64);
        let (q_pref, _) = make_lcg_array(prefill_shape, 0xF0F0_F0F0_u64);

        let step_shape: &[i32] = &[1, n_kv_heads, 1, head_dim];
        let (queries, _) = make_lcg_array(step_shape, 0xC3C3_C3C3_u64);
        let (new_k, _) = make_lcg_array(step_shape, 0xD4D4_D4D4_u64);
        let (new_v, _) = make_lcg_array(step_shape, 0xE5E5_E5E5_u64);

        // The shared-source wrapper is the cross-layer-KV producer entry point,
        // so the cache this test drives is one: without the declaration the
        // Mixed bf16 mirror it surfaces is never built and the call is refused.
        let mut cache = KvCache::with_quant(KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        })
        .with_shares_kv(true);

        // Prefill through the shared-KV wrapper — must NOT error for Mixed.
        cache.enter_prefill();
        let (out_pref, k_pre_full, v_pre_full) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(
                    &q_pref, &k_pref, &v_pref, scale, "causal", None, device,
                )
                .expect("Mixed prefill via the shared-source path must succeed"),
        );
        cache.exit_prefill(device).expect("exit_prefill failed");

        assert_eq!(out_pref.shape(), vec![1, n_kv_heads, prefill_seq, head_dim]);
        assert_eq!(
            k_pre_full.shape(),
            vec![1, n_kv_heads, prefill_seq, head_dim]
        );
        assert_eq!(
            v_pre_full.shape(),
            vec![1, n_kv_heads, prefill_seq, head_dim]
        );
        assert_eq!(cache.offset(), prefill_seq, "prefill must advance offset");
        assert!(
            array_to_vec(&out_pref.astype(Dtype::F32, device).unwrap())
                .iter()
                .all(|x| x.is_finite()),
            "prefill SDPA output has non-finite values"
        );

        // One decode step: extend the prefix by exactly one token.
        let (out, k_full, v_full) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(&queries, &new_k, &new_v, scale, "", None, device)
                .expect("update_and_sdpa_shared_source Mixed decode must succeed"),
        );

        let total_kv: i32 = prefill_seq + 1;
        assert_eq!(out.shape(), vec![1, n_kv_heads, 1, head_dim]);
        assert_eq!(
            k_full.shape(),
            vec![1, n_kv_heads, total_kv, head_dim],
            "accumulated K must extend by exactly one token per decode step"
        );
        assert_eq!(v_full.shape(), vec![1, n_kv_heads, total_kv, head_dim]);
        assert_eq!(
            cache.offset(),
            total_kv,
            "decode must advance offset by one"
        );
        assert!(
            array_to_vec(&out.astype(Dtype::F32, device).unwrap())
                .iter()
                .all(|x| x.is_finite()),
            "decode SDPA output has non-finite values"
        );
    }

    // ── 2-bit V (asymmetric K=8 / V=2) round-trip ──────────────────────

    /// 2-bit V quantization via the Mixed path (K=8-bit, V=2-bit affine, g=64).
    /// 2-bit is the lossiest rung — assert the dequantized output is finite,
    /// shape-correct, and bounded (no NaN/Inf), not bit-exact. The MLX affine
    /// quantizer (`mx.quantize`) handles 2-bit packing (16 vals/u32) natively;
    /// no rMLX kernel change was needed.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_cache_mixed_v2_roundtrip_finite_and_bounded() {
        let device = Device::Cpu;
        let n_kv_heads: i32 = 4;
        let head_dim: i32 = 128; // 128 % 16 == 0 (2-bit packing) and % 64 == 0.
        let prefill_seq: i32 = 16;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let prefill_shape: &[i32] = &[1, n_kv_heads, prefill_seq, head_dim];
        let (k_pref, _) = make_lcg_array(prefill_shape, 0x2B2B_0001_u64);
        let (v_pref, _) = make_lcg_array(prefill_shape, 0x2B2B_0002_u64);
        let (q_pref, _) = make_lcg_array(prefill_shape, 0x2B2B_0003_u64);

        // Driven through the shared-source wrapper, so this is a cross-layer-KV
        // producer layer and must say so.
        let mut cache = KvCache::with_quant(KvQuant::Mixed {
            k_bits: 8,
            v_bits: 2,
            k_group_size: 64,
            v_group_size: 64,
        })
        .with_shares_kv(true);

        cache.enter_prefill();
        let (out_pref, _k_full, v_full) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(
                    &q_pref, &k_pref, &v_pref, scale, "causal", None, device,
                )
                .expect("2-bit V Mixed prefill must succeed"),
        );
        cache.exit_prefill(device).expect("exit_prefill failed");

        assert_eq!(v_full.shape(), vec![1, n_kv_heads, prefill_seq, head_dim]);
        assert_eq!(out_pref.shape(), vec![1, n_kv_heads, prefill_seq, head_dim]);

        // 2-bit dequant must stay finite and bounded — the source data is in
        // [-1, 1], so a sane affine reconstruction is well within [-2, 2].
        let recon = array_to_vec(&v_full.astype(Dtype::F32, device).unwrap());
        assert!(
            recon.iter().all(|x| x.is_finite()),
            "2-bit V dequant produced non-finite values"
        );
        assert!(
            recon.iter().all(|&x| x.abs() < 2.0),
            "2-bit V dequant out of expected bound (source in [-1,1])"
        );
        assert!(
            array_to_vec(&out_pref.astype(Dtype::F32, device).unwrap())
                .iter()
                .all(|x| x.is_finite()),
            "2-bit V SDPA output has non-finite values"
        );
    }

    // ── KvQuant Display / FromStr ──────────────────────────────────────────────

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_none() {
        use std::str::FromStr;
        assert_eq!(KvQuant::None.to_string(), "none");
        assert_eq!(KvQuant::from_str("none").unwrap(), KvQuant::None);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_from_str_accepts_bf16_alias() {
        use std::str::FromStr;
        assert_eq!(KvQuant::from_str("bf16").unwrap(), KvQuant::None);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_from_str_accepts_f16_alias() {
        use std::str::FromStr;
        assert_eq!(KvQuant::from_str("f16").unwrap(), KvQuant::None);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_k8v4() {
        use std::str::FromStr;
        assert_eq!(KvQuant::K8V4.to_string(), "k8v4");
        assert_eq!(KvQuant::from_str("k8v4").unwrap(), KvQuant::K8V4);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_k8v8() {
        use std::str::FromStr;
        assert_eq!(KvQuant::K8V8.to_string(), "k8v8");
        assert_eq!(KvQuant::from_str("k8v8").unwrap(), KvQuant::K8V8);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_planar() {
        use std::str::FromStr;
        assert_eq!(KvQuant::Planar.to_string(), "planar");
        assert_eq!(KvQuant::from_str("planar").unwrap(), KvQuant::Planar);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_mixed_8_4_128_64() {
        use std::str::FromStr;
        let m = KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 128,
            v_group_size: 64,
        };
        assert_eq!(m.to_string(), "mixed_k8g128_v4g64");
        assert_eq!(KvQuant::from_str("mixed_k8g128_v4g64").unwrap(), m);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_mixed_8_8_64_64() {
        use std::str::FromStr;
        let m = KvQuant::Mixed {
            k_bits: 8,
            v_bits: 8,
            k_group_size: 64,
            v_group_size: 64,
        };
        assert_eq!(m.to_string(), "mixed_k8g64_v8g64");
        assert_eq!(KvQuant::from_str("mixed_k8g64_v8g64").unwrap(), m);
    }

    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn kv_quant_display_round_trip_mixed_4_4_32_32() {
        use std::str::FromStr;
        let m = KvQuant::Mixed {
            k_bits: 4,
            v_bits: 4,
            k_group_size: 32,
            v_group_size: 32,
        };
        assert_eq!(m.to_string(), "mixed_k4g32_v4g32");
        assert_eq!(KvQuant::from_str("mixed_k4g32_v4g32").unwrap(), m);
    }

    #[test]
    fn kv_quant_from_str_unknown_returns_err() {
        use std::str::FromStr;
        assert!(KvQuant::from_str("kx99").is_err());
        assert!(KvQuant::from_str("").is_err());
        assert!(KvQuant::from_str("mixed_garbage").is_err());
        assert!(KvQuant::from_str("mixed_k8_v4").is_err());
        assert!(KvQuant::from_str("mixed_x8g64_v4g64").is_err());
    }

    /// `k8vturbo2` display + FromStr round-trip.
    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "test asserts the round-trip; .unwrap() failure is the test failure"
    )]
    fn kv_quant_display_round_trip_k8vturbo2() {
        use std::str::FromStr;
        assert_eq!(KvQuant::K8VTurbo2.to_string(), "k8vturbo2");
        assert_eq!(KvQuant::from_str("k8vturbo2").unwrap(), KvQuant::K8VTurbo2);
    }

    /// `KvQuantParseError::Unknown` mentions `k8vturbo2` in its
    /// `valid:` listing. Locks in the user-visible string surface.
    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "test asserts the Err arm; .unwrap_err() failure on Ok is the test failure"
    )]
    fn kv_quant_parse_error_unknown_lists_k8vturbo2() {
        use std::str::FromStr;
        let err = KvQuant::from_str("definitely-not-a-codec").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("k8vturbo2"),
            "Unknown KvQuant error must list k8vturbo2 as a valid value: {msg}"
        );
    }

    // ── KV memory reduction regression table ───────────────────────────
    //
    // Tests that the reduction ratio (bf16_codes_bytes / quant_codes_bytes) for
    // each KvQuant variant stays within documented tolerance bands.
    //
    // DESIGN: uses codes-only bytes (seq * bhd * (k_bits + v_bits) / 8),
    // NOT resident_bytes(), because:
    // 1. This table pins each codec's **nominal** bit-width ratio — the number
    // the reference table (mlx-vlm / docs §4) cites. Real residency also
    // carries per-group scales, warm-TTFT bf16 seeds and (for the ring-backed
    // codecs) a GPU ring; folding those in produces ratios < 1.0 (quantized
    // "uses more" than bf16), which is not what this table is about.
    // 2. The (k_bits, v_bits) mapping below IS the regression target: if any
    // variant's bits change, the codes formula changes → ratio changes → this
    // test fails.
    //
    // This is the ONE place a bits-per-element figure is legitimate: it is the
    // codec's advertised ratio, explicitly not a claim about resident memory.
    // For memory, ask the store — see `KvCache::resident_bytes`.
    //
    // FIXED SHAPE: B=1, kv_h=8, head_dim=128, seq=1024
    // bhd = B * kv_h * head_dim = 1 * 8 * 128 = 1024
    // bf16_codes = 1024 * 1024 * (16+16)/8 = 4_194_304 bytes
    //
    // CODEC TABLE (derived, then bands set ± epsilon):
    //
    // Codec k_bits v_bits codes_bytes ratio_vs_bf16 band
    // bf16 (None) 16 16 4_194_304 1.000× —
    // K8V8 8 8 2_097_152 2.000× [1.9, 2.1]
    // K8V4 8 4 1_572_864 2.667× [2.5, 2.8]
    // Planar 8 4 1_572_864 2.667× [2.5, 2.8]
    // Mixed{k8,v4,g64} 8 4 1_572_864 2.667× [2.5, 2.8]
    // Mixed{k8,v2,g64} 8 2 1_310_720 3.200× [3.0, 3.5]
    // Mixed{k4,v4,g64} 4 4 1_048_576 4.000× [3.8, 4.2]
    // Mixed{k3,v3,g64} 3 3 786_432 5.333× [5.0, 5.7]
    // Mixed{k2,v2,g64} 2 2 524_288 8.000× [7.5, 8.5]
    // RotK{v4,g64} 8 4 1_572_864 2.667× [2.5, 2.8]
    // RotK{v2,g64} 8 2 1_310_720 3.200× [3.0, 3.5]
    //
    // NOTE on "q2 ≈ 8×" reference (mlx-vlm README, docs §4):
    // The mlx-vlm "8×" figure assumes symmetric 2-bit K+V combined: (16+16)/(2+2) = 8×.
    // rMLX's q2_g64 is V-side only (k=8-bit, v=2-bit): (16+16)/(8+2) = 3.2×.
    // The V-only compression is 16/2 = 8× (codes) / ~6.4× effective (at g=64, scale overhead
    // adds ~1 byte per 64 elements). Both figures are correct for their context; the test
    // uses the K+V combined ratio (3.2×), which is the codecs nominal figure.
    // See docs/KV_CACHE.md §4 (q2_g64 row) and §5.10 (pure-2-bit-K gated).

    /// Pure-formula helper: codes-only KV bytes for a given shape and bit widths.
    /// No GPU, no MLX operations. Pure arithmetic over the codec bit-widths.
    fn codes_bytes(seq: u64, bhd: u64, k_bits: u64, v_bits: u64) -> u64 {
        seq * bhd * (k_bits + v_bits) / 8
    }

    #[test]
    fn kv_reduction_ratios_match_table() {
        // Fixed shape — chosen so all divisibility constraints hold:
        // seq=1024, B=1, kv_h=8, head_dim=128 → bhd = 1024
        // head_dim=128 satisfies: 128 % 64 == 0 (group=64), 128 % (32/2) = 128 % 16 == 0 (2-bit packing).
        let seq: u64 = 1024;
        let bhd: u64 = 1024; // B=1, kv_h=8, head_dim=128

        let bf16_bytes = codes_bytes(seq, bhd, 16, 16);
        assert_eq!(bf16_bytes, 4_194_304, "bf16 baseline sanity check");

        // Helper: assert the reduction ratio is within [lo, hi].
        let check = |label: &str, k_bits: u64, v_bits: u64, lo: f64, hi: f64| {
            let qbytes = codes_bytes(seq, bhd, k_bits, v_bits);
            assert!(
                qbytes > 0,
                "{label}: quant codes_bytes must be > 0 (got {qbytes})"
            );
            let ratio = bf16_bytes as f64 / qbytes as f64;
            assert!(
                ratio >= lo && ratio <= hi,
                "{label}: reduction ratio {ratio:.4}× out of expected band [{lo}, {hi}] \
                 (bf16={bf16_bytes} bytes, quant={qbytes} bytes, k_bits={k_bits}, v_bits={v_bits})"
            );
        };

        // ── Named presets ─────────────────────────────────────────────────────

        // K8V8: k=8, v=8 → codes = 1024*1024*2 = 2_097_152 → ratio = 2.0×
        // Band [1.9, 2.1] — tight around the exact integer ratio.
        check("K8V8", 8, 8, 1.9, 2.1);

        // K8V4 / Planar / Mixed{k8,v4}: k=8, v=4 → codes = 1_572_864 → ratio = 2.667×
        // Band [2.5, 2.8] — covers both K8V4, Planar (same bits), and Mixed{k8,v4}.
        check("K8V4 / Planar / Mixed{k8,v4}", 8, 4, 2.5, 2.8);

        // ── Mixed asymmetric / q2 ───────────────────────────────────────

        // Mixed{k=8,v=2}: rMLX q2_g64. K stays 8-bit (pure-2-bit K gated §5.10).
        // codes = 1024*1024*(8+2)/8 = 1_310_720 → ratio = 32/10 = 3.2×
        // Band [3.0, 3.5].
        check("Mixed{k8,v2} (q2_g64)", 8, 2, 3.0, 3.5);

        // Mixed{k=4,v=4}: symmetric 4-bit — matches mlx-vlm "4 ≈ 4×" rung.
        // codes = 1024*1024*1 = 1_048_576 → ratio = 4.0×
        // Band [3.8, 4.2].
        check("Mixed{k4,v4} (q4 symmetric)", 4, 4, 3.8, 4.2);

        // Mixed{k=3,v=3}: symmetric 3-bit — near mlx-vlm "3 ≈ 5×" rung.
        // codes = 1024*1024*(3+3)/8 = 786_432 → ratio = 32/6 ≈ 5.333×
        // Band [5.0, 5.7].
        check("Mixed{k3,v3} (q3 symmetric)", 3, 3, 5.0, 5.7);

        // Mixed{k=2,v=2}: symmetric 2-bit — mlx-vlm "2 ≈ 8×" rung.
        // codes = 1024*1024*4/8 = 524_288 → ratio = 8.0×
        // Band [7.5, 8.5].
        // NOTE: pure-2-bit K is gated in rMLX (§5.10 / ); this row covers the
        // formula for the symmetric case referenced in the mlx-vlm table.
        check("Mixed{k2,v2} (q2 symmetric, mlx-vlm ref)", 2, 2, 7.5, 8.5);

        // ── RotK variants ─────────────────────────────────────────────────────

        // RotK{v_bits=4}: K=8-bit in rotated basis, V=4-bit → same bits as K8V4.
        // ratio = 2.667× — band [2.5, 2.8].
        check("RotK{v4}", 8, 4, 2.5, 2.8);

        // RotK{v_bits=2}: K=8-bit rotated, V=2-bit → same bits as Mixed{k8,v2}.
        // ratio = 3.2× — band [3.0, 3.5].
        check("RotK{v2}", 8, 2, 3.0, 3.5);

        // ── Baseline self-check ───────────────────────────────────────────────

        // bf16 / bf16 = 1.0× exactly.
        check("bf16 self", 16, 16, 0.99, 1.01);
    }

    /// regression: a Mixed-quant SWA (rotating) cache driven across a
    /// window-crossing prefill chunk must NOT broadcast-fail.
    ///
    /// Previously, `update_and_sdpa[_shared_source]` short-circuited Mixed to
    /// `update_prefill_raw` (full, uncapped K of length `offset + seq`) BEFORE
    /// honoring the rotating ring. But the Gemma4 SWA attention mask is sized
    /// to the ring's window-capped K (`offset.min(window-1) + seq`). On the
    /// chunk that crosses the window the two disagreed:
    /// `add: [broadcast_shapes] (…,seq,offset+seq) and (…,seq,window-1+seq)`.
    /// The fix routes rotating caches through `update()` (ring first) for every
    /// quant, so K matches the capped mask. This drives exactly that path with
    /// a tiny window and asserts Ok + window-capped K shape.
    #[test]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn mixed_swa_window_crossing_chunk_no_broadcast() {
        let device = Device::Cpu;
        let n_kv_heads: i32 = 2;
        let head_dim: i32 = 128;
        let window: i32 = 8;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        // Mixed-quant SWA layer: rotating ring is created regardless of quant
        // (mlx-lm RotatingKVCache stays bf16). max_seq is ignored on the ring.
        let mut cache = KvCache::with_quant_max_seq_window(
            KvQuant::Mixed {
                k_bits: 8,
                v_bits: 4,
                k_group_size: 64,
                v_group_size: 64,
            },
            4096,
            Some(window),
        );
        assert!(cache.is_rotating(), "SWA layer must use the rotating ring");

        // Chunk 1: fill exactly to the window (offset 0 -> window).
        let c1: i32 = window;
        let shape1: &[i32] = &[1, n_kv_heads, c1, head_dim];
        let (q1, _) = make_lcg_array(shape1, 0x1111_1111_u64);
        let (k1, _) = make_lcg_array(shape1, 0x2222_2222_u64);
        let (v1, _) = make_lcg_array(shape1, 0x3333_3333_u64);
        // First chunk: offset == 0 -> "causal", no explicit mask (matches
        // Gemma4 SWA prefill at offset 0).
        let (_o1, _k1f, _v1f) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(&q1, &k1, &v1, scale, "causal", None, device)
                .expect("chunk 1 prefill must succeed"),
        );
        assert_eq!(cache.offset(), c1);

        // Chunk 2: crosses the window. Build the SWA mask the way Gemma4 does:
        // effective (capped) offset = offset.min(window - 1).
        let c2: i32 = window; // second chunk same size; total > window
        let shape2: &[i32] = &[1, n_kv_heads, c2, head_dim];
        let (q2, _) = make_lcg_array(shape2, 0x4444_4444_u64);
        let (k2, _) = make_lcg_array(shape2, 0x5555_5555_u64);
        let (v2, _) = make_lcg_array(shape2, 0x6666_6666_u64);
        let eff_offset = cache.offset().min(window - 1);
        let mask = crate::layers::build_swa_prefill_mask(eff_offset, c2, window as usize, device)
            .expect("build swa prefill mask");

        // Pre-fix: this errored with the (…,c2,c2+offset) vs (…,c2,eff+c2)
        // broadcast mismatch. Post-fix: Ok, K capped at the window.
        let (out, k_full, v_full) = split_bf16_share(
            cache
                .update_and_sdpa_shared_source(&q2, &k2, &v2, scale, "array", Some(&mask), device)
                .expect("window-crossing Mixed SWA chunk must not broadcast-fail"),
        );

        // The crux: the ring's K length must equal the mask's key dimension
        // (`eff_offset + c2`), i.e. the window-capped length the mlx-lm
        // `_update_concat` produces on the wrapping chunk (`window - 1 + seq`).
        // This is exactly what would have mismatched pre-fix.
        let kv_len = k_full.shape()[2];
        let mask_keys = mask.shape()[3];
        assert_eq!(
            kv_len, mask_keys,
            "ring K length must equal the SWA mask key dimension (no off-by-one)"
        );
        assert_eq!(
            kv_len,
            eff_offset + c2,
            "wrapping concat caps K at window-1+seq"
        );
        assert_eq!(v_full.shape()[2], kv_len, "V len must match K len");
        assert_eq!(out.shape(), vec![1, n_kv_heads, c2, head_dim]);
        assert!(
            array_to_vec(&out.astype(Dtype::F32, device).unwrap())
                .iter()
                .all(|x| x.is_finite()),
            "SDPA output across the window boundary must be finite"
        );
    }

    // ── KvCacheBuilder::with_calibration + lookup_layer_calibration ──────────

    /// Build a `KvCalibration` via JSON round-trip to avoid `#[non_exhaustive]`
    /// restriction (struct literal construction is banned outside the defining crate).
    #[allow(
        clippy::expect_used,
        reason = "test helper: JSON is a compile-time constant; panic on parse failure is the correct behavior"
    )]
    fn make_t18_calib() -> KvCalibration {
        let json = r#"{
          "version": 1,
          "recipe": "turboquant35",
          "head_size": 128,
          "model_name": "test",
          "transform_version": "v1",
          "codebook_version": "v1",
          "layers": {
            "model.layers.0.self_attn": {
              "key_high_precision_indices": [[0, 1, 2]],
              "value_high_precision_indices": [[3, 4, 5]]
            }
          },
          "calibration": {
            "method": "weight_norm",
            "objective": "l2_norm",
            "num_prompts": 0,
            "max_seq_len": 0,
            "batch_size": 0,
            "num_observed_tokens": 0,
            "dtype": "bfloat16",
            "device": "cpu",
            "prompts_sha256": ""
          }
        }"#;
        serde_json::from_str(json).expect("make_t18_calib: valid JSON")
    }

    /// Default KvCacheBuilder has no calibration.
    #[test]
    fn kv_cache_builder_default_has_no_calibration() {
        let b = KvCacheBuilder::default();
        assert!(
            b.calibration.is_none(),
            "default KvCacheBuilder.calibration must be None"
        );
    }

    /// with_calibration(Some(_)) stores calibration on the builder.
    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "test: is_some() asserted on the line above; unwrap cannot fail"
    )]
    fn kv_cache_builder_with_calibration_stores_value() {
        let calib = make_t18_calib();
        let b = KvCacheBuilder::default().with_calibration(Some(calib));
        assert!(
            b.calibration.is_some(),
            "calibration should be Some after with_calibration"
        );
        assert_eq!(b.calibration.as_ref().unwrap().head_size, 128);
    }

    /// with_calibration(None) leaves calibration as None.
    #[test]
    fn kv_cache_builder_with_calibration_none_leaves_none() {
        let b = KvCacheBuilder::default().with_calibration(None);
        assert!(
            b.calibration.is_none(),
            "with_calibration(None) should leave calibration as None"
        );
    }

    /// Exact-match lookup succeeds.
    #[test]
    #[allow(
        clippy::unwrap_used,
        reason = "test: is_some() asserted on the line above; unwrap cannot fail"
    )]
    fn lookup_layer_calibration_exact_match() {
        let calib = make_t18_calib();
        let entry = lookup_layer_calibration(&calib, "model.layers.0.self_attn");
        assert!(entry.is_some(), "exact match should succeed");
        assert_eq!(
            entry.unwrap().key_high_precision_indices,
            vec![vec![0u32, 1, 2]]
        );
    }

    /// Case-insensitive 3-dotted-prefix match.
    #[test]
    fn lookup_layer_calibration_case_insensitive_fuzzy_match() {
        let calib = make_t18_calib();
        // Query uses uppercase — should still match model.layers.0.self_attn
        let entry = lookup_layer_calibration(&calib, "MODEL.LAYERS.0.self_attn");
        assert!(
            entry.is_some(),
            "case-insensitive 3-prefix match should succeed for MODEL.LAYERS.0"
        );
    }

    /// Near-miss with extra component still matches via 3-prefix.
    #[test]
    fn lookup_layer_calibration_extra_component_matches_prefix() {
        let calib = make_t18_calib();
        // "model.layers.0.self_attn.k_proj" — 3-prefix is "model.layers.0" which matches
        let entry = lookup_layer_calibration(&calib, "model.layers.0.self_attn.k_proj");
        assert!(
            entry.is_some(),
            "query with extra trailing component should match via 3-prefix"
        );
    }

    /// Missing layer key returns None.
    #[test]
    fn lookup_layer_calibration_no_match_returns_none() {
        let calib = make_t18_calib();
        let entry = lookup_layer_calibration(&calib, "model.layers.99.self_attn");
        assert!(entry.is_none(), "non-existent layer key must return None");
    }

    /// Query with fewer than 3 dot-components returns None on fuzzy path.
    #[test]
    fn lookup_layer_calibration_short_query_returns_none() {
        let calib = make_t18_calib();
        let entry = lookup_layer_calibration(&calib, "model.layers");
        assert!(
            entry.is_none(),
            "query with < 3 dot-components should return None"
        );
    }

    /// "model.layers.0" has exactly 3 components — must match via fuzzy path.
    ///
    /// Regression test for the doc/code contradiction in lookup_layer_calibration:
    /// the guard is `< 3` (strict), so a 3-component query passes and matches
    /// "model.layers.0.self_attn" by shared 3-prefix.
    #[test]
    fn lookup_layer_calibration_three_component_query_matches() {
        let calib = make_t18_calib();
        let entry = lookup_layer_calibration(&calib, "model.layers.0");
        assert!(
            entry.is_some(),
            "3-component query should match via 3-prefix (guard is < 3, not <= 3)"
        );
    }

    // ── Issue #25: kv_max_seq_and_ceiling policy ──────────────────────────────

    /// A large `--max-ctx` becomes a ceiling, NOT the initial ring size.
    ///
    /// The ring must start at the lazy default while the ceiling carries the
    /// operator's `--max-ctx` (clamped to mpe). This is the core fix: a 140k
    /// override on a 262k-mpe model starts the ring at 4096, ceiling 140k.
    #[test]
    fn ceiling_large_override_starts_lazy() {
        let (initial, ceiling) = kv_max_seq_and_ceiling(Some(140_000), 262_144);
        assert_eq!(initial, KV_MAX_SEQ_DEFAULT, "ring starts at lazy default");
        assert_eq!(ceiling, 140_000, "ceiling = override (under mpe)");
    }

    /// `--max-ctx` is clamped to the model's positional capacity.
    #[test]
    fn ceiling_override_clamped_to_mpe() {
        let (initial, ceiling) = kv_max_seq_and_ceiling(Some(500_000), 131_072);
        assert_eq!(initial, KV_MAX_SEQ_DEFAULT);
        assert_eq!(ceiling, 131_072, "ceiling never exceeds mpe");
    }

    /// A sub-default ceiling must also cap the initial ring (don't pre-grow
    /// past a small ceiling).
    #[test]
    fn ceiling_below_default_caps_initial() {
        let (initial, ceiling) = kv_max_seq_and_ceiling(Some(2048), 131_072);
        assert_eq!(initial, 2048, "initial capped at the sub-default ceiling");
        assert_eq!(ceiling, 2048);
    }

    /// No override: ceiling falls back to `min(mpe, default)`, initial = that.
    #[test]
    fn ceiling_no_override_uses_mpe_default_chain() {
        // mpe above default → clamp to default.
        let (initial, ceiling) = kv_max_seq_and_ceiling(None, 131_072);
        assert_eq!(initial, KV_MAX_SEQ_DEFAULT);
        assert_eq!(ceiling, KV_MAX_SEQ_DEFAULT);
        // mpe below default → use mpe.
        let (initial, ceiling) = kv_max_seq_and_ceiling(None, 2048);
        assert_eq!(initial, 2048);
        assert_eq!(ceiling, 2048);
    }

    /// Unknown mpe (arch reports 0) is ignored; override stands alone.
    #[test]
    fn ceiling_unknown_mpe_ignored() {
        let (initial, ceiling) = kv_max_seq_and_ceiling(Some(64_000), 0);
        assert_eq!(initial, KV_MAX_SEQ_DEFAULT);
        assert_eq!(ceiling, 64_000, "no mpe clamp when mpe unknown");
        // No override + unknown mpe → bare default.
        let (initial, ceiling) = kv_max_seq_and_ceiling(None, 0);
        assert_eq!(initial, KV_MAX_SEQ_DEFAULT);
        assert_eq!(ceiling, KV_MAX_SEQ_DEFAULT);
    }

    // ── Issue #34: KV-codec net-benefit decision (policy layer) ───────────────

    /// Build the Gemma4 e2b layer mix: 7 global (head_dim=256, 1 kv head) +
    /// 28 windowed (window=512). Model-agnostic helper — keyed on geometry.
    fn e2b_layer_mix() -> Vec<KvLayerShape> {
        let mut v = Vec::with_capacity(35);
        for i in 0..35 {
            // e2b pattern: 1 full-attention per 6 (5 sliding + 1 full); here we
            // only need the COUNT (7 global / 28 windowed) to exercise the sum.
            if i % 6 == 5 || i == 34 {
                v.push(KvLayerShape {
                    head_dim: 256,
                    kv_heads: 1,
                    window: None,
                });
            } else {
                v.push(KvLayerShape {
                    head_dim: 256,
                    kv_heads: 1,
                    window: Some(512),
                });
            }
        }
        v
    }

    /// On the windowed+global mix, `Mixed` warns exactly when the architecture
    /// shares K/V — the same discriminator the allocation uses.
    ///
    /// The e2b mix is a Gemma4 shape, and Gemma4 does share: its global layers
    /// keep both bf16 mirrors for their consumer layers, so the packed store's
    /// codes and scales are pure addition on top of buffers that are already
    /// bf16-sized and the advisory fires. Give the same mix to a stack that does
    /// not share and the mirrors are not built, leaving the store alone — a
    /// saving, and the advisory must stay silent.
    ///
    /// Both arms are asserted because the failure that matters is asymmetric: a
    /// warn that fires on a dense stack is the estimator describing bytes
    /// nothing allocated, which is exactly the drift the shared flag exists to
    /// prevent. The windowed layers contribute 0 under both.
    #[test]
    fn net_negative_warn_on_swa_mix_follows_shared_kv() {
        let layers = e2b_layer_mix();
        let n_windowed = layers.iter().filter(|l| l.window.is_some()).count();
        let mixed = KvQuant::Mixed {
            k_bits: 8,
            v_bits: 8,
            k_group_size: 64,
            v_group_size: 64,
        };

        let (shared, n_global, n_win) = kv_codec_net_saving_total(mixed, &layers, 4096, true);
        assert_eq!(n_win, n_windowed);
        assert_eq!(n_global, 35 - n_windowed);
        assert!(
            shared < 0,
            "shared-KV: a store + both-mirror codec on the SWA+global mix at 4096 ctx \
             must be net-negative (warn fires); got {shared}"
        );

        let (dense, _, _) = kv_codec_net_saving_total(mixed, &layers, 4096, false);
        assert!(
            dense > 0,
            "dense: with no mirror to keep, the same mix must be a net saving and the \
             advisory must stay silent; got {dense}"
        );

        // The gap is the mirror pair on the global layers only — the windowed
        // ones run the bf16 ring under both topologies and contribute nothing.
        let mirror_pair: i64 = layers
            .iter()
            .filter(|l| l.window.is_none())
            .map(|l| 2 * 4096 * l.head_dim * l.kv_heads * 2)
            .sum::<u64>() as i64;
        assert_eq!(
            dense - shared,
            mirror_pair,
            "the whole difference between the two topologies is the global layers' \
             bf16 mirror pair"
        );
    }

    /// A mirror-only codec never warns on any layer mix or context: it builds no
    /// packed store, so it holds exactly the bf16 bytes and the saving is 0.
    /// The advisory used to fire on every one of these runs, which is what the
    /// operator was being told to work around.
    #[test]
    fn net_negative_warn_silent_for_mirror_only_codec() {
        let layers = e2b_layer_mix();
        for quant in [KvQuant::K8V4, KvQuant::K8V8, KvQuant::Planar] {
            for eff_seq in [512, 4096, 131_072] {
                let (saving, _, _) = kv_codec_net_saving_total(quant, &layers, eff_seq, false);
                assert_eq!(
                    saving, 0,
                    "{quant:?} at eff_seq={eff_seq} holds exactly the bf16 bytes — the \
                     net-negative advisory must stay silent"
                );
            }
        }
    }

    /// bf16 (`None`) never warns — saving is exactly 0 against its own baseline.
    #[test]
    fn net_negative_warn_silent_for_bf16() {
        let layers = e2b_layer_mix();
        let (saving, _, _) = kv_codec_net_saving_total(KvQuant::None, &layers, 4096, false);
        assert_eq!(saving, 0, "bf16 must never be net-negative against itself");
    }

    /// On one and the same pure-global layer mix at large context, the two
    /// seed-free K-only codecs land on opposite sides of the advisory: rotor
    /// fires it and iso does not.
    ///
    /// That is the point of running them together. Both are "a K-only codec
    /// with a per-group sideband on a mix with no windowed layers", so a
    /// decision keyed on the layer mix — or on the codec being K-only — would
    /// have to give them the same answer. The sign comes from each codec's own
    /// group geometry: rotor spends one `u32` code word per 3 head-dim slots
    /// (10.67 bits per value before any sideband), iso spends one per 4 (8),
    /// and both carry the same two sideband planes at the stored sideband
    /// dtype.
    #[test]
    fn net_negative_warn_splits_the_two_k_only_codecs_global_only() {
        // All-global mix (no windowed layers), large context.
        let layers: Vec<KvLayerShape> = (0..16)
            .map(|_| KvLayerShape {
                head_dim: 128,
                kv_heads: 8,
                window: None,
            })
            .collect();
        let (rotor, n_global, n_win) =
            kv_codec_net_saving_total(KvQuant::RotorKOnly4, &layers, 16_384, false);
        assert_eq!(n_global, 16);
        assert_eq!(n_win, 0);
        assert!(
            rotor < 0,
            "RotorKOnly4 stores 16.25 bits per value against bf16's 16 → net-negative even \
             all-global; got {rotor}"
        );
        let (iso, _, _) = kv_codec_net_saving_total(KvQuant::IsoKOnly4, &layers, 16_384, false);
        assert!(
            iso > 0,
            "IsoKOnly4 stores 12.125 bits per value on the same mix → the advisory must stay \
             silent; got {iso}"
        );
    }

    /// An all-windowed mix can never be net-negative for any codec: every
    /// windowed layer runs the bf16 ring, so the codec is a no-op (saving 0).
    #[test]
    fn all_windowed_mix_never_net_negative() {
        let layers: Vec<KvLayerShape> = (0..24)
            .map(|_| KvLayerShape {
                head_dim: 256,
                kv_heads: 1,
                window: Some(512),
            })
            .collect();
        for q in [KvQuant::K8V4, KvQuant::K8V8, KvQuant::Planar] {
            let (saving, n_global, n_win) = kv_codec_net_saving_total(q, &layers, 4096, false);
            assert_eq!(n_global, 0);
            assert_eq!(n_win, 24);
            assert_eq!(saving, 0, "{q:?} all-windowed mix must net to 0 (no warn)");
        }
    }
}
