use super::*;
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

/// Build a small dummy `SparseAttnInputs` for the gate-OFF unit tests.
///
/// The gate check (`!sparse_attn_enabled()`) fires before any input
/// validation or kernel dispatch — so these arrays only need to be
/// constructible, not coherent.  We make them tiny f32 arrays sized for
/// `b=1, kv_h=1, heads_per_kv=1, kv_seq=2, head_dim=64`.
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Array build is correct"
)]
fn make_dummy_inputs() -> (Array, Array, Array, Array, Array) {
    let head_dim = 64usize;
    let kv_seq = 2usize;
    let q_bytes: Vec<u8> = vec![0u8; head_dim * 4];
    let q = Array::from_bytes(&q_bytes, &[1, 1, 1, head_dim as i32], Dtype::F32).expect("dummy Q");
    // PlanarQuant K codes: (head_dim / 32) * 4 = 8 u32 words per token, times 2 tokens = 16.
    let codes_per_tok = (head_dim / 32) * 4;
    let codes_total = kv_seq * codes_per_tok;
    let codes_bytes: Vec<u8> = vec![0u8; codes_total * 4];
    let k_codes =
        Array::from_bytes(&codes_bytes, &[codes_total as i32], Dtype::U32).expect("dummy K codes");
    let scales_total = kv_seq * head_dim / 2;
    let scales_bytes: Vec<u8> = vec![0u8; scales_total * 4];
    let k_scales = Array::from_bytes(&scales_bytes, &[scales_total as i32], Dtype::F32)
        .expect("dummy K scales");
    let rot_total = kv_seq * head_dim / 16;
    let rot_bytes: Vec<u8> = vec![0u8; rot_total * 4];
    let k_rot32 =
        Array::from_bytes(&rot_bytes, &[rot_total as i32], Dtype::U32).expect("dummy K rot32");
    let v_total = kv_seq * head_dim;
    let v_bytes: Vec<u8> = vec![0u8; v_total * 4];
    let v = Array::from_bytes(
        &v_bytes,
        &[1, 1, kv_seq as i32, head_dim as i32],
        Dtype::F32,
    )
    .expect("dummy V");
    (q, k_codes, k_scales, k_rot32, v)
}

fn make_dummy_sparse_inputs<'a>(
    q: &'a Array,
    k_codes: &'a Array,
    k_scales: &'a Array,
    k_rot32: &'a Array,
    v: &'a Array,
) -> SparseAttnInputs<'a> {
    SparseAttnInputs {
        query: q,
        k_codes,
        k_scales,
        k_rot32,
        v,
        b: 1,
        kv_h: 1,
        kv_seq: 2,
        head_dim: 64,
        heads_per_kv: 1,
        layer_idx: 0,
        scale: 0.125,
        device: Device::Cpu,
    }
}

// ── Table completeness ────────────────────────────────────────────────────────

/// The table lists exactly the codecs that can reach the fused-QK path.
#[test]
fn fused_qk_table_lists_only_reachable_codecs() {
    assert_eq!(
        FUSED_QK_TABLE.len(),
        6,
        "FUSED_QK_TABLE must have 6 entries (K8V4, K8V8, TurboSym3, TurboSym4, \
         RotorK3Asym, RotorK4Asym)"
    );
}

/// Every entry has a kernel, and every entry's codec keeps the bf16 K mirror
/// the fused-QK shadow is seeded from.
///
/// The mirror check is the load-bearing half: a codec with no bf16 K can never
/// reach this path at any shape on any arch, so listing one would be listing a
/// kernel nothing dispatches. That was the table's state for the iso and rotor
/// `*Sym` / `*KOnly` codecs, each of which decodes through its own
/// flash-decode-over-quant kernel instead.
#[test]
fn fused_qk_table_entries_are_dispatchable() {
    for entry in FUSED_QK_TABLE {
        assert!(
            entry.kernel.is_some(),
            "entry for {:?} must have kernel=Some",
            entry.kv_quant
        );
        assert!(
            entry.kv_quant.feeds_bf16_k_at_decode(),
            "entry for {:?} keeps no bf16 K mirror, so the fused-QK shadow can never be \
             seeded for it — the entry is unreachable",
            entry.kv_quant
        );
    }
}

/// `lookup_fused_qk` resolves every codec that has a fused-QK kernel, and
/// refuses the ones that keep no bf16 K mirror.
#[test]
fn lookup_fused_qk_resolves_the_reachable_codecs() {
    for kq in [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::TurboSym3,
        KvQuant::TurboSym4,
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
    ] {
        assert!(
            lookup_fused_qk(kq).is_some(),
            "lookup_fused_qk({kq:?}) must return Some"
        );
    }
    // These decode through their own flash-decode-over-quant kernel and keep
    // no bf16 K, so the fused-QK shadow can never be seeded for them.
    for kq in [
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
    ] {
        assert!(
            lookup_fused_qk(kq).is_none(),
            "lookup_fused_qk({kq:?}) must return None — the codec keeps no bf16 K mirror"
        );
    }
}

/// The rotor-asym entries are matched by variant, not by V-side payload.
///
/// The table spells one `(v_bits, v_group_size)` pair per rotor-asym entry.
/// The V codec never reaches a K-side kernel, so every other V configuration
/// must resolve to the same kernel; a `PartialEq` lookup would silently return
/// `None` for all of them.
#[test]
fn lookup_fused_qk_ignores_the_rotor_asym_v_payload() {
    for (v_bits, v_group_size) in [(2_u8, 64_u16), (3, 64), (4, 32), (4, 128)] {
        assert!(
            lookup_fused_qk(KvQuant::RotorK3Asym {
                v_bits,
                v_group_size
            })
            .is_some(),
            "RotorK3Asym(v_bits={v_bits}, v_group_size={v_group_size}) must resolve"
        );
        assert!(
            lookup_fused_qk(KvQuant::RotorK4Asym {
                v_bits,
                v_group_size
            })
            .is_some(),
            "RotorK4Asym(v_bits={v_bits}, v_group_size={v_group_size}) must resolve"
        );
    }
}

// ── sparse_attn_dispatch_if_enabled stub tests ────────────────────────────────
//
// All four cases return None in Exec A — the dispatch site is a placeholder
// until Exec B wires in real phase-1 / phase-2 MSL kernels.
//
// OnceLock note: `sparse_attn_enabled()` latches its env read on first call.
// In a fresh test process the env-var is unset → returns false → these
// assertions are stable regardless of test ordering. The CLI env-setter
// tests live in `rmlx-cli::commands::serve_tests` (no overlap).

/// Build a `HeadBudgets` via JSON round-trip (the struct is
/// `#[non_exhaustive]` so direct construction is banned outside
/// `rmlx-loader`).
#[allow(
    clippy::expect_used,
    reason = "test helper: JSON is a compile-time constant; panic on parse failure is the correct behavior"
)]
fn make_test_head_budgets() -> HeadBudgets {
    let json = r#"{
      "version": 1,
      "model_name": "test",
      "num_layers": 1,
      "num_heads": 2,
      "calibration": {
        "method": "softmax_mass",
        "prompt_set_sha256": "ab",
        "num_prompts": 1,
        "max_seq_len": 128,
        "mass_threshold": 0.95
      },
      "per_layer_per_head_budget": [[16, 16]]
    }"#;
    serde_json::from_str(json).expect("parse test head_budgets fixture")
}

#[test]
fn sparse_attn_dispatch_none_when_gate_off_and_budgets_absent() {
    // Default state in a fresh test process: env-var unset, no budgets.
    // The function must return None because the gate short-circuits before
    // touching the inputs (Exec B: gate ON path runs the kernels).
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, None).is_none());
}

#[test]
fn sparse_attn_dispatch_none_when_gate_off_with_budgets_present() {
    // Even with budgets present, the gate is off (OnceLock latches OFF
    // in this test process) → must return None.
    let budgets = make_test_head_budgets();
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, Some(&budgets)).is_none());
}

#[test]
fn sparse_attn_dispatch_short_circuits_on_missing_budgets() {
    // Even if a future test in the same process latched the OnceLock to
    // true, missing budgets must still produce None (correctness invariant).
    let (q, kc, ks, kr, v) = make_dummy_inputs();
    let inputs = make_dummy_sparse_inputs(&q, &kc, &ks, &kr, &v);
    assert!(sparse_attn_dispatch_if_enabled(&inputs, None).is_none());
}

/// lookup_fused_qk returns None for non-table KvQuant variants.
#[test]
fn lookup_fused_qk_returns_none_for_non_table_variants() {
    // These variants have no fused-QK table entry.
    for kq in [KvQuant::K8VTurbo3, KvQuant::Planar, KvQuant::PlanarK] {
        assert!(
            lookup_fused_qk(kq).is_none(),
            "lookup_fused_qk({kq:?}) must return None (not a fused-QK target)"
        );
    }
}

// ── Table entry correctness ───────────────────────────────────────────────────

/// The table contains an entry for every codec with a fused-QK kernel.
#[test]
fn fused_qk_table_contains_all_spec_entries() {
    let required = [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::TurboSym3,
        KvQuant::TurboSym4,
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
    ];
    for kq in required {
        let want = std::mem::discriminant(&kq);
        let found = FUSED_QK_TABLE
            .iter()
            .any(|e| std::mem::discriminant(&e.kv_quant) == want);
        assert!(found, "FUSED_QK_TABLE must contain an entry for {kq:?}");
    }
}
