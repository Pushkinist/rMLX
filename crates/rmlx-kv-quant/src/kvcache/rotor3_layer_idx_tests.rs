//! Thread real `layer_idx` into rotor3/rotor4 KV-cache construction.
//!
//! Pins two behaviours:
//!
//! * **Distinct layer tables** — when `n` caches are constructed with
//!   `KvQuant::Rotor3` and each receives a different `layer_idx` via
//!   `with_layer_idx`, the underlying rotor seeds differ. Verified by
//!   comparing the tables produced by `make_rotor_table` at different
//!   layer indices.
//!
//! * **Builder wiring** — `KvCache::with_quant_max_seq(…).with_layer_idx(i)`
//!   compiles and the `QuantRotorV3::new` / `QuantRotorV4::new` constructors
//!   accept distinct layer indices without error. Verified via append + dequant.

use rmlx_mlx::{Device, Dtype};

use crate::clifford::make_rotor_table;
use crate::kvcache::KvCache;
use crate::storage::{KvStorage, QuantRotorV3, QuantRotorV4};
use crate::KvQuant;
use rmlx_core::DispatchPolicy;

// ── helpers ───────────────────────────────────────────────────────────────────

fn f32_arr(data: &[f32], shape: &[i32]) -> rmlx_mlx::Array {
    // SAFETY:
    //   (1) Lifetime: `bytes` borrows `data` only for the duration of this fn;
    //       `Array::from_bytes` copies the slice before returning, so the
    //       borrow does not outlive `data`.
    //   (2) Layout: `f32` has align 4 and no padding; `data.len() * 4` covers
    //       exactly the contiguous backing buffer, matching the slice length.
    //   (3) Read-only: the resulting `&[u8]` is never mutated by this code or
    //       by MLX through this pointer — `Array::from_bytes` only reads it.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    rmlx_mlx::Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// `make_rotor_table` produces distinct tables for distinct layer
/// indices. This is the mathematical foundation: if seeding is correct,
/// cross-layer decorrelation exists.
#[test]
fn rotor3_make_rotor_table_distinct_for_different_layers() {
    // head_dim=64 → n_groups = ceil(64/3) = 22.
    let head_dim = 64usize;
    let n_groups = head_dim.div_ceil(3);

    let table_l0 = make_rotor_table(0, 0, n_groups);
    let table_l7 = make_rotor_table(7, 0, n_groups);

    // Tables must differ (proves seeding logic is wired correctly).
    // rotor_seed(0, 0, 0) ≠ rotor_seed(7, 0, 0) because (0 << 32) ≠ (7 << 32).
    assert_ne!(
        table_l0[..4],
        table_l7[..4],
        "layer_idx=0 and layer_idx=7 must produce different first-rotor entries"
    );

    // All n_groups entries must collectively differ (not just the first).
    assert_ne!(
        table_l0, table_l7,
        "layer_idx=0 and layer_idx=7 must produce entirely different rotor tables"
    );
}

/// `with_layer_idx` builder compiles and the resulting cache can
/// enter / exit prefill with a rotor3 codec without error.
#[test]
fn rotor3_with_layer_idx_builder_prefill_roundtrip() {
    // Construct two caches at different layer indices.
    let mut cache_l3 = KvCache::with_quant_max_seq(KvQuant::Rotor3, 1024).with_layer_idx(3);
    let mut cache_l11 = KvCache::with_quant_max_seq(KvQuant::Rotor3, 1024).with_layer_idx(11);

    let device = Device::Cpu;
    let shape = [1i32, 1, 4, 64]; // B=1, kv_h=1, seq=4, head_dim=64
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k_data = vec![0.1_f32; n];
    let v_data = vec![0.2_f32; n];
    let k = f32_arr(&k_data, &shape);
    let v = f32_arr(&v_data, &shape);

    // Prefill both caches.
    cache_l3.enter_prefill();
    cache_l11.enter_prefill();
    cache_l3.update(&k, &v, device).expect("l3 prefill update");
    cache_l11
        .update(&k, &v, device)
        .expect("l11 prefill update");
    cache_l3.exit_prefill(device).expect("l3 exit_prefill");
    cache_l11.exit_prefill(device).expect("l11 exit_prefill");

    // Both caches should have offset=4 after prefill.
    assert_eq!(cache_l3.offset(), 4, "l3 cache offset after prefill");
    assert_eq!(cache_l11.offset(), 4, "l11 cache offset after prefill");
}

/// `QuantRotorV3::new` at distinct layer indices generates different
/// rotor tables on first `append`, confirming that the `layer_idx` field is
/// used correctly when the codec initialises its rotor table.
#[test]
fn rotor3_storage_append_uses_layer_idx() {
    let head_dim = 64usize;
    let shape = vec![1_i32, 1, 0, head_dim as i32];
    let n_groups = head_dim.div_ceil(3);
    let n_tokens = 2usize;
    let v_data = vec![0.1_f32; n_tokens * head_dim];

    let mut qv0 = QuantRotorV3::new(shape.clone(), 16, 0);
    let mut qv5 = QuantRotorV3::new(shape, 16, 5);

    let token_shape = vec![1_i32, 1, n_tokens as i32, head_dim as i32];
    qv0.append(&v_data, &token_shape)
        .expect("qv0 append failed");
    qv5.append(&v_data, &token_shape)
        .expect("qv5 append failed");

    // After first append the rotor tables must be populated.
    assert_eq!(
        qv0.rotors.len(),
        n_groups * 4,
        "qv0 rotor table length after append"
    );
    assert_eq!(
        qv5.rotors.len(),
        n_groups * 4,
        "qv5 rotor table length after append"
    );

    // Tables must differ for different layer indices.
    assert_ne!(
        qv0.rotors[..4],
        qv5.rotors[..4],
        "QuantRotorV3 rotor tables must differ for layer_idx=0 vs layer_idx=5"
    );
}

/// Rotor4 counterpart of `rotor3_storage_append_uses_layer_idx`.
#[test]
fn rotor4_storage_append_uses_layer_idx() {
    let head_dim = 64usize;
    let shape = vec![1_i32, 1, 0, head_dim as i32];
    let n_groups = head_dim.div_ceil(3);
    let n_tokens = 2usize;
    let v_data = vec![0.1_f32; n_tokens * head_dim];

    let mut qv0 = QuantRotorV4::new(shape.clone(), 16, 0);
    let mut qv8 = QuantRotorV4::new(shape, 16, 8);

    let token_shape = vec![1_i32, 1, n_tokens as i32, head_dim as i32];
    qv0.append(&v_data, &token_shape)
        .expect("qv0 rotor4 append failed");
    qv8.append(&v_data, &token_shape)
        .expect("qv8 rotor4 append failed");

    assert_eq!(
        qv0.rotors.len(),
        n_groups * 4,
        "qv0 rotor4 table length after append"
    );
    assert_ne!(
        qv0.rotors[..4],
        qv8.rotors[..4],
        "QuantRotorV4 rotor tables must differ for layer_idx=0 vs layer_idx=8"
    );
}

/// All `n_layers` expected rotor tables (computed directly via
/// `make_rotor_table`) are pairwise distinct. Verifies that no two layers
/// collide in the seeding space at the mathematical layer (no cache involved).
#[test]
fn rotor3_multi_layer_all_distinct() {
    let n_layers = 8usize;
    let head_dim = 64usize;
    let n_groups = head_dim.div_ceil(3);

    // Gather the first rotor from each layer's expected table.
    let first_rotors: Vec<[f32; 4]> = (0..n_layers)
        .map(|i| {
            let table = make_rotor_table(i as u32, 0, n_groups);
            [table[0], table[1], table[2], table[3]]
        })
        .collect();

    // Assert all entries are pairwise distinct (no layer collision).
    for i in 0..n_layers {
        for j in (i + 1)..n_layers {
            assert_ne!(
                first_rotors[i], first_rotors[j],
                "layers {i} and {j} must have distinct rotor tables"
            );
        }
    }
}

/// End-to-end: build `n_layers` caches via the arch-builder pattern
/// (`with_quant_max_seq(Rotor3, _).with_layer_idx(i)`), drive each through a
/// real `update` so the rotor table is populated, then read the per-layer
/// rotor tables from the live `KvStorage::RotorV3` and assert they are all
/// pairwise distinct. This catches the layer-threading defect class at the
/// integration layer: if the builder fails to thread `layer_idx`, every
/// layer's table collapses to the `layer_idx=0` seed and these assertions fire.
/// Bracketed counterpart: the same builder, driven through the **production**
/// prefill path.
///
/// The unbracketed test below drives the codec body so a rotor table exists to
/// compare. That leaves the path a real serve takes untested, which is exactly
/// the path this change moved — so it is asserted here instead: a prefilled
/// `Rotor3` cache builds no store at all, and its `layer_idx` still threads
/// through the builder (the field the table is derived from).
#[test]
fn rotor3_multi_layer_prefill_builds_no_store_and_keeps_layer_idx() {
    let n_layers = 4usize;
    let device = Device::Cpu;
    let head_dim = 64i32;
    let shape = [1i32, 1, 2, head_dim];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k_data = vec![0.1_f32; n];
    let v_data = vec![0.2_f32; n];

    for i in 0..n_layers {
        let mut cache = KvCache::with_quant_max_seq(KvQuant::Rotor3, 1024).with_layer_idx(i);
        cache.enter_prefill();
        cache
            .update(&f32_arr(&k_data, &shape), &f32_arr(&v_data, &shape), device)
            .expect("prefill update");
        cache.exit_prefill(device).expect("exit_prefill");

        assert_eq!(
            cache.layer_idx(),
            i,
            "builder must thread layer_idx through the prefill path too"
        );
        assert_eq!(
            cache.storage().resident_bytes(),
            0,
            "layer {i}: Rotor3 decodes off the bf16 mirror, so a prefilled cache \
             must hold no packed store — and therefore no rotor table"
        );
        assert!(
            cache.decode_fp16_kv().is_some(),
            "layer {i}: the mirror is what decode reads, so it must be live"
        );
    }
}

#[test]
fn rotor3_multi_layer_builder_populates_distinct_tables() {
    let n_layers = 4usize;
    let device = Device::Cpu;
    let head_dim = 64i32;
    let shape = [1i32, 1, 2, head_dim]; // B=1, kv_h=1, seq=2, head_dim=64
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k_data = vec![0.1_f32; n];
    let v_data = vec![0.2_f32; n];

    // Build n_layers caches via the arch-builder pattern and append once so the
    // rotor table is generated. The append is deliberately NOT bracketed by
    // `enter_prefill`/`exit_prefill`: a prefilled Rotor3 cache decodes off the
    // bf16 mirror, so `exit_prefill` builds no packed store and there would be
    // no rotor table to read. The unbracketed append is the same path a
    // hydrated cache takes, which is where the store is load-bearing.
    let mut caches: Vec<KvCache> = (0..n_layers)
        .map(|i| KvCache::with_quant_max_seq(KvQuant::Rotor3, 1024).with_layer_idx(i))
        .collect();
    for cache in &mut caches {
        let k = f32_arr(&k_data, &shape);
        let v = f32_arr(&v_data, &shape);
        cache.update(&k, &v, device).expect("codec append");
    }

    // Read the first rotor (4 floats) from each layer's live storage.
    let first_rotors: Vec<[f32; 4]> = caches
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let KvStorage::RotorV3 { v, .. } = c.storage() else {
                panic!("layer {i}: expected RotorV3 storage");
            };
            let qv = v.as_ref().expect("QuantRotorV3 populated after append");
            assert!(
                qv.rotors.len() >= 4,
                "layer {i}: rotor table populated after append"
            );
            [qv.rotors[0], qv.rotors[1], qv.rotors[2], qv.rotors[3]]
        })
        .collect();

    // Pairwise-distinct: any collision means layer_idx threading failed.
    for i in 0..n_layers {
        for j in (i + 1)..n_layers {
            assert_ne!(
                first_rotors[i], first_rotors[j],
                "builder-driven layers {i} and {j} must have distinct rotor tables"
            );
        }
    }
}

/// `KvCache::from_storage` (the SSD hydrate constructor) records the
/// caller-supplied `layer_idx`. The SSD-block reader passes `enumerate()`'s
/// index, so a re-quantize fired after hydration uses the correct rotor seed.
#[test]
fn rotor3_from_storage_records_layer_idx() {
    // KvStorage::None is the simplest variant; we only need to verify the
    // builder field passthrough. The codec-specific storage round-trip is
    // exercised by the existing block_io_tests hydrate suite.
    let cache_l0 = KvCache::from_storage(
        KvStorage::None { max_seq: 1024 },
        KvQuant::Rotor3,
        0,
        0,
        DispatchPolicy::default(),
        false,
    );
    let cache_l7 = KvCache::from_storage(
        KvStorage::None { max_seq: 1024 },
        KvQuant::Rotor3,
        0,
        7,
        DispatchPolicy::default(),
        false,
    );

    assert_eq!(cache_l0.layer_idx, 0, "from_storage layer_idx=0 preserved");
    assert_eq!(cache_l7.layer_idx, 7, "from_storage layer_idx=7 preserved");
}
