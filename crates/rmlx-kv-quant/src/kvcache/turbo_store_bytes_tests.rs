//! Byte-level lock on the TurboQuant storage layer, at the deepest seam a
//! caller can reach on CPU.
//!
//! Sibling of `rotor_store_bytes_tests.rs` and `iso_store_bytes_tests.rs`. The
//! method is the same and is not restated here; `docs/KV_ROTOR_TWINS.md` holds
//! it. This file states what the turbo family pins and where it stops, and
//! `docs/KV_TURBO_TWINS.md` holds the decisions it is the oracle for.
//!
//! # What cannot move
//!
//! Every turbo spelling that exists before the K-storage unification exists
//! after it, and for each spelling, at each shape:
//!
//! * the **packed store bytes** are identical — every `codes` byte, every
//!   `scales` float, the block boundaries, the per-block bit tag, the
//!   accumulated shape, the GPU-buffer bookkeeping and the affine q8_0
//!   companion plane on the asymmetric spellings;
//! * the **K and V rows the attention receives** from that store are
//!   identical, bit for bit, at the chunk append and at every decode step;
//! * `resident_bytes()` is identical;
//! * and the served temp-0 token stream is identical.
//!
//! There is no named exception on this axis. The collapse's one intended
//! observable change is on the SSD hydrate path, outside this file:
//! `crates/rmlx-kv-ssd/src/block_io_turbo_hydrate_tests.rs` holds it.
//!
//! Two pairs of pin rows are identical and are meant to be: a TCQ spelling
//! writes the same bytes as its plain sibling, for a structural reason the
//! last-but-one test states and measures. That is a property of the TCQ
//! encoder, not of this collapse.
//!
//! # Why the served digest is not the oracle
//!
//! Stronger here than on any other family. All six turbo spellings report
//! `decode_reads_packed_store() == false`, `feeds_bf16_k_at_decode(false)`
//! and `feeds_bf16_v_at_decode(false)`, so `materialises_packed_store()` is
//! false for every one of them, and the form that takes is stronger than "the
//! store stops being read". `exit_prefill` returns at its
//! `materialises_packed_store()` gate, **before** every arm that would bulk
//! encode one, and clears whatever payload the cache arrived carrying. A
//! served prefill therefore writes no turbo store at all, and the decode
//! entries short-circuit to `update_decode_fp16` while the bf16 seed is live.
//! **No served capture can read a single turbo store byte**, at any width, on
//! any model: a run against an empty store and a run against a correct one
//! emit the same tokens. The store bytes are observable here and nowhere
//! else, which is why this file drives `update` with `in_prefill` false and no
//! bf16 seed — that is the one CPU route on which all six codecs write their
//! store.
//!
//! # Shapes
//!
//! Two, the same pair the rotor and iso pins use, so a reviewer reading the
//! three files side by side compares like with like:
//!
//! | shape | `kv_h` | `head_dim` | why |
//! |---|---|---|---|
//! | A | 1 | 128 | single KV head (shared-KV arch); power-of-two `head_dim` |
//! | B | 4 | 96 | `kv_h > 1`; non-power-of-two `head_dim` |
//!
//! **No turbo shape has a ragged group.** The codec groups `GROUP_SIZE` (32)
//! elements over the last axis, and both `head_dim` values divide by 32
//! exactly, so the padded-last-group case cannot occur here and there is no
//! third shape to add for it.
//!
//! The other axis of each spelling sets the floor: the q8_0 K side of the four
//! asymmetric spellings needs `B * kv_h * seq * head_dim % 128 == 0` on
//! **every** chunk including a one-token step. Shape A gives 3072 and 128;
//! shape B gives 9216 and 384. All four are multiples of 128.
//!
//! # What this pin cannot see
//!
//! Named so the collapse's reviewer knows where this file stops. Every item is
//! inside, or reached from, the files the collapse unifies, and no assertion
//! here can turn red on a defect in it.
//!
//! * **The V-axis device split.** `tsym_update` resolves the V device from
//!   `BITS` — `Device::Cpu` at 3, the caller's device at 4. On a CPU drive the
//!   two are the same routing, so **nothing in this file can tell a body that
//!   keeps the rule from one that lost it.** The rule is
//!   load-bearing: `QuantV::append` enters its GPU branch on `device ==
//!   Device::Gpu` with no bit-width guard and then returns `Error::Quant` for
//!   `bits != 4`, so a 3-bit V handed `Device::Gpu` fails the append. The gate
//!   over it is `make gpu-test` and the served capture.
//! * **`exit_prefill`.** This file drives `update` with `in_prefill` false,
//!   the only CPU route on which all six spellings write a store — but it is
//!   not the route production takes, and on this family `exit_prefill` is also
//!   what *clears* every one of these stores.
//! * **The fused flash-decode arms** and the turbo K fused-QK kernels
//!   (`turbo_k3_fused_qk` / `turbo_k4_fused_qk`). Gated on `Device::Gpu`; a CPU
//!   drive never reaches them.
//! * **The GPU append path** — `gpu_codes_buf` / `gpu_scales_buf` allocation,
//!   the paged growth, the hydrated-init upload branch (where the two stores'
//!   scale-byte encodings differ, one safe and one `unsafe`) and the MSL
//!   encode dispatch. Unreachable from a `Device::Cpu` drive, and the largest
//!   unseen surface in the storage pair.
//! * **`from_cpu_blocks` and `try_deep_clone`** — the SSD-hydrate and
//!   branch-clone constructors. Never called on this path. The hydrate half is
//!   pinned in `rmlx-kv-ssd`.
//! * **`max_seq`.** Deliberately not a digest field. Both K stores carry one,
//!   neither ever reads it — `append` sizes its buffer from its own `max_seq`
//!   parameter — so pinning it here would force a re-baseline over a field no
//!   decode path reads. Where it is *not* inert is the hydrate constructor,
//!   and that is where it is pinned.
//! * **Bit-exactness under a different toolchain.** The pins are f32 results
//!   from this host's codegen. They judge one change on one toolchain; they
//!   are not a portable golden.
//!
//! `truncate_to` **is** covered — the drive rolls back into the bulk chunk
//! after the decode steps, so both halves of the truncate plan run and the
//! post-truncate store bytes are a pinned column.

use super::core::KvCache;
use crate::storage::KvStorage;
use crate::test_utils::{
    array_bytes, env_lock, f32_arr, fnv1a64, lcg_data, push_quant_k, StoreBytes, TEST_SEED,
};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

/// Layer index every cell is built at. No turbo store seeds anything from it
/// today; the cells are built at a fixed one so a change that starts to would
/// move the pin rather than pass quietly.
const TEST_LAYER_IDX: usize = 3;
/// Storage capacity. Larger than the driven length so no ring wrap is in play.
const TEST_MAX_SEQ: i32 = 512;
/// Positions in the single bulk append.
const CHUNK_SEQ: i32 = 24;
/// One-token appends driven after the chunk.
const DECODE_STEPS: usize = 3;

/// `(kv_h, head_dim)` — see the shape table in the module doc.
const SHAPE_A: (i32, i32) = (1, 128);
const SHAPE_B: (i32, i32) = (4, 96);

/// Serialise a TurboQuant block store — the K twin and `QuantV` carry the same
/// payload type (`TurboBlocks`) and the same header fields, so one body covers
/// both axes.
///
/// A macro, not a fn: the K store is `QuantKTurbo<BITS>` and the V store is
/// `QuantV`, two unrelated types, so one fn cannot take both. The two K widths
/// are one type since the K-storage collapse and need no arm of their own.
macro_rules! push_turbo {
    ($out:expr, $side:expr, $store:expr) => {{
        let s = $store;
        $out.tag($side);
        $out.i32s("shape", &s.shape);
        $out.u8_("bits", s.bits);
        $out.u8_("gpu_codes_live", u8::from(s.gpu_codes_buf.is_some()));
        $out.u8_("gpu_scales_live", u8::from(s.gpu_scales_buf.is_some()));
        $out.usize_("gpu_words_per_step", s.gpu_words_per_step as usize);
        $out.usize_("gpu_scales_per_step", s.gpu_scales_per_step as usize);
        $out.usize_("gpu_capacity", s.gpu_capacity as usize);
        $out.usize_("n_blocks", s.blocks.len());
        for (i, b) in s.blocks.iter().enumerate() {
            $out.usize_("block", i);
            $out.u8s("codes", &b.codes);
            $out.f32s("scales", &b.scales);
            $out.i32s("original_shape", &b.original_shape);
            $out.u8_("block_bits", b.bits);
        }
    }};
}

/// `(bits, code bytes, scales)` summed over a store's blocks.
///
/// A macro for the same reason as [`push_turbo`].
macro_rules! turbo_geometry {
    ($store:expr) => {{
        let s = $store;
        (
            s.bits,
            s.blocks.iter().map(|b| b.codes.len()).sum::<usize>(),
            s.blocks.iter().map(|b| b.scales.len()).sum::<usize>(),
        )
    }};
}

/// Every byte of the storage a turbo spelling holds, in one digest.
///
/// A store the spelling did not populate serialises as its own `absent` tag,
/// so "empty" and "one empty block" are different digests.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant: the storage variant is the one the KvQuant selected; any other is a construction bug, and the explicit panic names it sooner than a wrong digest would"
)]
fn store_digest(storage: &KvStorage) -> u64 {
    let mut out = StoreBytes::default();
    macro_rules! opt_q8 {
        ($slot:expr) => {
            match $slot {
                Some(s) => push_quant_k(&mut out, "k_q8", s),
                None => out.tag("k_q8_absent"),
            }
        };
    }
    macro_rules! opt_turbo {
        ($slot:expr, $name:expr) => {
            match $slot {
                Some(s) => push_turbo!(out, $name, s),
                None => out.tag(concat!($name, "_absent")),
            }
        };
    }
    match storage {
        KvStorage::K8VTurbo3 { k, v, .. }
        | KvStorage::K8VTurbo3Tcq { k, v, .. }
        | KvStorage::K8VTurbo2 { k, v, .. }
        | KvStorage::K8VTurbo2Tcq { k, v, .. } => {
            opt_q8!(k.as_ref());
            opt_turbo!(v.as_ref(), "v_turbo");
        }
        KvStorage::TurboSym3 { k, v, .. } => {
            opt_turbo!(k.as_ref(), "k_turbo");
            opt_turbo!(v.as_ref(), "v_turbo");
        }
        KvStorage::TurboSym4 { k, v, .. } => {
            opt_turbo!(k.as_ref(), "k_turbo");
            opt_turbo!(v.as_ref(), "v_turbo");
        }
        other => panic!(
            "not a turbo storage variant: {}",
            super::helpers::storage_variant_name(other)
        ),
    }
    out.digest()
}

/// What one cell observes.
struct CellObservation {
    /// Store digest after the bulk chunk.
    store_after_chunk: u64,
    /// Store digest after the last decode step.
    store_after_decode: u64,
    /// Store digest after truncating back into the bulk chunk.
    ///
    /// `truncate_to` is inside the files the collapse unifies and is reached by
    /// no other assertion in the tree — the block-truncate suite states that it
    /// covers the plan, not the turbo stores' calls to it.
    store_after_truncate: u64,
    /// Digest of the K/V rows the attention received, chunk and every step.
    rows: u64,
    /// `KvCache::resident_bytes` after the last decode step.
    resident_bytes: u64,
}

/// Drive one spelling at one shape with `in_prefill` false throughout.
///
/// The prefill bracket is deliberately absent. `exit_prefill` clears the
/// payload of every spelling whose `materialises_packed_store()` is false —
/// which on this family is all six — so a prefill-bracketed drive would pin
/// six empty stores and read as a clean scan. Appending straight into the
/// decode dispatch is the one CPU route on which all six write their store.
#[allow(
    clippy::expect_used,
    reason = "test driver: every append here is on a shape the spelling accepts, so a failure is the defect under test and the panic names it"
)]
fn drive(quant: KvQuant, shape: (i32, i32)) -> CellObservation {
    let (kv_h, head_dim) = shape;
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ).with_layer_idx(TEST_LAYER_IDX);

    let mut rows = Vec::new();

    let chunk_shape = [1_i32, kv_h, CHUNK_SEQ, head_dim];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n_chunk, TEST_SEED), &chunk_shape);
    let v = f32_arr(&lcg_data(n_chunk, TEST_SEED ^ 0x5a5a), &chunk_shape);
    let (k_out, v_out) = cache.update(&k, &v, device).expect("bulk append");
    rows.extend_from_slice(&array_bytes(&k_out));
    rows.extend_from_slice(&array_bytes(&v_out));
    let store_after_chunk = store_digest(&cache.storage);

    let step_shape = [1_i32, kv_h, 1, head_dim];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    for step in 0..DECODE_STEPS {
        let seed = TEST_SEED.wrapping_add(step as u64 + 1);
        let ks = f32_arr(&lcg_data(n_step, seed), &step_shape);
        let vs = f32_arr(&lcg_data(n_step, seed ^ 0x5a5a), &step_shape);
        let (ko, vo) = cache.update(&ks, &vs, device).expect("decode step");
        rows.extend_from_slice(&array_bytes(&ko));
        rows.extend_from_slice(&array_bytes(&vo));
    }

    let store_after_decode = store_digest(&cache.storage);
    let resident_bytes = cache.resident_bytes();

    // Roll back into the bulk chunk: drops whole decode blocks and splits the
    // chunk block, so both halves of the truncate plan run.
    cache
        .truncate_to(CHUNK_SEQ + 1)
        .expect("truncate into the bulk chunk");

    CellObservation {
        store_after_chunk,
        store_after_decode,
        store_after_truncate: store_digest(&cache.storage),
        rows: fnv1a64(&rows),
        resident_bytes,
    }
}

/// Every turbo spelling the enum can spell, in `ALL_KV_QUANTS` order.
///
/// The filter is the enum's own `Display`, not a hand-written variant list. A
/// list would name only the variants that existed when it was written, so a
/// seventh turbo variant would never enter `want` and the census below would
/// stay green with nothing pinned — a gate that cannot fail on the one event
/// it exists for.
///
/// **Two tokens, not one.** The family spells itself two ways: the asymmetric
/// members read `k8vturbo*` and the symmetric ones read `tsym*`, which shares
/// no substring with `turbo`. A single `contains("turbo")` filter would find
/// four of the six and miss **exactly the two that build the K-side store this
/// oracle exists for**. No other spelling's `Display` text contains either
/// token, so the two together are the membership test.
fn turbo_spellings() -> Vec<KvQuant> {
    ALL_KV_QUANTS
        .iter()
        .copied()
        .filter(|q| {
            let s = q.to_string();
            s.contains("turbo") || s.contains("tsym")
        })
        .collect()
}

/// Turbo spellings `ALL_KV_QUANTS` holds today.
const TURBO_SPELLING_COUNT: usize = 6;

/// Turbo spellings whose K side is the store the collapse unifies.
const TURBO_K_TWIN_COUNT: usize = 2;

/// One pinned cell: `(spelling, kv_h, head_dim, store_after_chunk,
/// store_after_decode, store_after_truncate, rows, resident_bytes)`.
type Pin = (&'static str, i32, i32, u64, u64, u64, u64, u64);

/// The "before" side of the oracle, captured on the tree the unification
/// starts from. Any generic type that produces a different byte anywhere in a
/// turbo store, hands attention a different row, or changes a cell's residency
/// turns one of these red and names the cell.
///
/// Every row is held at both widths. The collapse has no licence to move one:
/// its single intended observable change is on the hydrate path, which no cell
/// here reaches.
const PINS: &[Pin] = &[
    (
        "k8vturbo3",
        1,
        128,
        0xc978cd9e3b6ea997,
        0xfa7a4a48703d1920,
        0xa1c0014511d8877f,
        0x57e3f8e7c5855aa4,
        5292,
    ),
    (
        "k8vturbo3",
        4,
        96,
        0xa84500cd0c4668c2,
        0x96c636db7c2c1998,
        0x12f25db93bf7bed7,
        0x0553295cb8aa3448,
        15876,
    ),
    (
        "k8vturbo3tcq",
        1,
        128,
        0xc978cd9e3b6ea997,
        0xfa7a4a48703d1920,
        0xa1c0014511d8877f,
        0x57e3f8e7c5855aa4,
        5292,
    ),
    (
        "k8vturbo3tcq",
        4,
        96,
        0xa84500cd0c4668c2,
        0x96c636db7c2c1998,
        0x12f25db93bf7bed7,
        0x0553295cb8aa3448,
        15876,
    ),
    (
        "k8vturbo2",
        1,
        128,
        0xb114acd71150baba,
        0x01f6d9cf70840ce3,
        0x87f9d45c75d9b483,
        0x223b929dc161c235,
        4860,
    ),
    (
        "k8vturbo2",
        4,
        96,
        0x8086b874eb46762d,
        0x04490d83c3d191be,
        0x64eb0d6ccff312e3,
        0x6be14156bf61667d,
        14580,
    ),
    (
        "k8vturbo2tcq",
        1,
        128,
        0xb114acd71150baba,
        0x01f6d9cf70840ce3,
        0x87f9d45c75d9b483,
        0x223b929dc161c235,
        4860,
    ),
    (
        "k8vturbo2tcq",
        4,
        96,
        0x8086b874eb46762d,
        0x04490d83c3d191be,
        0x64eb0d6ccff312e3,
        0x6be14156bf61667d,
        14580,
    ),
    (
        "tsym3",
        1,
        128,
        0xe525e0e0bd6a6469,
        0x6acf7a24385304ed,
        0x474959ce87b2ab24,
        0x6629bbeb17efad1b,
        3456,
    ),
    (
        "tsym3",
        4,
        96,
        0xfbf1bdba9a24db2d,
        0xd6a38b537d602e9d,
        0x9f9f0b549f2f131e,
        0xa488a46c89e0f7ef,
        10368,
    ),
    (
        "tsym4",
        1,
        128,
        0xc5f5b4175ed98e1a,
        0x820b35f80bbe0988,
        0x3ca45144c3e90aa5,
        0x3b93983043ab2f90,
        4320,
    ),
    (
        "tsym4",
        4,
        96,
        0x2ce01a190745e393,
        0x82815b7647065d1a,
        0xb697676213e4fce8,
        0x08eb6752f2e72a9d,
        12960,
    ),
];

/// Look a pin up by spelling and shape.
fn pin_for(name: &str, shape: (i32, i32)) -> Option<&'static Pin> {
    PINS.iter()
        .find(|p| p.0 == name && p.1 == shape.0 && p.2 == shape.1)
}

/// Every turbo spelling, at both shapes, holds the bytes it held before.
#[test]
fn turbo_store_bytes_are_pinned_per_spelling_and_shape() {
    let _guard = env_lock();
    let mut missing = Vec::new();
    let mut observed = Vec::new();
    for quant in turbo_spellings() {
        let name = quant.to_string();
        for shape in [SHAPE_A, SHAPE_B] {
            let obs = drive(quant, shape);
            observed.push(format!(
                "    (\"{}\", {}, {}, {:#018x}, {:#018x}, {:#018x}, {:#018x}, {}),",
                name,
                shape.0,
                shape.1,
                obs.store_after_chunk,
                obs.store_after_decode,
                obs.store_after_truncate,
                obs.rows,
                obs.resident_bytes
            ));
            let Some(pin) = pin_for(&name, shape) else {
                missing.push(format!("{name} @ kv_h={} head_dim={}", shape.0, shape.1));
                continue;
            };
            assert_eq!(
                obs.store_after_chunk, pin.3,
                "{name} @ kv_h={} head_dim={}: packed store bytes after the bulk append moved",
                shape.0, shape.1
            );
            assert_eq!(
                obs.store_after_decode, pin.4,
                "{name} @ kv_h={} head_dim={}: packed store bytes after {DECODE_STEPS} decode \
                 steps moved",
                shape.0, shape.1
            );
            assert_eq!(
                obs.store_after_truncate, pin.5,
                "{name} @ kv_h={} head_dim={}: packed store bytes after truncate_to moved",
                shape.0, shape.1
            );
            assert_eq!(
                obs.rows, pin.6,
                "{name} @ kv_h={} head_dim={}: the K/V rows attention receives moved",
                shape.0, shape.1
            );
            assert_eq!(
                obs.resident_bytes, pin.7,
                "{name} @ kv_h={} head_dim={}: resident_bytes moved",
                shape.0, shape.1
            );
        }
    }
    assert!(
        missing.is_empty(),
        "no pin for {} cell(s): {missing:?}\nobserved table:\n{}",
        missing.len(),
        observed.join("\n")
    );
}

/// The pin table names every turbo spelling the enum can spell, at both
/// shapes — and nothing else.
///
/// Without this, a turbo spelling added to `ALL_KV_QUANTS` and left out of
/// `PINS` would be unpinned and the suite would still be green.
#[test]
fn every_turbo_spelling_is_pinned_at_both_shapes() {
    let want: Vec<String> = turbo_spellings().iter().map(ToString::to_string).collect();
    assert_eq!(
        want.len(),
        TURBO_SPELLING_COUNT,
        "turbo spelling census moved — a turbo variant was added to or removed from \
         ALL_KV_QUANTS and this file's pins did not follow: {want:?}"
    );
    for name in &want {
        for shape in [SHAPE_A, SHAPE_B] {
            assert!(
                pin_for(name, shape).is_some(),
                "{name} @ kv_h={} head_dim={} has no pin",
                shape.0,
                shape.1
            );
        }
    }
    assert_eq!(
        PINS.len(),
        want.len() * 2,
        "PINS holds {} rows for {} spellings x 2 shapes — a stale row pins nothing",
        PINS.len(),
        want.len()
    );
}

/// The K-side store the collapse unifies is built by exactly two spellings,
/// and the scope of the whole change rests on that.
///
/// It sweeps **`ALL_KV_QUANTS`, not the `Display`-filtered subset**, and that
/// is the point. The filter and this anchor would otherwise share one blind
/// spot: a seventh spelling that builds `KvStorage::TurboSym*` under a third
/// token would enter neither, so the census would stay green with nothing
/// pinned and the scope anchor would stay green with the spelling outside its
/// scope. Sweeping the enum closes the loop — the anchor asserts that every
/// quant whose storage is a symmetric turbo variant is a member of
/// `turbo_spellings()`, so the filter is checked against the thing it claims
/// to select rather than against itself.
///
/// It reads the storage `KvStorage::new` built at construction and drives
/// nothing: the variant is decided there, so a drive would only add the cost
/// of an append the claim does not rest on.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the two arms named are the claim under test; every other storage variant is the negative case and needs no arm of its own"
)]
#[test]
fn only_the_symmetric_spellings_build_the_k_side_turbo_store() {
    let _guard = env_lock();
    let census: Vec<String> = turbo_spellings().iter().map(ToString::to_string).collect();
    let mut with_k_twin = Vec::new();
    for quant in ALL_KV_QUANTS.iter().copied() {
        let cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        if matches!(
            cache.storage,
            KvStorage::TurboSym3 { .. } | KvStorage::TurboSym4 { .. }
        ) {
            let name = quant.to_string();
            assert!(
                census.contains(&name),
                "{name} is backed by the K-side turbo store and the Display filter does not \
                 select it — the census pins nothing for it and the count below cannot see it"
            );
            with_k_twin.push(name);
        }
    }
    assert_eq!(
        with_k_twin.len(),
        TURBO_K_TWIN_COUNT,
        "the set of spellings backed by the K-side turbo store moved: {with_k_twin:?}. \
         The collapse is scoped to exactly these, so a change here is a scope change"
    );
}

/// The store geometry follows the spelling's own bit width, derived from
/// arithmetic written out here rather than from a pinned constant or from the
/// code under test.
///
/// This is the half of the oracle that survives a re-baseline: a const-generic
/// instantiated at the wrong width writes a code plane of the wrong length,
/// which this catches without knowing what the right bytes are.
#[allow(
    clippy::expect_used,
    reason = "test: a store the driver just populated is present, and a panic names the spelling that failed to populate it"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see store_digest"
)]
#[test]
fn turbo_store_geometry_follows_the_codec_bit_width() {
    let _guard = env_lock();
    for quant in turbo_spellings() {
        let name = quant.to_string();
        for (kv_h, head_dim) in [SHAPE_A, SHAPE_B] {
            let device = Device::Cpu;
            let mut cache =
                KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ).with_layer_idx(TEST_LAYER_IDX);
            let shape = [1_i32, kv_h, CHUNK_SEQ, head_dim];
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let k = f32_arr(&lcg_data(n, TEST_SEED), &shape);
            let v = f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &shape);
            cache.update(&k, &v, device).expect("bulk append");

            let hd = head_dim as usize;
            let tokens = (kv_h as usize) * (CHUNK_SEQ as usize);

            // `(bits, code bytes, scales)` of whichever turbo plane the
            // spelling writes. The symmetric spellings are read on the K axis,
            // which is the store the collapse unifies; the asymmetric ones
            // have no K-side turbo plane and are read on V.
            let (bits, codes, scales): (u8, usize, usize) = match &cache.storage {
                KvStorage::TurboSym3 { k, .. } => {
                    let s = k.as_ref().expect("turbo K store");
                    turbo_geometry!(s)
                }
                KvStorage::TurboSym4 { k, .. } => {
                    let s = k.as_ref().expect("turbo K store");
                    turbo_geometry!(s)
                }
                KvStorage::K8VTurbo3 { v, .. }
                | KvStorage::K8VTurbo3Tcq { v, .. }
                | KvStorage::K8VTurbo2 { v, .. }
                | KvStorage::K8VTurbo2Tcq { v, .. } => {
                    let s = v.as_ref().expect("turbo V store");
                    turbo_geometry!(s)
                }
                other => panic!(
                    "{name}: not a turbo storage variant: {}",
                    super::helpers::storage_variant_name(other)
                ),
            };

            let want_bits = expected_bits(&name);
            assert_eq!(
                bits, want_bits,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: store bit tag"
            );
            assert_eq!(
                codes,
                expected_code_bytes(tokens * hd, want_bits),
                "{name} @ kv_h={kv_h} head_dim={head_dim}: the packed code plane must be \
                 ceil(values * {want_bits} / 8) bytes"
            );
            assert_eq!(
                scales,
                tokens * hd / GROUP_SIZE,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: one scale per group of {GROUP_SIZE}"
            );
        }
    }
}

/// Elements one TurboQuant scale covers. Written out rather than imported: a
/// test that asks the code under test what shape it should be is not an
/// oracle.
const GROUP_SIZE: usize = 32;

/// Bit width a spelling encodes at, read off its own `Display` text.
///
/// The text is the width's one public statement — `tsym3` and `k8vturbo3` are
/// 3-bit, `tsym4` is 4-bit, `k8vturbo2` and its TCQ sibling are 2-bit — so the
/// expectation is derived from the spelling rather than from a table that
/// would have to be edited beside the code it checks.
fn expected_bits(name: &str) -> u8 {
    if name.contains('2') {
        2
    } else if name.contains('3') {
        3
    } else {
        4
    }
}

/// Bytes a bit-packed code plane of `values` codes at `bits` bits occupies.
const fn expected_code_bytes(values: usize, bits: u8) -> usize {
    (values * bits as usize).div_ceil(8)
}

/// Width twins hold different stores.
///
/// The positive control for the pin table: a unification that collapsed two
/// widths onto one instantiation would leave every assertion above satisfied
/// by a re-baseline, and this one red.
///
/// The two encoder pairs — `k8vturbo3` against `k8vturbo3tcq`, and the 2-bit
/// pair — are deliberately **not** here. They write the same bytes today, and
/// the test below is what says so and why.
#[test]
fn turbo_width_twins_hold_different_stores() {
    let _guard = env_lock();
    let pairs: [(KvQuant, KvQuant); 2] = [
        (KvQuant::TurboSym3, KvQuant::TurboSym4),
        (KvQuant::K8VTurbo3, KvQuant::K8VTurbo2),
    ];
    for (left, right) in pairs {
        for shape in [SHAPE_A, SHAPE_B] {
            let a = drive(left, shape);
            let b = drive(right, shape);
            assert_ne!(
                a.store_after_chunk, b.store_after_chunk,
                "{left} and {right} @ kv_h={} head_dim={}: the two wrote the same store \
                 bytes — one codec setting is not being applied",
                shape.0, shape.1
            );
            assert_ne!(
                a.store_after_decode, b.store_after_decode,
                "{left} and {right} @ kv_h={} head_dim={}: the two wrote the same store \
                 bytes after decode",
                shape.0, shape.1
            );
        }
    }
}

/// Driving the same spelling twice gives the same bytes.
///
/// Without this, every assertion above could be pinning a value that is not
/// reproducible, and a red cell would be read as flake rather than defect.
///
/// It drives both arms itself rather than reusing the pin test's observations.
/// Borrowing them would make this test's verdict depend on that test having
/// run, and a test whose outcome depends on which other tests ran is a shape
/// this crate has removed elsewhere. The cost is one extra drive per cell.
#[test]
fn a_turbo_cell_is_reproducible() {
    let _guard = env_lock();
    for quant in turbo_spellings() {
        for shape in [SHAPE_A, SHAPE_B] {
            let a = drive(quant, shape);
            let b = drive(quant, shape);
            assert_eq!(
                a.store_after_truncate, b.store_after_truncate,
                "{quant} @ kv_h={} head_dim={}: post-truncate store bytes differ between two \
                 identical drives",
                shape.0, shape.1
            );
            assert_eq!(
                a.store_after_decode, b.store_after_decode,
                "{quant} @ kv_h={} head_dim={}: store bytes differ between two identical drives",
                shape.0, shape.1
            );
            assert_eq!(
                a.rows, b.rows,
                "{quant} @ kv_h={} head_dim={}: attention rows differ between two identical \
                 drives",
                shape.0, shape.1
            );
            assert_eq!(
                a.resident_bytes, b.resident_bytes,
                "{quant} @ kv_h={} head_dim={}: resident_bytes differs between two identical \
                 drives",
                shape.0, shape.1
            );
        }
    }
}

/// The TCQ encoder writes the same bytes as nearest-centroid assignment.
///
/// Measured, not assumed, and it is the reason the pin table carries two pairs
/// of identical rows. The trellis `build_transition_table` in `crate::tcq`
/// gives every state an outgoing edge for **every** level — the transition
/// picks the next state from `level & 1` and forbids no level — so the
/// additive Viterbi cost is minimised position by position, which is the
/// greedy nearest-centroid assignment the plain encoder makes. The trellis
/// constrains nothing, so the Viterbi pass is an expensive way to reach the
/// same codes.
///
/// The flag is asserted beside the bytes so the two halves cannot be confused:
/// the TCQ spelling really does set `use_tcq` and really does run the Viterbi
/// encoder; what it does not do is produce a different store.
///
/// This is not the turbo K collapse's business and nothing here changes it.
/// The test exists so the identity is a recorded measurement with its reason
/// attached rather than a silent coincidence, and so that a change which makes
/// the trellis constrain the level set turns red and names itself.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see store_digest"
)]
#[allow(
    clippy::expect_used,
    reason = "test: a store the driver just populated is present, and a panic names the spelling that failed to populate it"
)]
#[test]
fn the_tcq_spellings_set_the_flag_and_still_write_the_plain_bytes() {
    let _guard = env_lock();
    let pairs: [(KvQuant, KvQuant); 2] = [
        (KvQuant::K8VTurbo3, KvQuant::K8VTurbo3Tcq),
        (KvQuant::K8VTurbo2, KvQuant::K8VTurbo2Tcq),
    ];
    for (plain, tcq) in pairs {
        for flagged in [false, true] {
            let quant = if flagged { tcq } else { plain };
            let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
            let (kv_h, head_dim) = SHAPE_A;
            let shape = [1_i32, kv_h, CHUNK_SEQ, head_dim];
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let k = f32_arr(&lcg_data(n, TEST_SEED), &shape);
            let v = f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &shape);
            cache.update(&k, &v, Device::Cpu).expect("bulk append");
            let use_tcq = match &cache.storage {
                KvStorage::K8VTurbo3 { v, .. }
                | KvStorage::K8VTurbo3Tcq { v, .. }
                | KvStorage::K8VTurbo2 { v, .. }
                | KvStorage::K8VTurbo2Tcq { v, .. } => v.as_ref().expect("turbo V store").use_tcq,
                other => panic!(
                    "{quant}: not an asymmetric turbo storage variant: {}",
                    super::helpers::storage_variant_name(other)
                ),
            };
            assert_eq!(
                use_tcq, flagged,
                "{quant}: the Viterbi assignment flag on the V store is not what the \
                 spelling selects"
            );
        }
        for shape in [SHAPE_A, SHAPE_B] {
            let a = drive(plain, shape);
            let b = drive(tcq, shape);
            assert_eq!(
                a.store_after_chunk, b.store_after_chunk,
                "{plain} and {tcq} @ kv_h={} head_dim={}: the Viterbi assignment now writes \
                 different bytes from nearest-centroid. That is a codec change, and the pin \
                 rows for the TCQ spelling are no longer copies of their plain sibling's",
                shape.0, shape.1
            );
        }
    }
}

/// No turbo spelling reads its packed store at decode, and none materialises
/// one past `exit_prefill`.
///
/// This is the premise the whole oracle rests on: it is why a served digest
/// cannot judge this change and why the store-bytes pin above is the only
/// thing that can. A spelling that moved into the store-reading class would
/// make a served capture meaningful for that cell and would need its own
/// row in the real-model table, so the move must be named here rather than
/// discovered later.
#[test]
fn every_turbo_spelling_is_decode_inert() {
    for quant in turbo_spellings() {
        assert!(
            !quant.decode_reads_packed_store(),
            "{quant} now reads its packed store at decode — the store-bytes pin is no longer \
             the only oracle for it, and the real-model table owes it a row"
        );
        assert!(
            !quant.materialises_packed_store(),
            "{quant} now keeps its packed store past exit_prefill — a served capture can see \
             its bytes, which the oracle's premise says it cannot"
        );
    }
}
