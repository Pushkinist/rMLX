//! What the turbo family claims about its own stores, beyond what the shared
//! store-bytes oracle pins.
//!
//! The pin table, the drive and the census live in `store_bytes_tests.rs` and
//! cover every spelling `ALL_KV_QUANTS` holds. This file holds the five claims
//! that are about the TurboQuant codec and about nothing else: that exactly two
//! spellings build the K-side store, that the store geometry follows the
//! spelling's own bit width, that the widths write different stores, that the
//! TCQ spellings set the Viterbi flag and still write the plain bytes, and that
//! every turbo spelling is decode-inert.
//!
//! # Why a served digest cannot judge this family
//!
//! Stronger here than on any other. All six turbo spellings report
//! `decode_reads_packed_store() == false`, `feeds_bf16_k_at_decode(false)` and
//! `feeds_bf16_v_at_decode(false)`, so `materialises_packed_store()` is false
//! for every one of them. `exit_prefill` returns at its
//! `materialises_packed_store()` gate **before** every arm that would bulk
//! encode a store, and clears whatever payload the cache arrived carrying. A
//! served prefill therefore writes no turbo store at all, and the decode
//! entries short-circuit to `update_decode_fp16` while the bf16 seed is live.
//! **No served capture can read a single turbo store byte**, at any width, on
//! any model.
//!
//! Two pairs of the oracle's pin rows are identical and are meant to be: a TCQ
//! spelling writes the same bytes as its plain sibling, for a structural reason
//! the TCQ test below states and measures. That is a property of the encoder,
//! not of any collapse.
//!
//! # Shapes
//!
//! The oracle's two:
//!
//! | shape | `kv_h` | `head_dim` | why |
//! |---|---|---|---|
//! | A | 1 | 128 | single KV head (shared-KV arch); power-of-two `head_dim` |
//! | B | 4 | 96 | `kv_h > 1`; non-power-of-two `head_dim` |
//!
//! # What this file cannot see
//!
//! Named so a reader knows where the turbo coverage stops. No assertion here
//! or in the shared oracle can turn red on a defect in any of it.
//!
//! * **The V-axis device split.** `tsym_update` resolves the V device from
//!   `BITS` — `Device::Cpu` at 3, the caller's device at 4. On a CPU drive the
//!   two are the same routing, so **nothing on this path can tell a body that
//!   keeps the rule from one that lost it.** The rule is
//!   load-bearing: `QuantV::append` enters its GPU branch on `device ==
//!   Device::Gpu` with no bit-width guard and then returns `Error::Quant` for
//!   `bits != 4`, so a 3-bit V handed `Device::Gpu` fails the append. The gate
//!   over it is `make gpu-test` and the served capture.
//! * **`exit_prefill`.** The drive appends with `in_prefill` false, the only
//!   CPU route on which all six spellings write a store — but it is not the
//!   route production takes, and on this family `exit_prefill` is also
//!   what *clears* every one of these stores. Every turbo spelling is
//!   decode-inert, so its bulk-encode arm is behind the gate and no drive
//!   anywhere runs it.
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
//! * **`max_seq`.** Deliberately not a digest field. Both K stores carry one
//!   and neither ever reads it — `append` sizes its buffer from its own
//!   `max_seq` parameter. Where it is *not* inert is the hydrate constructor,
//!   and that is where it is pinned.

use super::core::KvCache;
use super::store_bytes_tests::{
    assert_width_twins_differ, drive, shapes_for, CHUNK_SEQ, SHAPE_A, TEST_LAYER_IDX, TEST_MAX_SEQ,
};
use crate::storage::KvStorage;
use crate::test_utils::{env_lock, f32_arr, lcg_data, TEST_SEED};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

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

/// Turbo spellings whose K side is the store the collapse unifies.
const TURBO_K_TWIN_COUNT: usize = 2;

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
        for (kv_h, head_dim) in shapes_for(quant) {
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

/// The turbo width pairs, handed to the shared control.
///
/// The pair list is the only part of the claim that is about the turbo codec;
/// the drive, the columns and the assertions are the oracle's, in
/// `store_bytes_tests.rs`.
///
/// The two encoder pairs — `k8vturbo3` against `k8vturbo3tcq`, and the 2-bit
/// pair — are deliberately **not** here. They write the same bytes today, and
/// the test below is what says so and why.
#[test]
fn turbo_width_twins_hold_different_stores() {
    assert_width_twins_differ(
        &[
            (KvQuant::TurboSym3, KvQuant::TurboSym4),
            (KvQuant::K8VTurbo3, KvQuant::K8VTurbo2),
        ],
        "one codec setting",
    );
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
        for shape in shapes_for(plain) {
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
