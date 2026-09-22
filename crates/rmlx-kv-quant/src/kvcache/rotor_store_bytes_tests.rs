//! What the rotor family claims about its own stores, beyond what the shared
//! store-bytes oracle pins.
//!
//! The pin table, the drive and the census live in `store_bytes_tests.rs` and
//! cover every spelling `ALL_KV_QUANTS` holds, the two extra asymmetric
//! parameterisations included. This file holds the two claims that are about
//! the rotor codec and about nothing else: that the store geometry follows the
//! spelling's own bit width, derived from the codec's published layout rather
//! than from the code under test, and that the two widths write different
//! stores — the positive control without which a collapse of the widths would
//! satisfy every pin by re-baselining it.
//!
//! # Shapes
//!
//! The oracle's two, both legal for all eight spellings:
//!
//! | shape | `kv_h` | `head_dim` | why |
//! |---|---|---|---|
//! | A | 1 | 128 | single KV head (shared-KV arch); power-of-two `head_dim`, so the last multivector group is ragged (`128 = 42*3 + 2`) |
//! | B | 4 | 96 | `kv_h > 1`; non-power-of-two `head_dim` divisible by the group size, so no group is padded |
//!
//! # What this file cannot see
//!
//! * **The `exit_prefill` bulk-encode arms of the four mirror-family
//!   spellings.** The gate returns before them, so no CPU route runs them.
//!   The other four — `rotor3_sym`, `rotor4_sym`, `k_rotor3`, `k_rotor4` —
//!   do run theirs, and the shared oracle's second drive is what pins the
//!   bytes those arms write.
//! * **`gpu_append` and `gpu_packed_view`**, and the ring-readback branch of
//!   `synced_rotor_v_blocks` / `synced_rotor_k_blocks`. Unreachable from a
//!   `Device::Cpu` drive, and the largest unseen surface in the storage pair.
//!   Only the `#[ignore]` GPU suite reaches them.
//! * **`from_cpu_blocks` and `try_deep_clone`** — the SSD-hydrate and
//!   branch-clone constructors. Never called on this path.
//! * **The six unpinned asymmetric V configurations.**
//!   `validate_rotor_k_asym_v` accepts `(4, 128|64|32)` and `(3|2, 64)`: ten
//!   cells across the two asym K widths, of which four are driven (`v4_g64`,
//!   from `ALL_KV_QUANTS`, and `v2_g64`). A defect that appears only at
//!   `v4_g128`, `v4_g32` or `v3_g64` is not seen.

use super::core::KvCache;
use super::store_bytes_tests::{
    assert_width_twins_differ, pinned_spellings, shapes_for, CHUNK_SEQ, TEST_LAYER_IDX,
    TEST_MAX_SEQ,
};
use crate::rotor_qjl::rotor_qjl_enabled;
use crate::storage::KvStorage;
use crate::test_utils::{env_lock, f32_arr, lcg_data, TEST_SEED};
use crate::KvQuant;
use rmlx_mlx::Device;

/// `(bits, code words, scales, norms)` summed over a rotor store's blocks.
///
/// A macro, not a fn: the 3-bit and 4-bit stores are distinct types today, so
/// one fn cannot take both. When they become one const-generic type this can
/// become an ordinary fn — which is the point of the exercise.
macro_rules! rotor_geometry {
    ($store:expr) => {{
        let s = $store;
        (
            s.bits,
            s.blocks.iter().map(|b| b.codes.len()).sum::<usize>(),
            s.blocks.iter().map(|b| b.scales.len()).sum::<usize>(),
            s.blocks.iter().map(|b| b.norms.len()).sum::<usize>(),
        )
    }};
}

/// Every rotor cell the store-bytes oracle pins, in its order.
///
/// The filter is the enum's own `Display`, not a hand-written variant list: a
/// list would name only the variants that existed when it was written, so a
/// ninth rotor variant would never enter it and the tests below would stay
/// green with nothing checked. Every `Display` arm whose text contains `rotor`
/// is a rotor spelling and no other arm's text does (`RotK` renders
/// `rot_k_v8g64`), so the text is the membership test.
///
/// It filters the oracle's own pinned population rather than `ALL_KV_QUANTS`,
/// so the two extra asymmetric parameterisations that population carries are
/// checked here too, and neither file holds a second copy of that list.
fn rotor_spellings() -> Vec<KvQuant> {
    pinned_spellings()
        .into_iter()
        .filter(|q| q.to_string().contains("rotor"))
        .collect()
}

/// Assert the QJL residual is off, so the K-side pins mean what they say.
///
/// The toggle is process-global and read at every store construction, so a
/// pin taken with it on would be a different store. It is never installed in a
/// test process; the assert is here so a change that installs it fails loudly
/// instead of re-baselining the pins by accident.
fn assert_qjl_off() {
    assert!(
        !rotor_qjl_enabled(),
        "rotor K-side QJL residual is enabled in this process — the K-store pins \
         below were taken with it off and do not describe this store"
    );
}

/// Multivector groups one row of `head_dim` values occupies.
///
/// The rotor codec works on `Cl(3,0)` multivectors, so a row is cut into groups
/// of three components and the last group is zero-padded when `head_dim` is not
/// a multiple of three. Written out rather than imported: a test that asks the
/// code under test what shape it should be is not an oracle.
const fn expected_n_groups(head_dim: usize) -> usize {
    head_dim.div_ceil(3)
}

/// `u32` words one row of `head_dim` values occupies in the dense code plane.
///
/// Three codes per group, `bits` bits per code, rows padded to a whole word.
const fn expected_code_words(head_dim: usize, bits: u8) -> usize {
    let codes_per_row = expected_n_groups(head_dim) * 3;
    (codes_per_row * bits as usize).div_ceil(32)
}

/// The store geometry follows the spelling's own bit width, derived from
/// arithmetic written out here rather than from a pinned constant or from the
/// code under test.
///
/// This is the half of the oracle that survives a re-baseline: a const-generic
/// instantiated at the wrong width writes a code plane of the wrong length,
/// which this catches without knowing what the right bytes are.
///
/// The expected lengths are restated from the codec's published layout, not
/// read back from `crate::rotorquant`. Calling `n_groups_for` /
/// `row_words_for` would make the storage layer agree with the helper that
/// sizes it — true for free, and silent if the collapse rewrote both. See
/// [`expected_code_words`].
#[allow(
    clippy::expect_used,
    reason = "test: a store the driver just populated is present, and a panic names the spelling that failed to populate it"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see store_digest"
)]
#[test]
fn rotor_store_geometry_follows_the_codec_bit_width() {
    let _guard = env_lock();
    assert_qjl_off();
    for quant in rotor_spellings() {
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
            let n_groups = expected_n_groups(hd);

            // `(bits, codes, scales, norms)` of whichever rotor plane the
            // spelling writes.
            let (bits, codes, scales, norms): (u8, usize, usize, usize) = match &cache.storage {
                KvStorage::RotorV3 { v, .. } => {
                    let s = v.as_ref().expect("rotor V store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorV4 { v, .. } => {
                    let s = v.as_ref().expect("rotor V store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorSym3 { k, .. } => {
                    let s = k.as_ref().expect("rotor K store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorSym4 { k, .. } => {
                    let s = k.as_ref().expect("rotor K store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorKOnly3 { k, .. } => {
                    let s = k.as_ref().expect("rotor K store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorKOnly4 { k, .. } => {
                    let s = k.as_ref().expect("rotor K store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorKAsym3 { k, .. } => {
                    let s = k.as_ref().expect("rotor K store");
                    rotor_geometry!(s)
                }
                KvStorage::RotorKAsym4 { k, .. } => {
                    let s = k.as_ref().expect("rotor K store");
                    rotor_geometry!(s)
                }
                other => panic!(
                    "{name}: not a rotor storage variant: {}",
                    super::helpers::storage_variant_name(other)
                ),
            };

            let want_bits = if name.contains('3') { 3_u8 } else { 4_u8 };
            assert_eq!(
                bits, want_bits,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: store bit tag"
            );
            assert_eq!(
                codes,
                tokens * expected_code_words(hd, want_bits),
                "{name} @ kv_h={kv_h} head_dim={head_dim}: code-plane words must be \
                 tokens * ceil(3 * ceil(head_dim / 3) * {want_bits} / 32)"
            );
            assert_eq!(
                scales,
                tokens * n_groups,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: one scale per multivector group"
            );
            assert_eq!(
                norms, tokens,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: one norm per token"
            );
        }
    }
}

/// The rotor width pairs, handed to the shared control.
///
/// The pair list is the only part of the claim that is about the rotor codec;
/// the drive, the columns and the assertions are the oracle's, in
/// `store_bytes_tests.rs`.
#[test]
fn three_and_four_bit_twins_hold_different_stores() {
    assert_qjl_off();
    assert_width_twins_differ(
        &[
            (KvQuant::Rotor3, KvQuant::Rotor4),
            (KvQuant::Rotor3Sym, KvQuant::Rotor4Sym),
            (KvQuant::RotorKOnly3, KvQuant::RotorKOnly4),
            (
                KvQuant::RotorK3Asym {
                    v_bits: 4,
                    v_group_size: 64,
                },
                KvQuant::RotorK4Asym {
                    v_bits: 4,
                    v_group_size: 64,
                },
            ),
        ],
        "one K-side width",
    );
}
