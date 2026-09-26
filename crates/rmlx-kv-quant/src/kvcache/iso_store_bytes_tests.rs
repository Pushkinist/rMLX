//! What the iso family claims about its own stores, beyond what the shared
//! store-bytes oracle pins.
//!
//! The pin table, the drive and the census live in `store_bytes_tests.rs` and
//! cover every spelling `ALL_KV_QUANTS` holds. This file holds the three claims
//! that are about the iso codec and about nothing else: that the store geometry
//! follows the spelling's own bit width, that the two widths write different
//! stores — the positive control without which a collapse of the widths would
//! satisfy every pin by re-baselining it — and that the GPU-resident mirror is
//! off in production.
//!
//! # The one named exception the collapse carried
//!
//! `QuantIsoV3` and `QuantIsoK3` carry a GPU dequant entry (`dequant_gpu`, and
//! for the V store `dequant_on`) that the 4-bit stores did not, so the 4-bit
//! decode host-decoded the whole prefix where its 3-bit sibling dispatched a
//! kernel. Unifying the types gave the 4-bit width that entry. Nothing on this
//! path can see it: the exception lives on `Device::Gpu` and every cell drives
//! `Device::Cpu`, so the pins hold at both widths and a moved 4-bit pin is a
//! defect, not the intended change.
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
//! **No iso shape has a ragged group.** The codec rejects any `head_dim` that
//! is not a multiple of `ISO_QUAT_BLOCK_SIZE` (4) with
//! `IsoQuantError::HeadDimNotMultipleOf4`, so the padded-last-group case the
//! rotor family's shape A exists for cannot occur here.
//!
//! # What this file cannot see
//!
//! Named so a reader knows where the iso coverage stops. No assertion here or
//! in the shared oracle can turn red on a defect in any of it.
//!
//! * **The `exit_prefill` bulk-encode arms of `iso3` and `iso4`**
//!   (`kvcache/update.rs`, the `Iso*` arms of the `exit_prefill` match). The
//!   gate returns before those two, so no CPU route runs them. The other four
//!   — `iso3_sym`, `iso4_sym`, `k_iso3`, `k_iso4` — do run theirs, and the
//!   shared oracle's second drive is what pins the bytes those arms write.
//! * **The fused flash-decode arms.** `update_and_sdpa`'s iso K-only and iso
//!   symmetric arms are gated on `device == Device::Gpu`, `q_seq == 1` and
//!   `iso_flash_shape_ok` — which requires `b == 1`, `head_dim % 4 == 0`,
//!   `head_dim <= 512` and `head_dim` a **power of two**. Where all four hold
//!   they are the production decode route for `k_iso3`, `k_iso4`, `iso3_sym`
//!   and `iso4_sym` at both widths. A CPU drive never reaches them, and
//!   **shape B is a shape they reject**: `head_dim = 96` is not a power of two,
//!   so a served request at that head dimension takes the same host decode this
//!   file drives. The GPU test that covers the arms is
//!   `kvcache::iso_flash_dispatch_tests`.
//! * **`gpu_append`, `gpu_packed_view` and `reconcile_ring`** on all four
//!   stores, and the ring-readback branch of `synced_iso_v_blocks`.
//!   Unreachable from a `Device::Cpu` drive, and the largest unseen surface in
//!   each storage pair.
//! * **`QuantIsoV3::append_gpu` and its GPU-resident mirror.** The mirror write
//!   is behind `crate::gpu_resident_iso_enabled`, whose production value is
//!   `crate::GPU_RESIDENT_ISO_PRODUCTION` — `false`, so it writes no byte in
//!   production at either width. That is the one thing about the mirror this
//!   file does assert; see the last test.
//! * **`dequant_gpu` / `dequant_on`**, which is the one live behaviour
//!   difference between the widths. See "What cannot move".
//! * **`from_cpu_blocks` and `try_deep_clone`** — the SSD-hydrate and
//!   branch-clone constructors. Never called on this path.
//! * **`max_seq`.** Deliberately not a digest field. `QuantIsoK3`, `QuantIsoK4`
//!   and `QuantIsoV4` carry one and their own docs call it inert;
//!   `QuantIsoV3` deliberately carries none, because a cached provisioned
//!   window goes stale the moment the window grows. Pinning it would force a
//!   re-baseline over a field no decode path reads, in whichever direction the
//!   collapse resolves the asymmetry.
//! * **Bit-exactness under a different toolchain.** The pins are f32 results
//!   from this host's codegen. They judge one change on one toolchain; they are
//!   not a portable golden.
//!
//! `truncate_to` **is** covered — the drive rolls back into the bulk chunk
//! after the decode steps, so both halves of the truncate plan run and the
//! post-truncate store bytes are a pinned column.

use super::core::KvCache;
use super::store_bytes_tests::{
    assert_width_twins_differ, shapes_for, CHUNK_SEQ, TEST_LAYER_IDX, TEST_MAX_SEQ,
};
use crate::storage::KvStorage;
use crate::test_utils::{env_lock, f32_arr, lcg_data, TEST_SEED};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

/// `(bits, code words, scales, quaternions, norms)` summed over a store's
/// blocks.
///
/// A macro for the same reason as [`push_iso`].
macro_rules! iso_geometry {
    ($store:expr) => {{
        let s = $store;
        (
            s.bits,
            s.blocks.iter().map(|b| b.codes.len()).sum::<usize>(),
            s.blocks.iter().map(|b| b.scales.len()).sum::<usize>(),
            s.blocks.iter().map(|b| b.quaternions.len()).sum::<usize>(),
            s.blocks.iter().map(|b| b.norms.len()).sum::<usize>(),
        )
    }};
}

/// Every iso spelling the enum can spell, in `ALL_KV_QUANTS` order.
///
/// The filter is the enum's own `Display`, not a hand-written variant list. A
/// list would name only the variants that existed when it was written, so a
/// seventh iso variant would never enter `want` and the census below would stay
/// green with nothing pinned — a gate that cannot fail on the one event it
/// exists for. Every `Display` arm whose text contains `iso` is an iso spelling
/// and no other arm's text does, so the text is the membership test.
/// `ISO_SPELLING_COUNT` is the anchor beside it: a variant added to the enum
/// and not to `ALL_KV_QUANTS` moves the count rather than passing quietly.
fn iso_spellings() -> Vec<KvQuant> {
    ALL_KV_QUANTS
        .iter()
        .copied()
        .filter(|q| q.to_string().contains("iso"))
        .collect()
}

/// Quaternion groups one row of `head_dim` values occupies.
///
/// The iso codec works on quaternions, so a row is cut into groups of four
/// components and `head_dim` must divide by four exactly — there is no padded
/// last group. Written out rather than imported: a test that asks the code
/// under test what shape it should be is not an oracle.
const fn expected_n_groups(head_dim: usize) -> usize {
    head_dim / 4
}

/// `u32` words one row of `head_dim` values occupies in the dense code plane.
///
/// One code per value, `bits` bits per code, rows padded to a whole word.
const fn expected_code_words(head_dim: usize, bits: u8) -> usize {
    (head_dim * bits as usize).div_ceil(32)
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
/// read back from `crate::storage::iso_row_words` / `iso_n_groups_for`. Calling
/// those would make the storage layer agree with the helper that sizes it —
/// true for free, and silent if the collapse rewrote both.
#[allow(
    clippy::expect_used,
    reason = "test: a store the driver just populated is present, and a panic names the spelling that failed to populate it"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see store_digest"
)]
#[test]
fn iso_store_geometry_follows_the_codec_bit_width() {
    let _guard = env_lock();
    for quant in iso_spellings() {
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

            // `(bits, codes, scales, quaternions, norms)` of whichever iso
            // plane the spelling writes.
            let (bits, codes, scales, quaternions, norms): (u8, usize, usize, usize, usize) =
                match &cache.storage {
                    KvStorage::IsoV3 { v, .. } => {
                        let s = v.as_ref().expect("iso V store");
                        iso_geometry!(s)
                    }
                    KvStorage::IsoV4 { v, .. } => {
                        let s = v.as_ref().expect("iso V store");
                        iso_geometry!(s)
                    }
                    KvStorage::IsoSym3 { k, .. } => {
                        let s = k.as_ref().expect("iso K store");
                        iso_geometry!(s)
                    }
                    KvStorage::IsoSym4 { k, .. } => {
                        let s = k.as_ref().expect("iso K store");
                        iso_geometry!(s)
                    }
                    KvStorage::IsoKOnly3 { k, .. } => {
                        let s = k.as_ref().expect("iso K store");
                        iso_geometry!(s)
                    }
                    KvStorage::IsoKOnly4 { k, .. } => {
                        let s = k.as_ref().expect("iso K store");
                        iso_geometry!(s)
                    }
                    other => panic!("{name}: not an iso storage variant: {}", other.view().name),
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
                 tokens * ceil(head_dim * {want_bits} / 32)"
            );
            assert_eq!(
                scales,
                tokens * n_groups,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: one scale per quaternion group"
            );
            assert_eq!(
                quaternions,
                tokens * n_groups * 4,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: one 4-component quaternion per group"
            );
            assert_eq!(
                norms, tokens,
                "{name} @ kv_h={kv_h} head_dim={head_dim}: one norm per token"
            );
        }
    }
}

/// The iso width pairs, handed to the shared control.
///
/// The pair list is the only part of the claim that is about the iso codec;
/// the drive, the columns and the assertions are the oracle's, in
/// `store_bytes_tests.rs`.
#[test]
fn three_and_four_bit_twins_hold_different_stores() {
    assert_width_twins_differ(
        &[
            (KvQuant::Iso3, KvQuant::Iso4),
            (KvQuant::Iso3Sym, KvQuant::Iso4Sym),
            (KvQuant::IsoKOnly3, KvQuant::IsoKOnly4),
        ],
        "one width",
    );
}

/// The GPU-resident iso V mirror writes no byte in production.
///
/// `QuantIsoV3::append_gpu` guards its mirror write on
/// `crate::gpu_resident_iso_enabled`. The collapse hands the mirror to the
/// 4-bit width as a consequence of making the store one type, and this is what
/// says that hand-off changes no production byte.
///
/// It reads the constant, not the fn. Under `cfg(test)` the fn is a different
/// body — an override flag a sibling test can turn on — so asserting on the fn
/// would assert on that flag and pass whichever way production was set. Reading
/// the constant also removes the only shared mutable state this test touched,
/// so its outcome no longer depends on which other tests ran.
#[allow(
    clippy::assertions_on_constants,
    reason = "the constant is the point: this asserts a documented production value, so that changing it is a named failure rather than a silent one. The lint's usual target — an assertion that cannot fail — is what this would be if the value were inlined here instead of read from its one definition."
)]
#[test]
fn the_production_gpu_resident_iso_mirror_is_off() {
    assert!(
        !crate::GPU_RESIDENT_ISO_PRODUCTION,
        "the GPU-resident iso V mirror is on in production — the pins above were taken \
         with it off, and the collapse hands it to the 4-bit width"
    );
}
