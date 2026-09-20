//! Byte-level lock on the iso storage layer, at the deepest seam a caller can
//! reach on CPU.
//!
//! Sibling of `rotor_store_bytes_tests.rs`. The method is the same and is not
//! restated here; `docs/KV_ROTOR_TWINS.md` holds it. This file states what the
//! iso family pins and where it stops.
//!
//! # What cannot move
//!
//! Every iso spelling that exists before the storage/update unification exists
//! after it, and for each spelling, at each shape:
//!
//! * the **packed store bytes** are identical — every `codes` word, every
//!   `scales`, `quaternions` and `norms` float, the block boundaries, the bit
//!   tag, the accumulated shape and the affine q8_0 companion plane;
//! * the **K and V rows the attention receives** from that store are identical,
//!   bit for bit, at the chunk append and at every decode step;
//! * `resident_bytes()` is identical;
//! * and the served temp-0 token stream is identical.
//!
//! There is **one named exception**, and it is the whole reason the iso
//! collapse is not a rename. `QuantIsoV3` and `QuantIsoK3` carry a GPU dequant
//! entry (`dequant_gpu`, and for the V store `dequant_on`) that `QuantIsoV4`
//! and `QuantIsoK4` do not, so `update_iso4_sym` and `update_iso_k_only_4`
//! host-decode the whole prefix where their 3-bit siblings dispatch a kernel.
//! Unifying the types gives the 4-bit width that entry. Nothing in this file
//! can see it: the exception lives on `Device::Gpu` and every cell here drives
//! `Device::Cpu`, so these pins hold at both widths across the collapse and a
//! moved 4-bit pin is a defect, not the intended change. The GPU half is owed
//! as its own test — see "What this pin cannot see".
//!
//! # Why the served digest is not the oracle
//!
//! Two of the six iso spellings — `Iso3` and `Iso4` — report
//! `decode_reads_packed_store() == false` and feed both axes from the bf16
//! mirror, so `materialises_packed_store()` is false and `exit_prefill` clears
//! their payload outright. A served generation on those two emits the same
//! token ids as `--kv-quant none`, whatever the storage layer does. Their bytes
//! are observable here and nowhere else, which is why this file drives `update`
//! with `in_prefill` false and no bf16 seed: that is the one CPU route on which
//! all six codecs write their store.
//!
//! The four that do read the packed store at decode — `Iso3Sym`, `Iso4Sym`,
//! `IsoKOnly3`, `IsoKOnly4` — are pinned here as well, so the two oracles
//! overlap rather than partition.
//!
//! # Shapes
//!
//! Two, the same pair the rotor pin uses, so a reviewer reading the two files
//! side by side compares like with like:
//!
//! | shape | `kv_h` | `head_dim` | why |
//! |---|---|---|---|
//! | A | 1 | 128 | single KV head (shared-KV arch); power-of-two `head_dim` |
//! | B | 4 | 96 | `kv_h > 1`; non-power-of-two `head_dim` |
//!
//! **No iso shape has a ragged group.** The codec rejects any `head_dim` that
//! is not a multiple of `ISO_QUAT_BLOCK_SIZE` (4) with
//! `IsoQuantError::HeadDimNotMultipleOf4`, so the padded-last-group case the
//! rotor pin's shape A exists for cannot occur on this family and there is no
//! third shape to add for it.
//!
//! The other axis of each spelling sets the floor: the q8_0 K side of
//! `Iso{3,4}` needs `B * kv_h * seq * head_dim % 128 == 0` on **every** chunk
//! including a one-token step. Shape A gives 3072 and 128; shape B gives 9216
//! and 384. Both are multiples of 128.
//!
//! # What this pin cannot see
//!
//! Named so the collapse's reviewer knows where this file stops. Every item is
//! inside, or reached from, the four files the collapse unifies, and no
//! assertion here can turn red on a defect in it.
//!
//! * **The `exit_prefill` bulk-encode arms** (`kvcache/update.rs`, the `Iso*`
//!   arms of the `exit_prefill` match). This file drives `update` with
//!   `in_prefill` false, the only CPU route on which all six spellings write a
//!   store — but it is not the route production takes. For the four spellings
//!   that read their store at decode, the store a served request reads is the
//!   one those arms bulk-encode.
//! * **The fused flash-decode arms.** `update_and_sdpa`'s iso K-only and iso
//!   symmetric arms are gated on `device == Device::Gpu` and `q_seq == 1`, and
//!   they are the production decode route for `k_iso3`, `k_iso4`, `iso3_sym`
//!   and `iso4_sym` at both widths. A CPU drive never reaches them. The GPU
//!   test owed for them is
//!   `kvcache::iso_flash_dispatch_tests`, which already covers the K-only arm.
//! * **`gpu_append`, `gpu_packed_view` and `reconcile_ring`** on all four
//!   stores, and the ring-readback branch of `synced_iso_v_blocks`.
//!   Unreachable from a `Device::Cpu` drive, and the largest unseen surface in
//!   each storage pair.
//! * **`QuantIsoV3::append_gpu` and its GPU-resident mirror.** The mirror write
//!   is behind `crate::gpu_resident_iso_enabled`, which is a `false` constant
//!   outside `cfg(test)`, so it writes no byte in production at either width.
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
use crate::storage::KvStorage;
use crate::test_utils::{
    array_bytes, env_lock, f32_arr, fnv1a64, lcg_data, push_quant_k, StoreBytes, TEST_SEED,
};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

/// Layer index every cell is built at. No iso store seeds anything from it
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

/// Serialise an iso store — the V and K stores carry the same payload type
/// (`IsoBlocks`) and the same header fields, so one body covers both axes.
///
/// A macro, not a fn: the four iso stores are distinct types today, so one fn
/// cannot take them all. When they become two const-generic types this can
/// become an ordinary fn — which is the point of the exercise.
macro_rules! push_iso {
    ($out:expr, $side:expr, $store:expr) => {{
        let s = $store;
        $out.tag($side);
        $out.i32s("shape", &s.shape);
        $out.u8_("bits", s.bits);
        $out.u8_("gpu_ring_live", u8::from(s.gpu.is_allocated()));
        $out.usize_("n_blocks", s.blocks.len());
        for (i, b) in s.blocks.iter().enumerate() {
            $out.usize_("block", i);
            $out.u32s("codes", &b.codes);
            $out.f32s("scales", &b.scales);
            $out.f32s("quaternions", &b.quaternions);
            $out.f32s("norms", &b.norms);
            $out.usize_("n_tokens", b.n_tokens);
        }
    }};
}

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

/// Every byte of the storage an iso spelling holds, in one digest.
///
/// A store the spelling did not populate serialises as its own `absent` tag, so
/// "empty" and "one empty block" are different digests.
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
    macro_rules! opt_iso {
        ($slot:expr, $name:expr) => {
            match $slot {
                Some(s) => push_iso!(out, $name, s),
                None => out.tag(concat!($name, "_absent")),
            }
        };
    }
    match storage {
        KvStorage::IsoV3 { k, v, .. } => {
            opt_q8!(k.as_ref());
            opt_iso!(v.as_ref(), "v_iso");
        }
        KvStorage::IsoV4 { k, v, .. } => {
            opt_q8!(k.as_ref());
            opt_iso!(v.as_ref(), "v_iso");
        }
        KvStorage::IsoSym3 { k, v, .. } => {
            opt_iso!(k.as_ref(), "k_iso");
            opt_iso!(v.as_ref(), "v_iso");
        }
        KvStorage::IsoSym4 { k, v, .. } => {
            opt_iso!(k.as_ref(), "k_iso");
            opt_iso!(v.as_ref(), "v_iso");
        }
        KvStorage::IsoKOnly3 { k, .. } => opt_iso!(k.as_ref(), "k_iso"),
        KvStorage::IsoKOnly4 { k, .. } => opt_iso!(k.as_ref(), "k_iso"),
        other => panic!(
            "not an iso storage variant: {}",
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
    /// covers the plan, not the iso stores' calls to it.
    store_after_truncate: u64,
    /// Digest of the K/V rows the attention received, chunk and every step.
    rows: u64,
    /// `KvCache::resident_bytes` after the last decode step.
    resident_bytes: u64,
}

/// Drive one spelling at one shape with `in_prefill` false throughout.
///
/// The prefill bracket is deliberately absent. `exit_prefill` clears the
/// payload of every spelling whose `materialises_packed_store()` is false — two
/// of the six — so a prefill-bracketed drive would pin two empty stores and
/// read as a clean scan. Appending straight into the decode dispatch is the one
/// CPU route on which all six write their store.
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

/// Iso spellings `ALL_KV_QUANTS` holds today.
const ISO_SPELLING_COUNT: usize = 6;

/// One pinned cell: `(spelling, kv_h, head_dim, store_after_chunk,
/// store_after_decode, store_after_truncate, rows, resident_bytes)`.
type Pin = (&'static str, i32, i32, u64, u64, u64, u64, u64);

/// The "before" side of the oracle, captured on the tree the unification starts
/// from. Any generic type that produces a different byte anywhere in an iso
/// store, hands attention a different row, or changes a cell's residency turns
/// one of these red and names the cell.
///
/// The 4-bit rows are the reference the GPU dequant entry the collapse gives
/// the 4-bit width is compared against. They are not allowed to move: the new
/// entry is on `Device::Gpu` and these cells are `Device::Cpu`.
const PINS: &[Pin] = &[
    (
        "iso3",
        1,
        128,
        0x85abcb321652890e,
        0x301872898a190a2a,
        0xcc7c9bdf7c90d718,
        0x28277aafc5279657,
        22248,
    ),
    (
        "iso3",
        4,
        96,
        0x6da9d32e7a888e7b,
        0xc80798538692f800,
        0x3c57ed5bd63ff66c,
        0x162f4d9efb8b3dd4,
        66852,
    ),
    (
        "iso4",
        1,
        128,
        0x19b2462e7efccf6d,
        0x35831bf5e4b86e28,
        0x2e8a7f6bae2acc96,
        0x295c1a74dc7e1dde,
        22680,
    ),
    (
        "iso4",
        4,
        96,
        0x598b87e13fdea073,
        0x55a28838376b1eb2,
        0x1d03bd3f9e9ee018,
        0xcb2bfcdf25575f26,
        68148,
    ),
    (
        "iso3_sym",
        1,
        128,
        0x7535dbab9dc2124b,
        0x7f73eb48fc0abda8,
        0x51b53354d8100a32,
        0xcf696b2d642dbdda,
        37368,
    ),
    (
        "iso3_sym",
        4,
        96,
        0xcef21e1322c66a08,
        0x4ca434f12e2a40f3,
        0xcfffe388b94e94b1,
        0x82cf2d0364d47008,
        112320,
    ),
    (
        "iso4_sym",
        1,
        128,
        0xaa1ec45e837e3ece,
        0x794fa950342fbf79,
        0x39e8332b3e5e97ed,
        0x690aee61b10ef9ca,
        38232,
    ),
    (
        "iso4_sym",
        4,
        96,
        0x15b29d7aca0a3207,
        0x89895047c80340a1,
        0x6adf6c5223b9869a,
        0x55c44d5d72693501,
        114912,
    ),
    (
        "k_iso3",
        1,
        128,
        0xd34b3553f13568ac,
        0x3144fea59419dfbf,
        0x9a814066378802ef,
        0x4fc83ecb89cf7e1a,
        32508,
    ),
    (
        "k_iso3",
        4,
        96,
        0xad6d14f8012b417c,
        0xc5752988f421810b,
        0xb6f11723874ff849,
        0x5cbb01f9a7b72ee9,
        97632,
    ),
    (
        "k_iso4",
        1,
        128,
        0xded71e7e59d4770e,
        0x973a4218b23db58a,
        0x8c1efe2834e52234,
        0x7d12358980993deb,
        32940,
    ),
    (
        "k_iso4",
        4,
        96,
        0x7b82776a85755241,
        0xcaf9b3684437b9b3,
        0xcfa31092b4f07d48,
        0xc4667109651894e6,
        98928,
    ),
];

/// Look a pin up by spelling and shape.
fn pin_for(name: &str, shape: (i32, i32)) -> Option<&'static Pin> {
    PINS.iter()
        .find(|p| p.0 == name && p.1 == shape.0 && p.2 == shape.1)
}

/// Every iso spelling, at both shapes, holds the bytes it held before.
#[test]
fn iso_store_bytes_are_pinned_per_spelling_and_shape() {
    let _guard = env_lock();
    let mut missing = Vec::new();
    let mut observed = Vec::new();
    for quant in iso_spellings() {
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

/// The pin table names every iso spelling the enum can spell, at both shapes —
/// and nothing else.
///
/// Without this, an iso spelling added to `ALL_KV_QUANTS` and left out of
/// `PINS` would be unpinned and the suite would still be green.
#[test]
fn every_iso_spelling_is_pinned_at_both_shapes() {
    let want: Vec<String> = iso_spellings().iter().map(ToString::to_string).collect();
    assert_eq!(
        want.len(),
        ISO_SPELLING_COUNT,
        "iso spelling census moved — an iso variant was added to or removed from \
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
                    other => panic!(
                        "{name}: not an iso storage variant: {}",
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

/// The 3-bit and 4-bit members of each twin pair are not the same store.
///
/// The positive control for the pin table: a unification that collapsed both
/// widths onto one instantiation would leave every assertion above satisfied by
/// a re-baseline, and this one red.
#[test]
fn three_and_four_bit_twins_hold_different_stores() {
    let _guard = env_lock();
    let pairs: [(KvQuant, KvQuant); 3] = [
        (KvQuant::Iso3, KvQuant::Iso4),
        (KvQuant::Iso3Sym, KvQuant::Iso4Sym),
        (KvQuant::IsoKOnly3, KvQuant::IsoKOnly4),
    ];
    for (three, four) in pairs {
        for shape in [SHAPE_A, SHAPE_B] {
            let a = drive(three, shape);
            let b = drive(four, shape);
            assert_ne!(
                a.store_after_chunk, b.store_after_chunk,
                "{three} and {four} @ kv_h={} head_dim={}: the two widths wrote the same \
                 store bytes — one width is not being applied",
                shape.0, shape.1
            );
            assert_ne!(
                a.store_after_decode, b.store_after_decode,
                "{three} and {four} @ kv_h={} head_dim={}: the two widths wrote the same \
                 store bytes after decode",
                shape.0, shape.1
            );
        }
    }
}

/// Driving the same spelling twice gives the same bytes.
///
/// Without this, every assertion above could be pinning a value that is not
/// reproducible, and a red cell would be read as flake rather than defect.
#[test]
fn an_iso_cell_is_reproducible() {
    let _guard = env_lock();
    for quant in iso_spellings() {
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

/// The GPU-resident iso V mirror writes no byte in production.
///
/// `QuantIsoV3::append_gpu` guards its mirror write on
/// [`crate::gpu_resident_iso_enabled`], which is a `false` constant outside
/// `cfg(test)`. The collapse hands the mirror to the 4-bit width as a
/// consequence of making the store one type, and this is what says that hand-off
/// changes no production byte. It is a device-policy assertion, not a dispatch:
/// nothing here touches Metal.
#[test]
fn the_gpu_resident_iso_mirror_is_off_unless_a_test_forces_it() {
    let _guard = env_lock();
    assert!(
        !crate::gpu_resident_iso_enabled(),
        "the GPU-resident iso V mirror is on in this process — the pins above were taken \
         with it off, and production reads a `false` constant"
    );
}
