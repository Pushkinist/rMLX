//! Byte-level lock on the rotor storage layer, at the deepest seam a caller
//! can reach on CPU.
//!
//! # What cannot move
//!
//! Every rotor spelling that exists before the storage/update unification
//! exists after it, and for each spelling, at each shape:
//!
//! * the **packed store bytes** are identical — every `codes` word, every
//!   `scales` and `norms` float, the rotor table, the QJL residual planes, the
//!   affine/turbo companion plane, the block boundaries and the accumulated
//!   shape;
//! * the **K and V rows the attention receives** from that store are identical,
//!   bit for bit, at the chunk append and at every decode step;
//! * `resident_bytes()` is identical;
//! * and the served temp-0 token stream is identical.
//!
//! There is no intended behaviour change. The 3-bit files hold the shared
//! payload types (`RotorBlocks`, `RotorKBlocks`), the `BlockRows` impl and the
//! ring-sync helpers, which the 4-bit files import; both widths already carry
//! `gpu_append`, `gpu_packed_view`, `from_cpu_blocks`, `truncate_to`,
//! `try_deep_clone` and `dequant`. Unifying the two widths therefore moves no
//! code path from one width to the other, and this file is what says so.
//!
//! # Why the served digest is not the oracle
//!
//! Four of the eight rotor spellings — `Rotor3`, `Rotor4`, `RotorK3Asym`,
//! `RotorK4Asym` — report `decode_reads_packed_store() == false` and feed the
//! bf16 mirror on both axes, so `materialises_packed_store()` is false and
//! `exit_prefill` clears their payload outright. A served generation on those
//! four decodes off the mirror and emits the same token ids as `--kv-quant
//! none`, whatever the storage layer does. Their bytes are observable here and
//! nowhere else, which is why this file drives `update` with `in_prefill`
//! false and no bf16 seed: that is the one route on which those four codecs
//! write their store.
//!
//! The four that do read the packed store at decode — `Rotor3Sym`,
//! `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4` — are pinned here as well, so the
//! two oracles overlap rather than partition.
//!
//! # Shapes
//!
//! Two, both legal for all eight spellings:
//!
//! | shape | `kv_h` | `head_dim` | why |
//! |---|---|---|---|
//! | A | 1 | 128 | single KV head (shared-KV arch); power-of-two `head_dim`, so the last multivector group is ragged (`128 = 42*3 + 2`) |
//! | B | 4 | 96 | `kv_h > 1`; non-power-of-two `head_dim` divisible by the group size, so no group is padded |
//!
//! The other axis of each spelling sets the floor: the q8_0 K side of
//! `Rotor{3,4}` needs `B * kv_h * seq * head_dim % 128 == 0` on **every**
//! chunk including a one-token step, and the turbo V side of the asym pair
//! needs `head_dim % 32 == 0`. Both shapes satisfy both.
//!
//! # What this pin cannot see
//!
//! Named so the collapse's reviewer knows where this file stops. Every item
//! here is inside, or reached from, the four files the collapse unifies, and no
//! assertion in this file can turn red on a defect in it.
//!
//! * **The `exit_prefill` bulk-encode arms** (`kvcache/update.rs`, the
//!   `Rotor*` arms of the `exit_prefill` match). This file drives `update`
//!   with `in_prefill` false, which is the only CPU route on which all eight
//!   spellings write a store — but it is *not* the route production takes.
//!   For the four spellings that do read their store at decode, the store a
//!   served request reads is the one those arms bulk-encode. A defect confined
//!   to an arm is invisible here and shows only in a served digest.
//! * **`gpu_append` and `gpu_packed_view`** on all four stores, and the
//!   ring-readback branch of `synced_rotor_v_blocks` / `synced_rotor_k_blocks`.
//!   Unreachable from a `Device::Cpu` drive. These are six of the differing
//!   sites in each storage pair, so they are the largest unseen surface.
//!   Only the `#[ignore]` GPU suite reaches them.
//! * **`from_cpu_blocks` and `try_deep_clone`** — the SSD-hydrate and
//!   branch-clone constructors. Never called on this path.
//! * **The five unpinned asymmetric V configurations.**
//!   `validate_rotor_k_asym_v` accepts `(4, 128|64|32)` and `(3|2, 64)`: ten
//!   cells across the two asym K widths, of which four are pinned here
//!   (`v4_g64`, from `ALL_KV_QUANTS`, and `v2_g64`). A defect that appears only
//!   at `v4_g128`, `v4_g32` or `v3_g64` is not seen.
//! * **Bit-exactness under a different toolchain.** The pins are f32 results
//!   from this host's codegen. They judge one change on one toolchain; they are
//!   not a portable golden.
//!
//! `truncate_to` **is** covered — the drive rolls back into the bulk chunk
//! after the decode steps, so both halves of the truncate plan run and the
//! post-truncate store bytes are a pinned column.

use super::core::KvCache;
use crate::rotor_qjl::rotor_qjl_enabled;
use crate::storage::KvStorage;
use crate::test_utils::{
    array_bytes, env_lock, f32_arr, fnv1a64, lcg_data, push_quant_k, StoreBytes, TEST_SEED,
};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

/// Layer index every cell is built at. The rotor table is seeded from
/// `(layer_idx, head_idx)`, so the pin is only meaningful at a fixed one.
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

/// Serialise a rotor V store (3-bit or 4-bit — the two carry the same fields).
macro_rules! push_rotor_v {
    ($out:expr, $side:expr, $store:expr) => {{
        let s = $store;
        $out.tag($side);
        $out.f32s("rotors", &s.rotors);
        $out.i32s("shape", &s.shape);
        $out.u8_("bits", s.bits);
        $out.usize_("layer_idx", s.layer_idx as usize);
        $out.usize_("head_idx", s.head_idx as usize);
        $out.usize_("n_blocks", s.blocks.len());
        for (i, b) in s.blocks.iter().enumerate() {
            $out.usize_("block", i);
            $out.u32s("codes", &b.codes);
            $out.f32s("scales", &b.scales);
            $out.f32s("norms", &b.norms);
            $out.usize_("n_tokens", b.n_tokens);
        }
    }};
}

/// Serialise a rotor K store. Same fields as the V side plus the two QJL
/// residual planes.
macro_rules! push_rotor_k {
    ($out:expr, $side:expr, $store:expr) => {{
        let s = $store;
        $out.tag($side);
        $out.f32s("rotors", &s.rotors);
        $out.i32s("shape", &s.shape);
        $out.u8_("bits", s.bits);
        $out.usize_("layer_idx", s.layer_idx as usize);
        $out.usize_("head_idx", s.head_idx as usize);
        match s.qjl_s_matrix.as_ref() {
            Some(m) => $out.f32s("qjl_s", m),
            None => $out.tag("qjl_s_none"),
        }
        $out.usize_("n_blocks", s.blocks.len());
        for (i, b) in s.blocks.iter().enumerate() {
            $out.usize_("block", i);
            $out.u32s("codes", &b.codes);
            $out.f32s("scales", &b.scales);
            $out.f32s("norms", &b.norms);
            $out.u8s("qjl_codes", &b.qjl_codes);
            $out.f32s("qjl_norms", &b.qjl_norms);
            $out.usize_("n_tokens", b.n_tokens);
        }
    }};
}

/// Serialise the turbo V companion plane of the asym spellings.
fn push_quant_v(out: &mut StoreBytes, side: &str, s: &crate::storage::QuantV) {
    out.tag(side);
    out.usize_("n_blocks", s.blocks.len());
    for (i, b) in s.blocks.iter().enumerate() {
        out.usize_("block", i);
        out.u8s("codes", &b.codes);
        out.f32s("scales", &b.scales);
        out.i32s("original_shape", &b.original_shape);
        out.u8_("bits", b.bits);
    }
    out.i32s("shape", &s.shape);
}

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

/// Every byte of the storage a rotor spelling holds, in one digest.
///
/// A store the spelling did not populate serialises as its own `absent` tag,
/// so "empty" and "one empty block" are different digests.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant: the storage variant is the one the KvQuant selected; any other is a construction bug, and the explicit panic names it sooner than a wrong digest would"
)]
fn store_digest(storage: &KvStorage) -> u64 {
    let mut out = StoreBytes::default();
    macro_rules! opt {
        ($slot:expr, $name:expr, $push:expr) => {
            match $slot {
                Some(s) => $push(&mut out, $name, s),
                None => out.tag(concat!($name, "_absent")),
            }
        };
    }
    match storage {
        KvStorage::RotorV3 { k, v, .. } => {
            opt!(k.as_ref(), "k_q8", push_quant_k);
            match v.as_ref() {
                Some(s) => push_rotor_v!(out, "v_rotor", s),
                None => out.tag("v_rotor_absent"),
            }
        }
        KvStorage::RotorV4 { k, v, .. } => {
            opt!(k.as_ref(), "k_q8", push_quant_k);
            match v.as_ref() {
                Some(s) => push_rotor_v!(out, "v_rotor", s),
                None => out.tag("v_rotor_absent"),
            }
        }
        KvStorage::RotorSym3 { k, v, .. } => {
            match k.as_ref() {
                Some(s) => push_rotor_k!(out, "k_rotor", s),
                None => out.tag("k_rotor_absent"),
            }
            match v.as_ref() {
                Some(s) => push_rotor_v!(out, "v_rotor", s),
                None => out.tag("v_rotor_absent"),
            }
        }
        KvStorage::RotorSym4 { k, v, .. } => {
            match k.as_ref() {
                Some(s) => push_rotor_k!(out, "k_rotor", s),
                None => out.tag("k_rotor_absent"),
            }
            match v.as_ref() {
                Some(s) => push_rotor_v!(out, "v_rotor", s),
                None => out.tag("v_rotor_absent"),
            }
        }
        KvStorage::RotorKOnly3 { k, .. } => match k.as_ref() {
            Some(s) => push_rotor_k!(out, "k_rotor", s),
            None => out.tag("k_rotor_absent"),
        },
        KvStorage::RotorKOnly4 { k, .. } => match k.as_ref() {
            Some(s) => push_rotor_k!(out, "k_rotor", s),
            None => out.tag("k_rotor_absent"),
        },
        KvStorage::RotorKAsym3 { k, v, .. } => {
            match k.as_ref() {
                Some(s) => push_rotor_k!(out, "k_rotor", s),
                None => out.tag("k_rotor_absent"),
            }
            opt!(v.as_ref(), "v_turbo", push_quant_v);
        }
        KvStorage::RotorKAsym4 { k, v, .. } => {
            match k.as_ref() {
                Some(s) => push_rotor_k!(out, "k_rotor", s),
                None => out.tag("k_rotor_absent"),
            }
            opt!(v.as_ref(), "v_turbo", push_quant_v);
        }
        other => panic!(
            "not a rotor storage variant: {}",
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
    /// covers the plan, not the rotor stores' calls to it.
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
/// four of the eight — so a prefill-bracketed drive would pin four empty
/// stores and read as a clean scan. Appending straight into the decode
/// dispatch is the one CPU route on which all eight write their store.
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

/// Every rotor spelling the enum can spell, in `ALL_KV_QUANTS` order.
///
/// The filter is the enum's own `Display`, not a hand-written variant list. A
/// list would name only the variants that existed when it was written, so a
/// ninth rotor variant would never enter `want` and the census below would stay
/// green with nothing pinned — a gate that cannot fail on the one event it
/// exists for. Every `Display` arm whose text contains `rotor` is a rotor
/// spelling and no other arm's text does (`RotK` renders `rot_k_v8g64`), so the
/// text is the membership test. `ROTOR_SPELLING_COUNT` is the anchor beside it:
/// a variant added to the enum and not to `ALL_KV_QUANTS` moves the count
/// rather than passing quietly.
fn rotor_spellings() -> Vec<KvQuant> {
    ALL_KV_QUANTS
        .iter()
        .copied()
        .filter(|q| q.to_string().contains("rotor"))
        .collect()
}

/// Rotor spellings `ALL_KV_QUANTS` holds today.
const ROTOR_SPELLING_COUNT: usize = 8;

/// Legal asymmetric-V configurations pinned beyond the one `ALL_KV_QUANTS`
/// lists.
///
/// `validate_rotor_k_asym_v` accepts `(4, 128|64|32)` and `(3|2, 64)` — five V
/// configurations per asym K width, ten cells, of which `ALL_KV_QUANTS` names
/// two (`v4_g64`). One more per width is pinned here so the turbo V companion
/// plane is exercised at a second width; the remaining six are named as a blind
/// spot in the module doc.
const EXTRA_ASYM_CELLS: &[KvQuant] = &[
    KvQuant::RotorK3Asym {
        v_bits: 2,
        v_group_size: 64,
    },
    KvQuant::RotorK4Asym {
        v_bits: 2,
        v_group_size: 64,
    },
];

/// Every cell this file pins: the census population plus the extra asym ones.
fn pinned_spellings() -> Vec<KvQuant> {
    let mut v = rotor_spellings();
    v.extend_from_slice(EXTRA_ASYM_CELLS);
    v
}

/// One pinned cell: `(spelling, kv_h, head_dim, store_after_chunk,
/// store_after_decode, store_after_truncate, rows, resident_bytes)`.
type Pin = (&'static str, i32, i32, u64, u64, u64, u64, u64);

/// The "before" side of the oracle, captured on the tree the unification
/// starts from. Any generic type that produces a different byte anywhere in a
/// rotor store, hands attention a different row, or changes a cell's residency
/// turns one of these red and names the cell.
const PINS: &[Pin] = &[
    (
        "rotor3",
        1,
        128,
        0x1ed845adb123e5ee,
        0x1e571e07547424e2,
        0xf82cf5a28169ab93,
        0xe7647b0b20c759c3,
        10408,
    ),
    (
        "rotor3",
        4,
        96,
        0x5a9a75ad98dbff6e,
        0xb5a91bb8732a70e3,
        0xe883cf5c009430d5,
        0x2eaa7c3738755823,
        29348,
    ),
    (
        "rotor4",
        1,
        128,
        0x5019a8983d996afb,
        0x3b9449664bfe5ee9,
        0xfbaa64d468b60e22,
        0x09cb079e9d9c0162,
        10840,
    ),
    (
        "rotor4",
        4,
        96,
        0x5604a3481e53d582,
        0x957c315504d7762e,
        0xbe7687612cfa6f42,
        0x0734d38d67915639,
        30644,
    ),
    (
        "rotor3_sym",
        1,
        128,
        0x089c86e490eae161,
        0x243891f2ec0372d3,
        0x86c7ac2a3ff92278,
        0x3f584ea60f053542,
        13688,
    ),
    (
        "rotor3_sym",
        4,
        96,
        0x00e0eb309de8abef,
        0x083dcfad1d9e7cd1,
        0x6c90492265b304ab,
        0xb67d5fffa6001c79,
        37312,
    ),
    (
        "rotor4_sym",
        1,
        128,
        0x7e7bab52e333a78e,
        0xc8b1ce91d3ca5cd0,
        0xf590e158285dc43d,
        0x3214878d3d23cbb0,
        14552,
    ),
    (
        "rotor4_sym",
        4,
        96,
        0x136ab1a37d733720,
        0xf2bd9beafc3a6fad,
        0x4947b7f23c9d75bd,
        0x58fc19598bce7220,
        39904,
    ),
    (
        "k_rotor3",
        1,
        128,
        0xe15421eb720f5bf6,
        0x8e1e446cf171a70c,
        0xab0fbda0b7ce237a,
        0x9114ea2b45c7252a,
        20668,
    ),
    (
        "k_rotor3",
        4,
        96,
        0x2ce1794f8a160e1e,
        0x881874691537c40c,
        0x96894cb8d7eec498,
        0x98dd682b49ca4a0f,
        60128,
    ),
    (
        "k_rotor4",
        1,
        128,
        0x21b7b28b9ccbbc50,
        0x097e70bb08c9d3ec,
        0x801ca86955ac7a4e,
        0x955a08ac7bb6b53d,
        21100,
    ),
    (
        "k_rotor4",
        4,
        96,
        0x196729fd5de0a073,
        0x78b22765f4d5d5c9,
        0x07eb676905b1d9cd,
        0x0aa5a5e6d96acf64,
        61424,
    ),
    (
        "rotor_k_3_asym_v4_g64",
        1,
        128,
        0x23c9d791bd7f63b1,
        0x248e9ec82dbfce08,
        0x560c46b783a4dbf1,
        0xbebe17013e29623a,
        9004,
    ),
    (
        "rotor_k_3_asym_v4_g64",
        4,
        96,
        0x89e2ca6de16a00fd,
        0xdf1edf05cd8a6c9c,
        0xf909ef33322bd90e,
        0x014e5234606b831e,
        25136,
    ),
    (
        "rotor_k_4_asym_v4_g64",
        1,
        128,
        0xae9d7a1b3db30a4b,
        0x0bdb862c5dc9a3e8,
        0x29e88962f8f5467d,
        0x7c094ddc85b88e75,
        9436,
    ),
    (
        "rotor_k_4_asym_v4_g64",
        4,
        96,
        0xce80650dc2fd9a9e,
        0x51049c4dabafe681,
        0xc9ef829eb8b2a65f,
        0x917c916ab86c2e89,
        26432,
    ),
    (
        "rotor_k_3_asym_v2_g64",
        1,
        128,
        0x81aa509679c2f5fb,
        0x5f88a19815279032,
        0x386546877a5f6a40,
        0x74451fea33532a74,
        8140,
    ),
    (
        "rotor_k_3_asym_v2_g64",
        4,
        96,
        0x0aa34997cbaa2b26,
        0x7e3dba5449561f16,
        0x6a7a97da4547e2fd,
        0xbced5e69c4fb3c23,
        22544,
    ),
    (
        "rotor_k_4_asym_v2_g64",
        1,
        128,
        0x65436689dd57d7b9,
        0x1b7486829a862092,
        0xae961aa1f44a8ad4,
        0xb8c83ffd009a9d6f,
        8572,
    ),
    (
        "rotor_k_4_asym_v2_g64",
        4,
        96,
        0x1a4839b920be76e9,
        0x31b37924da91e83b,
        0x49ad9decdcdf5adc,
        0x007c598a3eaac36c,
        23840,
    ),
];

/// Look a pin up by spelling and shape.
fn pin_for(name: &str, shape: (i32, i32)) -> Option<&'static Pin> {
    PINS.iter()
        .find(|p| p.0 == name && p.1 == shape.0 && p.2 == shape.1)
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

/// Every rotor spelling, at both shapes, holds the bytes it held before.
#[test]
fn rotor_store_bytes_are_pinned_per_spelling_and_shape() {
    let _guard = env_lock();
    assert_qjl_off();
    let mut missing = Vec::new();
    let mut observed = Vec::new();
    for quant in pinned_spellings() {
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

/// The pin table names every rotor spelling the enum can spell, at both
/// shapes — and nothing else.
///
/// Without this, a rotor spelling added to `ALL_KV_QUANTS` and left out of
/// `PINS` would be unpinned and the suite would still be green.
#[test]
fn every_rotor_spelling_is_pinned_at_both_shapes() {
    let want: Vec<String> = rotor_spellings().iter().map(ToString::to_string).collect();
    assert_eq!(
        want.len(),
        ROTOR_SPELLING_COUNT,
        "rotor spelling census moved — a rotor variant was added to or removed from \
         ALL_KV_QUANTS and this file's pins did not follow: {want:?}"
    );
    let all: Vec<String> = pinned_spellings().iter().map(ToString::to_string).collect();
    for name in &all {
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
        all.len() * 2,
        "PINS holds {} rows for {} spellings x 2 shapes — a stale row pins nothing",
        PINS.len(),
        all.len()
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
    for quant in pinned_spellings() {
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

/// The 3-bit and 4-bit members of each twin pair are not the same store.
///
/// The positive control for the pin table: a unification that collapsed both
/// widths onto one instantiation would leave every assertion above satisfied
/// by a re-baseline, and this one red.
#[test]
fn three_and_four_bit_twins_hold_different_stores() {
    let _guard = env_lock();
    assert_qjl_off();
    let pairs: [(KvQuant, KvQuant); 4] = [
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
fn a_rotor_cell_is_reproducible() {
    let _guard = env_lock();
    assert_qjl_off();
    for quant in pinned_spellings() {
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
                "{quant} @ kv_h={} head_dim={}: store bytes differ between two identical \
                 drives",
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
