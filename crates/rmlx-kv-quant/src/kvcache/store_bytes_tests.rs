//! Byte-level lock on the KV storage layer, at the deepest seam a caller can
//! reach on CPU, for **every** spelling `ALL_KV_QUANTS` holds.
//!
//! # Why one file and not one per family
//!
//! The rotor, iso and turbo collapses each landed a pin file of their own.
//! Those three files held one machinery three times — the same drive, the same
//! five observed columns, the same census shape — and differed in the spelling
//! filter, the family name and the per-variant field list. That is the twin
//! shape the repo's own rule names, so the machinery lives here once and the
//! three family files keep only the claims that are about their family. The
//! width-twin control is the same shape and moved here too, as
//! [`assert_width_twins_differ`]; what stays with each family is its pair
//! list. Their three `*_store_geometry_follows_the_codec_bit_width` tests stay
//! apart, because each restates a different codec's published layout
//! arithmetic and one body cannot carry three.
//!
//! The method is stated in [`docs/KV_ROTOR_TWINS.md`] and is not restated here.
//! What this file adds to it is coverage: the three family files pinned 20 of
//! the 28 spellings, and the eight they left out — `none`, `k8v4`, `k8v8`,
//! `planar`, `planar3`, `planar_k`, the mixed pair — are the ones a restructure
//! of the update path is most likely to move, because they are the oldest
//! bodies in it.
//!
//! # What cannot move
//!
//! For every spelling, at each shape:
//!
//! * the **packed store bytes** after the bulk append, after three decode
//!   steps, and after a truncate back into the bulk chunk;
//! * the **K and V rows the attention receives**, bit for bit, at the chunk
//!   append and at every decode step;
//! * `resident_bytes()`.
//!
//! A restructure that moves any of them is a defect, not a re-baseline. The
//! pin values the three family files captured are carried here unchanged; a
//! reader comparing this table against the deleted ones finds the same 44 rows
//! with the same five numbers each.
//!
//! # Two drives, two populations
//!
//! `exit_prefill` returns at its `materialises_packed_store()` gate for every
//! spelling that reports `false`, **before** every arm that would bulk encode a
//! store, and clears whatever payload the cache arrived carrying. That splits
//! every spelling into two populations, and each needs a drive of its own.
//!
//! **The 18 that report `false`.** A served prefill writes them no packed
//! store and the decode entries short-circuit to the bf16 seed, so a served
//! capture against an empty store and one against a correct store emit the
//! same tokens. Their bytes are observable here and nowhere else, and the one
//! route on which they write a store at all is a decode-dispatch append with
//! `in_prefill` false — which is what [`drive`] does. Their `exit_prefill`
//! arms are unreachable, and that the gate leaves them empty is pinned by
//! `warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`.
//!
//! **The 10 that report `true`.** Their `exit_prefill` arms run, and the bytes
//! those arms bulk-encode are what a served decode reads. [`drive`] never
//! reaches those arms, and the guard above reads only whether the store is
//! non-empty, so before this table nothing in the tree read a byte one of them
//! wrote. [`drive_prefill`] brackets the chunk and pins three columns per
//! spelling per shape: the store the arm bulk-encoded, `resident_bytes()` at
//! the same point, and the rows the first decode step after the bracket hands
//! back.
//!
//! Both counts are derived from the predicate and asserted below rather than
//! written into prose, so they move when the dispositions move.
//!
//! # Shapes
//!
//! Two, the pair the three family files used, so a reviewer comparing the
//! tables reads like with like:
//!
//! | shape | `kv_h` | `head_dim` | why |
//! |---|---|---|---|
//! | A | 1 | 128 | single KV head (shared-KV arch); power-of-two `head_dim` |
//! | B | 4 | 96 | `kv_h > 1`; non-power-of-two `head_dim` |
//!
//! Both satisfy the floors the codecs impose on a one-token step: the affine
//! q8_0 K side needs `B * kv_h * seq * head_dim % 128 == 0` (3072 and 128 at
//! shape A, 9216 and 384 at shape B), and the planar codecs need
//! `head_dim % 32 == 0` (128 and 96). For the codecs that group by a power of
//! two neither shape leaves a ragged group. The rotor family is the exception
//! and the reason shape A is kept: it groups by three, so `head_dim = 128`
//! pads its last group (`128 = 42*3 + 2`) while `head_dim = 96` does not.
//!
//! # What this pin cannot see
//!
//! * **`KvStorage::Paged`.** No spelling builds it: the routing reads a
//!   process-global the CLI latches once, and a test that set it could not
//!   unset it for the rest of the binary. The last test states that, so a
//!   change which starts routing a spelling there turns red here rather than
//!   going unpinned. The paged path's own coverage is `crate::paged`.
//! * **Every GPU path** — the MSL encode dispatch, the resident ring buffers,
//!   the fused flash-decode arms, the hydrated-init upload branch. A
//!   `Device::Cpu` drive reaches none of them, and they are the largest unseen
//!   surface in the storage layer. `make gpu-test` is the gate over them.
//! * **The `exit_prefill` arms of the 18 decode-inert spellings.** The gate
//!   returns before them and no CPU route reaches them, so they are dead code
//!   under every drive here. The guard named above is what says the gate keeps
//!   them that way.
//! * **`from_cpu_blocks` and `try_deep_clone`** — the SSD-hydrate and
//!   branch-clone constructors, pinned in `rmlx-kv-ssd`.
//! * **`max_seq`.** Deliberately not a digest field: the stores that carry one
//!   never read it on this path, so pinning it would force a re-baseline over
//!   an inert field.
//! * **Bit-exactness under a different toolchain.** The pins are f32 results
//!   from this host's codegen. They judge one change on one toolchain; they are
//!   not a portable golden.
//!
//! `truncate_to` **is** covered: the drive rolls back into the bulk chunk after
//! the decode steps, so both halves of the truncate plan run and the
//! post-truncate store bytes are a pinned column.
//!
//! [`docs/KV_ROTOR_TWINS.md`]: ../../../../docs/KV_ROTOR_TWINS.md

use super::core::KvCache;
use crate::storage::KvStorage;
use crate::test_utils::{
    array_bytes, env_lock, f32_arr, fnv1a64, lcg_data, push_quant_k, StoreBytes, TEST_SEED,
};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

/// Layer index every cell is built at. The rotor table is seeded from
/// `(layer_idx, head_idx)`, so the pin is only meaningful at a fixed one.
pub(super) const TEST_LAYER_IDX: usize = 3;
/// Storage capacity. Larger than the driven length so no ring wrap is in play.
pub(super) const TEST_MAX_SEQ: i32 = 512;
/// Positions in the single bulk append.
pub(super) const CHUNK_SEQ: i32 = 24;
/// One-token appends driven after the chunk.
const DECODE_STEPS: usize = 3;

/// `(kv_h, head_dim)` — see the shape table in the module doc.
pub(super) const SHAPE_A: (i32, i32) = (1, 128);
pub(super) const SHAPE_B: (i32, i32) = (4, 96);
/// The second shape for a spelling whose codec cannot group 96 elements. Same
/// `kv_h > 1` as [`SHAPE_B`], power-of-two `head_dim`.
pub(super) const SHAPE_B_POW2: (i32, i32) = (4, 128);

/// Group sizes a spelling carries in its own parameters.
///
/// Only the mixed pair parameterises one; every other spelling fixes its group
/// size inside the codec, where a shape cannot disagree with it.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the two arms named are the only spellings that carry a group size in the enum; every other one fixes it inside its codec and has nothing to report here"
)]
fn group_sizes(quant: KvQuant) -> Vec<i32> {
    match quant {
        KvQuant::Mixed {
            k_group_size,
            v_group_size,
            ..
        } => vec![i32::from(k_group_size), i32::from(v_group_size)],
        KvQuant::RotK { v_group_size, .. } => vec![i32::from(v_group_size)],
        _ => Vec::new(),
    }
}

/// The two shapes one spelling is driven at.
///
/// [`SHAPE_B`] moves two properties at once — `kv_h` off 1 and `head_dim` off a
/// power of two. A spelling whose group size does not divide 96 cannot be
/// driven there at all: `MixedKvState::init_quant` rejects the append before
/// any store is written. Those spellings keep the `kv_h > 1` half and give up
/// the non-power-of-two half, which is derived from the spelling's own
/// parameters rather than written into a list that would go stale.
pub(super) fn shapes_for(quant: KvQuant) -> [(i32, i32); 2] {
    let second = if group_sizes(quant)
        .iter()
        .all(|g| *g > 0 && SHAPE_B.1 % g == 0)
    {
        SHAPE_B
    } else {
        SHAPE_B_POW2
    };
    [SHAPE_A, second]
}

/// Serialise a TurboQuant block store. `QuantKTurbo<BITS>` and `QuantV` carry
/// the same payload type and header fields, so one body covers both axes.
///
/// A macro, not a fn: the two are unrelated types and one fn cannot take both.
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

/// Serialise an iso store — the V and K stores carry the same payload type
/// (`IsoBlocks`) and the same header fields, so one body covers both axes.
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

/// Serialise a rotor V store.
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

/// Serialise a planar store. `QuantPlanarK` and `QuantPlanarV` carry the same
/// block payload; the V side carries a bit width the K side does not, so the
/// caller passes it in.
macro_rules! push_planar {
    ($out:expr, $side:expr, $store:expr, $bits:expr) => {{
        let s = $store;
        $out.tag($side);
        $out.i32s("shape", &s.shape);
        $out.u8_("bits", $bits);
        $out.u8_("gpu_codes_live", u8::from(s.gpu_codes_buf.is_some()));
        $out.u8_("gpu_scales_live", u8::from(s.gpu_scales_buf.is_some()));
        $out.u8_(
            "gpu_rotations_live",
            u8::from(s.gpu_rotations_buf.is_some()),
        );
        $out.usize_("gpu_capacity", s.gpu_capacity as usize);
        $out.usize_("n_blocks", s.blocks.len());
        for (i, b) in s.blocks.iter().enumerate() {
            $out.usize_("block", i);
            $out.u8s("codes", &b.codes);
            $out.f32s("scales", &b.scales);
            $out.u8s("rotations", &b.rotations);
            $out.i32s("original_shape", &b.original_shape);
        }
    }};
}

/// Serialise the turbo V companion plane of the rotor asym spellings.
///
/// The field list is the rotor family's, not [`push_turbo`]'s, and the two are
/// deliberately not merged: each family's pin rows were captured with its own
/// serialisation, and one encoding for both would re-baseline every row of the
/// other. A serialiser is an identity here, not an interface.
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

/// Serialise the mixed-precision state: the two quantized 3-tuples, the codec
/// settings they were produced under, and the K-side rotation matrix.
fn push_mixed(out: &mut StoreBytes, s: &crate::mixed_quant::MixedKvState) {
    out.tag("mixed");
    out.usize_("k_bits", s.k_bits as usize);
    out.usize_("v_bits", s.v_bits as usize);
    out.usize_("k_group_size", s.k_group_size as usize);
    out.usize_("v_group_size", s.v_group_size as usize);
    out.usize_("offset", s.offset as usize);
    out.u8_("rotate_k", u8::from(s.rotate_k));
    for (name, slot) in [("keys", s.keys.as_ref()), ("values", s.values.as_ref())] {
        match slot {
            Some(t) => {
                out.tag(name);
                out.u8s("codes", &array_bytes(&t.codes));
                out.u8s("scales", &array_bytes(&t.scales));
                out.u8s("biases", &array_bytes(&t.biases));
            }
            None => out.tag("tuple_absent"),
        }
    }
    match s.k_rotation.as_ref() {
        Some(r) => out.u8s("k_rotation", &array_bytes(r)),
        None => out.tag("k_rotation_absent"),
    }
}

/// Every byte of the storage a spelling holds, in one digest.
///
/// A store the spelling did not populate serialises as its own `absent` tag,
/// so "empty" and "one empty block" are different digests.
///
/// The `match` is exhaustive and names no wildcard on purpose: a storage
/// variant added to the enum then fails to compile here, which is how a new
/// codec is made to bring its pin rows with it.
fn store_digest(storage: &KvStorage) -> u64 {
    let mut out = StoreBytes::default();
    macro_rules! opt_q8 {
        ($slot:expr, $name:expr) => {
            match $slot {
                Some(s) => push_quant_k(&mut out, $name, s),
                None => out.tag(concat!($name, "_absent")),
            }
        };
    }
    macro_rules! opt {
        ($slot:expr, $name:expr, $push:ident) => {
            match $slot {
                Some(s) => $push!(out, $name, s),
                None => out.tag(concat!($name, "_absent")),
            }
        };
    }
    match storage {
        KvStorage::K8V4 { k, v, .. }
        | KvStorage::K8VTurbo3 { k, v, .. }
        | KvStorage::K8VTurbo3Tcq { k, v, .. }
        | KvStorage::K8VTurbo2 { k, v, .. }
        | KvStorage::K8VTurbo2Tcq { k, v, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            opt!(v.as_ref(), "v_turbo", push_turbo);
        }
        KvStorage::K8V8 { k, v, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            opt_q8!(v.as_ref(), "v_q8");
        }
        KvStorage::TurboSym3 { k, v, .. } => {
            opt!(k.as_ref(), "k_turbo", push_turbo);
            opt!(v.as_ref(), "v_turbo", push_turbo);
        }
        KvStorage::TurboSym4 { k, v, .. } => {
            opt!(k.as_ref(), "k_turbo", push_turbo);
            opt!(v.as_ref(), "v_turbo", push_turbo);
        }
        KvStorage::Planar { k, v, bits, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            match v.as_ref() {
                Some(s) => push_planar!(out, "v_planar", s, *bits),
                None => out.tag("v_planar_absent"),
            }
        }
        KvStorage::PlanarK { k, .. } => match k.as_ref() {
            // `QuantPlanarK` is 4-bit by construction and carries no width
            // field, so the width is written out here rather than read back
            // from a store that cannot disagree with it.
            Some(s) => push_planar!(out, "k_planar", s, 4_u8),
            None => out.tag("k_planar_absent"),
        },
        KvStorage::IsoV3 { k, v, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            opt!(v.as_ref(), "v_iso", push_iso);
        }
        KvStorage::IsoV4 { k, v, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            opt!(v.as_ref(), "v_iso", push_iso);
        }
        KvStorage::IsoSym3 { k, v, .. } => {
            opt!(k.as_ref(), "k_iso", push_iso);
            opt!(v.as_ref(), "v_iso", push_iso);
        }
        KvStorage::IsoSym4 { k, v, .. } => {
            opt!(k.as_ref(), "k_iso", push_iso);
            opt!(v.as_ref(), "v_iso", push_iso);
        }
        KvStorage::IsoKOnly3 { k, .. } => opt!(k.as_ref(), "k_iso", push_iso),
        KvStorage::IsoKOnly4 { k, .. } => opt!(k.as_ref(), "k_iso", push_iso),
        KvStorage::RotorV3 { k, v, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            opt!(v.as_ref(), "v_rotor", push_rotor_v);
        }
        KvStorage::RotorV4 { k, v, .. } => {
            opt_q8!(k.as_ref(), "k_q8");
            opt!(v.as_ref(), "v_rotor", push_rotor_v);
        }
        KvStorage::RotorSym3 { k, v, .. } => {
            opt!(k.as_ref(), "k_rotor", push_rotor_k);
            opt!(v.as_ref(), "v_rotor", push_rotor_v);
        }
        KvStorage::RotorSym4 { k, v, .. } => {
            opt!(k.as_ref(), "k_rotor", push_rotor_k);
            opt!(v.as_ref(), "v_rotor", push_rotor_v);
        }
        KvStorage::RotorKOnly3 { k, .. } => opt!(k.as_ref(), "k_rotor", push_rotor_k),
        KvStorage::RotorKOnly4 { k, .. } => opt!(k.as_ref(), "k_rotor", push_rotor_k),
        KvStorage::RotorKAsym3 { k, v, .. } => {
            opt!(k.as_ref(), "k_rotor", push_rotor_k);
            match v.as_ref() {
                Some(s) => push_quant_v(&mut out, "v_turbo", s),
                None => out.tag("v_turbo_absent"),
            }
        }
        KvStorage::RotorKAsym4 { k, v, .. } => {
            opt!(k.as_ref(), "k_rotor", push_rotor_k);
            match v.as_ref() {
                Some(s) => push_quant_v(&mut out, "v_turbo", s),
                None => out.tag("v_turbo_absent"),
            }
        }
        KvStorage::Mixed { state, .. } => push_mixed(&mut out, state),
        // The bf16 cache holds its buffers on the parent `KvCache`, not in the
        // storage. Its store column is this constant tag at every step, and
        // the rows and residency columns are what pin the cell — which is
        // stated here so a reader does not mistake a constant column for a
        // cell that is not driven.
        KvStorage::None { .. } => out.tag("bf16_on_the_parent_cache"),
        KvStorage::Paged { .. } => panic!(
            "a spelling built the paged storage: the paged routing is latched by the CLI and \
             no test in this binary can set it, so this cell has no pin and the paged path's \
             own suite is where it belongs"
        ),
    }
    out.digest()
}

/// What one cell observes.
pub(super) struct CellObservation {
    /// Store digest after the bulk chunk.
    pub(super) store_after_chunk: u64,
    /// Store digest after the last decode step.
    pub(super) store_after_decode: u64,
    /// Store digest after truncating back into the bulk chunk.
    pub(super) store_after_truncate: u64,
    /// Digest of the K/V rows the attention received, chunk and every step.
    pub(super) rows: u64,
    /// `KvCache::resident_bytes` after the last decode step.
    pub(super) resident_bytes: u64,
}

/// Attention scale every mixed-path cell is driven at. Fixed, because the pin
/// is of the store the append wrote, not of the attention arithmetic.
const TEST_SCALE: f32 = 0.125;

/// Append one chunk and collect what the attention received.
///
/// Two routes, because the cache refuses one of them. Every spelling but the
/// mixed pair hands its K and V rows back from `update`, and those rows are
/// what the attention reads. `KvQuant::Mixed` and `KvQuant::RotK` reject
/// `update` outright — a direct call leaves their state inconsistent, so the
/// cache returns a contract violation — and the one entry that appends to them
/// is `update_and_sdpa`, which returns the attention **output** rather than the
/// rows. For those two the `rows` column is therefore the output of one
/// attention over the store the append just wrote: a value that moves on the
/// same defects, one step further downstream.
#[allow(
    clippy::expect_used,
    reason = "test driver: every append here is on a shape the spelling accepts, so a failure is the defect under test and the panic names it"
)]
fn append(
    cache: &mut KvCache,
    quant: KvQuant,
    k: &rmlx_mlx::Array,
    v: &rmlx_mlx::Array,
    rows: &mut Vec<u8>,
    device: Device,
) {
    if quant.uses_mixed_path() {
        let out = cache
            .update_and_sdpa(k, k, v, TEST_SCALE, "", None, device)
            .expect("mixed append");
        rows.extend_from_slice(&array_bytes(&out));
    } else {
        let (k_out, v_out) = cache.update(k, v, device).expect("append");
        rows.extend_from_slice(&array_bytes(&k_out));
        rows.extend_from_slice(&array_bytes(&v_out));
    }
}

/// Drive one spelling at one shape with `in_prefill` false throughout.
///
/// The prefill bracket is deliberately absent here: `exit_prefill` clears the
/// payload of every spelling whose `materialises_packed_store()` is false, so a
/// prefill-bracketed drive would pin empty stores for them and read as a clean
/// scan. Appending straight into the decode dispatch is the one CPU route on
/// which every spelling writes its store. [`drive_prefill`] is the other half:
/// it runs the bracket for the spellings that do materialise one.
#[allow(
    clippy::expect_used,
    reason = "test driver: every append here is on a shape the spelling accepts, so a failure is the defect under test and the panic names it"
)]
pub(super) fn drive(quant: KvQuant, shape: (i32, i32)) -> CellObservation {
    let (kv_h, head_dim) = shape;
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ).with_layer_idx(TEST_LAYER_IDX);

    let mut rows = Vec::new();

    let chunk_shape = [1_i32, kv_h, CHUNK_SEQ, head_dim];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n_chunk, TEST_SEED), &chunk_shape);
    let v = f32_arr(&lcg_data(n_chunk, TEST_SEED ^ 0x5a5a), &chunk_shape);
    append(&mut cache, quant, &k, &v, &mut rows, device);
    let store_after_chunk = store_digest(&cache.storage);

    let step_shape = [1_i32, kv_h, 1, head_dim];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    for step in 0..DECODE_STEPS {
        let seed = TEST_SEED.wrapping_add(step as u64 + 1);
        let ks = f32_arr(&lcg_data(n_step, seed), &step_shape);
        let vs = f32_arr(&lcg_data(n_step, seed ^ 0x5a5a), &step_shape);
        append(&mut cache, quant, &ks, &vs, &mut rows, device);
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

/// What one prefill-bracketed cell observes.
///
/// Three columns, not the five [`CellObservation`] carries: the bracket runs
/// one bulk encode and one step, so there is no second append to observe and
/// no truncate plan in play. What it adds is the arm that wrote the bytes.
pub(super) struct PrefillObservation {
    /// Store digest after `exit_prefill` bulk-encoded the chunk.
    pub(super) store_after_exit_prefill: u64,
    /// `KvCache::resident_bytes` at the same point.
    pub(super) resident_bytes: u64,
    /// Digest of the rows the first decode step after the bracket handed back.
    pub(super) rows: u64,
}

/// Drive one spelling at one shape through the prefill bracket production
/// takes: `enter_prefill`, one bulk chunk, `exit_prefill`, one decode step.
///
/// [`drive`] appends straight into the decode dispatch. That is the one CPU
/// route on which *every* spelling writes a store, which is why the pin table
/// above uses it — and it is also the one route that runs no `exit_prefill`
/// bulk-encode arm at all. Those arms are what a served prefill executes, and
/// the bytes they write are what a served decode then reads.
///
/// Only a spelling whose `materialises_packed_store()` is true is driven here.
/// For the rest `exit_prefill` returns at its gate and clears the payload, so
/// their arms are unreachable; that store-is-empty property is pinned by
/// `warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`,
/// which reads the byte count and not the bytes.
#[allow(
    clippy::expect_used,
    reason = "test driver: every append here is on a shape the spelling accepts, so a failure is the defect under test and the panic names it"
)]
pub(super) fn drive_prefill(quant: KvQuant, shape: (i32, i32)) -> PrefillObservation {
    let (kv_h, head_dim) = shape;
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ).with_layer_idx(TEST_LAYER_IDX);
    cache.enter_prefill();

    let chunk_shape = [1_i32, kv_h, CHUNK_SEQ, head_dim];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n_chunk, TEST_SEED), &chunk_shape);
    let v = f32_arr(&lcg_data(n_chunk, TEST_SEED ^ 0x5a5a), &chunk_shape);
    // In prefill every spelling takes the same raw-accumulation route, the
    // mixed pair included, so `update` is the append here and the rows it
    // hands back are the raw chunk rather than a store read.
    cache.update(&k, &v, device).expect("prefill chunk");
    cache.exit_prefill(device).expect("exit_prefill");

    let store_after_exit_prefill = store_digest(&cache.storage);
    let resident_bytes = cache.resident_bytes();

    let step_shape = [1_i32, kv_h, 1, head_dim];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    let seed = TEST_SEED.wrapping_add(1);
    let ks = f32_arr(&lcg_data(n_step, seed), &step_shape);
    let vs = f32_arr(&lcg_data(n_step, seed ^ 0x5a5a), &step_shape);
    let mut rows = Vec::new();
    append(&mut cache, quant, &ks, &vs, &mut rows, device);

    PrefillObservation {
        store_after_exit_prefill,
        resident_bytes,
        rows: fnv1a64(&rows),
    }
}

/// The two members of each pair write different stores.
///
/// The positive control for the pin tables: a unification that collapsed two
/// settings onto one instantiation would leave every geometry assertion above
/// satisfied by a re-baseline, and this one red.
///
/// The drive and the columns are this file's, so the check is too. What stays
/// with each family is its pair list, which is the only part of the claim that
/// is about one codec. `what` names the setting the pair differs in, so a red
/// assertion says which one stopped being applied.
pub(super) fn assert_width_twins_differ(pairs: &[(KvQuant, KvQuant)], what: &str) {
    let _guard = env_lock();
    for &(left, right) in pairs {
        for shape in shapes_for(left) {
            let a = drive(left, shape);
            let b = drive(right, shape);
            assert_ne!(
                a.store_after_chunk, b.store_after_chunk,
                "{left} and {right} @ kv_h={} head_dim={}: the two wrote the same store bytes \
                 after the bulk append — {what} is not being applied",
                shape.0, shape.1
            );
            assert_ne!(
                a.store_after_decode, b.store_after_decode,
                "{left} and {right} @ kv_h={} head_dim={}: the two wrote the same store bytes \
                 after decode — {what} is not being applied",
                shape.0, shape.1
            );
        }
    }
}

/// Spellings `ALL_KV_QUANTS` holds today. An anchor beside the derived sweep:
/// a spelling added to the enum and not to `ALL_KV_QUANTS` moves this count
/// rather than passing quietly.
const SPELLING_COUNT: usize = 28;

/// Legal parameterisations pinned beyond the one `ALL_KV_QUANTS` lists.
///
/// `ALL_KV_QUANTS` carries one representative setting per field-carrying
/// variant, so a second setting is reachable only by naming it. These two are
/// the rotor asym K widths at a 2-bit V companion plane:
/// `validate_rotor_k_asym_v` accepts `(4, 128|64|32)` and `(3|2, 64)`, five V
/// configurations per asym K width, of which the enum's list names one. One
/// more per width is pinned so the turbo V companion plane is exercised at a
/// second width; the remaining six are a blind spot.
///
/// A named list beside a derived population, and it cannot go stale silently:
/// every cell here is swept and pinned like any other, so a row that stops
/// building turns the pin test red.
const EXTRA_PARAMETERISATIONS: &[KvQuant] = &[
    KvQuant::RotorK3Asym {
        v_bits: 2,
        v_group_size: 64,
    },
    KvQuant::RotorK4Asym {
        v_bits: 2,
        v_group_size: 64,
    },
];

/// Every cell this file pins: every spelling the enum can spell, plus the
/// extra parameterisations above.
pub(super) fn pinned_spellings() -> Vec<KvQuant> {
    let mut v: Vec<KvQuant> = ALL_KV_QUANTS.to_vec();
    v.extend_from_slice(EXTRA_PARAMETERISATIONS);
    v
}

/// The spellings whose `exit_prefill` arm runs — the population
/// [`drive_prefill`] sweeps.
///
/// Derived from the disposition predicate, not from a list: a codec that grows
/// a decode kernel over its own store flips that predicate and enters this
/// population, and the pin census below then says it owes rows.
pub(super) fn materialising_spellings() -> Vec<KvQuant> {
    pinned_spellings()
        .into_iter()
        .filter(KvQuant::materialises_packed_store)
        .collect()
}

/// One pinned cell: `(spelling, kv_h, head_dim, store_after_chunk,
/// store_after_decode, store_after_truncate, rows, resident_bytes)`.
type Pin = (&'static str, i32, i32, u64, u64, u64, u64, u64);

/// The "before" side of the oracle.
///
/// The 44 rows of the rotor, iso and turbo cells are the values those three
/// families' own pin files captured, carried here byte for byte — none was
/// re-baselined by the merge. The 16 rows of the eight spellings those files
/// never covered are new: `none`, `k8v4`, `k8v8`, `planar`, `planar3`,
/// `planar_k` and the mixed pair had no store-bytes pin at all before this
/// file.
const PINS: &[Pin] = &[
    (
        "none",
        1,
        128,
        0xd0f36f67914eadc2,
        0xd0f36f67914eadc2,
        0xd0f36f67914eadc2,
        0x96e6b3300b20f630,
        13824,
    ),
    (
        "none",
        4,
        96,
        0xd0f36f67914eadc2,
        0xd0f36f67914eadc2,
        0xd0f36f67914eadc2,
        0x33e64571e1f83a12,
        41472,
    ),
    (
        "k8v4",
        1,
        128,
        0x1d1e1bb05a047686,
        0x2cef5519fd44e2db,
        0xc5cba8628399fed6,
        0x4eb8b16203c0364f,
        5724,
    ),
    (
        "k8v4",
        4,
        96,
        0x866be6aead50194e,
        0xff8e4c9aa80eae6a,
        0xdc1f2891fcf1c31c,
        0x6b423a033362005c,
        17172,
    ),
    (
        "k8v8",
        1,
        128,
        0xd3acd12d1d750cb7,
        0x377abddfcdf277df,
        0xbcb3d7586a2af8aa,
        0xf9799f12f6232d13,
        7128,
    ),
    (
        "k8v8",
        4,
        96,
        0x5f72aaf8b23523cb,
        0x34a6161647792f15,
        0xf740b9874156f7d7,
        0xe7e805b620c53888,
        21384,
    ),
    (
        "planar",
        1,
        128,
        0xa460e6612c0ba377,
        0x6c840aa56d465257,
        0x3272c0da1322d760,
        0xd9b0190b1ed99d60,
        13068,
    ),
    (
        "planar",
        4,
        96,
        0x042e3502a1bf3379,
        0xe8152485207e0e34,
        0xea195bce35c17bd9,
        0x8548ff4b59578f1a,
        39204,
    ),
    (
        "planar3",
        1,
        128,
        0x204ab99c34f78fd9,
        0x32a035d062534e38,
        0xfba95299aa179771,
        0x4be142af8ff939e9,
        13068,
    ),
    (
        "planar3",
        4,
        96,
        0xcb79e70afd6321e5,
        0x2a71acc421f1a973,
        0x3a5990a3e719355e,
        0xde83dba4e59b585d,
        39204,
    ),
    (
        "planar_k",
        1,
        128,
        0x1e40335a9e551ecf,
        0x1e40335a9e551ecf,
        0x1e40335a9e551ecf,
        0x6a26c1a56d5ed055,
        22272,
    ),
    (
        "planar_k",
        4,
        96,
        0xb80bccefab6a6977,
        0xb80bccefab6a6977,
        0xb80bccefab6a6977,
        0x2f684c73a2b913d8,
        66816,
    ),
    (
        "mixed_k8g64_v4g64",
        1,
        128,
        0x910f41a35c42bdc3,
        0x2b739b3e60458b9b,
        0x857e6cba4f76c4f1,
        0xe3be1abd8d815479,
        57344,
    ),
    (
        "mixed_k8g64_v4g64",
        4,
        128,
        0x69bf6e035aeed087,
        0x52f3984721fa1529,
        0x9d7660a514ff95d7,
        0x05f8c39867fcd450,
        229376,
    ),
    (
        "rot_k_v8g64",
        1,
        128,
        0xe91b6c0aa0f34120,
        0x27dea87207c2b602,
        0xfa061ddad10d8b50,
        0x893b49f7f701eaaf,
        139264,
    ),
    (
        "rot_k_v8g64",
        4,
        128,
        0xf4abf0997c0c4185,
        0x7699f16ad97c6aa3,
        0xeb25bca4710f5751,
        0xe236f69516533d47,
        360448,
    ),
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

/// Every spelling, at both shapes, holds the bytes it held before.
#[test]
fn store_bytes_are_pinned_per_spelling_and_shape() {
    let _guard = env_lock();
    let mut missing = Vec::new();
    let mut observed = Vec::new();
    for quant in pinned_spellings() {
        let name = quant.to_string();
        for shape in shapes_for(quant) {
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

/// The pin table names every spelling the enum can spell, at both shapes — and
/// nothing else.
///
/// The population is `ALL_KV_QUANTS` itself, not a filtered subset: there is no
/// family filter left to go blind on a spelling whose `Display` text carries a
/// new token.
#[test]
fn every_spelling_is_pinned_at_both_shapes() {
    let want: Vec<String> = ALL_KV_QUANTS.iter().map(ToString::to_string).collect();
    assert_eq!(
        want.len(),
        SPELLING_COUNT,
        "the spelling census moved — a codec was added to or removed from ALL_KV_QUANTS and \
         this file's pins did not follow: {want:?}"
    );
    let pinned = pinned_spellings();
    for quant in pinned.iter().copied() {
        let name = quant.to_string();
        for shape in shapes_for(quant) {
            assert!(
                pin_for(&name, shape).is_some(),
                "{name} @ kv_h={} head_dim={} has no pin",
                shape.0,
                shape.1
            );
        }
    }
    assert_eq!(
        PINS.len(),
        pinned.len() * 2,
        "PINS holds {} rows for {} cells x 2 shapes — a stale row pins nothing",
        PINS.len(),
        pinned.len()
    );
}

/// One pinned prefill cell: `(spelling, kv_h, head_dim,
/// store_after_exit_prefill, resident_bytes, rows)`.
type PrefillPin = (&'static str, i32, i32, u64, u64, u64);

/// The "before" side of the prefill oracle.
///
/// Every row here is new. No file in the tree read the bytes an
/// `exit_prefill` arm wrote before this one: the pin table above drives a
/// route that runs none of those arms, and the guard that does run them reads
/// only whether the store is non-empty.
const PREFILL_PINS: &[PrefillPin] = &[
    (
        "mixed_k8g64_v4g64",
        1,
        128,
        0x8335af25efd0fef0,
        4992,
        0x3ec6ab35eb546b3f,
    ),
    (
        "mixed_k8g64_v4g64",
        4,
        128,
        0x731372a4cccf895a,
        19968,
        0xf0ae10a6fdafb465,
    ),
    (
        "rot_k_v8g64",
        1,
        128,
        0x03fa4ac7ed91f708,
        72064,
        0x9455f93c9ca69906,
    ),
    (
        "rot_k_v8g64",
        4,
        128,
        0x50e95c202355bfdb,
        91648,
        0xdf725b46270642d2,
    ),
    (
        "iso3_sym",
        1,
        128,
        0xadba4f7bb77a55ad,
        33216,
        0xad069e5080b3a3f7,
    ),
    (
        "iso3_sym",
        4,
        96,
        0x2689c78a863f3f06,
        99840,
        0xbbddb7ff340f3fd8,
    ),
    (
        "iso4_sym",
        1,
        128,
        0x5f848dd01a15acc7,
        33984,
        0xce6e0599d6eb1839,
    ),
    (
        "iso4_sym",
        4,
        96,
        0xf9e7089609b2ec37,
        102144,
        0x8ad58a674158c26f,
    ),
    (
        "k_iso3",
        1,
        128,
        0xbb9e90fee8d84779,
        22752,
        0x8e848d5595e061af,
    ),
    (
        "k_iso3",
        4,
        96,
        0x4b62a9e06e2b5772,
        68352,
        0xc4e7767e0d74beb8,
    ),
    (
        "k_iso4",
        1,
        128,
        0x05eb1b63429e6060,
        23136,
        0x4046f97170a2603a,
    ),
    (
        "k_iso4",
        4,
        96,
        0x2dda5c30f346be33,
        69504,
        0xd23846fa568fe3d5,
    ),
    (
        "rotor3_sym",
        1,
        128,
        0x6626a18505e29f77,
        12320,
        0xaba52798cfbc5408,
    ),
    (
        "rotor3_sym",
        4,
        96,
        0x415cdf8ffaadf097,
        33280,
        0x44626bcd24b0f4ba,
    ),
    (
        "rotor4_sym",
        1,
        128,
        0x75949e7159ba5c6c,
        13088,
        0x4eccdcf45ece1da8,
    ),
    (
        "rotor4_sym",
        4,
        96,
        0x8acc3dade9ad35ce,
        35584,
        0x5609c90cfab335a6,
    ),
    (
        "k_rotor3",
        1,
        128,
        0xff94a3ecc882a908,
        12304,
        0xe5c15bc5f72f2e55,
    ),
    (
        "k_rotor3",
        4,
        96,
        0x968acb5046ce0839,
        35072,
        0x1d4b89b7d79dcfd9,
    ),
    (
        "k_rotor4",
        1,
        128,
        0x3101b4f9c159866e,
        12688,
        0x7af8e17246f9c2eb,
    ),
    (
        "k_rotor4",
        4,
        96,
        0x974461cdf0716469,
        36224,
        0x98e2ba20ee211ea8,
    ),
];

/// Look a prefill pin up by spelling and shape.
fn prefill_pin_for(name: &str, shape: (i32, i32)) -> Option<&'static PrefillPin> {
    PREFILL_PINS
        .iter()
        .find(|p| p.0 == name && p.1 == shape.0 && p.2 == shape.1)
}

/// The bytes the live `exit_prefill` arms write, at both shapes.
#[test]
fn exit_prefill_bulk_encode_bytes_are_pinned_per_spelling_and_shape() {
    let _guard = env_lock();
    let mut missing = Vec::new();
    let mut observed = Vec::new();
    for quant in materialising_spellings() {
        let name = quant.to_string();
        for shape in shapes_for(quant) {
            let obs = drive_prefill(quant, shape);
            observed.push(format!(
                "    (\"{}\", {}, {}, {:#018x}, {}, {:#018x}),",
                name, shape.0, shape.1, obs.store_after_exit_prefill, obs.resident_bytes, obs.rows
            ));
            let Some(pin) = prefill_pin_for(&name, shape) else {
                missing.push(format!("{name} @ kv_h={} head_dim={}", shape.0, shape.1));
                continue;
            };
            assert_eq!(
                obs.store_after_exit_prefill, pin.3,
                "{name} @ kv_h={} head_dim={}: the store exit_prefill bulk-encoded moved",
                shape.0, shape.1
            );
            assert_eq!(
                obs.resident_bytes, pin.4,
                "{name} @ kv_h={} head_dim={}: resident_bytes after exit_prefill moved",
                shape.0, shape.1
            );
            assert_eq!(
                obs.rows, pin.5,
                "{name} @ kv_h={} head_dim={}: the rows the first decode step after \
                 exit_prefill handed back moved",
                shape.0, shape.1
            );
        }
    }
    assert!(
        missing.is_empty(),
        "no prefill pin for {} cell(s): {missing:?}\nobserved table:\n{}",
        missing.len(),
        observed.join("\n")
    );
}

/// The prefill pin table names every materialising spelling at both shapes —
/// and nothing else.
#[test]
fn every_materialising_spelling_is_pinned_through_the_prefill_bracket() {
    let live = materialising_spellings();
    for quant in live.iter().copied() {
        let name = quant.to_string();
        for shape in shapes_for(quant) {
            assert!(
                prefill_pin_for(&name, shape).is_some(),
                "{name} @ kv_h={} head_dim={} has no prefill pin",
                shape.0,
                shape.1
            );
        }
    }
    assert_eq!(
        PREFILL_PINS.len(),
        live.len() * 2,
        "PREFILL_PINS holds {} rows for {} cells x 2 shapes — a stale row pins nothing",
        PREFILL_PINS.len(),
        live.len()
    );
}

/// Driving the same spelling twice gives the same bytes.
///
/// Without this, every assertion above could be pinning a value that is not
/// reproducible, and a red cell would be read as flake rather than defect.
#[test]
fn a_cell_is_reproducible() {
    let _guard = env_lock();
    for quant in pinned_spellings() {
        for shape in shapes_for(quant) {
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
    for quant in materialising_spellings() {
        for shape in shapes_for(quant) {
            let a = drive_prefill(quant, shape);
            let b = drive_prefill(quant, shape);
            assert_eq!(
                a.store_after_exit_prefill, b.store_after_exit_prefill,
                "{quant} @ kv_h={} head_dim={}: the bulk-encoded store differs between two \
                 identical prefill drives",
                shape.0, shape.1
            );
            assert_eq!(
                a.rows, b.rows,
                "{quant} @ kv_h={} head_dim={}: the first decode step's rows differ between \
                 two identical prefill drives",
                shape.0, shape.1
            );
            assert_eq!(
                a.resident_bytes, b.resident_bytes,
                "{quant} @ kv_h={} head_dim={}: resident_bytes after exit_prefill differs \
                 between two identical prefill drives",
                shape.0, shape.1
            );
        }
    }
}

/// How many spellings a served digest cannot judge, derived rather than
/// asserted in prose.
///
/// A spelling whose `materialises_packed_store()` is false writes no packed
/// store past `exit_prefill`, so a served capture against an empty store and
/// one against a correct store emit the same tokens: its bytes are observable
/// in this file and nowhere else. The count is pinned so that a codec moving
/// into or out of that class is a red cell here, which is where the real-model
/// table learns it owes the cell a row.
const DECODE_INERT_SPELLINGS: usize = 18;

/// The complement: spellings whose `exit_prefill` arm runs and whose bytes a
/// served decode reads. [`PREFILL_PINS`] is their pin table.
const MATERIALISING_SPELLINGS: usize = 10;

/// The two populations partition every spelling, and each names what covers it.
///
/// The inert side is what a served capture cannot judge: its store is empty
/// past `exit_prefill`, so the pin table above is its only oracle. The
/// materialising side runs a live bulk-encode arm, and the prefill pin table
/// is that arm's only oracle. The split is derived from the disposition
/// predicate, so a codec crossing it turns a cell red here instead of quietly
/// changing what the real-model run is worth.
#[test]
fn the_two_populations_partition_every_spelling() {
    let (live, inert): (Vec<KvQuant>, Vec<KvQuant>) = ALL_KV_QUANTS
        .iter()
        .copied()
        .partition(KvQuant::materialises_packed_store);
    let inert: Vec<String> = inert.iter().map(ToString::to_string).collect();
    let live: Vec<String> = live.iter().map(ToString::to_string).collect();
    assert_eq!(
        inert.len(),
        DECODE_INERT_SPELLINGS,
        "the set of spellings whose packed store no served capture can read moved: {inert:?}. \
         A spelling that left the class owes the real-model table a row; one that entered it \
         owes this file the only coverage it has"
    );
    assert_eq!(
        live.len(),
        MATERIALISING_SPELLINGS,
        "the set of spellings whose exit_prefill arm runs moved: {live:?}. The prefill pin \
         table is that arm's only oracle, and it sweeps this population"
    );
    assert_eq!(
        inert.len() + live.len(),
        SPELLING_COUNT,
        "the two populations no longer cover every spelling"
    );
}

/// No spelling builds the paged storage under the process state a test runs in.
///
/// The paged routing reads a process-global the CLI latches once. This file
/// cannot set it and cannot unset it, so the paged arms of every update body
/// are outside its reach — stated as an assertion rather than as prose, so a
/// change that starts routing a spelling there fails here instead of leaving a
/// cell silently unpinned.
#[test]
fn no_spelling_builds_the_paged_storage_in_this_process() {
    let _guard = env_lock();
    for quant in ALL_KV_QUANTS.iter().copied() {
        let cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        assert!(
            !matches!(cache.storage, KvStorage::Paged { .. }),
            "{quant} built the paged storage: its cells here pin the paged path, which the \
             pin values were not captured against"
        );
    }
}
