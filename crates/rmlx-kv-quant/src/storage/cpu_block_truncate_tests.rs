//! Mid-sequence truncation of the CPU-side turbo / planar / affine KV stores.
//!
//! These stores accumulate their CPU payload independently of `shape[2]` — a
//! `Vec<TurboBlocks>`, a `Vec<PlanarBlocks>`, or (for `QuantK`) a flat
//! append-only `codes`/`scales` pair. Lowering `shape[2]` alone therefore does
//! **not** roll them back: the next `append` stacks on top of the rejected
//! tokens and the dequant reads a prefix of the stale buffer, so attention runs
//! over tokens the verifier rejected while the accepted correction is missing.
//! Nothing errors — the whole point of these tests is that "it did not error"
//! was the failure mode.
//!
//! Every oracle here is a **reference store built from only the retained
//! tokens**: same type, same shape, same codec, fed the accepted prefix and the
//! correction as its own appends. It shares no arithmetic with the truncation
//! logic, which never touches payload values.
//!
//! `head_dim` is 64 for the turbo / planar stores (a multiple of the 32-element
//! quant group, so a group never straddles a row and the split block is
//! bit-identical to a freshly encoded one) and 128 for `QuantK` (its q8 group is
//! 128 wide).
//!
//! Covered here: every `truncate_plan` branch — mid-block split, block-aligned
//! drop, zero target, exact-length no-op, and a target past the stored length
//! (clamped) — plus both refusals, `KvStorage::truncate_to` across five arms and
//! `KvStorage::reset`.
//!
//! What these cannot reach: the `b > 1` split (refused by design — only the
//! loud-abort contract is pinned below), the flat GPU buffers (which need no cut
//! and are not exercised without Metal), TCQ-encoded blocks (`use_tcq` blocks are
//! `TurboBlocks`, so the cut machinery is identical, but no test here drives the
//! Viterbi encoder), and anything above the store: the post-hydrate decode path
//! lives in `rmlx-kv-ssd/src/hydrate_tests.rs`, and the end-to-end speculative
//! round in `rmlx-server` is not covered by any test.

use rmlx_mlx::{zeros, Array, Device, Dtype};

use super::{QuantK, QuantKTurbo3, QuantKTurbo4, QuantPlanarK, QuantPlanarV, QuantV};

const MAX_SEQ: i32 = 512;

/// Head-major `[b, kv_h, n_tok, d]` chunk covering positions
/// `[first_tok, first_tok + n_tok)` — the layout every `append` receives.
///
/// `tag` separates the values a rejected draft token carries from the ones its
/// accepted replacement carries, by a margin far wider than quant noise, so a
/// store that kept the wrong one cannot come out looking right by accident.
fn chunk(b: usize, kv_h: usize, d: usize, first_tok: usize, n_tok: usize, tag: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * kv_h * n_tok * d];
    for bi in 0..b {
        for h in 0..kv_h {
            for t in 0..n_tok {
                for dd in 0..d {
                    let v = (tag as f32).mul_add(37.0, (first_tok + t) as f32 * 3.0)
                        + (h as f32) * 0.5
                        + ((dd % 7) as f32) * 0.125
                        + (bi as f32) * 11.0
                        + 1.0;
                    let idx = ((bi * kv_h + h) * n_tok + t) * d + dd;
                    if let Some(slot) = out.get_mut(idx) {
                        *slot = v;
                    }
                }
            }
        }
    }
    out
}

#[allow(
    clippy::expect_used,
    reason = "test: a zero-filled CPU array of a fixed in-bounds shape cannot fail to allocate"
)]
fn dummy(b: usize, kv_h: usize, n_tok: usize, d: usize) -> Array {
    zeros(
        &[b as i32, kv_h as i32, n_tok as i32, d as i32],
        Dtype::F32,
        Device::Cpu,
    )
    .expect("dummy array")
}

fn shape_of(b: usize, kv_h: usize, n_tok: usize, d: usize) -> [i32; 4] {
    [b as i32, kv_h as i32, n_tok as i32, d as i32]
}

fn init_shape(b: usize, kv_h: usize, d: usize) -> Vec<i32> {
    vec![b as i32, kv_h as i32, 0, d as i32]
}

// ── Per-store adapters ────────────────────────────────────────────────────────
//
// Each store has the same `append(f32_data, new_shape, arr, device, max_seq)` /
// `dequantize_choice(device, dtype)` pair but no common trait, so the scenario
// below is spelled out once per store rather than hidden behind one.

macro_rules! append_cpu {
    ($store:expr, $b:expr, $kv_h:expr, $d:expr, $first:expr, $n:expr, $tag:expr) => {{
        let data = chunk($b, $kv_h, $d, $first, $n, $tag);
        let arr = dummy($b, $kv_h, $n, $d);
        $store
            .append(
                &data,
                &shape_of($b, $kv_h, $n, $d),
                &arr,
                Device::Cpu,
                MAX_SEQ,
            )
            .expect("cpu append")
    }};
}

// ── QuantV (TurboQuant V blocks) ──────────────────────────────────────────────

/// A speculative partial accept must leave the store holding exactly the
/// accepted tokens plus the correction — not the rejected draft.
///
/// Sequence: append 1 prompt token, append a 4-token draft chunk, accept 2 of
/// it (`truncate_to(3)`), append the 2-token correction. `shape[2]` is back to
/// 5, and before the fix the blocks held 7 positions; `dequantize_choice` cut
/// that back to the declared 5 with `out.resize(total, 0.0)` and returned the
/// **original** five tokens — the two rejected draft tokens included, the
/// correction dropped, no error.
///
/// Mutation check: delete the `apply_truncate_plan` call in
/// `QuantV::truncate_to` (leaving only `shape[2] = n`) and the store decodes to
/// 7 positions against a declared 5, so the new coverage check in
/// `dequantize_choice` fires and `expect` panics. Restore the old
/// `out.resize(total, 0.0)` alongside it and the `assert_eq!` against the
/// reference fails instead, on the `stale` values the third assertion names.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_v_partial_accept_keeps_only_accepted_tokens() {
    let d = 64;
    for kv_h in [1_usize, 3_usize] {
        let mut store = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        append_cpu!(store, 1, kv_h, d, 0, 1, 0);
        append_cpu!(store, 1, kv_h, d, 1, 4, 0);
        assert_eq!(store.shape[2], 5);

        store.truncate_to(3);
        assert_eq!(store.shape[2], 3, "shape[2] lowered (kv_h={kv_h})");
        append_cpu!(store, 1, kv_h, d, 3, 2, 1);
        assert_eq!(store.shape[2], 5);

        let (decoded, arr) = store
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("dequant after a partial accept");
        assert!(arr.is_none(), "CPU dequant returns a flat vec");

        let mut reference = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        append_cpu!(reference, 1, kv_h, d, 0, 1, 0);
        append_cpu!(reference, 1, kv_h, d, 1, 2, 0);
        append_cpu!(reference, 1, kv_h, d, 3, 2, 1);
        let (expected, _) = reference
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("reference dequant");
        assert_eq!(
            decoded, expected,
            "the store must hold exactly the accepted prefix plus the correction (kv_h={kv_h})"
        );

        // Name the wrong answer explicitly: the pre-fix store returned the
        // original five draft tokens.
        let mut stale = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        append_cpu!(stale, 1, kv_h, d, 0, 1, 0);
        append_cpu!(stale, 1, kv_h, d, 1, 4, 0);
        let (stale_decoded, _) = stale
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("stale dequant");
        assert_ne!(
            decoded, stale_decoded,
            "premise: the rejected draft must decode differently from the correction, or this \
             test cannot tell them apart (kv_h={kv_h})"
        );
    }
}

/// A cut that lands on a block boundary keeps whole blocks and needs no split.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_v_block_aligned_truncate_drops_whole_blocks() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut store = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(store, 1, kv_h, d, 0, 2, 0);
    append_cpu!(store, 1, kv_h, d, 2, 3, 0);
    store.truncate_to(2);
    assert_eq!(store.blocks.len(), 1, "the trailing block is dropped whole");

    append_cpu!(store, 1, kv_h, d, 2, 1, 1);
    let (decoded, _) = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");

    let mut reference = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(reference, 1, kv_h, d, 0, 2, 0);
    append_cpu!(reference, 1, kv_h, d, 2, 1, 1);
    let (expected, _) = reference
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("reference dequant");
    assert_eq!(decoded, expected);
}

/// The row reading must not depend on which axis order the block's
/// `original_shape` happens to record.
///
/// The CPU append paths label a block with the sequence-major chunk shape
/// `[B, S_block, kv_h, D]`; the SSD hydrate paths label the identical
/// sequence-major bytes with the store's head-major `[B, kv_h, S, D]`. Only the
/// product is ever read back, which is why `block_rows` multiplies the leading
/// three axes instead of naming one of them.
///
/// Mutation check — the one this case uniquely catches: read the sequence
/// extent as `original_shape[1]` and scale it by the store's `kv_h`. That is
/// right for an append-labelled block (`[B, S_block, kv_h, D]`) and wrong for a
/// hydrate-labelled one (`[B, kv_h, S, D]`, where index 1 is the head count), so
/// every other test in this file stays green and only this one goes red.
/// (Dropping `apply_truncate_plan` outright also fails this case, but it fails
/// the partial-accept cases too — that is not what this one is here to pin.)
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_v_truncate_reads_rows_from_the_shape_product() {
    let (kv_h, d) = (3_usize, 64_usize);
    let mut store = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(store, 1, kv_h, d, 0, 5, 0);
    // Re-label the single block the way SSD hydrate does: head-major store
    // shape over the same sequence-major bytes.
    store.blocks.first_mut().expect("one block").original_shape = shape_of(1, kv_h, 5, d);

    store.truncate_to(2);
    append_cpu!(store, 1, kv_h, d, 2, 1, 1);
    let (decoded, _) = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");

    let mut reference = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(reference, 1, kv_h, d, 0, 2, 0);
    append_cpu!(reference, 1, kv_h, d, 2, 1, 1);
    let (expected, _) = reference
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("reference dequant");
    assert_eq!(
        decoded, expected,
        "a head-major-labelled block must still cut at the right sequence position"
    );
}

/// At `b > 1` a mid-block cut must stay **loud**, not become silently wrong.
///
/// Rows run batch-major, so a sequence prefix is not a row prefix and a split
/// store would decode scrambled with no error — the planner refuses and drops
/// the block whole, which leaves the store short of `shape[2]`, which the
/// coverage check turns into an abort.
///
/// Mutation check: drop the `b != 1` arm in `truncate_plan` and the store
/// splits, the coverage check passes, and `expect_err` goes RED on scrambled
/// values. Delete the coverage check and it goes RED on a zero-padded `Ok`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_v_truncate_at_b_gt_1_stays_loud() {
    let (b, kv_h, d) = (2_usize, 2_usize, 64_usize);
    let mut store = QuantV::new_affine_decode(init_shape(b, kv_h, d), 4, MAX_SEQ);
    append_cpu!(store, b, kv_h, d, 0, 2, 0);
    append_cpu!(store, b, kv_h, d, 2, 3, 0);
    store.truncate_to(3);
    assert_eq!(
        store.blocks.len(),
        1,
        "the trailing block is dropped whole at b > 1"
    );
    let err = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect_err("a b > 1 mid-block cut must abort, not decode a short store");
    assert!(
        err.to_string().contains("refusing to zero-pad / truncate"),
        "the abort must be the block-coverage error, got: {err}"
    );
}

// ── QuantKTurbo3 / QuantKTurbo4 (TurboQuant K blocks) ─────────────────────────

/// Same partial-accept scenario on the symmetric-turbo K stores. Both share
/// `TurboBlocks` with `QuantV`, so this pins that the K-side wiring routes
/// through the same cut rather than keeping its own bare `shape[2] = n`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_k_turbo3_partial_accept_keeps_only_accepted_tokens() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut store = QuantKTurbo3::new(init_shape(1, kv_h, d), MAX_SEQ);
    append_cpu!(store, 1, kv_h, d, 0, 1, 0);
    append_cpu!(store, 1, kv_h, d, 1, 4, 0);
    store.truncate_to(3);
    append_cpu!(store, 1, kv_h, d, 3, 2, 1);

    let (decoded, _) = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant after a partial accept");

    let mut reference = QuantKTurbo3::new(init_shape(1, kv_h, d), MAX_SEQ);
    append_cpu!(reference, 1, kv_h, d, 0, 1, 0);
    append_cpu!(reference, 1, kv_h, d, 1, 2, 0);
    append_cpu!(reference, 1, kv_h, d, 3, 2, 1);
    let (expected, _) = reference
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("reference dequant");
    assert_eq!(decoded, expected);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_k_turbo4_partial_accept_keeps_only_accepted_tokens() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut store = QuantKTurbo4::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
    append_cpu!(store, 1, kv_h, d, 0, 1, 0);
    append_cpu!(store, 1, kv_h, d, 1, 4, 0);
    store.truncate_to(3);
    append_cpu!(store, 1, kv_h, d, 3, 2, 1);

    let (decoded, _) = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant after a partial accept");

    let mut reference = QuantKTurbo4::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
    append_cpu!(reference, 1, kv_h, d, 0, 1, 0);
    append_cpu!(reference, 1, kv_h, d, 1, 2, 0);
    append_cpu!(reference, 1, kv_h, d, 3, 2, 1);
    let (expected, _) = reference
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("reference dequant");
    assert_eq!(decoded, expected);
}

// ── QuantPlanarK / QuantPlanarV (PlanarQuant blocks) ──────────────────────────

/// Same partial-accept scenario on the planar K store.
///
/// Planar has no `resize` backstop — `transpose_seq_heads` reads the first
/// `b * s * kv_h * d` elements and ignores the rest — so the pre-fix failure
/// was the same silent prefix read with one fewer line of code involved.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_planar_k_partial_accept_keeps_only_accepted_tokens() {
    let d = 64;
    for kv_h in [1_usize, 3_usize] {
        let mut store = QuantPlanarK::new(init_shape(1, kv_h, d), MAX_SEQ);
        append_cpu!(store, 1, kv_h, d, 0, 1, 0);
        append_cpu!(store, 1, kv_h, d, 1, 4, 0);
        store.truncate_to(3);
        append_cpu!(store, 1, kv_h, d, 3, 2, 1);

        let (decoded, _) = store
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("dequant after a partial accept");

        let mut reference = QuantPlanarK::new(init_shape(1, kv_h, d), MAX_SEQ);
        append_cpu!(reference, 1, kv_h, d, 0, 1, 0);
        append_cpu!(reference, 1, kv_h, d, 1, 2, 0);
        append_cpu!(reference, 1, kv_h, d, 3, 2, 1);
        let (expected, _) = reference
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("reference dequant");
        assert_eq!(
            decoded, expected,
            "the store must hold exactly the accepted prefix plus the correction (kv_h={kv_h})"
        );
    }
}

/// `--kv-quant planar` is an arch default, so the V side gets the same pin.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_planar_v_partial_accept_keeps_only_accepted_tokens() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut store = QuantPlanarV::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
    append_cpu!(store, 1, kv_h, d, 0, 1, 0);
    append_cpu!(store, 1, kv_h, d, 1, 4, 0);
    store.truncate_to(3);
    append_cpu!(store, 1, kv_h, d, 3, 2, 1);

    let (decoded, _) = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant after a partial accept");

    let mut reference = QuantPlanarV::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
    append_cpu!(reference, 1, kv_h, d, 0, 1, 0);
    append_cpu!(reference, 1, kv_h, d, 1, 2, 0);
    append_cpu!(reference, 1, kv_h, d, 3, 2, 1);
    let (expected, _) = reference
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("reference dequant");
    assert_eq!(decoded, expected);
}

/// The planar refusal path aborts instead of reading a short buffer — which
/// used to be an out-of-range panic inside `transpose_seq_heads`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_planar_k_truncate_at_b_gt_1_stays_loud() {
    let (b, kv_h, d) = (2_usize, 2_usize, 64_usize);
    let mut store = QuantPlanarK::new(init_shape(b, kv_h, d), MAX_SEQ);
    append_cpu!(store, b, kv_h, d, 0, 2, 0);
    append_cpu!(store, b, kv_h, d, 2, 3, 0);
    store.truncate_to(3);
    let err = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect_err("a b > 1 mid-block cut must abort, not decode a short store");
    assert!(
        err.to_string().contains("refusing to zero-pad / truncate"),
        "the abort must be the block-coverage error, got: {err}"
    );
}

// ── QuantK (affine q8_0, flat CPU codes) ──────────────────────────────────────

/// `QuantK` keeps no blocks — its CPU payload is one flat append-only
/// `codes`/`scales` pair — but it fails the same way: the next `append` lands
/// past the rejected tokens and the dequant returns a prefix of the stale
/// buffer.
///
/// Mutation check: delete the `retain_cpu_prefix` call in
/// `QuantK::truncate_to` and the codes cover 7 positions against a declared 5,
/// so the coverage check fires and `expect` panics. Delete the coverage check
/// too and the `assert_eq!` fails on the stale prefix.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_k_partial_accept_keeps_only_accepted_tokens() {
    let d = 128;
    for kv_h in [1_usize, 3_usize] {
        let mut store = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
        append_cpu!(store, 1, kv_h, d, 0, 1, 0);
        append_cpu!(store, 1, kv_h, d, 1, 4, 0);
        store.truncate_to(3);
        assert_eq!(
            store.codes.len(),
            3 * kv_h * d,
            "CPU codes cut to the accepted prefix (kv_h={kv_h})"
        );
        append_cpu!(store, 1, kv_h, d, 3, 2, 1);

        let (decoded, _) = store
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("dequant after a partial accept");

        let mut reference = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
        append_cpu!(reference, 1, kv_h, d, 0, 1, 0);
        append_cpu!(reference, 1, kv_h, d, 1, 2, 0);
        append_cpu!(reference, 1, kv_h, d, 3, 2, 1);
        let (expected, _) = reference
            .dequantize_choice(Device::Cpu, Dtype::F32)
            .expect("reference dequant");
        assert_eq!(
            decoded, expected,
            "the store must hold exactly the accepted prefix plus the correction (kv_h={kv_h})"
        );
    }
}

/// A cut that lands inside a q8 group cannot be expressed — one scale covers
/// `Q8_GROUP_SIZE` elements and the f32 source is gone — so it is refused and
/// the store is left over-covering on purpose, which the coverage check turns
/// into an abort.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_k_truncate_inside_a_q8_group_stays_loud() {
    // kv_h * d == 64 elements per sequence position, so an odd target lands
    // halfway through a 128-element group.
    let (kv_h, d) = (1_usize, 64_usize);
    let mut store = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
    append_cpu!(store, 1, kv_h, d, 0, 4, 0);
    store.truncate_to(3);
    assert_eq!(
        store.codes.len(),
        4 * kv_h * d,
        "the refused cut leaves the codes untouched"
    );
    let err = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect_err("an unexpressible cut must abort, not decode a stale prefix");
    assert!(
        err.to_string().contains("refusing to zero-pad / truncate"),
        "the abort must be the code-coverage error, got: {err}"
    );

    // The same store cut to an even target is expressible and stays quiet.
    let mut ok_store = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
    append_cpu!(ok_store, 1, kv_h, d, 0, 4, 0);
    ok_store.truncate_to(2);
    assert_eq!(ok_store.codes.len(), 2 * kv_h * d);
    ok_store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an aligned cut decodes");
}

/// `b > 1` is refused for the flat store for the same batch-major reason as the
/// block stores, and stays loud the same way.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn quant_k_truncate_at_b_gt_1_stays_loud() {
    let (b, kv_h, d) = (2_usize, 2_usize, 128_usize);
    let mut store = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(b, kv_h, d));
    append_cpu!(store, b, kv_h, d, 0, 4, 0);
    store.truncate_to(3);
    let err = store
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect_err("a b > 1 cut must abort, not decode a batch-interleaved prefix");
    assert!(
        err.to_string().contains("refusing to zero-pad / truncate"),
        "the abort must be the code-coverage error, got: {err}"
    );
}

// ── Truncation is monotone-decreasing ─────────────────────────────────────────

/// A target past what the store covers must not raise `shape[2]`.
///
/// It is reachable, not hypothetical: post-`exit_prefill` the codec store is
/// frozen at the prefill length while `KvCache::offset` keeps advancing on the
/// bf16 decode mirror, so a speculative rollback to a position inside the decode
/// window arrives with `n > shape[2]`. Raising the shape to meet it invents
/// coverage no payload backs — the dequant then reads past the blocks and an SSD
/// spill persists a header claiming more tokens than its bytes hold.
///
/// Mutation check: drop the `.min(covered)` in `clamp_truncate_target` and both
/// stores report `shape[2] == 9` with 5 positions of payload, so the `assert_eq!`
/// on the shape fails and the following `dequantize_choice` returns `Err`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn truncate_past_the_stored_length_is_a_no_op() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut v = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    let mut k = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
    append_cpu!(v, 1, kv_h, d, 0, 5, 0);
    append_cpu!(k, 1, kv_h, d, 0, 5, 0);
    let (v_before, _) = v
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant before");

    v.truncate_to(9);
    k.truncate_to(9);

    assert_eq!(v.shape[2], 5, "V shape must not grow past its payload");
    assert_eq!(k.shape[2], 5, "K shape must not grow past its payload");
    let (v_after, _) = v
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an over-long target must leave a decodable store");
    assert_eq!(v_before, v_after, "the store is unchanged");
    k.dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an over-long target must leave a decodable store");
}

/// The two `truncate_plan` branches the partial-accept cases never take: a zero
/// target (drop everything) and an exact-length target (keep everything).
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn truncate_to_zero_and_to_the_exact_length_are_both_clean() {
    let (kv_h, d) = (2_usize, 64_usize);

    let mut zeroed = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(zeroed, 1, kv_h, d, 0, 2, 0);
    append_cpu!(zeroed, 1, kv_h, d, 2, 3, 0);
    zeroed.truncate_to(0);
    assert_eq!(zeroed.shape[2], 0);
    assert!(zeroed.blocks.is_empty(), "every block dropped");
    let (flat, _) = zeroed
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an emptied store decodes to nothing");
    assert!(flat.is_empty());
    // And it is reusable: the next append starts from position 0.
    append_cpu!(zeroed, 1, kv_h, d, 0, 1, 1);
    let (reused, _) = zeroed
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant after reuse");
    let mut fresh = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(fresh, 1, kv_h, d, 0, 1, 1);
    let (fresh_flat, _) = fresh
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("fresh dequant");
    assert_eq!(reused, fresh_flat);

    let mut exact = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(exact, 1, kv_h, d, 0, 2, 0);
    append_cpu!(exact, 1, kv_h, d, 2, 3, 0);
    let (before, _) = exact
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant before");
    exact.truncate_to(5);
    assert_eq!(exact.blocks.len(), 2, "both blocks kept whole");
    let (after, _) = exact
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant after");
    assert_eq!(before, after);
}

// ── KvStorage dispatch ────────────────────────────────────────────────────────

/// Decode a `(K, V)` store pair and assert both match a reference pair.
///
/// The oracle is the reference store, never a recomputation of the cut — an
/// assertion on block row counts would share `block_rows` with the code under
/// test and pass any mutation that scales the cut and the reading together.
fn assert_same_decode(subject: (&[f32], &[f32]), reference: (&[f32], &[f32]), label: &str) {
    assert!(
        !subject.0.is_empty() && !subject.1.is_empty(),
        "{label}: empty"
    );
    assert_eq!(subject.0, reference.0, "{label}: K axis");
    assert_eq!(subject.1, reference.1, "{label}: V axis");
}

/// `KvStorage::truncate_to` routes its arms through the store-level cut.
///
/// Before the fix twelve arms set `shape[2]` directly and never touched the CPU
/// payload. Each case below drives the same partial accept through the enum —
/// append, `storage.truncate_to`, append the correction — and compares the
/// decoded K and V against reference stores fed only the retained tokens.
///
/// **Coverage, stated honestly.** Five arms are exercised: `K8V4` (`QuantK` +
/// `QuantV`), `TurboSym3` (`QuantKTurbo3` + 3-bit `QuantV`), `TurboSym4`
/// (`QuantKTurbo4`), `Planar` (`QuantPlanarV`) and `PlanarK` (`QuantPlanarK`,
/// a K-only arm shape). That covers all six **store types**, so a regression in
/// any store's `truncate_to` is caught — but it does **not** cover every enum
/// arm. Reverting any of `K8V8`, `K8VTurbo2`, `K8VTurbo3Tcq`, `K8VTurbo2Tcq`,
/// the K axis of `IsoV3` / `IsoV4` / `RotorV3` / `RotorV4`, or the V axis of
/// `RotorKAsym3` / `RotorKAsym4` to a bare `shape[2] = n` leaves this suite
/// green. Those ten are unpinned by construction: nothing in the workspace
/// drives `KvStorage::truncate_to` on them.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
#[allow(
    clippy::unreachable,
    reason = "test: each storage is matched against the variant it was constructed as; the else arm documents that rather than silently skipping the assertions"
)]
#[allow(
    clippy::too_many_lines,
    reason = "test: five independent enum-arm cases, each self-contained; splitting would hide that they are one contract"
)]
fn kv_storage_truncate_routes_every_arm_through_the_store_cut() {
    let (kv_h, d) = (2_usize, 64_usize);
    // `kv_h * d == 128` — one whole q8 group per sequence position, so the
    // `QuantK` cut is expressible at every target.
    let cpu = Device::Cpu;
    let f32d = Dtype::F32;

    // ── K8V4: QuantK + QuantV(4) ──────────────────────────────────────────
    {
        let mut k = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
        let mut v = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        append_cpu!(k, 1, kv_h, d, 0, 1, 0);
        append_cpu!(v, 1, kv_h, d, 0, 1, 0);
        append_cpu!(k, 1, kv_h, d, 1, 4, 0);
        append_cpu!(v, 1, kv_h, d, 1, 4, 0);
        let mut storage = super::KvStorage::K8V4 {
            k: Some(k),
            v: Some(v),
            max_seq: MAX_SEQ,
        };
        storage.truncate_to(3);
        let super::KvStorage::K8V4 { k, v, .. } = &mut storage else {
            unreachable!("constructed as K8V4")
        };
        let k = k.as_mut().expect("k side");
        let v = v.as_mut().expect("v side");
        append_cpu!(k, 1, kv_h, d, 3, 2, 1);
        append_cpu!(v, 1, kv_h, d, 3, 2, 1);
        let (kd, _) = k.dequantize_choice(cpu, f32d).expect("K dequant");
        let (vd, _) = v.dequantize_choice(cpu, f32d).expect("V dequant");

        let mut rk = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
        let mut rv = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        for (first, n, tag) in [(0, 1, 0), (1, 2, 0), (3, 2, 1)] {
            append_cpu!(rk, 1, kv_h, d, first, n, tag);
            append_cpu!(rv, 1, kv_h, d, first, n, tag);
        }
        let (rkd, _) = rk.dequantize_choice(cpu, f32d).expect("ref K dequant");
        let (rvd, _) = rv.dequantize_choice(cpu, f32d).expect("ref V dequant");
        assert_same_decode((&kd, &vd), (&rkd, &rvd), "K8V4");
    }

    // ── TurboSym3: QuantKTurbo3 + QuantV(3) ───────────────────────────────
    {
        let mut k = QuantKTurbo3::new(init_shape(1, kv_h, d), MAX_SEQ);
        let mut v = QuantV::new_affine_decode(init_shape(1, kv_h, d), 3, MAX_SEQ);
        append_cpu!(k, 1, kv_h, d, 0, 1, 0);
        append_cpu!(v, 1, kv_h, d, 0, 1, 0);
        append_cpu!(k, 1, kv_h, d, 1, 4, 0);
        append_cpu!(v, 1, kv_h, d, 1, 4, 0);
        let mut storage = super::KvStorage::TurboSym3 {
            k: Some(k),
            v: Some(v),
            max_seq: MAX_SEQ,
        };
        storage.truncate_to(3);
        let super::KvStorage::TurboSym3 { k, v, .. } = &mut storage else {
            unreachable!("constructed as TurboSym3")
        };
        let k = k.as_mut().expect("k side");
        let v = v.as_mut().expect("v side");
        append_cpu!(k, 1, kv_h, d, 3, 2, 1);
        append_cpu!(v, 1, kv_h, d, 3, 2, 1);
        let (kd, _) = k.dequantize_choice(cpu, f32d).expect("K dequant");
        let (vd, _) = v.dequantize_choice(cpu, f32d).expect("V dequant");

        let mut rk = QuantKTurbo3::new(init_shape(1, kv_h, d), MAX_SEQ);
        let mut rv = QuantV::new_affine_decode(init_shape(1, kv_h, d), 3, MAX_SEQ);
        for (first, n, tag) in [(0, 1, 0), (1, 2, 0), (3, 2, 1)] {
            append_cpu!(rk, 1, kv_h, d, first, n, tag);
            append_cpu!(rv, 1, kv_h, d, first, n, tag);
        }
        let (rkd, _) = rk.dequantize_choice(cpu, f32d).expect("ref K dequant");
        let (rvd, _) = rv.dequantize_choice(cpu, f32d).expect("ref V dequant");
        assert_same_decode((&kd, &vd), (&rkd, &rvd), "TurboSym3");
    }

    // ── TurboSym4: QuantKTurbo4 + QuantV(4) ───────────────────────────────
    {
        let mut k = QuantKTurbo4::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
        let mut v = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        append_cpu!(k, 1, kv_h, d, 0, 1, 0);
        append_cpu!(v, 1, kv_h, d, 0, 1, 0);
        append_cpu!(k, 1, kv_h, d, 1, 4, 0);
        append_cpu!(v, 1, kv_h, d, 1, 4, 0);
        let mut storage = super::KvStorage::TurboSym4 {
            k: Some(k),
            v: Some(v),
            max_seq: MAX_SEQ,
        };
        storage.truncate_to(3);
        let super::KvStorage::TurboSym4 { k, v, .. } = &mut storage else {
            unreachable!("constructed as TurboSym4")
        };
        let k = k.as_mut().expect("k side");
        let v = v.as_mut().expect("v side");
        append_cpu!(k, 1, kv_h, d, 3, 2, 1);
        append_cpu!(v, 1, kv_h, d, 3, 2, 1);
        let (kd, _) = k.dequantize_choice(cpu, f32d).expect("K dequant");
        let (vd, _) = v.dequantize_choice(cpu, f32d).expect("V dequant");

        let mut rk = QuantKTurbo4::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
        let mut rv = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
        for (first, n, tag) in [(0, 1, 0), (1, 2, 0), (3, 2, 1)] {
            append_cpu!(rk, 1, kv_h, d, first, n, tag);
            append_cpu!(rv, 1, kv_h, d, first, n, tag);
        }
        let (rkd, _) = rk.dequantize_choice(cpu, f32d).expect("ref K dequant");
        let (rvd, _) = rv.dequantize_choice(cpu, f32d).expect("ref V dequant");
        assert_same_decode((&kd, &vd), (&rkd, &rvd), "TurboSym4");
    }

    // ── Planar: QuantK + QuantPlanarV ─────────────────────────────────────
    {
        let mut k = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
        let mut v = QuantPlanarV::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
        append_cpu!(k, 1, kv_h, d, 0, 1, 0);
        append_cpu!(v, 1, kv_h, d, 0, 1, 0);
        append_cpu!(k, 1, kv_h, d, 1, 4, 0);
        append_cpu!(v, 1, kv_h, d, 1, 4, 0);
        let mut storage = super::KvStorage::Planar {
            k: Some(k),
            v: Some(v),
            max_seq: MAX_SEQ,
            bits: 4,
        };
        storage.truncate_to(3);
        let super::KvStorage::Planar { k, v, .. } = &mut storage else {
            unreachable!("constructed as Planar")
        };
        let k = k.as_mut().expect("k side");
        let v = v.as_mut().expect("v side");
        append_cpu!(k, 1, kv_h, d, 3, 2, 1);
        append_cpu!(v, 1, kv_h, d, 3, 2, 1);
        let (kd, _) = k.dequantize_choice(cpu, f32d).expect("K dequant");
        let (vd, _) = v.dequantize_choice(cpu, f32d).expect("V dequant");

        let mut rk = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
        let mut rv = QuantPlanarV::from_cpu_blocks(Vec::new(), init_shape(1, kv_h, d), 4);
        for (first, n, tag) in [(0, 1, 0), (1, 2, 0), (3, 2, 1)] {
            append_cpu!(rk, 1, kv_h, d, first, n, tag);
            append_cpu!(rv, 1, kv_h, d, first, n, tag);
        }
        let (rkd, _) = rk.dequantize_choice(cpu, f32d).expect("ref K dequant");
        let (rvd, _) = rv.dequantize_choice(cpu, f32d).expect("ref V dequant");
        assert_same_decode((&kd, &vd), (&rkd, &rvd), "Planar");
    }

    // ── PlanarK: QuantPlanarK, K-only arm (V is bf16 on the parent) ───────
    {
        let mut k = QuantPlanarK::new(init_shape(1, kv_h, d), MAX_SEQ);
        append_cpu!(k, 1, kv_h, d, 0, 1, 0);
        append_cpu!(k, 1, kv_h, d, 1, 4, 0);
        let mut storage = super::KvStorage::PlanarK {
            k: Some(k),
            max_seq: MAX_SEQ,
        };
        storage.truncate_to(3);
        let super::KvStorage::PlanarK { k, .. } = &mut storage else {
            unreachable!("constructed as PlanarK")
        };
        let k = k.as_mut().expect("k side");
        append_cpu!(k, 1, kv_h, d, 3, 2, 1);
        let (kd, _) = k.dequantize_choice(cpu, f32d).expect("K dequant");

        let mut rk = QuantPlanarK::new(init_shape(1, kv_h, d), MAX_SEQ);
        for (first, n, tag) in [(0, 1, 0), (1, 2, 0), (3, 2, 1)] {
            append_cpu!(rk, 1, kv_h, d, first, n, tag);
        }
        let (rkd, _) = rk.dequantize_choice(cpu, f32d).expect("ref K dequant");
        assert_eq!(kd, rkd, "PlanarK: K axis");
    }
}

/// `KvStorage::reset` is the same defect one screen above `truncate_to`, and it
/// gets the same treatment: it must clear the CPU payload, not just zero
/// `shape[2]`.
///
/// Mutation check: put `vs.shape[2] = 0;` back in the `K8V4` arm of `reset` and
/// the store decodes 5 positions against a declared 1, so `dequantize_choice`
/// returns `Err` and the `expect` panics.
///
/// Coverage: the `K8V4` arm only. `reset` has twenty-seven other arms (28 in
/// total, one per `KvStorage` variant) and nothing
/// drives them, so reverting any of those is not caught here.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
#[allow(
    clippy::unreachable,
    reason = "test: the storage was constructed as K8V4; the else arm documents that rather than silently skipping the assertions"
)]
fn kv_storage_reset_clears_the_cpu_payload() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut k = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
    let mut v = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(k, 1, kv_h, d, 0, 5, 0);
    append_cpu!(v, 1, kv_h, d, 0, 5, 0);
    let mut storage = super::KvStorage::K8V4 {
        k: Some(k),
        v: Some(v),
        max_seq: MAX_SEQ,
    };

    storage.reset();

    let super::KvStorage::K8V4 { k, v, .. } = &mut storage else {
        unreachable!("constructed as K8V4")
    };
    let k = k.as_mut().expect("k side");
    let v = v.as_mut().expect("v side");
    assert!(k.codes.is_empty(), "K codes cleared");
    assert!(v.blocks.is_empty(), "V blocks cleared");

    // A reset store must behave like a fresh one for the next request.
    append_cpu!(k, 1, kv_h, d, 0, 1, 1);
    append_cpu!(v, 1, kv_h, d, 0, 1, 1);
    let (kd, _) = k
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("K dequant after reset");
    let (vd, _) = v
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("V dequant after reset");

    let mut rk = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(1, kv_h, d));
    let mut rv = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(rk, 1, kv_h, d, 0, 1, 1);
    append_cpu!(rv, 1, kv_h, d, 0, 1, 1);
    let (rkd, _) = rk
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("ref K dequant");
    let (rvd, _) = rv
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("ref V dequant");
    assert_same_decode((&kd, &vd), (&rkd, &rvd), "reset");
}

/// A clear must clear, even at `b > 1` where every ordinary cut is refused.
///
/// `QuantK`'s refusals exist to stop a wrong prefix being kept. A zero target
/// keeps no prefix, so no refusal can be correct there — and since
/// `KvStorage::reset` routes through `truncate_to(0)`, refusing would leave
/// `codes` populated under `shape[2] == 0`: permanently un-decodable, and
/// asymmetric with the block stores, whose planner drops every block regardless
/// of `b`.
///
/// Mutation check: move the `n <= 0` clear in `retain_cpu_prefix` below the
/// `b != 1` refusal and the K codes survive at full length while the V blocks
/// are gone, so the `is_empty` assertion fails and the dequant returns `Err`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
fn truncate_to_zero_clears_even_where_a_cut_would_be_refused() {
    let (b, kv_h, d) = (2_usize, 2_usize, 64_usize);
    let mut k = QuantK::from_cpu_parts(Vec::new(), Vec::new(), init_shape(b, kv_h, d));
    let mut v = QuantV::new_affine_decode(init_shape(b, kv_h, d), 4, MAX_SEQ);
    append_cpu!(k, b, kv_h, d, 0, 4, 0);
    append_cpu!(v, b, kv_h, d, 0, 4, 0);

    k.truncate_to(0);
    v.truncate_to(0);

    assert!(k.codes.is_empty(), "K codes cleared at b > 1");
    assert!(k.scales.is_empty(), "K scales cleared at b > 1");
    assert!(v.blocks.is_empty(), "V blocks cleared at b > 1");
    k.dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an emptied K store decodes to nothing");
    v.dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an emptied V store decodes to nothing");
}

/// `RotKTq4V` resets its V store to zero positions rather than to `n`, and now
/// routes that through `truncate_to(0)` so the blocks go with it.
///
/// This is a real behaviour change: the arm previously zeroed `shape[2]` and
/// left the blocks in place — the defect mechanism verbatim, one arm over.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: append/dequant of fixed in-bounds fixtures cannot fail; expect documents that"
)]
#[allow(
    clippy::unreachable,
    reason = "test: the storage was constructed as RotKTq4V; the else arm documents that rather than silently skipping the assertions"
)]
fn kv_storage_rot_k_tq4v_truncate_clears_the_v_blocks() {
    let (kv_h, d) = (2_usize, 64_usize);
    let mut v = QuantV::new_affine_decode(init_shape(1, kv_h, d), 4, MAX_SEQ);
    append_cpu!(v, 1, kv_h, d, 0, 5, 0);
    let mut storage = super::KvStorage::RotKTq4V {
        k_state: crate::mixed_quant::MixedKvState::new(8, 4, 64, 32),
        v: Some(v),
        max_seq: MAX_SEQ,
    };

    storage.truncate_to(3);

    let super::KvStorage::RotKTq4V { v, .. } = &mut storage else {
        unreachable!("constructed as RotKTq4V")
    };
    let v = v.as_mut().expect("v side");
    assert_eq!(v.shape[2], 0, "the V axis resets to zero, not to n");
    assert!(
        v.blocks.is_empty(),
        "the blocks go with it — leaving them is the defect this file exists for"
    );
    v.dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("an emptied store decodes to nothing");
}
