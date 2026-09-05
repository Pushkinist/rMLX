//! Unit tests for [`QuantRotorK3`].
//!
//! Mirror of `quant_iso_k_tests.rs` adapted to the rotor3 K codec. The
//! `_with_qjl` / `_no_qjl` variants exercise both QJL branches.
#![allow(unsafe_code)]

use crate::clifford::make_rotor_table;
use crate::rotorquant::{n_groups_for, rotor3_decode, rotor3_encode};
use crate::storage::quant_rotor_k3::{QuantRotorK3, ROTOR3_K_BITS};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

#[test]
fn quant_rotor_k3_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let q = QuantRotorK3::new(init_shape.clone(), 0);
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.bits, ROTOR3_K_BITS);
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert!(!q.use_qjl(), "use_qjl false before first append");
}

#[test]
fn quant_rotor_k3_roundtrip_no_qjl_matches_v_side() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 9; // n_groups = 3, exact
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![b as i32, kv_h as i32, 0_i32, head_dim as i32], 0);
    qk.append(&data, &new_shape).expect("append");
    assert_eq!(qk.blocks.len(), 1);
    assert_eq!(qk.shape[2], n_seq as i32);
    assert!(!qk.use_qjl(), "QJL off via env");
    let decoded = qk.dequant().expect("dequant");

    // Reference: V-side rotor3 codec produces the same output (codec is
    // axis-agnostic when QJL is off).
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(0, 0, n_groups);
    let (ref_codes, ref_scales, ref_norms) = rotor3_encode(&data, &rotors, head_dim).unwrap();
    let reference = rotor3_decode(&ref_codes, &ref_scales, &ref_norms, &rotors, head_dim).unwrap();

    assert_eq!(decoded.len(), reference.len());
    let max_abs_err = decoded
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err < 1e-5,
        "rotor3_k (no QJL) vs rotor3 V: max_abs_err = {max_abs_err:.6} (>= 1e-5)"
    );

    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// The `blocks == shape[2]` invariant is enforced loudly, never zero-padded.
///
/// A store whose CPU `blocks` cover fewer tokens than `shape[2]` claims, with
/// no GPU ring to supply the tail, is the forbidden state (the naive "skip the
/// per-step download" bug). Both `dequant()` and `try_deep_clone()` must reject
/// it with an `Error` rather than fabricate a zeroed gap (`dequant`) or persist
/// a truncated store (`try_deep_clone` — the SSD spill / prompt-cache clone).
///
/// Mutation check: revert `dequant()` to decode `self.blocks` + `out.resize(_,
/// 0.0)` (the old zero-pad), or drop the `synced_rotor_k_blocks` reconcile —
/// then `dequant()` returns `Ok` with a zeroed tail and this assertion flips
/// RED.
#[test]
fn quant_rotor_k3_short_blocks_without_ring_is_loud_not_zero_padded() {
    let kv_h = 2_i32;
    let head_dim = 9_i32; // n_groups = 3, exact
    let data = lcg_data((kv_h * 2 * head_dim) as usize, TEST_SEED);

    // A real single block covering 2 tokens (QJL off via env-independent seed).
    let mut src = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    src.append(&data, &[1, kv_h, 2, head_dim]).expect("append");
    assert_eq!(src.blocks.len(), 1, "one block for the 2-token append");

    // Reassemble a store that *claims* shape[2] == 4 while its blocks cover only
    // 2 tokens and no ring exists — the ring-only tail with the ring missing.
    let truncated = QuantRotorK3::from_cpu_blocks(
        src.rotors.clone(),
        None,
        src.blocks.clone(),
        vec![1, kv_h, 4, head_dim],
        0,
    );
    assert!(
        !truncated.gpu.is_allocated(),
        "precondition: no ring to cover the tail"
    );

    let dq = truncated.dequant();
    assert!(
        dq.is_err(),
        "dequant must reject a short-blocks/no-ring store loudly, not zero-pad the tail; \
         got Ok(len={:?})",
        dq.map(|v| v.len())
    );
    assert!(
        truncated.try_deep_clone().is_err(),
        "try_deep_clone must refuse to materialise a truncated store (SSD spill / snapshot)"
    );
}

/// Encode + decode run under whatever QJL setting is ambient.
///
/// The name used to say "default on", which is false — `crate::rotor_qjl`
/// defaults **off** because the sideband has no MSL kernel and forces the whole
/// rotor K path onto the CPU. The body never asserted a default either; the
/// toggle itself is covered by the lift test below.
#[test]
fn quant_rotor_k3_roundtrip_under_ambient_qjl_setting() {
    let _guard = crate::test_utils::env_lock();
    // No env override: whatever the process default resolves to.
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };

    let head_dim = 9;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![1, 1, 0, head_dim as i32], 0);
    qk.append(&data, &new_shape).unwrap();
    // The CLI may have installed QJL=on or QJL=off at startup; only check that
    // the encoder/decoder ran without panic. The QJL toggle lift is verified
    // by the dedicated lift test below.
    let _decoded = qk.dequant().unwrap();
    assert_eq!(qk.blocks.len(), 1);
}

#[test]
fn quant_rotor_k3_reset_clears_seq() {
    let head_dim = 9;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![1, 1, 0, head_dim as i32], 0);
    qk.append(&data, &new_shape).unwrap();
    assert_eq!(qk.shape[2], n_seq as i32);

    qk.reset();
    assert_eq!(qk.shape[2], 0);
    assert!(qk.blocks.is_empty());
}

/// Empirical cosine floor for the rotor3_k codec at head_dim=128, QJL off.
/// Matches V-side rotor3 (axis-agnostic) at this seed/shape: measured ≈
/// 0.985. Gate at 0.97 (measured − 0.015 floor, accommodates LCG drift +
/// the rotor3 single-codebook simplification noise).
#[test]
fn quant_rotor_k3_cosine_empirical_floor_head_dim_128_no_qjl() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![1, 1, 0, head_dim as i32], 0);
    qk.append(&data, &new_shape).unwrap();
    let decoded = qk.dequant().unwrap();
    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    assert!(
        stats.min >= 0.65,
        "rotor3_k cosine min={:.6} below empirical floor 0.65",
        stats.min
    );
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant), with the QJL
/// sideband ON. The static rotor table and QJL projection are
/// group/projection-keyed (not token); the per-token QJL sideband (qjl_codes /
/// qjl_norms) reorders with the token rows. Per-(head, token, dim) distinct
/// values surface any head transposition as a large error.
#[test]
fn quant_rotor_k3_multi_append_matches_single_shot_gqa_with_qjl() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };

    let kv_h = 3_usize;
    let head_dim = 128_usize;
    let chunk_a = 2_usize;
    let chunk_b = 3_usize;
    let s_total = chunk_a + chunk_b;
    let val = |h: usize, s: usize, d: usize| {
        (h as f32) * 100.0 + (s as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };
    let build = |s_lo: usize, s_hi: usize| -> Vec<f32> {
        let s = s_hi - s_lo;
        let mut out = vec![0.0_f32; kv_h * s * head_dim];
        for h in 0..kv_h {
            for si in 0..s {
                for d in 0..head_dim {
                    out[(h * s + si) * head_dim + d] = val(h, s_lo + si, d);
                }
            }
        }
        out
    };
    let mut qref = QuantRotorK3::new(vec![1, kv_h as i32, 0, head_dim as i32], 7);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .unwrap();
    assert!(qref.use_qjl(), "QJL must be ON for this test");
    let reference = qref.dequant().unwrap();

    let mut qv = QuantRotorK3::new(vec![1, kv_h as i32, 0, head_dim as i32], 7);
    qv.append(
        &build(0, chunk_a),
        &[1, kv_h as i32, chunk_a as i32, head_dim as i32],
    )
    .unwrap();
    qv.append(
        &build(chunk_a, s_total),
        &[1, kv_h as i32, chunk_b as i32, head_dim as i32],
    )
    .unwrap();
    let multi = qv.dequant().unwrap();

    assert_eq!(multi.len(), reference.len());
    let max_abs = multi
        .iter()
        .zip(reference.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_abs < 1.0,
        "rotor3_k (QJL on) multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

// ── GPU ring lifecycle ────────────────────────────────────────────────────────
//
// `reset` / `truncate_to` shorten `blocks` + `shape[2]` while a longer GPU ring
// may still be live. The ring must be dropped, not left behind: a stale ring is
// longer than the store claims, so the next `gpu_append` takes `prev_seq` from
// the (shorter) shape and writes into the middle of a ring whose tail still
// holds the truncated tokens — `packed_view` would then hand the kernel stale
// keys past the truncation point. Silent wrong answer, no error.

/// Provisioned window for the ring-lifecycle tests; larger than any chunk they append.
const RING_TEST_MAX_SEQ: i32 = 64;

#[allow(
    clippy::expect_used,
    reason = "test helper: a failure here is the assertion"
)]
fn seed_ring_via_gpu_append(
    ks: &mut QuantRotorK3,
    kv_h: i32,
    head_dim: i32,
    n_tokens: i32,
    max_seq: i32,
) {
    use rmlx_mlx::{Array, Device, Dtype};
    let n_groups = n_groups_for(head_dim as usize) as i32;
    // The code plane and the scale plane have their own per-token strides.
    let code_words =
        crate::rotorquant::row_words_for(head_dim as usize, crate::rotorquant::ROTOR3_BITS) as i32;
    let cps = (kv_h * code_words * n_tokens) as usize;
    let sps = (kv_h * n_groups * n_tokens) as usize;
    let nps = (kv_h * n_tokens) as usize;
    // Scales and norms start at 1, not 0. `rotor3_decode` multiplies the
    // reconstruction by both, so a zero in either makes the decoded token
    // identically zero — and a value oracle built from the same zeros then
    // compares zeros to zeros and cannot tell a correct ring readback from a
    // zero-padded gap, which is the exact failure these tests exist to catch.
    let codes_b: Vec<u8> = (0..cps).flat_map(|i| (i as u32).to_le_bytes()).collect();
    let scales_b: Vec<u8> = (0..sps)
        .flat_map(|i| ((i + 1) as f32).to_le_bytes())
        .collect();
    let norms_b: Vec<u8> = (0..nps)
        .flat_map(|i| ((i + 1) as f32).to_le_bytes())
        .collect();
    let codes = Array::from_bytes(&codes_b, &[cps as i32], Dtype::U32).expect("codes");
    let scales = Array::from_bytes(&scales_b, &[sps as i32], Dtype::F32).expect("scales");
    let norms = Array::from_bytes(&norms_b, &[nps as i32], Dtype::F32).expect("norms");
    ks.gpu_append(
        &codes,
        &scales,
        &norms,
        kv_h,
        head_dim,
        0,
        n_tokens,
        max_seq,
        Device::Gpu,
    )
    .expect("gpu_append");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_rotor_k3 -- --ignored --test-threads=1"]
fn quant_rotor_k3_reset_drops_the_gpu_ring() {
    if crate::test_utils::skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, head_dim) = (2_i32, 6_i32);
    let mut ks = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    ks.rotors = make_rotor_table(0, 0, n_groups_for(head_dim as usize));
    seed_ring_via_gpu_append(&mut ks, kv_h, head_dim, 3, RING_TEST_MAX_SEQ);
    assert!(ks.gpu.is_allocated(), "ring should be live before reset");

    ks.reset();
    assert!(
        !ks.gpu.is_allocated(),
        "reset() must drop the ring — a ring outliving the blocks it mirrors \
         would serve stale keys on the next append"
    );
}

/// `truncate_to()` must KEEP the GPU ring, not drop it.
///
/// Reproduces the fused decode path's state exactly: the store carries a
/// **ring-only tail** — `blocks` empty (dropped once the ring went live, the way
/// `drop_blocks_when_ring_live_k3` does it) while the ring holds the whole
/// prefix and `shape[2]` tracks it. The ring is then the only copy of every
/// token. Clearing it in `truncate_to` (the pre-fix behaviour) strands the kept
/// prefix with nothing to rebuild it from, and `dequant()` / an SSD spill hits
/// the "blocks short of shape[2], no ring" guard and aborts instead of returning
/// the kept tokens.
///
/// Note this is the ring-only regime, **not** the mid-block split: with `blocks`
/// empty there is nothing to split, so `truncate_to` only lowers `shape[2]`. The
/// split is covered on the CPU side by
/// [`quant_rotor_k3_truncate_mid_block_splits_at_this_shape`] — which runs in
/// `make ci`, unlike this one — and by
/// `quant_rotor_k3_truncate_mid_block_splits_instead_of_dropping`.
///
/// `kv_h = 1` keeps `RotorKBlocks::n_tokens` (`b * kv_h * seq`) directly
/// comparable to the truncate target `n` (a raw sequence position).
///
/// Mutation check: re-introduce `self.gpu.clear()` in `truncate_to`. The ring
/// (the only remaining copy of the kept token) is then gone, so `dequant()`
/// hits the no-ring guard and returns `Err` instead of `Ok` — RED.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_rotor_k3 -- --ignored --test-threads=1"]
fn quant_rotor_k3_truncate_to_keeps_the_gpu_ring() {
    if crate::test_utils::skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, head_dim) = (1_i32, 6_i32);
    let mut ks = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    ks.rotors = make_rotor_table(0, 0, n_groups_for(head_dim as usize));

    // One CPU-side append covering 2 tokens.
    let per_tok = (kv_h * head_dim) as usize;
    let d1 = lcg_data(per_tok * 2, TEST_SEED);
    ks.append(&d1, &[1, kv_h, 2, head_dim]).expect("append");
    assert_eq!(ks.shape[2], 2);

    // A live ring covering both tokens, then drop the redundant CPU blocks —
    // this is what the fused decode append does once the ring is allocated, and
    // it is what makes the ring the sole copy.
    seed_ring_via_gpu_append(&mut ks, kv_h, head_dim, 2, RING_TEST_MAX_SEQ);
    assert!(ks.gpu.is_allocated());
    ks.blocks.clear();

    // Truncate to seq=1. Nothing to split (no blocks), so the kept token exists
    // only in the ring and must be rebuilt from it — not zero-padded, not an error.
    ks.truncate_to(1);
    assert_eq!(ks.shape[2], 1);
    assert!(
        ks.gpu.is_allocated(),
        "truncate_to() must KEEP the ring — it is the source of truth for a \
         ring-only decode tail; dropping it here would strand the only copy \
         of the kept token and abort the next dequant/spill"
    );

    // dequant() must rebuild the kept token from the ring — not error, not
    // zero-pad — and must return the ring's actual bytes, not garbage.
    let dq = ks
        .dequant()
        .expect("dequant after truncate must read the kept token from the ring");
    let n_groups = n_groups_for(head_dim as usize);
    let ref_codes: Vec<u32> = (0..n_groups as u32).collect();
    let ref_scales: Vec<f32> = (0..n_groups).map(|i| (i + 1) as f32).collect();
    let ref_norms = vec![1.0_f32];
    let reference = rotor3_decode(
        &ref_codes,
        &ref_scales,
        &ref_norms,
        &ks.rotors,
        head_dim as usize,
    )
    .expect("reference decode of the seeded ring token");
    // The oracle only discriminates if it is not all zeros — `rotor3_decode`
    // scales by both the per-group scale and the per-token norm, so a zero in
    // either would make this comparison zeros-vs-zeros and unable to tell a
    // correct readback from a zero-padded gap.
    assert!(
        reference.iter().any(|v| v.abs() > 1e-6),
        "the reference token decoded to all zeros — the seeded scales/norms went back \
         to starting at 0 and this assertion is vacuous"
    );
    assert_eq!(
        dq, reference,
        "dequant() after truncate must return the ring's actual content for the \
         kept token, not a zero-padded/garbage buffer"
    );
}

/// Falsifies #284: at `kv_h > 1`, `truncate_to(n)` must keep exactly the
/// leading blocks covering sequence `[0, n)`, not `floor(n / kv_h)` of them.
///
/// Builds one block per token (CPU-only, no GPU ring ever touched), truncates
/// mid-sequence at a block boundary, and requires the result to exactly match
/// a reference store built from only the first `keep_tokens` — same shape,
/// same layer (so the same static rotor table), fully deterministic encode.
///
/// Runs both `kv_h == 1` (the historical, accidentally-correct case) and
/// `kv_h > 1` (where the pre-fix code compared row-counted `n_tokens` against
/// a raw sequence target and undercounted) in one test.
///
/// Mutation check: reverting the `truncate_to` body to compare
/// `acc + blk.n_tokens <= n as usize` (raw, not row-scaled) makes the
/// `kv_h > 1` case RED — `blocks.len()` drops to `keep_tokens / kv_h` and
/// `dequant()` returns `Err` (blocks short of `shape[2]`, no ring to rebuild
/// from).
#[test]
fn quant_rotor_k3_truncate_to_kv_h_gt_1_keeps_exact_prefix() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let head_dim = 9_usize; // n_groups = 3, exact
    let total_tokens = 4_usize;
    let keep_tokens = 2_usize;
    let val = |h: usize, tok: usize, d: usize| {
        (h as f32) * 100.0 + (tok as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };

    for kv_h in [1_usize, 4_usize] {
        let token_data = |tok: usize| -> Vec<f32> {
            let mut out = vec![0.0_f32; kv_h * head_dim];
            for h in 0..kv_h {
                for d in 0..head_dim {
                    out[h * head_dim + d] = val(h, tok, d);
                }
            }
            out
        };
        let new_shape = [1_i32, kv_h as i32, 1, head_dim as i32];

        let mut store = QuantRotorK3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 5);
        for tok in 0..total_tokens {
            store.append(&token_data(tok), &new_shape).unwrap();
        }
        assert_eq!(
            store.blocks.len(),
            total_tokens,
            "one block per token append (kv_h={kv_h})"
        );

        store.truncate_to(keep_tokens as i32);

        assert_eq!(
            store.shape[2], keep_tokens as i32,
            "shape[2] must equal keep_tokens (kv_h={kv_h})"
        );
        assert_eq!(
            store.blocks.len(),
            keep_tokens,
            "truncate_to must keep exactly keep_tokens blocks, not floor(keep_tokens / kv_h) (kv_h={kv_h})"
        );
        let kept_rows: usize = store.blocks.iter().map(|blk| blk.n_tokens).sum();
        assert_eq!(
            kept_rows,
            keep_tokens * kv_h,
            "kept rows must equal keep_tokens * b * kv_h (kv_h={kv_h})"
        );

        let decoded = store
            .dequant()
            .expect("dequant must succeed after truncate at kv_h>1 (#284)");

        let mut reference = QuantRotorK3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 5);
        for tok in 0..keep_tokens {
            reference.append(&token_data(tok), &new_shape).unwrap();
        }
        let ref_decoded = reference.dequant().unwrap();

        assert_eq!(
            decoded, ref_decoded,
            "truncated store must exactly match a store built from only the \
             first keep_tokens (kv_h={kv_h})"
        );
    }

    // SAFETY: env lock still held.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// A speculative partial accept truncates the K store **inside** the verifier's
/// append block; the block must be split, not dropped.
///
/// The verifier appends its whole multi-token chunk as one block, then the
/// round keeps only the accepted prefix. Dropping the block discards the
/// accepted tokens with the rejected ones, leaving `blocks` covering fewer rows
/// than `shape[2]`. With no GPU ring live — the CPU append path, which is what
/// a QJL-carrying store and every `--kv-quant rotor*_sym` legacy append use —
/// `synced_rotor_k_blocks` has nothing to rebuild the gap from and aborts the
/// request with "CPU blocks cover N tokens but shape[2] needs M".
///
/// Covers the QJL sideband explicitly: `qjl_codes` / `qjl_norms` are per-token
/// buffers with a different stride from `codes` / `scales`, so a split that only
/// handled the main payload would leave them over-long.
///
/// The oracle is a reference store built from only the retained tokens — same
/// shape, same layer index (so the same static rotor table and QJL projection),
/// deterministic encode. It shares no arithmetic with the truncation logic,
/// which never reads a payload value.
///
/// Mutation check: restore the whole-block drop and `dequant()` returns `Err`
/// with that message, so the `expect` below goes RED.
#[test]
fn quant_rotor_k3_truncate_mid_block_splits_instead_of_dropping() {
    let _guard = crate::test_utils::env_lock();
    if crate::rotor_qjl::rotor_qjl_cli_is_set() {
        return;
    }
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };

    let head_dim = 12_usize; // n_groups = 4, exact
    let kv_h = 2_usize;
    let chunk = |first_tok: usize, n_tok: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; kv_h * n_tok * head_dim];
        for h in 0..kv_h {
            for t in 0..n_tok {
                for d in 0..head_dim {
                    out[(h * n_tok + t) * head_dim + d] =
                        (h as f32) * 100.0 + (first_tok + t) as f32 * 10.0 + (d as f32) * 0.25;
                }
            }
        }
        out
    };

    // 46 prefill positions, then a 5-position verifier chunk; 4 accepted.
    let mut store = QuantRotorK3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 0);
    store
        .append(&chunk(0, 46), &[1, kv_h as i32, 46, head_dim as i32])
        .expect("prefill append");
    store
        .append(&chunk(46, 5), &[1, kv_h as i32, 5, head_dim as i32])
        .expect("verifier append");
    assert!(store.use_qjl(), "QJL sideband active for this store");
    assert_eq!(store.shape[2], 51);

    store.truncate_to(50);

    let decoded = store
        .dequant()
        .expect("dequant must succeed after a mid-block speculative rollback");

    assert_eq!(store.shape[2], 50);
    let kept_rows: usize = store.blocks.iter().map(|blk| blk.n_tokens).sum();
    assert_eq!(kept_rows, 50 * kv_h, "blocks must cover shape[2] exactly");
    let qjl_rows: usize = store.blocks.iter().map(|blk| blk.qjl_norms.len()).sum();
    assert_eq!(
        qjl_rows,
        50 * kv_h,
        "the QJL sideband must be cut to the same row count as the main payload"
    );

    let mut reference = QuantRotorK3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 0);
    reference
        .append(&chunk(0, 46), &[1, kv_h as i32, 46, head_dim as i32])
        .expect("prefill append");
    reference
        .append(&chunk(46, 4), &[1, kv_h as i32, 4, head_dim as i32])
        .expect("accepted-prefix append");
    let ref_decoded = reference.dequant().expect("reference dequant");

    assert_eq!(
        decoded, ref_decoded,
        "the split block must reconstruct the retained prefix exactly"
    );

    // SAFETY: env lock held.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// The same shape the ring test uses, on the CPU side, where `make ci` runs it.
///
/// `quant_rotor_k3_truncate_to_keeps_the_gpu_ring` needs a Metal context, so it
/// carries `#[ignore]` and only `make gpu-test` / `make ci-perf` execute it. A
/// truncation-semantics change is therefore invisible to the per-commit gate
/// unless the same state transition is also asserted without a GPU. This is that
/// assertion: one 2-token block, `truncate_to(1)`, no ring anywhere.
///
/// Mutation check: restore the whole-block drop and `blocks` is empty while
/// `shape[2] == 1`, so `dequant()` returns the no-ring `Err` and the `expect`
/// goes RED.
#[test]
fn quant_rotor_k3_truncate_mid_block_splits_at_this_shape() {
    let (kv_h, head_dim) = (1_i32, 6_i32);
    let mut ks = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    ks.rotors = make_rotor_table(0, 0, n_groups_for(head_dim as usize));
    let per_tok = (kv_h * head_dim) as usize;
    let d1 = lcg_data(per_tok * 2, TEST_SEED);
    ks.append(&d1, &[1, kv_h, 2, head_dim]).expect("append");

    ks.truncate_to(1);

    assert_eq!(ks.shape[2], 1);
    let kept_rows: usize = ks.blocks.iter().map(|blk| blk.n_tokens).sum();
    assert_eq!(
        kept_rows, 1,
        "the 2-token block must be split to 1 row, not dropped"
    );
    let decoded = ks.dequant().expect("dequant after a mid-block truncate");

    let mut reference = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    reference.rotors = make_rotor_table(0, 0, n_groups_for(head_dim as usize));
    reference
        .append(&d1[..per_tok], &[1, kv_h, 1, head_dim])
        .expect("reference append");
    let ref_decoded = reference.dequant().expect("reference dequant");
    assert_eq!(
        decoded, ref_decoded,
        "the split block must reconstruct the retained token exactly"
    );
}

// ── Batch-axis block-boundary parity ──────────────────────────────────

/// Two appends must decode exactly like one append of the same tokens, at
/// `B > 1` as well as `B == 1`.
///
/// Each block covers `[B, S_block, kv_h, D]`, so the concatenation of two
/// blocks is not one `[B, S_total, kv_h, D]` run — reading it as one maps the
/// second block's batch-0 rows onto batch-1 sequence slots. The single-append
/// store holds exactly one block and therefore concatenates nothing, which is
/// what makes it the oracle here.
///
/// Mutation check: put `seq_layout::transpose_seq_heads` over the whole
/// concatenation back in `QuantRotorK3::dequant` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_rotor_k3_two_block_decode_matches_one_block_at_b_gt_1() {
    // The QJL sideband is read from the process env at each `append`, so a
    // concurrent env-mutating test could otherwise encode the two stores under
    // different settings. Hold the lock and pin both settings explicitly — the
    // per-token sideband has to reorder with its token rows either way.
    let _guard = crate::test_utils::env_lock();
    for (b, kv_h, qjl) in [
        (1_usize, 1_usize, "0"),
        (1, 2, "0"),
        (2, 1, "0"),
        (2, 2, "0"),
        (1, 2, "1"),
        (2, 2, "1"),
    ] {
        // SAFETY: env lock held — no concurrent env reader/writer in this binary.
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", qjl) };
        let head_dim = 128_usize;
        let (n0, n1) = (2_usize, 3_usize);
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];

        let mut one = QuantRotorK3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 5);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
        )
        .expect("single append");
        let oracle = one.dequant().expect("one-block dequant");

        let mut two = QuantRotorK3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 5);
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0, head_dim),
            &shape(n0),
        )
        .expect("append chunk 0");
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, n0, n1, head_dim),
            &shape(n1),
        )
        .expect("append chunk 1");
        let got = two.dequant().expect("two-block dequant");

        assert_eq!(
            got, oracle,
            "two-block decode must equal the one-block oracle at b={b} kv_h={kv_h}, qjl={qjl}"
        );
    }
}
