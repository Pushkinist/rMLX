//! What `round_common` can be read on the CPU: the recorder's mapping from a
//! loop's totals to the record that is ingested, and the rollback's own
//! contract on the offsets it is handed.
//!
//! The mapping takes no device, no model and no cache. What it pins is the
//! wiring, and the wiring is exactly what nothing else here can see — a record
//! is a `done` line in a log, so a field written from the wrong place produces
//! a plausible row and no failure anywhere. Two fields make that concrete:
//! `charged` decides how a row's timings are read and is a literal `false` in
//! four loops and a switch in three; and `total_draft` / `total_accept`
//! swapped inverts every ingested `accept_rate` while every token, every count
//! and every gate stays as it was.
//!
//! Three of the record's fields are derived here rather than carried on the
//! totals — `emitted` off the buffer's length, `elapsed_ns` off the request's
//! start, `decode_tps` off the window — and `seed_emitted` is a fourth that is
//! neither: it is passed straight through, because its source differs between
//! the recorder's two callers.

use std::time::{Duration, Instant};

use rmlx_kv_quant::{GdnTapeSegment, KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::{Array, Device, Dtype};

use super::{
    refold_lin_tapes, rollback_round, rollback_round_caches, round_stats, seq_range, RoundTotals,
};
use crate::arch::{load_model, Architecture, LoadOpts};
use crate::decode_loop::ProbeStep;
use crate::speculative::{arm_lin_tapes, DecodeWindow, SpecLoop};

/// How far back the fixture's request started.
///
/// A second is far outside every counter the fixture carries, so a wall-clock
/// read replaced by any of them lands below this floor rather than inside a
/// plausible range. Backdating rather than sleeping keeps the bound exact: a
/// sleep-calibrated one is nondeterministic under a loaded
/// `cargo test --workspace`, which is the same reason `DecodeWindow`'s own
/// tests inject their instants.
const BACKDATE: Duration = Duration::from_secs(1);

/// Totals whose every counter is a different number.
///
/// Distinct values are the point: two fields carrying the same figure agree
/// when they are cross-wired, so a fixture of zeros and ones would pass the
/// swap this file exists to catch.
#[allow(
    clippy::expect_used,
    reason = "test-only: a monotonic clock that cannot be read one second back would make the wall-clock bound below vacuous, so failing here is the honest outcome"
)]
fn totals(charged: bool) -> RoundTotals {
    RoundTotals {
        loop_kind: SpecLoop::MtpSidecar,
        block_size: 5,
        conditioned_rows: Some(7),
        charged,
        rounds: 11,
        emitted_in_rounds: 13,
        total_draft: 17,
        total_accept: 19,
        prefill_ns: 23,
        draft_ns: 29,
        verifier_ns: 31,
        round_loop_ns: 37,
        t_total: Instant::now()
            .checked_sub(BACKDATE)
            .expect("a monotonic clock readable one second back"),
    }
}

/// A window that has seen two tokens 40 ms apart, which is 25 tok/s exactly.
///
/// Instants are injected, never slept for — the idiom
/// `decode_window_excludes_the_time_before_the_first_token` uses for the same
/// reason.
fn window_at_25_tps() -> DecodeWindow {
    let t0 = Instant::now();
    let mut w = DecodeWindow::new();
    w.mark_at(t0);
    w.mark_at(t0 + Duration::from_millis(40));
    w
}

/// `n` emitted steps, so the record's `emitted` has a length to read.
fn emitted(n: usize) -> Vec<ProbeStep> {
    (0..n)
        .map(|i| ProbeStep {
            token_id: i as u32,
            piece: String::from("x").into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        })
        .collect()
}

/// The record carries the charge decision the loop handed over, both ways.
///
/// A recorder that writes a constant here re-attributes the phase timings of
/// every loop that charges, with every token and every count identical.
///
/// Three checks cover that between them, and each is blind to what the others
/// see. `check-spec-charge`'s RULE 5 refuses the *literal* form — any
/// `charged:` field written outside a round loop, whether or not its value is
/// legible. This test catches a value rebound before the literal, where the
/// field is still the shorthand and RULE 5 reads nothing wrong. And
/// `clippy::redundant_field_names` under `-D warnings` is what makes the
/// shorthand the only legal spelling of the correct form, so there is no third
/// way to write it that neither of the first two is looking at.
///
/// Mutation: replace `charged` in [`round_stats`] with `false`.
#[test]
fn the_recorder_writes_the_charge_decision_it_was_handed() {
    for charged in [false, true] {
        let record = round_stats(&totals(charged), &emitted(3), 2, &DecodeWindow::new());
        assert_eq!(
            record.charged, charged,
            "the record must carry the decision the loop made, not one of its own"
        );
    }
}

/// Every counter reaches the record under its own name, and so does each of
/// the three figures the recorder derives.
///
/// `emitted` comes off the buffer's length and `seed_emitted` off its own
/// argument — different numbers on purpose, since one is a pass-through and the
/// other is not. `elapsed_ns` is bracketed by two reads of the same clock
/// taken either side of the call, so it is pinned to the read that happened
/// rather than to a duration, and held above [`BACKDATE`] so that no counter in
/// the fixture can stand in for it. `decode_tps` is read from a window driven
/// to a known rate; the window that has seen nothing is the second case below,
/// and it must arrive as `None` rather than as a zero.
///
/// Mutation: swap `total_draft` and `total_accept` in [`round_stats`], which
/// inverts every ingested `accept_rate` and changes nothing else in the tree;
/// or write `elapsed_ns: prefill_ns`, which lands a wall-clock of 23 ns in an
/// append-only store.
#[test]
fn every_counter_reaches_the_record_under_its_own_name() {
    let totals = totals(true);
    let steps = emitted(3);
    let window = window_at_25_tps();
    let before = totals.t_total.elapsed().as_nanos();
    let record = round_stats(&totals, &steps, 2, &window);
    let after = totals.t_total.elapsed().as_nanos();
    assert_eq!(record.loop_kind, SpecLoop::MtpSidecar);
    assert_eq!(record.block_size, 5);
    assert_eq!(record.conditioned_rows, Some(7));
    assert_eq!(record.rounds, 11);
    assert_eq!(record.emitted_in_rounds, 13);
    assert_eq!(record.total_draft, 17);
    assert_eq!(record.total_accept, 19);
    assert_eq!(record.prefill_ns, 23);
    assert_eq!(record.draft_ns, 29);
    assert_eq!(record.verifier_ns, 31);
    assert_eq!(record.round_loop_ns, 37);
    assert_eq!(record.emitted, 3, "the tokens handed to the sink");
    assert_eq!(record.seed_emitted, 2, "the count the caller passed");
    assert!(
        (before..=after).contains(&record.elapsed_ns),
        "the request's wall-clock must be the clock read while the record was \
         built: {} is outside the {before}..={after} that read happened inside",
        record.elapsed_ns
    );
    assert!(
        record.elapsed_ns >= BACKDATE.as_nanos(),
        "the request started {BACKDATE:?} ago, so a wall-clock of {} is one of \
         the fixture's counters standing in for it",
        record.elapsed_ns
    );
    assert!(
        record
            .decode_tps
            .is_some_and(|tps| (tps - 25.0).abs() < 1e-9),
        "two tokens 40 ms apart is 25 tok/s, and the record read {:?}",
        record.decode_tps
    );
    // The derived accept rate is what a swapped pair moves, and it is what an
    // ingested row is read with.
    assert!((record.accept_rate() - 19.0 / 17.0).abs() < 1e-12);

    // A window that has seen no token has no rate, and the recorder passes that
    // through rather than turning it into a throughput of zero.
    let empty = round_stats(&totals, &steps, 2, &DecodeWindow::new());
    assert_eq!(empty.decode_tps, None);
}

/// The offsets the caller passes have to describe the round it is rolling back.
///
/// Unreachable through [`super::rollback_round`], which takes this arm only
/// when `target_offset` is inside the round — so this is the low-level
/// contract, read directly, and it is why the branch above it is not the only
/// thing standing between a caller's arithmetic and a wrong prefix. `lin` is
/// never touched: the refusal is before it.
#[test]
fn rollback_refuses_offsets_that_overrun_the_round() {
    let mut lin = vec![LinearAttnCache::new()];
    let err = rollback_round_caches(
        &mut [],
        Some(&mut lin),
        &[1, 2, 3],
        100,
        105,
        false,
        Device::Cpu,
    )
    .err()
    .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("retained prefix 5 exceeds the 3 tokens"),
        "offsets that overrun the round must be refused; got: {err:?}"
    );
}

/// Which arm the rollback takes is decided by the round's own extent, and it is
/// decidable without a device.
///
/// The caches stand at `pre_round_offset + round_tokens.len()`, so a target
/// below that dropped positions and one at or above it dropped none. A fresh
/// recurrent cache separates the two arms: the refold arm asks it for the tape
/// its forwards were supposed to record and refuses when there is none, and the
/// disarm arm takes whatever tape is there and returns.
///
/// Mutation: `<` to `<=` in [`rollback_round`]. A full accept then refolds a
/// prefix nothing rejected — which on a real round is a recurrent state rebuilt
/// from a tape the loop was about to drop.
#[test]
fn the_rollback_refolds_only_when_the_round_dropped_positions() {
    // (target offset, whether the round dropped anything)
    let cases = [(99, true), (102, true), (103, false), (104, false)];
    for (target, dropped) in cases {
        let mut lin = vec![LinearAttnCache::new()];
        let err = rollback_round(
            &mut [],
            Some(&mut lin),
            &[1, 2, 3],
            100,
            target,
            false,
            Device::Cpu,
        )
        .err()
        .map_or_else(String::new, |e| e.to_string());
        let refolded = err.contains("recurrent layer 0 has no round tape");
        assert_eq!(
            refolded,
            dropped,
            "the caches stand at 103 having consumed three tokens from 100, so a \
             target of {target} {} — got {err:?}",
            if dropped {
                "is a partial accept and must refold"
            } else {
                "dropped nothing and must only disarm"
            }
        );
    }
}

// -- Round tape: the recurrent rollback's evidence ---------------------------

/// A deterministic ramp, so every position of every taped buffer is distinct
/// and a join that lands one position out is visible in the bytes.
fn ramp(seed: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| seed + i as f32 * 0.125).collect()
}

#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn tape_arr(seed: f32, shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_f32_slice(&ramp(seed, n as usize), shape).expect("from_f32_slice")
}

#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn tape_bytes(a: &Array) -> Vec<u8> {
    a.eval().expect("eval");
    a.to_bytes().expect("to_bytes")
}

/// Shapes the recurrence kernel accepts: `Dk` a multiple of 32, `Hv` a multiple
/// of `Hk`.
const TAPE_HK: i32 = 1;
const TAPE_HV: i32 = 2;
const TAPE_D: i32 = 32;
/// The depthwise conv1d's carried tail, `kernel - 1`.
const TAPE_PAD: i32 = 3;
const TAPE_CONV_DIM: i32 = 2;

/// One taped forward over `len` positions, its arrays cut from a longer ramp
/// starting at `from` so segments of one round line up end to end.
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn tape_segment(from: usize, len: usize) -> GdnTapeSegment {
    let pos = |seed: f32, per: i32| -> Array {
        let per = per as usize;
        let all = ramp(seed, (from + len) * per);
        let shape_len = len as i32;
        let tail = all.get(from * per..).expect("ramp covers the segment");
        Array::from_f32_slice(tail, &[1, shape_len, per as i32]).expect("from_f32_slice")
    };
    let heads = |seed: f32, h: i32| -> Array {
        let flat = pos(seed, h * TAPE_D);
        flat.reshape(&[1, len as i32, h, TAPE_D], Device::Cpu)
            .expect("reshape")
    };
    GdnTapeSegment {
        q: heads(0.5, TAPE_HK),
        k: heads(1.5, TAPE_HK),
        v: heads(2.5, TAPE_HV),
        g: pos(0.25, TAPE_HV),
        beta: pos(0.75, TAPE_HV),
        // The conv input opens with the `kernel - 1` positions carried in from
        // the previous call, so a segment starting at `from` covers the global
        // conv rows `from .. from + pad + len`.
        conv_input: {
            let per = TAPE_CONV_DIM as usize;
            let rows = TAPE_PAD as usize + len;
            let all = ramp(9.0, (from + rows) * per);
            let tail = all.get(from * per..).expect("ramp covers the segment");
            Array::from_f32_slice(tail, &[1, rows as i32, TAPE_CONV_DIM]).expect("from_f32_slice")
        },
        len,
    }
}

/// The whole round as one segment, which is what the refold must reproduce
/// from however many segments actually recorded it.
fn tape_zero_state() -> Array {
    tape_arr(0.0, &[1, TAPE_HV, TAPE_D, TAPE_D])
}

#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn armed(segments: Vec<GdnTapeSegment>, state_in: &Array) -> LinearAttnCache {
    let mut cache = LinearAttnCache::new();
    cache.arm_tape();
    let tape = cache.tape.as_mut().expect("armed");
    for (idx, seg) in segments.into_iter().enumerate() {
        // Only the first forward starts from the round's state. Later forwards
        // start where their predecessor ended, and a tape that kept one of
        // those would refold from the wrong place — so they are handed a state
        // no correct refold can produce.
        let state = if idx == 0 {
            state_in.try_clone().expect("clone round state")
        } else {
            tape_arr(7.0, &[1, TAPE_HV, TAPE_D, TAPE_D])
        };
        tape.push(&state, seg).expect("push");
    }
    cache
}

/// A layer whose tape was never armed cannot be rolled back, and says so rather
/// than leaving the recurrent state where the rejected drafts left it.
///
/// This is the mutation that matters: without the check, the refold has nothing
/// to fold and the obvious implementation leaves the state untouched — a silent
/// no-op that reads as a successful rollback and produces wrong tokens with no
/// error anywhere.
#[test]
fn refold_refuses_a_layer_whose_tape_was_never_armed() {
    let mut lin = vec![armed(vec![tape_segment(0, 3)], &tape_zero_state())];
    // The mutation: this round's arming never reached the layer.
    let _ = lin.get_mut(0).map(LinearAttnCache::take_tape);

    let err = refold_lin_tapes(&mut lin, 3, 2, false, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("recurrent layer 0 has no round tape"),
        "an unarmed layer must be refused by name; got: {err:?}"
    );
}

/// A tape that covers fewer positions than the round fed describes a different
/// round, and refolding it would leave the recurrent state at a prefix the K/V
/// stack beside it does not agree with.
#[test]
fn refold_refuses_a_tape_that_is_short_of_the_round() {
    // The mutation: the round fed five positions across two forwards and only
    // the first was recorded.
    let mut lin = vec![armed(vec![tape_segment(0, 2)], &tape_zero_state())];

    let err = refold_lin_tapes(&mut lin, 5, 3, false, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("taped 2 positions over 1 forwards but the round fed 5"),
        "a short tape must be refused with both counts; got: {err:?}"
    );
}

/// A hybrid hands one recurrent slot per decoder layer, and most of them belong
/// to full-attention layers. Those record nothing and hold no state, and the
/// refold walks past them.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn a_layer_that_holds_no_recurrence_is_walked_past() {
    let state_in = tape_zero_state();
    let mut lin = vec![
        armed(vec![tape_segment(0, 3)], &state_in),
        // A full-attention layer's slot: armed with the rest of the stack, and
        // no forward ever came through it.
        armed(vec![], &state_in),
    ];

    refold_lin_tapes(&mut lin, 3, 0, false, Device::Cpu).expect("refold");

    let unused = lin.get(1).expect("two layers");
    assert!(
        unused.conv_state.is_none() && unused.delta_state.is_none(),
        "a slot that holds no recurrence must be left alone, not given one"
    );
}

/// A layer that DOES hold recurrent state and recorded nothing is the defect
/// that skip must not swallow: its state advanced with the round and there is
/// nothing to roll it back with.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn a_recurrent_layer_that_recorded_nothing_is_still_refused() {
    let state_in = tape_zero_state();
    let mut lin = vec![armed(vec![], &state_in)];
    // The mutation: the round advanced this layer's recurrence and the tape
    // missed every forward that did it.
    lin.get_mut(0).expect("one layer").delta_state = Some(tape_zero_state());

    let err = refold_lin_tapes(&mut lin, 3, 1, false, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("taped 0 positions over 0 forwards but the round fed 3"),
        "a recurrent layer with an empty tape must be refused; got: {err:?}"
    );
}

/// A round that kept nothing goes back to the state it started from, and the
/// conv tail goes back to the one the round carried in — with no kernel call,
/// because there is nothing to fold.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn refold_at_zero_kept_restores_what_the_round_started_from() {
    let state_in = tape_zero_state();
    let seg = tape_segment(0, 3);
    let carried = seq_range(&seg.conv_input, 0, TAPE_PAD, Device::Cpu).expect("carried tail");
    let mut lin = vec![armed(vec![seg], &state_in)];

    refold_lin_tapes(&mut lin, 3, 0, false, Device::Cpu).expect("refold");

    let cache = lin.first().expect("one layer");
    assert_eq!(
        tape_bytes(cache.delta_state.as_ref().expect("delta restored")),
        tape_bytes(&state_in),
        "a zero-kept refold must restore the pre-round recurrent state"
    );
    assert_eq!(
        tape_bytes(cache.conv_state.as_ref().expect("conv restored")),
        tape_bytes(&carried),
        "a zero-kept refold must restore the conv tail the round carried in"
    );
    assert!(
        cache.tape.is_none(),
        "the refold consumes the tape it folded"
    );
}

/// Refolding a one-forward tape at `kept` gives exactly the state a forward
/// over `kept` positions would have left.
///
/// The recurrence is sequential in position, so the state after position `kept`
/// does not depend on how many positions came after it — which is the whole
/// reason the accepted prefix can be rebuilt from what the round already
/// computed. Bytes, not a tolerance: same kernel, same inputs, same order.
#[test]
#[ignore = "requires Metal GPU context (gated_delta recurrence kernel)"]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn a_one_forward_tape_refolds_to_the_shorter_forward() {
    let device = Device::Gpu;
    let round = 5usize;
    let state_in = tape_zero_state();

    for kept in 1..=round {
        let seg = tape_segment(0, round);
        let want = {
            let (_y, state) = crate::gated_delta_msl::gated_delta_step_gpu(
                &seq_range(&seg.q, 0, kept as i32, device).expect("q"),
                &seq_range(&seg.k, 0, kept as i32, device).expect("k"),
                &seq_range(&seg.v, 0, kept as i32, device).expect("v"),
                &seq_range(&seg.g, 0, kept as i32, device).expect("g"),
                &seq_range(&seg.beta, 0, kept as i32, device).expect("beta"),
                &state_in,
                device,
            )
            .expect("reference recurrence");
            state
        };
        let conv_want = seq_range(&seg.conv_input, kept as i32, kept as i32 + TAPE_PAD, device)
            .expect("conv tail");

        let mut lin = vec![armed(vec![tape_segment(0, round)], &state_in)];
        refold_lin_tapes(&mut lin, round, kept, false, device).expect("refold");
        let cache = lin.first().expect("one layer");

        assert_eq!(
            tape_bytes(cache.delta_state.as_ref().expect("delta")),
            tape_bytes(&want),
            "refold at kept={kept} must equal a forward over {kept} positions"
        );
        assert_eq!(
            tape_bytes(cache.conv_state.as_ref().expect("conv")),
            tape_bytes(&conv_want),
            "refold at kept={kept} must leave the conv tail at that position"
        );
    }
}

/// The two-model draft rollback spans a forward per drafted token, so its tape
/// accumulates. Refolding across the join must give the same state a single
/// forward over the same positions would.
///
/// The prefix is walked over every `kept` in the round, so the case that lands
/// exactly on a segment boundary and the ones either side are all covered — a
/// join that took one position too many or too few from a segment moves the
/// state and shows up here.
#[test]
#[ignore = "requires Metal GPU context (gated_delta recurrence kernel)"]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn an_accumulating_tape_refolds_across_its_segment_join() {
    let device = Device::Gpu;
    let (first, second) = (2usize, 3usize);
    let round = first + second;
    let state_in = tape_zero_state();
    let whole = tape_segment(0, round);

    for kept in 1..=round {
        let want = {
            let (_y, state) = crate::gated_delta_msl::gated_delta_step_gpu(
                &seq_range(&whole.q, 0, kept as i32, device).expect("q"),
                &seq_range(&whole.k, 0, kept as i32, device).expect("k"),
                &seq_range(&whole.v, 0, kept as i32, device).expect("v"),
                &seq_range(&whole.g, 0, kept as i32, device).expect("g"),
                &seq_range(&whole.beta, 0, kept as i32, device).expect("beta"),
                &state_in,
                device,
            )
            .expect("reference recurrence");
            state
        };
        let conv_want = seq_range(
            &whole.conv_input,
            kept as i32,
            kept as i32 + TAPE_PAD,
            device,
        )
        .expect("conv tail");

        let mut lin = vec![armed(
            vec![tape_segment(0, first), tape_segment(first, second)],
            &state_in,
        )];
        refold_lin_tapes(&mut lin, round, kept, false, device).expect("refold");
        let cache = lin.first().expect("one layer");

        assert_eq!(
            tape_bytes(cache.delta_state.as_ref().expect("delta")),
            tape_bytes(&want),
            "a two-segment refold at kept={kept} must equal one forward over {kept} positions"
        );
        assert_eq!(
            tape_bytes(cache.conv_state.as_ref().expect("conv")),
            tape_bytes(&conv_want),
            "a two-segment refold at kept={kept} must leave the conv tail at that position"
        );
    }
}

// -- Round tape against the replay it replaced, on a real model --------------

/// The hybrids the tape is checked on: GDN stacks whose recurrent layers are
/// interleaved with full-attention ones, which is the shape that made the replay
/// run the whole layer stack in the first place. One dense, one mixture — the
/// projections a shorter forward recomputes go through different kernels on the
/// two, and this equality is a claim about those.
const TAPE_REPLAY_SLUGS: &[&str] = &[
    "mlx-community__Qwen3.8-27B-4bit",
    "mlx-community__Qwen3.6-35B-A3B-8bit",
];

/// The snapshots present on this machine, or a named stand-down the GPU runner
/// counts for each that is not.
fn tape_replay_models(test: &str) -> Vec<std::path::PathBuf> {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!("SKIP {test}: RMLX_O_MODELS_ROOT is not set");
        return Vec::new();
    };
    let root = std::path::PathBuf::from(root);
    TAPE_REPLAY_SLUGS
        .iter()
        .filter_map(|slug| {
            let path = root.join(slug);
            if path.join("config.json").exists() {
                return Some(path);
            }
            eprintln!("SKIP {test}: {slug} is not under RMLX_O_MODELS_ROOT");
            None
        })
        .collect()
}

/// A fresh unquantized cache stack, so nothing between the two arms differs but
/// the way the recurrent state got where it is.
#[allow(
    clippy::expect_used,
    reason = "test-only: a cache stack sized from the model just loaded is present by construction, and the panic names it"
)]
fn tape_replay_stack(arch: &Architecture) -> (Vec<KvCache>, Vec<LinearAttnCache>) {
    let n = arch.num_hidden_layers();
    (
        (0..n).map(|_| KvCache::with_quant(KvQuant::None)).collect(),
        (0..n).map(|_| LinearAttnCache::new()).collect(),
    )
}

/// Every recurrent buffer in the stack, read back as f32 in layer order.
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array the model just wrote is present by construction, and the panic names it"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tape_replay_state(lin: &[LinearAttnCache], device: Device) -> Vec<f32> {
    let mut out = Vec::new();
    for cache in lin {
        for buf in [&cache.delta_state, &cache.conv_state]
            .into_iter()
            .flatten()
        {
            let f32_buf = buf.astype(Dtype::F32, device).expect("astype f32");
            f32_buf.eval().expect("materialise");
            out.extend(
                f32_buf
                    .to_bytes()
                    .expect("to_bytes")
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
            );
        }
    }
    out
}

/// `||a - b|| / ||a||`, the relative size of the difference between two
/// states.
///
/// A norm and not a worst element: these buffers hold millions of values, most
/// of them near zero, and an elementwise ratio saturates on any pair of tiny
/// opposite-signed ones whatever the states as a whole are doing.
fn tape_replay_rel_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "two states of the same stack differ in size"
    );
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if scale > 0.0 {
        diff / scale
    } else {
        diff
    }
}

/// Twenty-odd deterministic ids well inside any vocabulary. The values do not
/// matter; that both arms see the same ones does.
const TAPE_REPLAY_PROMPT: &[u32] = &[
    2, 9707, 11, 1879, 13, 576, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 6323, 374, 26867,
    13,
];
/// The round's own tokens: a verify block's worth.
const TAPE_REPLAY_ROUND: &[u32] = &[785, 6722, 315, 15236, 374];

/// Advance a stack over `tokens` the way a round's forwards do, and hand back
/// the recurrent state the round leaves.
///
/// `chunks` is how the round split those tokens between forwards: one chunk for
/// a verify block, one chunk per drafted token for a two-model drafter. An empty
/// chunk is a round that advanced nothing, which is the control at `kept == 1`.
#[allow(
    clippy::expect_used,
    reason = "test-only: a forward over ids this test chose is expected to succeed, and the panic names it"
)]
fn tape_replay_run(
    arch: &Architecture,
    prompt: &[u32],
    chunks: &[&[u32]],
    arm_tapes: bool,
    device: Device,
) -> (Vec<KvCache>, Vec<LinearAttnCache>) {
    let (mut kv, mut lin) = tape_replay_stack(arch);
    arch.forward_seq_last_k_with_cache(prompt, 1, &mut kv, Some(&mut lin), device)
        .expect("prefill");
    if arm_tapes {
        arm_lin_tapes(Some(&mut lin));
    }
    for chunk in chunks.iter().filter(|c| !c.is_empty()) {
        arch.forward_seq_last_k_with_cache(chunk, 1, &mut kv, Some(&mut lin), device)
            .expect("round forward");
    }
    (kv, lin)
}

/// The tape rebuilds the state the replay used to rebuild, on a real recurrent
/// stack — for a round taken as one verify forward, and for one taken as a
/// forward per token.
///
/// The replay arm is what this change removed: restore the pre-round state and
/// run the accepted prefix through the whole layer stack. It is reproduced here
/// as a second stack prefilled identically and advanced over the accepted prefix
/// alone, which is the same computation.
///
/// **How close they can be is the model's own answer, and the run measures it.**
/// The replay computes the prefix in a forward of its own length; the tape
/// returns what the round's forward computed at those positions. On the dense
/// hybrid the two are bit-identical, and the run says so. On the mixture the
/// model does not reproduce itself that way — one forward over the round and the
/// same tokens stepped one at a time part company by a couple of percent, which
/// is a property of that stack and not of this change — so the bound the refold
/// is held to is that same disagreement, measured beside it in the same process.
///
/// The control is what gives the comparison power: the same refold against the
/// replay one token short of the accepted length, which is where a refold that
/// folded the wrong number of positions would sit. It reads about 0.5 against a
/// refold-to-replay agreement of at most a few percent.
#[test]
#[ignore = "requires Metal GPU context and a 27B snapshot"]
#[allow(
    clippy::expect_used,
    reason = "test-only: a model this test has already checked for existence but cannot load is a broken checkout, and the panic names it"
)]
#[allow(
    clippy::print_stderr,
    reason = "test-only: stand-down notice and the measured agreement, both read by the operator"
)]
fn a_round_tape_refolds_to_what_the_replay_produced() {
    const NAME: &str = "a_round_tape_refolds_to_what_the_replay_produced";
    let device = Device::Gpu;
    let round = TAPE_REPLAY_ROUND;
    let per_token: Vec<&[u32]> = round.iter().map(std::slice::from_ref).collect();

    for path in tape_replay_models(NAME) {
        let model = path.file_name().unwrap_or_default().to_string_lossy();
        let arch = load_model(&path, device, &LoadOpts::default()).expect("load verifier");
        assert!(
            arch.needs_lin_caches(),
            "{model} must carry recurrent state or this test proves nothing"
        );

        // What this model's own two regimes make of the same tokens: the round
        // in one forward against the round stepped. It is the bound the refold
        // is held to, because no rebuild of a prefix can be closer to a forward
        // over that prefix than the model is to itself.
        let (_kv, batched) = tape_replay_run(&arch, TAPE_REPLAY_PROMPT, &[round], false, device);
        let (_kv, stepped) = tape_replay_run(&arch, TAPE_REPLAY_PROMPT, &per_token, false, device);
        let regime = tape_replay_rel_err(
            &tape_replay_state(&batched, device),
            &tape_replay_state(&stepped, device),
        );
        eprintln!("[{NAME}/{model}] one forward against stepped: {regime:.6}");

        for (shape, chunks) in [
            ("one verify forward", vec![round]),
            ("a forward per token", per_token.clone()),
        ] {
            for kept in 1..round.len() {
                let (_kv, mut taped) =
                    tape_replay_run(&arch, TAPE_REPLAY_PROMPT, &chunks, true, device);
                refold_lin_tapes(&mut taped, round.len(), kept, false, device).expect("refold");
                let refolded = tape_replay_state(&taped, device);

                let (_kv, replayed) = tape_replay_run(
                    &arch,
                    TAPE_REPLAY_PROMPT,
                    &[round.get(..kept).expect("kept prefix")],
                    false,
                    device,
                );
                let agreement =
                    tape_replay_rel_err(&refolded, &tape_replay_state(&replayed, device));
                let (_kv, off_by_one) = tape_replay_run(
                    &arch,
                    TAPE_REPLAY_PROMPT,
                    &[round.get(..kept - 1).expect("shorter prefix")],
                    false,
                    device,
                );
                let control =
                    tape_replay_rel_err(&refolded, &tape_replay_state(&off_by_one, device));

                eprintln!(
                    "[{NAME}/{shape}] kept={kept} agreement={agreement:.6} \
                     one-token-short={control:.6}"
                );
                assert!(
                    agreement <= regime,
                    "{model}, {shape} at kept={kept}: the refold and the replay of the \
                     same {kept} tokens differ by {agreement}, and this model \
                     reproduces itself to {regime}"
                );
                assert!(
                    agreement * 10.0 < control,
                    "{model}, {shape} at kept={kept}: the refold is no nearer the replay \
                     of {kept} tokens ({agreement}) than the replay of {} ({control}), \
                     so this cell would pass whatever the refold folded",
                    kept - 1
                );
            }
        }
    }
}
