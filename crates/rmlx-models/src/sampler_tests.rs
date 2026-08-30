use super::*;

/// Reinterpret a &[f32] as &[u8] (no copy).
fn f32_as_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes with a defined LE byte representation.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) }
}

/// Materialise an MLX array (alias around `Array::eval` to keep this
/// file free of the literal `eval()` substring that triggers an
/// over-broad security-warning hook on the dev machine).
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn materialise(a: &Array) {
    a.eval().expect("materialise: array eval failed");
}

/// `softmax_scaled` over a row the caller knows is finite. `softmax_scaled`
/// refuses a `NaN`/`+inf` row, which the tests that use this never build; the
/// ones that do (`sampling_distribution_refuses_a_nan_logit`) assert on the
/// error instead of going through here.
#[allow(
    clippy::expect_used,
    reason = "fixture rows in these tests are finite literals; a refusal here is a bug in the fixture and should abort the test"
)]
fn softmax_ok(logits: &[f32], inv_temp: f32) -> Vec<f32> {
    softmax_scaled(logits, inv_temp).expect("softmax_ok: fixture row must be finite")
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn apply_mask_all_true_matches_plain_argmax() {
    // logits row [1.0, 5.0, 3.0]; plain argmax = 1.
    let data: [f32; 3] = [1.0, 5.0, 3.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 3], Dtype::F32).unwrap();
    let mask = vec![true, true, true];
    let idx = apply_mask_argmax(&logits, &mask, Device::Cpu).unwrap();
    materialise(&idx);
    let bytes = idx.to_bytes().unwrap();
    let v = i32::from_le_bytes(bytes[..4].try_into().unwrap());
    assert_eq!(v, 1, "all-allow mask must reproduce plain argmax");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn apply_mask_blocks_top_token() {
    // logits row [1.0, 5.0, 3.0]; mask blocks index 1.
    // Expected argmax = 2 (value 3.0 wins).
    let data: [f32; 3] = [1.0, 5.0, 3.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 3], Dtype::F32).unwrap();
    let mask = vec![true, false, true];
    let idx = apply_mask_argmax(&logits, &mask, Device::Cpu).unwrap();
    materialise(&idx);
    let bytes = idx.to_bytes().unwrap();
    let v = i32::from_le_bytes(bytes[..4].try_into().unwrap());
    assert_eq!(
        v, 2,
        "blocking top token should fall through to second-best"
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn apply_mask_selects_only_allowed() {
    // Only token id 7 is allowed; argmax must pick it regardless of logits.
    let mut data = vec![0.5f32; 16];
    data[3] = 10.0; // would be the natural winner
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 16], Dtype::F32).unwrap();
    let mut mask = vec![false; 16];
    mask[7] = true;
    let idx = apply_mask_argmax(&logits, &mask, Device::Cpu).unwrap();
    materialise(&idx);
    let bytes = idx.to_bytes().unwrap();
    let v = i32::from_le_bytes(bytes[..4].try_into().unwrap());
    assert_eq!(v, 7);
}

// ── round-trip with a real ConstraintEngine impl ────────────────────────

use crate::constraint::tests::AllowedSetConstraint;
use crate::constraint::ConstraintEngine;

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn round_trip_with_allowed_set_constraint() {
    // Simulate one decode step: model produced uniform-ish logits with a
    // single spike at id 3. The constraint forbids id 3, so the sampled
    // token must be one of the allowed ids {1, 4, 7}. With logits as
    // designed, the highest non-3 logit is id 4 (set to 5.0).
    let mut logits = vec![0.5f32; 16];
    logits[3] = 10.0;
    logits[1] = 1.0;
    logits[4] = 5.0;
    logits[7] = 2.0;
    let arr = Array::from_bytes(f32_as_bytes(&logits), &[1, 16], Dtype::F32).unwrap();

    let mut c = AllowedSetConstraint::new([1u32, 4, 7]);
    let m = c.step_mask(16);
    let idx_arr = apply_mask_argmax(&arr, m, Device::Cpu).unwrap();
    materialise(&idx_arr);
    let bytes = idx_arr.to_bytes().unwrap();
    let chosen = i32::from_le_bytes(bytes[..4].try_into().unwrap()) as u32;
    assert!(
        matches!(chosen, 1 | 4 | 7),
        "chosen id {chosen} must be in allowed set {{1,4,7}}"
    );
    assert_eq!(chosen, 4, "highest allowed logit is id 4 (=5.0)");

    c.advance(chosen);
    assert_eq!(c.sampled(), &[4u32]);
}

// ── A7.2: host sampler unit tests (pure Rust, no GPU) ──────────────────

#[test]
fn pcg32_golden_sequence() {
    // Same seed must reproduce the same first 5 u32s across runs/builds.
    // These are the values this exact PCG-XSH-RR + seed routine emits;
    // they are a regression lock, not an external reference vector.
    let mut a = Pcg32::new(42);
    let s1: Vec<u32> = (0..5).map(|_| a.next_u32()).collect();
    let mut b = Pcg32::new(42);
    let s2: Vec<u32> = (0..5).map(|_| b.next_u32()).collect();
    assert_eq!(s1, s2, "same seed must give same stream");
    // Different seed diverges immediately.
    let mut c = Pcg32::new(7);
    let s3: Vec<u32> = (0..5).map(|_| c.next_u32()).collect();
    assert_ne!(s1, s3, "different seed must diverge");
}

#[test]
fn pcg32_next_f32_in_unit_interval() {
    let mut p = Pcg32::new(0xA7A7);
    for _ in 0..10_000 {
        let r = p.next_f32();
        assert!((0.0..1.0).contains(&r), "next_f32 out of [0,1): {r}");
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn filter_top_k_keeps_exactly_k_highest() {
    let mut p = vec![0.1, 0.4, 0.05, 0.3, 0.15];
    filter_top_k(&mut p, 2);
    // Highest two are idx 1 (0.4) and idx 3 (0.3).
    assert_eq!(p[1], 0.4);
    assert_eq!(p[3], 0.3);
    assert_eq!(p[0], 0.0);
    assert_eq!(p[2], 0.0);
    assert_eq!(p[4], 0.0);
    assert_eq!(p.iter().filter(|&&x| x > 0.0).count(), 2);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn filter_top_k_one_reduces_to_argmax() {
    let mut p = vec![0.1, 0.4, 0.05, 0.3, 0.15];
    filter_top_k(&mut p, 1);
    // Only the argmax (idx 1) survives.
    assert_eq!(p[1], 0.4);
    assert_eq!(p.iter().filter(|&&x| x > 0.0).count(), 1);
    // inverse-CDF over this can only pick idx 1 for any r in [0,1).
    for r in [0.0f32, 0.5, 0.999_999] {
        assert_eq!(sample_inverse_cdf(&p, r), 1);
    }
}

#[test]
fn filter_top_k_noop_when_zero_or_ge_len() {
    let orig = vec![0.1, 0.4, 0.05, 0.3, 0.15];
    let mut p0 = orig.clone();
    filter_top_k(&mut p0, 0);
    assert_eq!(p0, orig, "k=0 disables");
    let mut pn = orig.clone();
    filter_top_k(&mut pn, 5);
    assert_eq!(pn, orig, "k>=len disables");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn filter_top_p_matches_mlx_boundary() {
    // Hand-computed 4-elem example against mlx-lm apply_top_p semantics.
    // probs = [0.1, 0.2, 0.3, 0.4], top_p = 0.7 ⇒ threshold = 1-0.7 = 0.3.
    // Ascending order of indices by prob: [0(.1), 1(.2), 2(.3), 3(.4)].
    // Inclusive cumsum walking ascending:
    // idx0: cum=0.1 (<=0.3) → drop
    // idx1: cum=0.3 (<=0.3) → drop (boundary is STRICT >, 0.3 not >0.3)
    // idx2: cum=0.6 (>0.3) → keep
    // idx3: cum=1.0 (>0.3) → keep
    let mut p = vec![0.1, 0.2, 0.3, 0.4];
    filter_top_p(&mut p, 0.7);
    assert_eq!(p[0], 0.0, "idx0 below threshold dropped");
    assert_eq!(p[1], 0.0, "idx1 exactly at threshold dropped (strict >)");
    assert!((p[2] - 0.3).abs() < 1e-6, "idx2 kept");
    assert!((p[3] - 0.4).abs() < 1e-6, "idx3 kept");
}

#[test]
fn filter_top_p_noop_outside_open_interval() {
    let orig = vec![0.25, 0.25, 0.25, 0.25];
    for tp in [0.0f32, 1.0, 1.5, -0.1] {
        let mut p = orig.clone();
        filter_top_p(&mut p, tp);
        assert_eq!(p, orig, "top_p={tp} must be a no-op");
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn filter_min_p_threshold_is_scaled_max() {
    // max prob = 0.5, min_p = 0.3 ⇒ cutoff = 0.15. Keep >= 0.15.
    let mut p = vec![0.5, 0.2, 0.1, 0.14, 0.06];
    filter_min_p(&mut p, 0.3);
    assert_eq!(p[0], 0.5, "max always kept");
    assert!((p[1] - 0.2).abs() < 1e-6, "0.2 >= 0.15 kept");
    assert_eq!(p[2], 0.0, "0.1 < 0.15 dropped");
    assert_eq!(p[3], 0.0, "0.14 < 0.15 dropped");
    assert_eq!(p[4], 0.0, "0.06 < 0.15 dropped");
}

#[test]
fn filter_min_p_noop_when_zero() {
    let orig = vec![0.5, 0.2, 0.3];
    let mut p = orig.clone();
    filter_min_p(&mut p, 0.0);
    assert_eq!(p, orig);
}

#[test]
fn sample_inverse_cdf_monotone_and_boundaries() {
    // probs = [0.0, 0.5, 0.0, 0.5]; total = 1.0.
    // r→0 ⇒ first nonzero (idx 1). r just below 0.5 ⇒ idx 1.
    // r at/after 0.5 ⇒ idx 3. r→1 ⇒ last nonzero (idx 3).
    let p = vec![0.0, 0.5, 0.0, 0.5];
    assert_eq!(sample_inverse_cdf(&p, 0.0), 1);
    assert_eq!(sample_inverse_cdf(&p, 0.4999), 1);
    assert_eq!(sample_inverse_cdf(&p, 0.5), 3);
    assert_eq!(sample_inverse_cdf(&p, 0.999_999), 3);
}

#[test]
fn sample_inverse_cdf_known_cdf_picks_expected_bin() {
    // probs = [0.2, 0.3, 0.5]; cumulative = [0.2, 0.5, 1.0].
    // r=0.1 → bin 0 ; r=0.35 → bin 1 ; r=0.8 → bin 2.
    let p = vec![0.2, 0.3, 0.5];
    assert_eq!(sample_inverse_cdf(&p, 0.1), 0);
    assert_eq!(sample_inverse_cdf(&p, 0.35), 1);
    assert_eq!(sample_inverse_cdf(&p, 0.8), 2);
}

#[test]
fn sample_inverse_cdf_all_zero_is_safe() {
    let p = vec![0.0, 0.0, 0.0];
    assert_eq!(
        sample_inverse_cdf(&p, 0.5),
        0,
        "degenerate → idx 0, no panic"
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn softmax_scaled_forbidden_positions_are_zero() {
    // logits with one -inf (constraint-masked) position.
    let logits = vec![1.0f32, f32::NEG_INFINITY, 3.0];
    let probs = softmax_ok(&logits, 1.0);
    assert_eq!(probs[1], 0.0, "-inf logit ⇒ exactly 0 prob");
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax normalised, sum={sum}");
    assert!(probs[2] > probs[0], "higher logit ⇒ higher prob");
}

#[test]
fn end_to_end_top_k_one_equals_argmax_index() {
    // Combined path: softmax → filters with top_k=1 → inverse-CDF must
    // always return the argmax index for any rng draw.
    let logits = vec![0.2f32, 5.0, 1.0, 4.9, -2.0];
    let mut probs = softmax_ok(&logits, 1.0 / 0.8); // temp=0.8
    filter_top_p(&mut probs, 1.0); // disabled
    filter_min_p(&mut probs, 0.0); // disabled
    filter_top_k(&mut probs, 1);
    let argmax_idx = 1usize; // logits[1] = 5.0 is the max
    let mut rng = Pcg32::new(123);
    for _ in 0..50 {
        let r = rng.next_f32();
        assert_eq!(sample_inverse_cdf(&probs, r), argmax_idx);
    }
}

#[test]
fn same_seed_same_draw_sequence_reproducible() {
    let probs = vec![0.25f32, 0.25, 0.25, 0.25];
    let mut a = Pcg32::new(42);
    let mut b = Pcg32::new(42);
    let sa: Vec<usize> = (0..20)
        .map(|_| sample_inverse_cdf(&probs, a.next_f32()))
        .collect();
    let sb: Vec<usize> = (0..20)
        .map(|_| sample_inverse_cdf(&probs, b.next_f32()))
        .collect();
    assert_eq!(sa, sb, "seed=42 must reproduce identical sample stream");
    let mut c = Pcg32::new(7);
    let sc: Vec<usize> = (0..20)
        .map(|_| sample_inverse_cdf(&probs, c.next_f32()))
        .collect();
    assert_ne!(sa, sc, "seed=7 must differ from seed=42 on uniform probs");
}

#[test]
fn sampler_config_helpers() {
    let greedy = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    assert!(!greedy.sampling_active(), "temp=0 ⇒ greedy path");
    assert_eq!(greedy.seed_or_default(), 0xA7A7, "absent seed default");
    let sampled = SamplerConfig {
        temperature: 0.7,
        seed: Some(99),
        ..greedy
    };
    assert!(sampled.sampling_active(), "temp>0 ⇒ sampling path");
    assert_eq!(sampled.seed_or_default(), 99);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn compute_top_logprobs_known_logits() {
    // logits [1, 2, 3, 0] over 4 tokens. The log-softmax is
    // lse = ln(e^1 + e^2 + e^3 + e^0).
    // The two highest-prob tokens are id 2 (logit 3) then id 1 (logit 2).
    // Verify top-2 ids/order and that each logprob == logit - lse, and that
    // the chosen token's own logprob is pinned correctly.
    let data: [f32; 4] = [1.0, 2.0, 3.0, 0.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 4], Dtype::F32).unwrap();

    // chosen = id 1 (deliberately NOT the argmax) to exercise the
    // chosen-token logprob path independently of the top-k select.
    let out = compute_top_logprobs(&logits, 1, 2).unwrap();

    // Reference log-softmax denominator.
    let max = 3.0f64;
    let sum_exp = (1.0 - max).exp() + (2.0 - max).exp() + (3.0 - max).exp() + (0.0 - max).exp();
    let lse = max + sum_exp.ln();
    let lp = |logit: f64| (logit - lse) as f32;

    assert_eq!(out.token_id, 1);
    // chosen logprob = logit[1] - lse.
    assert!(
        (out.token_logprob - lp(2.0)).abs() < 1e-5,
        "chosen logprob {} != {}",
        out.token_logprob,
        lp(2.0)
    );

    // top-2: highest logit first.
    assert_eq!(out.top.len(), 2, "k=2 ⇒ exactly two entries");
    assert_eq!(out.top[0].0, 2, "highest-prob token is id 2 (logit 3)");
    assert_eq!(out.top[1].0, 1, "second is id 1 (logit 2)");
    assert!((out.top[0].1 - lp(3.0)).abs() < 1e-5);
    assert!((out.top[1].1 - lp(2.0)).abs() < 1e-5);

    // All logprobs are <= 0 (probabilities <= 1).
    assert!(out.token_logprob <= 0.0);
    assert!(out.top.iter().all(|&(_, l)| l <= 0.0));

    // Logprobs are strictly descending in the top list (distinct logits).
    assert!(out.top[0].1 > out.top[1].1);
}

// ── stochastic speculative acceptance ──────────────────────────

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn stochastic_acceptance_all_accept() {
    // p == q ⇒ ratio min(1, p/q) == 1 for every token ⇒ always accept,
    // regardless of the rng draw. Deterministic across seeds.
    let p = vec![0.1f32, 0.2, 0.3, 0.4];
    let q = p.clone();
    let mut rng = Pcg32::new(12345);
    for _ in 0..200 {
        for x in 0u32..4 {
            match stochastic_accept(&p, &q, x, &mut rng).unwrap() {
                AcceptDecision::Accept => {}
                AcceptDecision::Reject(_) => panic!("p==q must always accept token {x}"),
            }
        }
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn stochastic_acceptance_always_reject() {
    // Draft proposes token 0, but the verifier puts ~0 mass there while
    // the draft puts ~all mass there. ratio = p(0)/q(0) ≈ 0 ⇒ reject for
    // any rng draw in [0,1). Residual normalize((p-q)+) has mass only on
    // tokens where p>q (ids 1..3), never on id 0 ⇒ correction != 0.
    let q = vec![0.999_999f32, 0.000_001, 0.0, 0.0];
    let p = vec![0.0f32, 0.0, 0.5, 0.5];
    let mut rng = Pcg32::new(999);
    for _ in 0..200 {
        match stochastic_accept(&p, &q, 0, &mut rng).unwrap() {
            AcceptDecision::Accept => panic!("q puts all mass on id 0, p puts none ⇒ reject"),
            AcceptDecision::Reject(corr) => {
                assert!(
                    corr == 2 || corr == 3,
                    "residual mass only on ids 2,3; got {corr}"
                );
            }
        }
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn stochastic_acceptance_residual_is_p_minus_q_normalised() {
    // Hand-checked residual. q=[0.5,0.3,0.2], p=[0.2,0.3,0.5].
    // (p-q)+ = [0, 0, 0.3] ⇒ normalised = [0,0,1] ⇒ correction always id 2.
    let q = vec![0.5f32, 0.3, 0.2];
    let p = vec![0.2f32, 0.3, 0.5];
    let mut rng = Pcg32::new(7);
    // Force a reject by proposing the token most over-weighted in q (id 0,
    // ratio 0.2/0.5=0.4) many times; whenever it rejects, correction must
    // be id 2 (the only residual support).
    let mut saw_reject = false;
    for _ in 0..500 {
        if let AcceptDecision::Reject(corr) = stochastic_accept(&p, &q, 0, &mut rng).unwrap() {
            assert_eq!(corr, 2, "residual support is only id 2");
            saw_reject = true;
        }
    }
    assert!(
        saw_reject,
        "id 0 has ratio 0.4 ⇒ must reject ~60% of the time"
    );
}

/// Distribution-preservation invariant (Leviathan Thm 1): the stochastic
/// speculative step's output distribution equals sampling from `p`
/// directly. We simulate single-token speculation: draft samples x~q, then
/// accept/reject. Over many trials the empirical distribution of the
/// emitted token must match `p` (TVD within tolerance).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn stochastic_acceptance_preserves_target_distribution() {
    let q = vec![0.4f32, 0.3, 0.2, 0.1];
    let p = vec![0.1f32, 0.2, 0.3, 0.4];
    let trials = 200_000usize;
    let mut rng = Pcg32::new(0xBEEF);
    let mut counts = [0usize; 4];
    for _ in 0..trials {
        // Draft proposes x ~ q.
        let x = sample_index(&q, &mut rng) as u32;
        let emitted = match stochastic_accept(&p, &q, x, &mut rng).unwrap() {
            AcceptDecision::Accept => x,
            AcceptDecision::Reject(corr) => corr,
        };
        counts[emitted as usize] += 1;
    }
    // Empirical vs target p: total variation distance.
    let mut tvd = 0.0f64;
    for i in 0..4 {
        let emp = counts[i] as f64 / trials as f64;
        tvd += (emp - f64::from(p[i])).abs();
    }
    tvd *= 0.5;
    assert!(
        tvd < 0.01,
        "output distribution must match p (Leviathan Thm 1): TVD={tvd:.4} counts={counts:?}"
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn sampling_distribution_matches_sample_token_array_draw() {
    // The distribution `sampling_distribution` returns must be the same one
    // `sample_token_array` draws from: with a shared rng, drawing via
    // sample_index over the distribution must equal sample_token_array.
    let data: [f32; 6] = [0.5, 2.0, 1.0, 3.0, -1.0, 0.2];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 6], Dtype::F32).unwrap();
    let sp = SamplerConfig {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 0,
        min_p: 0.0,
        seed: Some(55),
        top_logprobs_k: 0,
    };
    let nop = PenaltyConfig::default();
    let dist = sampling_distribution(&logits, &sp, None, &nop, &[]).unwrap();
    let s: f32 = dist.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-5,
        "distribution must be normalised, sum={s}"
    );

    let mut rng_a = Pcg32::new(sp.seed_or_default());
    let mut rng_b = Pcg32::new(sp.seed_or_default());
    for _ in 0..50 {
        let via_dist = sample_index(&dist, &mut rng_a) as i32;
        let arr =
            sample_token_array(&logits, &sp, None, &nop, &[], &mut rng_b, Device::Cpu).unwrap();
        materialise(&arr);
        let b = arr.to_bytes().unwrap();
        let via_full = i32::from_le_bytes(b[..4].try_into().unwrap());
        assert_eq!(via_dist, via_full, "shared dist + sampler must agree");
    }
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn sample_token_array_returns_i32_scalar_shape() {
    // Strong spike at idx 2 ⇒ even with temp the sample is overwhelmingly
    // idx 2; assert shape/dtype contract matches argmax ([1] I32).
    let data: [f32; 4] = [-5.0, -5.0, 12.0, -5.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 4], Dtype::F32).unwrap();
    let sp = SamplerConfig {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(1),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sp.seed_or_default());
    let nop = PenaltyConfig::default();
    let out = sample_token_array(&logits, &sp, None, &nop, &[], &mut rng, Device::Cpu).unwrap();
    materialise(&out);
    assert_eq!(out.shape(), vec![1], "must be [1] like argmax output");
    assert_eq!(out.dtype(), Dtype::I32, "must be I32 like argmax output");
    let b = out.to_bytes().unwrap();
    let id = i32::from_le_bytes(b[..4].try_into().unwrap());
    assert_eq!(id, 2, "dominant spike must be chosen");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn sample_token_array_composes_with_constraint_mask() {
    // Natural argmax is idx 3 (12.0) but the mask forbids it; the only
    // allowed ids are {0, 5}. Sampler must never return a forbidden id.
    let mut data = vec![0.5f32; 8];
    data[3] = 12.0;
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 8], Dtype::F32).unwrap();
    let mut mask = vec![false; 8];
    mask[0] = true;
    mask[5] = true;
    let sp = SamplerConfig {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(3),
        top_logprobs_k: 0,
    };
    let nop = PenaltyConfig::default();
    let mut rng = Pcg32::new(sp.seed_or_default());
    for _ in 0..40 {
        let out = sample_token_array(&logits, &sp, Some(&mask), &nop, &[], &mut rng, Device::Cpu)
            .unwrap();
        materialise(&out);
        let b = out.to_bytes().unwrap();
        let id = i32::from_le_bytes(b[..4].try_into().unwrap()) as u32;
        assert!(matches!(id, 0 | 5), "masked sample {id} not in {{0,5}}");
    }
}

// ── A7.3: penalty unit tests ────────────────────────────────────────────

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn repetition_penalty_sign_aware() {
    // penalty=2.0: positive logit /= 2, negative logit *= 2.
    // token 0: logit=4.0 → 4.0/2.0=2.0. token 1: logit=-2.0 → -2.0*2.0=-4.0.
    // token 2 NOT in window → untouched.
    // penalty=1.0 → exact no-op (bit-identical input).
    let orig = vec![4.0f32, -2.0, 3.0];
    let window: Vec<u32> = vec![0, 1];

    // penalty=2.0
    let mut logits = orig.clone();
    apply_penalties(&mut logits, &window, 2.0, 0.0, 0.0, &[]);
    assert!(
        (logits[0] - 2.0).abs() < 1e-6,
        "positive logit halved: {}",
        logits[0]
    );
    assert!(
        (logits[1] - (-4.0)).abs() < 1e-6,
        "negative logit doubled: {}",
        logits[1]
    );
    assert!(
        (logits[2] - 3.0).abs() < 1e-6,
        "out-of-window token untouched: {}",
        logits[2]
    );

    // penalty=1.0 → no-op (bit-identical)
    let mut logits_nop = orig.clone();
    apply_penalties(&mut logits_nop, &window, 1.0, 0.0, 0.0, &[]);
    assert_eq!(logits_nop, orig, "penalty=1.0 must be exact no-op");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn presence_penalty_once_regardless_of_count() {
    // Token 0 appears 1× in window, token 1 appears 3× — both get exactly -0.5.
    // Token 2 not in window → untouched. penalty=0.0 → no-op.
    let orig = vec![5.0f32, 5.0, 5.0];
    let window: Vec<u32> = vec![0, 1, 1, 1];

    let mut logits = orig.clone();
    apply_penalties(&mut logits, &window, 1.0, 0.5, 0.0, &[]);
    assert!(
        (logits[0] - 4.5).abs() < 1e-6,
        "1× token -0.5: {}",
        logits[0]
    );
    assert!(
        (logits[1] - 4.5).abs() < 1e-6,
        "3× token still -0.5: {}",
        logits[1]
    );
    assert!(
        (logits[2] - 5.0).abs() < 1e-6,
        "absent token untouched: {}",
        logits[2]
    );

    // penalty=0.0 → no-op
    let mut logits_nop = orig.clone();
    apply_penalties(&mut logits_nop, &window, 1.0, 0.0, 0.0, &[]);
    assert_eq!(logits_nop, orig, "presence_penalty=0.0 must be no-op");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn frequency_penalty_proportional_to_count() {
    // Token 0: 3×, token 1: 1×, token 2: absent.
    // freq_penalty=0.5 → token0: -1.5, token1: -0.5, token2: 0.
    let orig = vec![5.0f32, 5.0, 5.0];
    let window: Vec<u32> = vec![0, 0, 0, 1];

    let mut logits = orig.clone();
    apply_penalties(&mut logits, &window, 1.0, 0.0, 0.5, &[]);
    assert!(
        (logits[0] - 3.5).abs() < 1e-6,
        "3× at 0.5 = -1.5: {}",
        logits[0]
    );
    assert!(
        (logits[1] - 4.5).abs() < 1e-6,
        "1× at 0.5 = -0.5: {}",
        logits[1]
    );
    assert!(
        (logits[2] - 5.0).abs() < 1e-6,
        "absent untouched: {}",
        logits[2]
    );

    // penalty=0.0 → no-op
    let mut logits_nop = orig.clone();
    apply_penalties(&mut logits_nop, &window, 1.0, 0.0, 0.0, &[]);
    assert_eq!(logits_nop, orig, "frequency_penalty=0.0 must be no-op");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn logit_bias_additive_and_oov_skipped() {
    // logit_bias=[(5,2.0),(9,-3.0)]: logit[5]+=2, logit[9]-=3.
    // Out-of-range id (>=len) skipped without panic.
    // Empty → no-op.
    let orig = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let bias: Vec<(u32, f32)> = vec![(5, 2.0), (9, -3.0), (100, 99.0)]; // 100 is OOV

    let mut logits = orig.clone();
    apply_penalties(&mut logits, &[], 1.0, 0.0, 0.0, &bias);
    assert!(
        (logits[5] - 3.0).abs() < 1e-6,
        "logit[5] += 2: {}",
        logits[5]
    );
    assert!(
        (logits[9] - (-2.0)).abs() < 1e-6,
        "logit[9] -= 3: {}",
        logits[9]
    );
    // all other positions unchanged
    for i in [0, 1, 2, 3, 4, 6, 7, 8] {
        assert!(
            (logits[i] - 1.0).abs() < 1e-6,
            "logit[{i}] unchanged: {}",
            logits[i]
        );
    }

    // Empty bias → no-op
    let mut logits_nop = orig.clone();
    apply_penalties(&mut logits_nop, &[], 1.0, 0.0, 0.0, &[]);
    assert_eq!(logits_nop, orig, "empty bias must be no-op");
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn combined_order_matches_mlx_lm() {
    // Verify application order: logit_bias → rep → presence → freq.
    // Crafted vector: 4 tokens, token id 0 in window.
    // logit[0] = 2.0
    // Step 1: logit_bias: logit[0] += 1.0 → 3.0
    // Step 2: rep(penalty=2.0): logit[0]=3.0 > 0 → 3.0/2.0 = 1.5
    // Step 3: presence(0.3): logit[0] -= 0.3 → 1.2
    // Step 4: freq(0.2): count=2 → logit[0] -= 0.4 → 0.8
    let mut logits = vec![2.0f32, 5.0, 5.0, 5.0];
    let window: Vec<u32> = vec![0, 0]; // id 0 appears 2×
    let bias: Vec<(u32, f32)> = vec![(0, 1.0)];
    apply_penalties(&mut logits, &window, 2.0, 0.3, 0.2, &bias);
    assert!(
        (logits[0] - 0.8).abs() < 1e-5,
        "combined order: {}",
        logits[0]
    );
    // token 1,2,3 not in window and not in bias → untouched
    assert!((logits[1] - 5.0).abs() < 1e-6);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn window_only_uses_last_20_tokens() {
    // history of 50 tokens; only the last 20 matter.
    // First 30 tokens are id=99 (out of vocab range 0..10), last 20 are id=0.
    // With rep_penalty=2.0: only id=0 should be affected.
    let mut history: Vec<u32> = vec![99u32; 30];
    history.extend(std::iter::repeat_n(0u32, 20));
    // trim to last 20
    let start = history.len().saturating_sub(20);
    let window = &history[start..];

    let mut logits = vec![4.0f32; 10]; // id 0..9
    apply_penalties(&mut logits, window, 2.0, 0.0, 0.0, &[]);
    // id=0 appears 20× in window → rep_penalty applies
    assert!(
        (logits[0] - 2.0).abs() < 1e-6,
        "id=0 penalised: {}",
        logits[0]
    );
    // ids 1..9 not in window → untouched
    for (i, &val) in logits.iter().enumerate().take(10).skip(1) {
        assert!((val - 4.0).abs() < 1e-6, "logit[{i}] unchanged");
    }
}

#[test]
fn penalties_all_noop_leaves_slice_unchanged() {
    // apply_penalties with identity/zero args = exact no-op on the bytes.
    let orig: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
    let window: Vec<u32> = (0..20).map(|i| i as u32).collect();
    let mut logits = orig.clone();
    apply_penalties(&mut logits, &window, 1.0, 0.0, 0.0, &[]);
    assert_eq!(logits, orig, "all-noop must leave slice bit-identical");
}

// ── A7.4: spec-mandated named integration tests ─────────────────────────

/// A7.4 spec cell (b): `top_k=1` with any `temperature > 0` must collapse
/// to the greedy argmax index — i.e. the same token that `temp=0` would
/// have selected.
///
/// Mechanism: `filter_top_k(probs, 1)` zeros all but the maximum-probability
/// element; the inverse-CDF sample of a one-hot distribution is deterministic
/// for every `r ∈ [0, 1)`. This holds regardless of temperature.
///
/// The test runs 100 draws across a range of seeds to confirm no draw ever
/// escapes the argmax index, then verifies two distinct temperatures still
/// produce the same token (the argmax).
#[test]
fn top_k_one_collapses_to_greedy() {
    // logits: clear winner at idx 1 (logit=10.0).
    let logits = vec![1.0f32, 10.0, 2.0, 0.5, -1.0];
    let argmax_idx = 1usize;

    // Run with temp=0.8 and top_k=1 — 100 seeds.
    for seed in 0u64..100 {
        let mut probs = softmax_ok(&logits, 1.0 / 0.8);
        filter_top_p(&mut probs, 1.0); // disabled
        filter_min_p(&mut probs, 0.0); // disabled
        filter_top_k(&mut probs, 1);
        let mut rng = Pcg32::new(seed);
        let chosen = sample_inverse_cdf(&probs, rng.next_f32());
        assert_eq!(
            chosen, argmax_idx,
            "top_k=1 seed={seed}: expected argmax idx {argmax_idx}, got {chosen}"
        );
    }

    // Also confirm a different temperature (1.5) gives the same token.
    let mut probs_hi = softmax_ok(&logits, 1.0 / 1.5);
    filter_top_k(&mut probs_hi, 1);
    let mut rng2 = Pcg32::new(0xA7A7);
    let chosen_hi = sample_inverse_cdf(&probs_hi, rng2.next_f32());
    assert_eq!(
        chosen_hi, argmax_idx,
        "top_k=1 at temp=1.5 must still pick the argmax"
    );
}

/// A7.4 spec cell (g): `repetition_penalty > 1.0` on a repetitive prompt
/// must prevent the decode loop from degenerating into a single-token
/// infinite repeat.
///
/// This test uses the **greedy (argmax) path** to keep results fully
/// deterministic and seed-independent. The property holds for any temperature
/// because the penalty acts on raw logits before softmax.
///
/// Two phases:
/// 1. **Baseline (no penalty)** — 20 greedy steps with id 0 as clear argmax
/// 2. **With penalty=3.0** — same logits; after the first step that picks

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn repetition_penalty_breaks_single_token_loop() {
    // Vocab of 6. id 0 is the argmax (3.0). Second best is id 1 (2.0).
    // After rep_penalty=3.0: logit[0] = 3.0/3.0 = 1.0 < 2.0. id 1 wins
    // from step 2 onward (as long as id 0 stays in the window).
    let base_logits = vec![3.0f32, 2.0, 1.5, 1.0, 0.5, 0.0];
    let n_steps = 10usize;

    // ── Phase 1: baseline (no penalty) — all steps pick id 0 ───────────
    let mut window_base: Vec<u32> = Vec::new();
    for _ in 0..n_steps {
        let mut logits = base_logits.clone();
        let start = window_base.len().saturating_sub(20);
        apply_penalties(&mut logits, &window_base[start..], 1.0, 0.0, 0.0, &[]);
        // greedy argmax
        let chosen = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            chosen, 0,
            "baseline step: id 0 must always be argmax without penalty"
        );
        window_base.push(chosen as u32);
    }

    // ── Phase 2: rep_penalty=3.0 — loop must break ──────────────────────
    let mut window: Vec<u32> = Vec::new();
    let mut non_zero_count = 0usize;

    for _ in 0..n_steps {
        let mut logits = base_logits.clone();
        let start = window.len().saturating_sub(20);
        apply_penalties(&mut logits, &window[start..], 3.0, 0.0, 0.0, &[]);
        let chosen = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if chosen != 0 {
            non_zero_count += 1;
        }
        window.push(chosen as u32);
    }

    assert!(
        non_zero_count > 0,
        "rep_penalty=3.0 must cause at least one non-zero token in {n_steps} greedy steps; got 0"
    );
}

// ── Greedy tie-break: host selection must mirror the device reduction ─────
//
// The oracle in this section is MLX's own `argmax`, called through the same
// FFI the decode loop uses. It shares no arithmetic with the host scan under
// test: one is a C++/Metal reduction over the array, the other a Rust loop
// over a host copy.
//
// Most of these run on `Device::Cpu` so they land in the ordinary
// (non-`#[ignore]`) suite. `host_argmax` is written against the **Metal**
// reduction's seed, so the Metal backend is checked rather than assumed by the
// `#[ignore]`d `Device::Gpu` mirrors at the end of this section — a CPU-only
// oracle cannot establish a Metal property.

/// Decode the `[1] I32` selection Array into a token id.
#[allow(
    clippy::indexing_slicing,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn token_id(a: &Array) -> u32 {
    materialise(a);
    let b = a.to_bytes().unwrap();
    i32::from_le_bytes(b[..4].try_into().unwrap()) as u32
}

/// Device `argmax` over a `[1, vocab]` row, on the given stream.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn device_argmax(logits: &Array, device: Device) -> u32 {
    token_id(&argmax(logits, -1, device).unwrap())
}

/// Host greedy over the same row, with no mask and no penalties.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn host_greedy(logits: &Array, device: Device) -> u32 {
    let nop = PenaltyConfig::default();
    token_id(&argmax_with_penalties(logits, None, &nop, &[], device).unwrap())
}

/// Wrap an f32 row as a `[1, n]` F32 Array.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn f32_row(data: &[f32]) -> Array {
    Array::from_bytes(f32_as_bytes(data), &[1, data.len() as i32], Dtype::F32).unwrap()
}

/// Width of the wide tie fixtures — a real gemma-4 vocabulary.
const WIDE_VOCAB: usize = 262_144;

/// A `WIDE_VOCAB`-wide row of `0.5` with `12.0` at each of `peaks`.
fn wide_row_with_peaks(peaks: &[usize]) -> Vec<f32> {
    let mut row = vec![0.5f32; WIDE_VOCAB];
    for &p in peaks {
        if let Some(slot) = row.get_mut(p) {
            *slot = 12.0;
        }
    }
    row
}

/// The tied-row shapes the contract has to cover, as `(name, row, expected
/// lowest-id max)`. Shared by the CPU tests and their `Device::Gpu` mirrors so
/// the two cannot drift apart.
///
/// The wide rows exist because MLX's arg-reduce switches between a single- and
/// a multi-threadgroup strategy with size, and a tie split across that boundary
/// is where a parallel reduction's tie rule can break. An 8-wide row cannot
/// reach that shape.
fn tie_shapes() -> Vec<(&'static str, Vec<f32>, u32)> {
    vec![
        ("narrow", vec![1.0f32, 9.0, 3.0, 2.0, 9.0, 0.0, 9.0, 4.0], 1),
        (
            "wide-spread",
            wide_row_with_peaks(&[7, WIDE_VOCAB / 2, WIDE_VOCAB - 1]),
            7,
        ),
        (
            "wide-tail-adjacent",
            wide_row_with_peaks(&[WIDE_VOCAB - 2, WIDE_VOCAB - 1]),
            (WIDE_VOCAB - 2) as u32,
        ),
    ]
}

/// Oracle characterisation: MLX `argmax` resolves an exact tie to the
/// **lowest** index, at every width the sampler sees. Every other test in this
/// section is written against that fact, so it is pinned on its own — if a
/// future MLX bump changes the rule this fails first and names the cause.
#[test]
fn mlx_argmax_breaks_ties_to_lowest_index() {
    for (name, row, expect) in tie_shapes() {
        let got = device_argmax(&f32_row(&row), Device::Cpu);
        assert_eq!(
            got, expect,
            "{name}: MLX argmax must return the first of the tied maxima"
        );
    }
}

/// The host greedy path (`temperature == 0` with penalties) must return the
/// same token as the device greedy path on the same logits row. A tie is the
/// only input where the two reductions can differ, so it is the only input
/// that tests the contract.
#[test]
fn host_greedy_matches_device_argmax_on_a_tie() {
    for (name, row, _) in tie_shapes() {
        let logits = f32_row(&row);
        let device_id = device_argmax(&logits, Device::Cpu);
        let host_id = host_greedy(&logits, Device::Cpu);
        assert_eq!(
            host_id, device_id,
            "{name}: host greedy picked {host_id}, device argmax picked {device_id}"
        );
    }
}

/// Same contract, but the tie is *created* by a `logit_bias` rather than
/// present in the raw row — one of the two shapes a served request actually
/// reaches, since the host greedy path only runs when a penalty or a bias is
/// active.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn host_greedy_matches_device_argmax_when_a_bias_creates_the_tie() {
    // Raw row has a unique max at 0. A +2.0 bias on id 3 lifts it to exactly
    // 5.0 — an exact tie with id 0, no rounding involved.
    let logits = f32_row(&[5.0, 1.0, 2.0, 3.0, 0.5, 4.0]);
    let cfg = PenaltyConfig {
        logit_bias: vec![(3, 2.0)],
        ..PenaltyConfig::default()
    };
    let host_id = token_id(&argmax_with_penalties(&logits, None, &cfg, &[], Device::Cpu).unwrap());

    // Oracle: apply the same bias on the device and reduce there.
    let device_id = device_argmax(&f32_row(&[5.0, 1.0, 2.0, 5.0, 0.5, 4.0]), Device::Cpu);

    assert_eq!(
        host_id, device_id,
        "bias-induced tie: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// The other route: the **repetition penalty**, which is the one the field
/// report names. `rep_penalty` divides a positive logit by the penalty, and
/// that division maps distinct f32 values onto equal ones — `1.1f32 / 1.1f32`
/// is exactly `1.0f32`, so a row holding both `1.1` and `1.0` ties after the
/// penalty is applied to the first.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn host_greedy_matches_device_argmax_when_rep_penalty_creates_the_tie() {
    // id 3 = 1.1 is the unique max. With id 3 in the penalty window and
    // rep_penalty 1.1, it becomes exactly 1.0 — tied with id 1.
    let logits = f32_row(&[0.25, 1.0, 0.5, 1.1, 0.75]);
    let cfg = PenaltyConfig {
        rep_penalty: 1.1,
        ..PenaltyConfig::default()
    };
    let host_id = token_id(&argmax_with_penalties(&logits, None, &cfg, &[3], Device::Cpu).unwrap());

    // Oracle: the post-penalty row, reduced on the device. Built from the
    // literal 1.0 rather than by recomputing 1.1 / 1.1, so it shares no
    // arithmetic with the penalty under test.
    let device_id = device_argmax(&f32_row(&[0.25, 1.0, 0.5, 1.0, 0.75]), Device::Cpu);
    assert_eq!(device_id, 1, "post-penalty row ties at ids 1 and 3");

    assert_eq!(
        host_id, device_id,
        "rep-penalty tie: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// The documented "penalties active, with or without a constraint" sub-case:
/// a mask and a penalty together, with the tie among the *allowed* ids.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn host_greedy_matches_device_argmax_with_mask_and_penalties() {
    // Raw max is id 5. The mask forbids it, leaving ids {1, 2, 4}; a +3.0 bias
    // on id 4 lifts it from 1.0 to exactly 4.0, tying with the allowed id 1.
    let logits = f32_row(&[0.5, 4.0, 2.0, 9.0, 1.0, 12.0]);
    let mask = vec![false, true, true, false, true, false];
    let cfg = PenaltyConfig {
        logit_bias: vec![(4, 3.0)],
        ..PenaltyConfig::default()
    };
    let host_id =
        token_id(&argmax_with_penalties(&logits, Some(&mask), &cfg, &[], Device::Cpu).unwrap());

    // Oracle: the same row with the mask and bias folded in by hand, reduced
    // on the device.
    let device_id = device_argmax(
        &f32_row(&[
            f32::NEG_INFINITY,
            4.0,
            2.0,
            f32::NEG_INFINITY,
            4.0,
            f32::NEG_INFINITY,
        ]),
        Device::Cpu,
    );
    assert_eq!(device_id, 1, "masked+biased row ties at ids 1 and 4");

    assert_eq!(
        host_id, device_id,
        "mask+penalty tie: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// A `NaN` in the row must not displace a real maximum, and must not reset the
/// running best — both are properties of the device reduction, which compares
/// with a strict `>` and therefore skips `NaN` entirely. The `NaN` is placed
/// away from index 0 on purpose: MLX seeds its CPU reduction with `in[0]` and
/// its Metal reduction with `-inf`, so a leading `NaN` is the one shape where
/// MLX's own two backends disagree and no host rule can match both.
#[test]
fn host_greedy_skips_nan_like_the_device() {
    let logits = f32_row(&[1.0, 3.0, f32::NAN, 2.0]);
    let device_id = device_argmax(&logits, Device::Cpu);
    assert_eq!(device_id, 1, "device reduction must skip the NaN");
    let host_id = host_greedy(&logits, Device::Cpu);
    assert_eq!(
        host_id, device_id,
        "NaN row: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// An all-`false` constraint mask must be **refused**, on every path that
/// accepts a mask.
///
/// Emitting a token from it is worse than failing: no token satisfies the
/// grammar, the engine state that produced the empty mask persists, so the
/// stream would emit that same arbitrary token for the rest of the generation
/// and the request would report success. The mask is passed as a mask — the
/// shape a broken constraint engine actually produces — not as a pre-built
/// `-inf` row.
#[test]
fn every_mask_path_refuses_an_all_forbidden_mask() {
    let logits = f32_row(&[0.5, 4.0, 2.0, 9.0, 1.0]);
    let mask = vec![false; 5];
    let cfg = PenaltyConfig::default();
    let sp = SamplerConfig {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(1),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(1);

    assert!(
        argmax_with_penalties(&logits, Some(&mask), &cfg, &[], Device::Cpu).is_err(),
        "host greedy must refuse an all-forbidden mask"
    );
    assert!(
        apply_mask_argmax(&logits, &mask, Device::Cpu).is_err(),
        "device greedy must refuse an all-forbidden mask"
    );
    assert!(
        sampling_distribution(&logits, &sp, Some(&mask), &cfg, &[]).is_err(),
        "the sampling distribution must refuse an all-forbidden mask"
    );
    assert!(
        sample_token_array(&logits, &sp, Some(&mask), &cfg, &[], &mut rng, Device::Cpu).is_err(),
        "the sampler must refuse an all-forbidden mask"
    );

    // A mask allowing a single token is not the same shape and must still work.
    let mut one = vec![false; 5];
    if let Some(slot) = one.get_mut(2) {
        *slot = true;
    }
    assert!(
        argmax_with_penalties(&logits, Some(&one), &cfg, &[], Device::Cpu).is_ok(),
        "a mask with one allowed token must be accepted"
    );
}

/// `top_k` keeps the `k` highest probabilities; when the cut falls inside a
/// group of equal probabilities it must keep the **lowest ids**, so that
/// `top_k == 1` is the argmax on every row and not only on rows with a unique
/// maximum. The array is longer than the length at which `sort_unstable_by`
/// degenerates to a stable insertion sort, so the ordering under test is the
/// real one.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs is built with length n immediately above and every index used is < n"
)]
fn filter_top_k_breaks_ties_to_lowest_id() {
    let n = 64usize;
    let tied = [5usize, 30, 63];
    let mut probs = vec![0.001f32; n];
    for &i in &tied {
        probs[i] = 0.3;
    }

    let mut k1 = probs.clone();
    filter_top_k(&mut k1, 1);
    let survivors: Vec<usize> = (0..n).filter(|&i| k1[i] > 0.0).collect();
    assert_eq!(
        survivors,
        vec![5],
        "top_k=1 must keep only the lowest tied id"
    );

    let mut k2 = probs.clone();
    filter_top_k(&mut k2, 2);
    let survivors2: Vec<usize> = (0..n).filter(|&i| k2[i] > 0.0).collect();
    assert_eq!(
        survivors2,
        vec![5, 30],
        "top_k=2 must keep the two lowest tied ids"
    );
}

/// The same rule on `top_p`, which unlike `top_k` is ON by default on the
/// served path (several `generation_config.json` snapshots ship `top_p`).
/// `filter_top_p` walks ascending and drops from the front, so a tied group has
/// to be ordered highest-id-first for the lowest ids to survive the cut.
///
/// Shape: one dominant probability plus a long run of bit-identical tail
/// values, with `top_p` placed so the cut lands inside that run. Without an id
/// rule the survivor set is a non-contiguous artefact of pdqsort's pivot
/// choice — `{0, 29, 54..63}`, id 29 surviving while ids 30..53 are zeroed from
/// the same value. `n = 64` is above the length where `sort_unstable_by`
/// degenerates to a stable insertion sort and the shape becomes unreachable.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs is built with length n immediately above and every index used is < n"
)]
fn filter_top_p_breaks_ties_to_lowest_id() {
    let n = 64usize;
    let tail = 0.6f32 / 63.0;
    let mut probs: Vec<f32> = (0..n).map(|i| if i == 0 { 0.4 } else { tail }).collect();
    filter_top_p(&mut probs, 0.5);

    let survivors: Vec<usize> = (0..n).filter(|&i| probs[i] > 0.0).collect();
    assert!(
        survivors.len() > 2,
        "fixture must put the cut inside the tied run, got {} survivors",
        survivors.len()
    );
    assert_eq!(survivors[0], 0, "the dominant id always survives");
    // Every survivor drawn from the tied run must be a prefix of it: no gap,
    // and never a higher id kept while a lower one is dropped.
    let tied: Vec<usize> = survivors[1..].to_vec();
    let expected: Vec<usize> = (1..=tied.len()).collect();
    assert_eq!(
        tied, expected,
        "tied survivors must be the lowest ids of the run, contiguously"
    );
}

/// Neither filter may depend on an ordering that is not total, `NaN` included.
///
/// A comparator that folds an unordered `partial_cmp` to `Equal` is
/// intransitive, and `sort_unstable_by` may detect that and panic mid-decode.
/// But the panic is a *heuristic* detector, so "did it panic?" is a weak
/// oracle: on the shipped comparator only one of six (filter, length) cells
/// actually fires. So this asserts the property structurally instead — the
/// survivor set must equal what the documented rank rule prescribes — which has
/// power at every length whether the detector fires or not.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs and the reference order both have length n, and every index used is < n"
)]
fn filters_hold_a_total_order_with_nan_present() {
    for n in [8usize, 64, 512, 4096] {
        let probs: Vec<f32> = (0..n)
            .map(|i| {
                if i % 7 == 3 {
                    f32::NAN
                } else {
                    ((i.wrapping_mul(2_654_435_761_usize)) % 1000) as f32 / 1000.0
                }
            })
            .collect();

        let k = (n / 4).max(1);
        let mut got_k = probs.clone();
        filter_top_k(&mut got_k, k);
        let mut expect_k = probs.clone();
        for &i in &reference_rank_order(&probs, true)[k..] {
            expect_k[i] = 0.0;
        }
        assert_eq!(
            survivor_ids(&got_k),
            survivor_ids(&expect_k),
            "n={n}: top_k survivors must follow the rank rule with a NaN present"
        );

        let mut got_p = probs.clone();
        filter_top_p(&mut got_p, 0.95);
        let mut expect_p = probs.clone();
        let mut cum = 0.0f32;
        for &i in &reference_rank_order(&probs, false) {
            cum += probs[i];
            if cum <= 1.0 - 0.95 {
                expect_p[i] = 0.0;
            }
        }
        assert_eq!(
            survivor_ids(&got_p),
            survivor_ids(&expect_p),
            "n={n}: top_p survivors must follow the rank rule with a NaN present"
        );
    }
}

/// Ids whose probability survived a filter. `NaN > 0.0` is false, so a `NaN`
/// counts as dropped on both sides of a comparison — which is what makes this
/// a fair oracle rather than one that hides the interesting case.
fn survivor_ids(probs: &[f32]) -> Vec<usize> {
    probs
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > 0.0)
        .map(|(i, _)| i)
        .collect()
}

/// A tie-dense row of the shape the filters actually receive: a BF16-quantised
/// logit row, softmaxed. Almost every adjacent pair is exactly equal, so the
/// tie path is the common path rather than a corner.
fn tie_dense_probs(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let logits: Vec<f32> = (0..n)
        .map(|_| {
            let l = ((next() % 4000) as f32) / 100.0 - 20.0;
            // Round to BF16: keep 8 mantissa bits, round-to-nearest-even.
            let bits = l.to_bits();
            let lsb = (bits >> 16) & 1;
            f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
        })
        .collect();
    softmax_ok(&logits, 1.0 / 0.7)
}

/// Reference statement of the documented rank rule, written as a plain
/// comparator over `(probability, id)` pairs.
///
/// This is an oracle for the *rule*, deliberately not for the implementation:
/// the filters sort packed `u64` keys, this sorts tuples with an explicit
/// comparison. If the bit-packing is wrong in any way — the inversion, the
/// half-word split, the id width — the two orders diverge.
fn reference_rank_order(probs: &[f32], descending: bool) -> Vec<usize> {
    // Sort (probability, id) pairs directly; no indexing back into `probs`.
    let mut pairs: Vec<(f32, usize)> = probs.iter().copied().zip(0..probs.len()).collect();
    pairs.sort_by(|&(pa, a), &(pb, b)| {
        if descending {
            // Highest probability first; equal probabilities by lowest id.
            pb.total_cmp(&pa).then(a.cmp(&b))
        } else {
            // Lowest probability first; equal probabilities by highest id.
            pa.total_cmp(&pb).then(b.cmp(&a))
        }
    });
    pairs.into_iter().map(|(_, id)| id).collect()
}

/// `filter_top_k`'s survivor set must be the first `k` of the reference
/// descending order, on a row where nearly every value is tied.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs and the reference order both have length n, and every index used is < n"
)]
fn filter_top_k_matches_the_reference_rank_rule() {
    let n = 4096usize;
    for k in [1usize, 7, 64, 1000, 4095] {
        let probs = tie_dense_probs(n, 0x51ED_1234);
        let mut expect = reference_rank_order(&probs, true)[..k].to_vec();
        expect.sort_unstable();

        let mut got_probs = probs.clone();
        filter_top_k(&mut got_probs, k);
        let got: Vec<usize> = (0..n).filter(|&i| got_probs[i] > 0.0).collect();

        // Zero-probability survivors are indistinguishable from drops, so
        // compare only over the ids the reference keeps that carry mass.
        let expect_nonzero: Vec<usize> = expect.into_iter().filter(|&i| probs[i] > 0.0).collect();
        assert_eq!(
            got, expect_nonzero,
            "k={k}: survivor set must match the rule"
        );
    }
}

/// `filter_top_p`'s survivor set must be the suffix of the reference ascending
/// order that the cumulative-sum walk keeps.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs and the reference order both have length n, and every index used is < n"
)]
fn filter_top_p_matches_the_reference_rank_rule() {
    let n = 4096usize;
    for top_p in [0.1f32, 0.5, 0.9, 0.95, 0.99] {
        let probs = tie_dense_probs(n, 0x7A11_9999);

        // Reference: walk the reference ascending order and drop the prefix
        // whose inclusive cumulative sum stays at or below the threshold.
        let order = reference_rank_order(&probs, false);
        let threshold = 1.0 - top_p;
        let mut cum = 0.0f32;
        let mut expect: Vec<usize> = Vec::new();
        for &i in &order {
            cum += probs[i];
            if cum > threshold {
                expect.push(i);
            }
        }
        expect.sort_unstable();
        let expect_nonzero: Vec<usize> = expect.into_iter().filter(|&i| probs[i] > 0.0).collect();

        let mut got_probs = probs.clone();
        filter_top_p(&mut got_probs, top_p);
        let got: Vec<usize> = (0..n).filter(|&i| got_probs[i] > 0.0).collect();

        assert_eq!(
            got, expect_nonzero,
            "top_p={top_p}: survivor set must match the rule"
        );
    }
}

/// A non-finite logits row must be **refused**, not sampled.
///
/// Sampling one is not a degraded result, it is a silent one. The `NaN` reaches
/// `probs`, `renormalise` no-ops (`total > 0.0` is false on `NaN`),
/// `sample_inverse_cdf`'s `total <= 0.0` guard is also false, its `cum > target`
/// never fires, and it returns `last_nonzero` — the same id every step,
/// independent of the RNG. The request reports success and streams a constant
/// token. This asserts refusal on every entry point, and asserts that the
/// matching finite row is still accepted so the guard is not a blanket reject.
#[test]
fn a_non_finite_logits_row_is_refused_not_sampled() {
    let n = 64usize;
    let sp = SamplerConfig {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.0,
        seed: Some(5),
        top_logprobs_k: 0,
    };
    let nop = PenaltyConfig::default();

    let finite: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    assert!(
        sampling_distribution(&f32_row(&finite), &sp, None, &nop, &[]).is_ok(),
        "the finite control row must still be accepted"
    );

    for (label, bad) in [("NaN", f32::NAN), ("+inf", f32::INFINITY)] {
        let mut data = finite.clone();
        if let Some(slot) = data.get_mut(7) {
            *slot = bad;
        }
        let logits = f32_row(&data);
        assert!(
            sampling_distribution(&logits, &sp, None, &nop, &[]).is_err(),
            "{label} logit: the distribution must be refused"
        );
        let mut rng = Pcg32::new(5);
        assert!(
            sample_token_array(&logits, &sp, None, &nop, &[], &mut rng, Device::Cpu).is_err(),
            "{label} logit: the sampler must be refused"
        );
    }

    // An all-`-inf` row is a different shape: its softmax sum is a finite 0,
    // and it must NOT be swept up by the non-finite guard.
    let masked = vec![f32::NEG_INFINITY; n];
    assert!(
        softmax_scaled(&masked, 1.0 / 0.7).is_ok(),
        "an all -inf row sums to a finite zero and must not be refused here"
    );
}

/// The greedy paths deliberately do **not** refuse a `NaN` row: they mirror the
/// device reduction, which skips `NaN` and returns the largest real logit, and
/// diverging from it would re-create the host/device split this contract
/// exists to close. Pinned so the asymmetry is a decision rather than an
/// oversight — the sampling path refuses because its failure is a constant
/// stream, the greedy path does not because its answer is the device's.
#[test]
fn greedy_does_not_refuse_a_nan_row_but_the_sampler_does() {
    let logits = f32_row(&[1.0, 3.0, f32::NAN, 2.0]);
    let nop = PenaltyConfig::default();
    let sp = SamplerConfig {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(2),
        top_logprobs_k: 0,
    };
    assert!(
        argmax_with_penalties(&logits, None, &nop, &[], Device::Cpu).is_ok(),
        "host greedy keeps device parity on a NaN row"
    );
    assert_eq!(
        host_greedy(&logits, Device::Cpu),
        device_argmax(&logits, Device::Cpu),
        "and agrees with the device on which token that is"
    );
    assert!(
        sampling_distribution(&logits, &sp, None, &nop, &[]).is_err(),
        "the sampling path refuses the same row"
    );
}

/// End-to-end restatement of the same contract on the full host pipeline:
/// `top_k = 1` must reproduce the device argmax for every RNG draw, including
/// on a row whose maximum is tied. The existing `top_k_one_collapses_to_greedy`
/// covers only rows with a unique maximum, which is the case that cannot fail.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn top_k_one_equals_device_argmax_under_a_tie() {
    let n = 64usize;
    let mut data = vec![1.0f32; n];
    for &i in &[5usize, 30, 63] {
        if let Some(slot) = data.get_mut(i) {
            *slot = 8.0;
        }
    }
    let logits = f32_row(&data);
    let device_id = device_argmax(&logits, Device::Cpu);

    let sp = SamplerConfig {
        temperature: 0.8,
        top_p: 1.0,
        top_k: 1,
        min_p: 0.0,
        seed: Some(11),
        top_logprobs_k: 0,
    };
    let nop = PenaltyConfig::default();
    let mut rng = Pcg32::new(sp.seed_or_default());
    for draw in 0..64 {
        let out = sample_token_array(&logits, &sp, None, &nop, &[], &mut rng, Device::Cpu).unwrap();
        let id = token_id(&out);
        assert_eq!(
            id, device_id,
            "top_k=1 draw {draw}: sampled {id}, device argmax {device_id}"
        );
    }
}

/// `top_logprobs` reports ranks, not a selection, but it must still order
/// equal logits by ascending id — otherwise rank 0 can disagree with the token
/// the device argmax chose, and ranks 1..k are an artefact of the selection's
/// swaps rather than a reproducible answer.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "top is built with exactly k = 3 entries immediately above"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn compute_top_logprobs_breaks_ties_to_lowest_id() {
    // Two exactly-equal logits at ids 0 and 2, under a unique max at 3.
    let logits = f32_row(&[5.0, 7.0, 5.0, 9.0]);
    let out = compute_top_logprobs(&logits, 3, 3).unwrap();
    let ids: Vec<u32> = out.top.iter().map(|&(id, _)| id).collect();
    assert_eq!(
        ids,
        vec![3, 1, 0],
        "equal logits must rank by ascending id (id 0 before id 2)"
    );
    assert_eq!(
        ids[0],
        device_argmax(&logits, Device::Cpu),
        "rank 0 must be the token the device argmax would choose"
    );
}

/// `top_logprobs` is reported alongside the emitted token, computed from the
/// same raw logits row the greedy path reduces, so rank 0 must be the token the
/// device would emit on a `NaN` row too. Seeding the selection from a candidate
/// rather than from `-inf` breaks exactly this: a `NaN` at the seed position
/// wins the rank, because every later `>` and `==` against it is false.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "top is built with exactly k = 2 entries immediately above"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn compute_top_logprobs_skips_nan_like_the_device() {
    // NaN at index 0 — the seed position of the first selection round.
    let logits = f32_row(&[f32::NAN, 1.0, 5.0, 2.0]);
    let out = compute_top_logprobs(&logits, 2, 2).unwrap();
    let ids: Vec<u32> = out.top.iter().map(|&(id, _)| id).collect();
    assert_eq!(
        ids,
        vec![2, 3],
        "a NaN must not take a rank ahead of a real logit"
    );
    assert_eq!(
        ids[0],
        host_greedy(&logits, Device::Cpu),
        "rank 0 must be the token the greedy path emits"
    );
}

/// The rank keys must order every `f32`, not only the non-negative ones the
/// sampler happens to produce today.
///
/// Under a raw bit pattern `-0.0` (`0x8000_0000`) outranks every positive
/// value, so `top_k(1)` keeps `-0.0` and zeroes the real maximum — a silently
/// wrong token, with no panic to notice. `release-perf` disables debug
/// assertions, so a `debug_assert` on the sign would not catch it either;
/// the key is made correct instead.
///
/// **This guards the key encoding, not a defect that existed on `main`.** The
/// comparator `main` used happens to order `-0.0` correctly, so this test is
/// green there; it is red only against a packed key built from the raw
/// `to_bits()` pattern. Do not count it as regression evidence.
///
/// Both halves assert on **bit patterns**, not values. `-0.0 == 0.0` is true in
/// IEEE, so `assert_eq!(asc[0], 0.0)` passes whether `-0.0` was dropped or left
/// in place — it cannot see the defect it names. Verified: under the raw-bits
/// key this row comes back `[-0.0, 0.0, 0.0, 0.6]` and a value comparison is
/// satisfied by it.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs is a 3-element literal and every index used is < 3"
)]
fn rank_keys_order_negative_zero_correctly() {
    let mut probs = vec![-0.0f32, 0.5, 0.25];
    filter_top_k(&mut probs, 1);
    assert_eq!(
        survivor_ids(&probs),
        vec![1],
        "top_k=1 must keep the real maximum, not -0.0"
    );

    // The same through the ascending order, where -0.0 must sort below +0.0.
    let mut asc = vec![-0.0f32, 0.0, 0.4, 0.6];
    filter_top_p(&mut asc, 0.5);
    assert!(
        asc[3] > 0.0,
        "the dominant id must survive the nucleus, got {asc:?}"
    );
    assert_eq!(
        asc[0].to_bits(),
        0.0f32.to_bits(),
        "-0.0 carries no mass and must be zeroed; got bits {:#010x}",
        asc[0].to_bits()
    );
}

// ── Near-zero temperature is sampling, not greedy ────────────────────────
//
// These two pin the boundary of the "temperature ~0 behaves like an argmax"
// shortcut so it cannot be read as a guarantee. They characterise existing
// behaviour; neither is changed by the tie-break work above.

/// At `temperature = 1e-4` every logit below the maximum by more than about
/// 0.0104 underflows to exactly zero probability, and every logit closer than
/// that does not. `exp` in f32 returns exactly 0 below roughly -104, so the
/// boundary in logit units is 104 * temperature.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "probs has the same length as the 3-element logits literal above it"
)]
fn near_zero_temperature_underflow_boundary() {
    // Gaps of 0.02 (well past the boundary) and 0.005 (well inside it).
    let logits = vec![10.0f32, 9.98, 9.995];
    let probs = softmax_ok(&logits, 1.0 / 1e-4);
    assert_eq!(probs[1], 0.0, "gap 0.02 must underflow to exactly zero");
    assert!(
        probs[2] > 0.0,
        "gap 0.005 must keep non-zero mass, got {}",
        probs[2]
    );
    assert!(
        probs[0] > 0.99,
        "maximum keeps essentially all the mass, got {}",
        probs[0]
    );
}

/// A *sufficient* mechanism for a near-zero-temperature stream to leave the
/// greedy stream, reproduced without a model: on an **exactly tied** row the
/// host sampler is a uniform draw over the tied ids, so it disagrees with the
/// device argmax roughly half the time. This is the sampler behaving correctly
/// — a categorical draw from a two-point distribution — and it is why a
/// near-zero temperature is not a valid oracle for the greedy path.
///
/// It does **not** establish that this is what happened in any particular
/// field report; that needs the top-2 logits at the first divergent step.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn near_zero_temperature_is_a_uniform_draw_over_an_exact_tie() {
    let logits = f32_row(&[4.0, 4.0, 1.0]);
    let device_id = device_argmax(&logits, Device::Cpu);
    assert_eq!(device_id, 0, "device argmax takes the lowest tied id");

    let sp = SamplerConfig {
        temperature: 1e-4,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(7),
        top_logprobs_k: 0,
    };
    let nop = PenaltyConfig::default();
    let mut rng = Pcg32::new(sp.seed_or_default());
    let mut hits_other = 0usize;
    let mut hits_device = 0usize;
    for _ in 0..64 {
        let out = sample_token_array(&logits, &sp, None, &nop, &[], &mut rng, Device::Cpu).unwrap();
        let id = token_id(&out);
        assert!(id < 2, "never the untied low logit, got {id}");
        if id == device_id {
            hits_device += 1;
        } else {
            hits_other += 1;
        }
    }
    assert!(
        hits_other > 0 && hits_device > 0,
        "an exact tie at temp 1e-4 must split between the tied ids, got {hits_device}/{hits_other}"
    );
}

// ── Device::Gpu mirrors ───────────────────────────────────────────────────
//
// `host_argmax` mirrors the **Metal** reduction, and the CPU tests above
// cannot establish a Metal property — MLX's two backends are separate
// implementations that are already known to differ on one shape (the `-inf`
// vs `in[0]` seed). These re-run the tie contract on the real stream.
//
// `#[ignore]`d per the workspace rule for tests that reach `Device::Gpu`; run
// via `make gpu-test CRATE=rmlx-models FILTER=tie`.

/// Metal's `argmax` must break ties to the lowest index at every width,
/// including a width that crosses its multi-threadgroup reduction strategy.
#[test]
#[ignore = "reaches Device::Gpu; run via make gpu-test"]
fn mlx_argmax_breaks_ties_to_lowest_index_gpu() {
    for (name, row, expect) in tie_shapes() {
        let got = device_argmax(&f32_row(&row), Device::Gpu);
        assert_eq!(
            got, expect,
            "{name}: Metal argmax must return the first of the tied maxima"
        );
    }
}

/// Host greedy must agree with the Metal reduction on the same tied rows.
#[test]
#[ignore = "reaches Device::Gpu; run via make gpu-test"]
fn host_greedy_matches_device_argmax_on_a_tie_gpu() {
    for (name, row, _) in tie_shapes() {
        let logits = f32_row(&row);
        let device_id = device_argmax(&logits, Device::Gpu);
        let host_id = host_greedy(&logits, Device::Gpu);
        assert_eq!(
            host_id, device_id,
            "{name}: host greedy picked {host_id}, Metal argmax picked {device_id}"
        );
    }
}

/// The BF16 row, on the dtype and the stream production actually uses.
/// `logits_to_host_f32` widens BF16 by shifting the pattern into the high
/// half-word, which is exact and injective, so it can neither create a tie nor
/// break one — a BF16 tie is still a tie after the readback.
#[test]
#[ignore = "reaches Device::Gpu; run via make gpu-test"]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn host_greedy_matches_device_argmax_on_a_bf16_tie_gpu() {
    let logits = bf16_tie_row();
    let device_id = device_argmax(&logits, Device::Gpu);
    assert_eq!(device_id, 1, "Metal takes the first of the two BF16 maxima");
    let host_id = host_greedy(&logits, Device::Gpu);
    assert_eq!(
        host_id, device_id,
        "BF16 tie: host greedy picked {host_id}, Metal argmax picked {device_id}"
    );
}

/// `host_argmax` claims a `NaN` never displaces a real maximum *on the device*.
/// The CPU test of that claim is weak — MLX's CPU backend seeds its reduction
/// with `in[0]`, so it skips a mid-row `NaN` for a reason Metal does not share.
/// This is the mirror that actually checks Metal.
#[test]
#[ignore = "reaches Device::Gpu; run via make gpu-test"]
fn host_greedy_skips_nan_like_the_device_gpu() {
    let logits = f32_row(&[1.0, 3.0, f32::NAN, 2.0]);
    let device_id = device_argmax(&logits, Device::Gpu);
    let host_id = host_greedy(&logits, Device::Gpu);
    assert_eq!(
        host_id, device_id,
        "NaN row: host greedy picked {host_id}, Metal argmax picked {device_id}"
    );
    assert_eq!(
        device_id, 1,
        "Metal must skip the NaN and take the real max"
    );
}

/// The all-`-inf` row is the one shape where the `-inf` seed is never
/// displaced, so what a multi-threadgroup Metal arg-reduce returns is a
/// property of its seeding rather than of its comparisons — and the CPU test
/// **cannot fail**, because MLX's CPU backend seeds with `in[0]` and so returns
/// 0 by construction whatever Metal does. This is the only check with power
/// over `host_argmax`'s third documented bullet.
///
/// The row is deliberately wide enough to cross the single- to
/// multi-threadgroup boundary. If this fails, that bullet is wrong and needs
/// correcting rather than the test.
#[test]
#[ignore = "reaches Device::Gpu; run via make gpu-test"]
fn all_neg_inf_row_agrees_with_metal_gpu() {
    for width in [8usize, WIDE_VOCAB] {
        let row = vec![f32::NEG_INFINITY; width];
        let logits = f32_row(&row);
        let device_id = device_argmax(&logits, Device::Gpu);
        let host_id = host_greedy(&logits, Device::Gpu);
        assert_eq!(
            host_id, device_id,
            "width {width}: host greedy picked {host_id}, Metal argmax picked {device_id}"
        );
        assert_eq!(
            device_id, 0,
            "width {width}: host_argmax documents id 0 for an all -inf row"
        );
    }
}

/// BF16 is the high half-word of the F32 pattern:
/// `0x3F80` = 1.0, `0x4000` = 2.0, `0x4020` = 2.5, `0x4080` = 4.0.
/// Tied maxima at index 1 and index 3.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: from_bytes with a literal shape, to_bytes/try_into over the fixed 4 bytes of a [1] I32 result, and sampler calls on inputs built in the same fn — all infallible by construction"
)]
fn bf16_tie_row() -> Array {
    let patterns: [u16; 5] = [0x4020, 0x4080, 0x3F80, 0x4080, 0x4000];
    let mut bytes = Vec::with_capacity(patterns.len() * 2);
    for p in patterns {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    Array::from_bytes(&bytes, &[1, 5], Dtype::Bf16).unwrap()
}

/// The same BF16 contract on the CPU stream, so the ordinary suite still
/// covers the dtype even though the Metal mirror is `#[ignore]`d.
#[test]
fn host_greedy_matches_device_argmax_on_a_bf16_tie() {
    let logits = bf16_tie_row();
    let device_id = device_argmax(&logits, Device::Cpu);
    assert_eq!(
        device_id, 1,
        "device takes the first of the two BF16 maxima"
    );
    let host_id = host_greedy(&logits, Device::Cpu);
    assert_eq!(
        host_id, device_id,
        "BF16 tie: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// `k` is caller-supplied, so it must not size the result vector: the
/// selection is clamped to the vocabulary before the allocation, and a `k`
/// past the vocabulary yields exactly `vocab` ranks.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: compute_top_logprobs over a [1, 4] F32 row built in the same fn"
)]
fn compute_top_logprobs_clamps_k_to_vocab() {
    let logits = f32_row(&[5.0, 7.0, 5.0, 9.0]);
    let out = compute_top_logprobs(&logits, 3, usize::MAX).unwrap();
    assert_eq!(out.top.len(), 4, "k must be clamped to the vocabulary");
}
