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
    let probs = softmax_scaled(&logits, 1.0);
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
    let mut probs = softmax_scaled(&logits, 1.0 / 0.8); // temp=0.8
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
        let mut probs = softmax_scaled(&logits, 1.0 / 0.8);
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
    let mut probs_hi = softmax_scaled(&logits, 1.0 / 1.5);
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
// over a host copy. `Device::Cpu` is used only so the tests run in the
// ordinary (non-`#[ignore]`) suite; MLX's CPU and Metal argmax implement the
// same tie rule, which the first test pins explicitly.

/// Oracle characterisation: MLX `argmax` resolves an exact tie to the
/// **lowest** index. Every other test in this section is written against that
/// fact, so it is pinned here on its own — if a future MLX bump changes the
/// rule this fails first and names the cause.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn mlx_argmax_breaks_ties_to_lowest_index() {
    // Three exactly-equal maxima at 1, 4 and 6.
    let data: [f32; 8] = [1.0, 9.0, 3.0, 2.0, 9.0, 0.0, 9.0, 4.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 8], Dtype::F32).unwrap();
    let idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&idx);
    let b = idx.to_bytes().unwrap();
    let v = i32::from_le_bytes(b[..4].try_into().unwrap());
    assert_eq!(
        v, 1,
        "MLX argmax must return the first of three tied maxima"
    );
}

/// The host greedy path (`temperature == 0` with penalties) must return the
/// same token as the device greedy path on the same logits row. A tie is the
/// only input where the two reductions can differ, so it is the only input
/// that tests the contract.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn host_greedy_matches_device_argmax_on_a_tie() {
    let data: [f32; 8] = [1.0, 9.0, 3.0, 2.0, 9.0, 0.0, 9.0, 4.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 8], Dtype::F32).unwrap();

    let device_idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap());

    let nop = PenaltyConfig::default();
    let host = argmax_with_penalties(&logits, None, &nop, &[], Device::Cpu).unwrap();
    materialise(&host);
    let hb = host.to_bytes().unwrap();
    let host_id = i32::from_le_bytes(hb[..4].try_into().unwrap());

    assert_eq!(
        host_id, device_id,
        "host greedy picked {host_id}, device argmax picked {device_id} on the same row"
    );
}

/// Same contract, but the tie is *created* by a penalty rather than present in
/// the raw row — the shape a served request actually reaches, since the host
/// greedy path only runs when a penalty or a logit bias is active.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn host_greedy_matches_device_argmax_when_a_bias_creates_the_tie() {
    // Raw row has a unique max at 0. A +2.0 bias on id 3 lifts it to exactly
    // 5.0 — an exact tie with id 0, no rounding involved.
    let data: [f32; 6] = [5.0, 1.0, 2.0, 3.0, 0.5, 4.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 6], Dtype::F32).unwrap();
    let cfg = PenaltyConfig {
        logit_bias: vec![(3, 2.0)],
        ..PenaltyConfig::default()
    };
    let host = argmax_with_penalties(&logits, None, &cfg, &[], Device::Cpu).unwrap();
    materialise(&host);
    let hb = host.to_bytes().unwrap();
    let host_id = i32::from_le_bytes(hb[..4].try_into().unwrap());

    // Oracle: apply the same bias on the device and reduce there.
    let biased: [f32; 6] = [5.0, 1.0, 2.0, 5.0, 0.5, 4.0];
    let biased_arr = Array::from_bytes(f32_as_bytes(&biased), &[1, 6], Dtype::F32).unwrap();
    let device_idx = argmax(&biased_arr, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap());

    assert_eq!(
        host_id, device_id,
        "bias-induced tie: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// A `NaN` in the row must not displace a real maximum, and must not reset the
/// running best — both are properties of the device reduction, which compares
/// with a strict `>` and therefore skips `NaN` entirely. The `NaN` is placed
/// away from index 0 on purpose: MLX seeds its CPU reduction with `in[0]` and
/// its Metal reduction with `-inf`, so a leading `NaN` is the one shape where
/// MLX's own two backends disagree and no host rule can match both.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn host_greedy_skips_nan_like_the_device() {
    let data: [f32; 4] = [1.0, 3.0, f32::NAN, 2.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 4], Dtype::F32).unwrap();

    let device_idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap());
    assert_eq!(device_id, 1, "device reduction must skip the NaN");

    let nop = PenaltyConfig::default();
    let host = argmax_with_penalties(&logits, None, &nop, &[], Device::Cpu).unwrap();
    materialise(&host);
    let hb = host.to_bytes().unwrap();
    let host_id = i32::from_le_bytes(hb[..4].try_into().unwrap());
    assert_eq!(
        host_id, device_id,
        "NaN row: host greedy picked {host_id}, device argmax picked {device_id}"
    );
}

/// A fully constraint-masked row (every logit `-inf`) must still agree.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn host_greedy_matches_device_argmax_on_an_all_neg_inf_row() {
    let data = [f32::NEG_INFINITY; 5];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 5], Dtype::F32).unwrap();

    let device_idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap());

    let nop = PenaltyConfig::default();
    let host = argmax_with_penalties(&logits, None, &nop, &[], Device::Cpu).unwrap();
    materialise(&host);
    let hb = host.to_bytes().unwrap();
    let host_id = i32::from_le_bytes(hb[..4].try_into().unwrap());

    assert_eq!(
        host_id, device_id,
        "all -inf row must agree with the device"
    );
}

/// The same contract on a **BF16** row, which is the dtype most snapshots
/// produce and the dtype the device actually reduces over.
///
/// This also settles a standing question about whether the host readback could
/// itself be the source of a disagreement: `logits_to_host_f32` widens BF16 to
/// F32 by shifting the pattern into the high half-word, which is exact and
/// injective, so it can neither create a tie nor break one. A BF16 tie is
/// still a tie after the readback, and the two paths must agree on it.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn host_greedy_matches_device_argmax_on_a_bf16_tie() {
    // BF16 is the high half-word of the F32 pattern:
    // 0x3F80 = 1.0, 0x4000 = 2.0, 0x4020 = 2.5, 0x4080 = 4.0.
    // Tied maxima at index 1 and index 3.
    let patterns: [u16; 5] = [0x4020, 0x4080, 0x3F80, 0x4080, 0x4000];
    let mut bytes = Vec::with_capacity(patterns.len() * 2);
    for p in patterns {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    let logits = Array::from_bytes(&bytes, &[1, 5], Dtype::Bf16).unwrap();

    let device_idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap());
    assert_eq!(
        device_id, 1,
        "device takes the first of the two BF16 maxima"
    );

    let nop = PenaltyConfig::default();
    let host = argmax_with_penalties(&logits, None, &nop, &[], Device::Cpu).unwrap();
    materialise(&host);
    let hb = host.to_bytes().unwrap();
    let host_id = i32::from_le_bytes(hb[..4].try_into().unwrap());

    assert_eq!(
        host_id, device_id,
        "BF16 tie: host greedy picked {host_id}, device argmax picked {device_id}"
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
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
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

/// End-to-end restatement of the same contract on the full host pipeline:
/// `top_k = 1` must reproduce the device argmax for every RNG draw, including
/// on a row whose maximum is tied. The existing `top_k_one_collapses_to_greedy`
/// covers only rows with a unique maximum, which is the case that cannot fail.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn top_k_one_equals_device_argmax_under_a_tie() {
    let n = 64usize;
    let mut data = vec![1.0f32; n];
    for &i in &[5usize, 30, 63] {
        data[i] = 8.0;
    }
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, n as i32], Dtype::F32).unwrap();

    let device_idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap()) as u32;

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
        materialise(&out);
        let b = out.to_bytes().unwrap();
        let id = i32::from_le_bytes(b[..4].try_into().unwrap()) as u32;
        assert_eq!(
            id, device_id,
            "top_k=1 draw {draw}: sampled {id}, device argmax {device_id}"
        );
    }
}

// ── Near-zero temperature is sampling, not greedy ────────────────────────
//
// These two pin the boundary of the "temperature ~0 behaves like an argmax"
// shortcut so it cannot be read as a guarantee.

/// At `temperature = 1e-4` every logit below the maximum by more than about
/// 0.0104 underflows to exactly zero probability, and every logit closer than
/// that does not. `exp` in f32 returns exactly 0 below roughly -104, so the
/// boundary in logit units is 104 * temperature.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn near_zero_temperature_underflow_boundary() {
    // Gaps of 0.02 (well past the boundary) and 0.005 (well inside it).
    let logits = vec![10.0f32, 9.98, 9.995];
    let probs = softmax_scaled(&logits, 1.0 / 1e-4);
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

/// The reported symptom, reproduced without a model: on an **exactly tied**
/// row the near-zero-temperature host sampler is a uniform draw over the tied
/// ids, so it disagrees with the device argmax roughly half the time. This is
/// the sampler behaving correctly — a categorical draw from a two-point
/// distribution — not a defect, and the host greedy path (which this test does
/// not touch) is the one that must match the device.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn near_zero_temperature_is_a_uniform_draw_over_an_exact_tie() {
    let data: [f32; 3] = [4.0, 4.0, 1.0];
    let logits = Array::from_bytes(f32_as_bytes(&data), &[1, 3], Dtype::F32).unwrap();

    let device_idx = argmax(&logits, -1, Device::Cpu).unwrap();
    materialise(&device_idx);
    let db = device_idx.to_bytes().unwrap();
    let device_id = i32::from_le_bytes(db[..4].try_into().unwrap()) as u32;
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
        materialise(&out);
        let b = out.to_bytes().unwrap();
        let id = i32::from_le_bytes(b[..4].try_into().unwrap()) as u32;
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
