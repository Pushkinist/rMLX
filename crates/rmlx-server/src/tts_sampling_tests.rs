//! Speech sampling: the resolved configuration of both heads, and the token
//! pick on synthetic logit rows. CPU only, no weights.

use super::*;
use rmlx_audio::tts::talker_allow_mask;
use std::path::Path;
use tempfile::{tempdir, TempDir};

const VOCAB: usize = 3072;
const EOS: u32 = 2150;
const SPECIAL_START: usize = 2048;

/// The checkpoint file of `mlx-community__Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`.
const CHECKPOINT_FILE: &str = r#"{
  "do_sample": true,
  "repetition_penalty": 1.05,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "subtalker_dosample": true,
  "subtalker_temperature": 0.9,
  "subtalker_top_p": 1.0,
  "subtalker_top_k": 50,
  "max_new_tokens": 8192
}"#;

fn snapshot(generation_config: Option<&str>) -> TempDir {
    let dir = tempdir().unwrap();
    if let Some(body) = generation_config {
        std::fs::write(dir.path().join("generation_config.json"), body).unwrap();
    }
    dir
}

fn resolve(dir: &Path, req: TtsSamplingRequest) -> TtsSampling {
    resolve_tts_sampling(dir, &req).unwrap()
}

fn request(
    temperature: Option<f32>,
    top_k: Option<u32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
) -> TtsSamplingRequest {
    TtsSamplingRequest {
        temperature,
        top_k,
        repetition_penalty,
        seed,
    }
}

/// `(temperature, top_k, top_p, repetition_penalty)` of one head.
fn values(h: &HeadSampling) -> (f32, u32, f32, f32) {
    (
        h.sampler.temperature,
        h.sampler.top_k,
        h.sampler.top_p,
        h.penalty.rep_penalty,
    )
}

fn assert_greedy(h: &HeadSampling, rep_penalty: f32, what: &str) {
    assert!(
        !h.sampler.sampling_active(),
        "{what}: temperature {} must be greedy",
        h.sampler.temperature
    );
    assert_eq!(
        h.penalty.rep_penalty, rep_penalty,
        "{what}: repetition penalty"
    );
}

// ── Resolution ──────────────────────────────────────────────────────────────

#[test]
fn checkpoint_values_are_the_defaults_of_both_heads() {
    let dir = snapshot(Some(CHECKPOINT_FILE));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.talker), (0.9, 50, 1.0, 1.05));
    assert_eq!(
        values(&s.code_predictor),
        (0.9, 50, 1.0, 1.0),
        "the reference passes no repetition penalty to the code predictor"
    );
    assert_eq!(s.talker.sampler.seed, None);
    assert_eq!(s.code_predictor.sampler.seed, None);
}

/// No file: the Qwen3-TTS reference defaults, not greedy decoding.
#[test]
fn no_file_uses_the_reference_defaults() {
    let dir = snapshot(None);
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.talker), (0.9, 50, 1.0, 1.05));
    assert_eq!(values(&s.code_predictor), (0.9, 50, 1.0, 1.0));
}

/// Each absent key falls back to its own reference default.
#[test]
fn a_file_with_only_a_temperature_takes_the_other_defaults() {
    let dir = snapshot(Some(r#"{"temperature": 0.9}"#));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.talker), (0.9, 50, 1.0, 1.05));
    assert_eq!(values(&s.code_predictor), (0.9, 50, 1.0, 1.0));
}

/// The code predictor falls back to the reference defaults, never to the
/// talker's values from the file.
#[test]
fn absent_subtalker_keys_take_the_reference_defaults_not_the_talker_values() {
    let dir = snapshot(Some(
        r#"{"temperature": 0.5, "top_k": 7, "top_p": 0.8, "repetition_penalty": 1.2}"#,
    ));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.talker), (0.5, 7, 0.8, 1.2));
    assert_eq!(values(&s.code_predictor), (0.9, 50, 1.0, 1.0));
}

#[test]
fn subtalker_keys_feed_the_code_predictor_only() {
    let dir = snapshot(Some(
        r#"{"temperature": 0.9, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
            "subtalker_temperature": 0.5, "subtalker_top_k": 7, "subtalker_top_p": 0.8}"#,
    ));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.code_predictor), (0.5, 7, 0.8, 1.0));
    assert_eq!(values(&s.talker), (0.9, 50, 1.0, 1.05));
}

#[test]
fn request_fields_win_over_the_file() {
    let dir = snapshot(Some(CHECKPOINT_FILE));
    let s = resolve(
        dir.path(),
        request(Some(0.3), Some(10), Some(1.2), Some(42)),
    );
    assert_eq!(values(&s.talker), (0.3, 10, 1.0, 1.2));
    assert_eq!(values(&s.code_predictor), (0.3, 10, 1.0, 1.0));
    assert_eq!(s.talker.sampler.seed, Some(42));
    assert_eq!(s.code_predictor.sampler.seed, Some(42));
}

/// Temperature 0 is greedy. The talker still takes the file's repetition
/// penalty, as HF greedy decoding applies its logits processors.
#[test]
fn request_temperature_zero_is_greedy_and_keeps_the_file_penalty() {
    let dir = snapshot(Some(CHECKPOINT_FILE));
    let s = resolve(dir.path(), request(Some(0.0), None, None, None));
    assert_greedy(&s.talker, 1.05, "talker");
    assert_greedy(&s.code_predictor, 1.0, "code predictor");
}

/// Today's decoding: greedy and no penalty on both heads.
#[test]
fn temperature_zero_with_penalty_one_is_plain_greedy() {
    let dir = snapshot(Some(CHECKPOINT_FILE));
    let s = resolve(dir.path(), request(Some(0.0), None, Some(1.0), None));
    for (h, what) in [(&s.talker, "talker"), (&s.code_predictor, "code predictor")] {
        assert!(!h.sampler.sampling_active(), "{what}: must be greedy");
        assert!(!h.penalty.penalties_active(), "{what}: no penalty");
    }
}

#[test]
fn request_penalty_applies_to_a_greedy_talker() {
    let dir = snapshot(Some(CHECKPOINT_FILE));
    let s = resolve(dir.path(), request(Some(0.0), None, Some(1.2), None));
    assert_greedy(&s.talker, 1.2, "talker");
    assert_greedy(&s.code_predictor, 1.0, "code predictor");
}

/// `do_sample: false` is greedy, and the file's repetition penalty still applies.
#[test]
fn do_sample_false_is_greedy_with_the_file_penalty() {
    let dir = snapshot(Some(
        r#"{"do_sample": false, "temperature": 0.9, "top_k": 50, "repetition_penalty": 1.2,
            "subtalker_dosample": true, "subtalker_temperature": 0.7, "subtalker_top_k": 50}"#,
    ));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_greedy(&s.talker, 1.2, "talker");
    assert_eq!(values(&s.code_predictor), (0.7, 50, 1.0, 1.0));
}

#[test]
fn a_request_temperature_wins_over_do_sample_false() {
    let dir = snapshot(Some(r#"{"do_sample": false, "subtalker_dosample": false}"#));
    let s = resolve(dir.path(), request(Some(0.3), None, None, None));
    assert_eq!(s.talker.sampler.temperature, 0.3);
    assert_eq!(s.code_predictor.sampler.temperature, 0.3);
}

#[test]
fn subtalker_dosample_false_makes_the_code_predictor_greedy() {
    let dir = snapshot(Some(
        r#"{"do_sample": true, "temperature": 0.9, "top_k": 50, "repetition_penalty": 1.05,
            "subtalker_dosample": false, "subtalker_temperature": 0.9, "subtalker_top_k": 50}"#,
    ));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_greedy(&s.code_predictor, 1.0, "code predictor");
    assert_eq!(values(&s.talker), (0.9, 50, 1.0, 1.05));
}

/// A file without `do_sample` samples with its values, as the reference does.
#[test]
fn absent_do_sample_keeps_the_file_values() {
    let dir = snapshot(Some(
        r#"{"temperature": 0.6, "top_k": 30, "repetition_penalty": 1.1,
            "subtalker_temperature": 0.4, "subtalker_top_k": 20}"#,
    ));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.talker), (0.6, 30, 1.0, 1.1));
    assert_eq!(values(&s.code_predictor), (0.4, 20, 1.0, 1.0));
}

#[test]
fn unparseable_file_is_an_error() {
    let dir = snapshot(Some("{ not json }"));
    let err = resolve_tts_sampling(dir.path(), &TtsSamplingRequest::default()).unwrap_err();
    assert!(
        err.to_string().contains("generation_config.json"),
        "the error must name the file, got: {err}"
    );
}

/// A value from the file passes the bounds of a request field, or the file is
/// refused with the key in the message.
#[test]
fn a_file_value_out_of_bounds_is_an_error_that_names_the_key() {
    for (body, key) in [
        (
            r#"{"repetition_penalty": 0.0, "temperature": 0.9}"#,
            "repetition_penalty",
        ),
        (r#"{"temperature": -3.0}"#, "temperature"),
        (r#"{"top_p": 9.0}"#, "top_p"),
        (r#"{"subtalker_temperature": 2.5}"#, "subtalker_temperature"),
        (r#"{"subtalker_top_p": -0.5}"#, "subtalker_top_p"),
        (r#"{"temperature": 1e-40}"#, "temperature"),
        (r#"{"temperature": 9.9e-7}"#, "temperature"),
        (r#"{"temperature": 1e39}"#, "temperature"),
        (r#"{"repetition_penalty": 1e-40}"#, "repetition_penalty"),
        (r#"{"repetition_penalty": 9.9e-7}"#, "repetition_penalty"),
        (r#"{"repetition_penalty": 1e39}"#, "repetition_penalty"),
        (
            r#"{"subtalker_temperature": 9.9e-7}"#,
            "subtalker_temperature",
        ),
    ] {
        let dir = snapshot(Some(body));
        let err = match resolve_tts_sampling(dir.path(), &TtsSamplingRequest::default()) {
            Err(err) => err.to_string(),
            Ok(s) => panic!("{body}: expected an error, got {s:?}"),
        };
        assert!(err.contains(&format!("`{key}`")), "{body}: {err}");
        assert!(err.contains("generation_config.json"), "{body}: {err}");
    }
}

/// The file's values at the edge of the bounds are accepted.
#[test]
fn file_values_at_the_bounds_are_accepted() {
    let dir = snapshot(Some(
        r#"{"temperature": 2.0, "top_p": 0.0, "repetition_penalty": 1e-6,
            "subtalker_temperature": 1e-6, "subtalker_top_p": 1.0}"#,
    ));
    let s = resolve(dir.path(), TtsSamplingRequest::default());
    assert_eq!(values(&s.talker), (2.0, 50, 0.0, 1e-6));
    assert_eq!(values(&s.code_predictor), (1e-6, 50, 1.0, 1.0));
}

#[test]
fn request_fields_deserialize_with_the_text_route_names() {
    let req: TtsSamplingRequest = serde_json::from_str(
        r#"{"temperature": 0.3, "top_k": 10, "repetition_penalty": 1.2, "seed": 7}"#,
    )
    .unwrap();
    assert_eq!(req.temperature, Some(0.3));
    assert_eq!(req.top_k, Some(10));
    assert_eq!(req.repetition_penalty, Some(1.2));
    assert_eq!(req.seed, Some(7));
}

// ── Token pick ──────────────────────────────────────────────────────────────

fn head(temperature: f32, top_k: u32, rep_penalty: f32, seed: Option<u64>) -> HeadSampling {
    HeadSampling {
        sampler: SamplerConfig {
            temperature,
            top_p: 1.0,
            top_k,
            min_p: 0.0,
            seed,
            top_logprobs_k: 0,
        },
        penalty: PenaltyConfig {
            rep_penalty,
            ..PenaltyConfig::default()
        },
    }
}

fn picker(talker: HeadSampling, code_predictor: HeadSampling) -> TtsPicker {
    TtsPicker::new(
        TtsSampling {
            talker,
            code_predictor,
        },
        Device::Cpu,
    )
}

fn values_row(base: impl Fn(usize) -> f32, set: &[(usize, f32)], vocab: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..vocab).map(base).collect();
    for &(id, x) in set {
        v[id] = x;
    }
    v
}

fn array(v: &[f32]) -> Array {
    Array::from_f32_slice(v, &[1, 1, v.len() as i32]).unwrap()
}

/// A `[1, 1, VOCAB]` f32 row: `base` everywhere, then `set` on top.
fn row(base: impl Fn(usize) -> f32, set: &[(usize, f32)]) -> Array {
    array(&values_row(base, set, VOCAB))
}

fn picks(
    p: &mut TtsPicker,
    h: PickHead,
    logits: &Array,
    allow: Option<&[bool]>,
    history: &[u32],
    n: usize,
) -> Vec<u32> {
    (0..n)
        .map(|_| p.pick(h, logits, allow, history).unwrap())
        .collect()
}

fn talker_mask() -> Vec<bool> {
    talker_allow_mask(VOCAB, EOS)
}

#[test]
fn same_seed_gives_the_same_tokens_and_another_seed_does_not() {
    let flat = row(|_| 0.0, &[]);
    let mask = talker_mask();
    let draw = |seed| {
        let mut p = picker(head(1.0, 0, 1.0, seed), head(1.0, 0, 1.0, seed));
        picks(&mut p, PickHead::Talker, &flat, Some(&mask), &[], 32)
    };
    let a = draw(Some(7));
    assert!(
        a.iter().any(|&id| id != a[0]),
        "32 draws from a flat row were all {}",
        a[0]
    );
    assert_eq!(a, draw(Some(7)), "one seed must give one token stream");
    assert_ne!(
        a,
        draw(Some(8)),
        "two seeds gave one stream over 32 flat draws"
    );
    assert_eq!(
        draw(None),
        draw(None),
        "an unseeded request must repeat: the shared default seed"
    );
}

/// The inverse CDF of `softmax(row / temperature)` at `r`, in the order of
/// operations of the shared sampler with no filter active.
fn oracle_pick(row: &[f32], temperature: f32, r: f32) -> u32 {
    let inv_t = 1.0 / temperature;
    let max = row
        .iter()
        .map(|&l| l * inv_t)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut p: Vec<f32> = row
        .iter()
        .map(|&l| {
            let e = (l * inv_t - max).exp();
            sum += e;
            e
        })
        .collect();
    let inv = 1.0 / sum;
    for x in &mut p {
        *x *= inv;
    }
    let total: f32 = p.iter().sum();
    let inv = 1.0 / total;
    for x in &mut p {
        *x *= inv;
    }
    let total: f32 = p.iter().sum();
    let target = r * total;
    let mut cum = 0.0f32;
    let mut last = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > 0.0 {
            last = i;
            cum += x;
            if cum > target {
                return i as u32;
            }
        }
    }
    last as u32
}

/// Pick k of a request is the inverse CDF at the k-th draw of one `Pcg32(seed)`
/// stream, whichever head picks. This pins the temperature of each head, the
/// one stream across heads, and one draw per pick.
#[test]
fn every_pick_is_the_inverse_cdf_at_the_next_draw_of_one_stream() {
    let talker_v = values_row(
        |_| f32::NEG_INFINITY,
        &[(10, 0.0), (11, 0.5), (12, 1.0), (13, 1.5)],
        VOCAB,
    );
    let cp_v = values_row(
        |_| f32::NEG_INFINITY,
        &[(20, 1.0), (21, 0.2), (22, 0.8), (23, 0.4)],
        2048,
    );
    let (talker_row, cp_row) = (array(&talker_v), array(&cp_v));
    let (talker_t, cp_t) = (1.0, 0.7);
    let seed = 21;
    let mut p = picker(
        head(talker_t, 0, 1.0, Some(seed)),
        head(cp_t, 0, 1.0, Some(seed)),
    );
    let mut stream = Pcg32::new(seed);
    let mask = talker_mask();
    let order = "TCCTCTTCCCTCTTTCCTCTCCTT";
    let mut seen = std::collections::BTreeSet::new();
    for (k, which) in order.chars().enumerate() {
        let r = stream.next_f32();
        let (got, want) = if which == 'T' {
            let got = p
                .pick(PickHead::Talker, &talker_row, Some(&mask), &[])
                .unwrap();
            (got, oracle_pick(&talker_v, talker_t, r))
        } else {
            let got = p.pick(PickHead::CodePredictor, &cp_row, None, &[]).unwrap();
            (got, oracle_pick(&cp_v, cp_t, r))
        };
        assert_eq!(got, want, "pick {k} ({which})");
        seen.insert(got);
    }
    assert!(
        seen.len() >= 5,
        "the draws must spread over both rows: {seen:?}"
    );
}

#[test]
fn a_masked_special_token_is_never_chosen() {
    let mask = talker_mask();
    // Every special scores far above every audio code; the end token scores low.
    let logits = row(
        |id| {
            if id >= SPECIAL_START {
                30.0
            } else {
                (id % 7) as f32
            }
        },
        &[(EOS as usize, -5.0)],
    );
    let mut sampled = picker(head(0.9, 50, 1.05, Some(1)), head(0.9, 50, 1.0, Some(1)));
    for id in picks(
        &mut sampled,
        PickHead::Talker,
        &logits,
        Some(&mask),
        &[],
        64,
    ) {
        assert!(mask[id as usize], "sampled a masked special token {id}");
    }
    let mut greedy = picker(head(0.0, 0, 1.0, None), head(0.0, 0, 1.0, None));
    let id = greedy
        .pick(PickHead::Talker, &logits, Some(&mask), &[])
        .unwrap();
    assert!(
        mask[id as usize],
        "greedy chose a masked special token {id}"
    );
}

#[test]
fn the_end_token_is_chosen_when_it_is_the_row_maximum() {
    let mask = talker_mask();
    let logits = row(|id| (id % 5) as f32, &[(EOS as usize, 20.0)]);
    let mut low_temp = picker(head(0.05, 50, 1.05, Some(3)), head(0.05, 50, 1.0, Some(3)));
    for id in picks(
        &mut low_temp,
        PickHead::Talker,
        &logits,
        Some(&mask),
        &[],
        16,
    ) {
        assert_eq!(id, EOS, "the end token must stay selectable under the mask");
    }
    let mut greedy = picker(head(0.0, 0, 1.0, None), head(0.0, 0, 1.0, None));
    assert_eq!(
        greedy
            .pick(PickHead::Talker, &logits, Some(&mask), &[])
            .unwrap(),
        EOS
    );
}

#[test]
fn the_repetition_penalty_makes_a_repeated_token_lose_to_a_close_neighbour() {
    let mask = talker_mask();
    // 10.0 / 1.05 = 9.52 < 9.99: the penalty flips the pair.
    let logits = row(|_| 0.0, &[(100, 10.0), (101, 9.99)]);
    let mut p = picker(head(0.0, 0, 1.05, None), head(0.0, 0, 1.0, None));
    assert_eq!(
        p.pick(PickHead::Talker, &logits, Some(&mask), &[]).unwrap(),
        100,
        "control: with no history the higher logit wins"
    );
    assert_eq!(
        p.pick(PickHead::Talker, &logits, Some(&mask), &[100])
            .unwrap(),
        101,
        "the repeated token must lose to its close neighbour"
    );
}

/// The penalty reads the whole history it is given: a token 70 ids back is
/// still penalised, so no trailing window is cut inside the pick.
#[test]
fn the_repetition_penalty_reads_the_full_history() {
    let mask = talker_mask();
    let logits = row(|_| 0.0, &[(100, 10.0), (101, 9.99)]);
    let history: Vec<u32> = std::iter::once(100).chain(300..370).collect();
    assert_eq!(history.len(), 71);
    let mut p = picker(head(0.0, 0, 1.05, None), head(0.0, 0, 1.0, None));
    assert_eq!(
        p.pick(PickHead::Talker, &logits, Some(&mask), &history)
            .unwrap(),
        101,
        "a token 70 ids back must still be penalised"
    );
}

/// The penalty also applies on the sampled path.
#[test]
fn the_repetition_penalty_applies_when_sampling() {
    let mask = talker_mask();
    // 10.0 / 1.05 = 9.524. With 9.76 both gaps are about 0.24, 24 at T = 0.01.
    let logits = row(|_| 0.0, &[(100, 10.0), (101, 9.76)]);
    let mut p = picker(head(0.01, 0, 1.05, Some(1)), head(0.01, 0, 1.0, Some(1)));
    for id in picks(&mut p, PickHead::Talker, &logits, Some(&mask), &[100], 32) {
        assert_eq!(id, 101, "the repeated token must lose when sampling");
    }
    for id in picks(&mut p, PickHead::Talker, &logits, Some(&mask), &[], 32) {
        assert_eq!(id, 100, "control: with no history the higher logit wins");
    }
}

#[test]
fn each_head_uses_its_own_penalty() {
    let logits = row(|_| 0.0, &[(100, 10.0), (101, 9.99)]);
    let mut p = picker(head(0.0, 0, 1.05, None), head(0.0, 0, 1.0, None));
    assert_eq!(
        p.pick(PickHead::Talker, &logits, None, &[100]).unwrap(),
        101
    );
    assert_eq!(
        p.pick(PickHead::CodePredictor, &logits, None, &[100])
            .unwrap(),
        100,
        "the code predictor has no repetition penalty"
    );
}

#[test]
fn each_head_uses_its_own_temperature() {
    let logits = row(|_| 0.0, &[(7, 3.0)]);
    let mut p = picker(head(5.0, 0, 1.0, Some(9)), head(0.0, 0, 1.0, Some(9)));
    let cp = picks(&mut p, PickHead::CodePredictor, &logits, None, &[], 32);
    assert!(
        cp.iter().all(|&id| id == 7),
        "a greedy code predictor picks the argmax: {cp:?}"
    );
    let talker = picks(&mut p, PickHead::Talker, &logits, None, &[], 32);
    assert!(
        talker.iter().any(|&id| id != 7),
        "a talker at temperature 5 must not pick the argmax 32 of 32 times"
    );
}

#[test]
fn the_top_k_cut_comes_after_the_mask() {
    let mask = talker_mask();
    // The 50 highest logits of the row are masked specials. Among allowed ids
    // the audio codes rise with the id, so the top 50 allowed are 1998..2048.
    let mut set: Vec<(usize, f32)> = (2200..2250).map(|id| (id, 50.0)).collect();
    set.push((EOS as usize, -10.0));
    let logits = row(
        |id| {
            if id < SPECIAL_START {
                id as f32 * 0.001
            } else {
                0.0
            }
        },
        &set,
    );
    let mut p = picker(head(0.9, 50, 1.0, Some(5)), head(0.9, 50, 1.0, Some(5)));
    for id in picks(&mut p, PickHead::Talker, &logits, Some(&mask), &[], 64) {
        assert!(
            (1998..2048).contains(&id),
            "picked {id}: the top-k cut must keep the 50 highest allowed ids"
        );
    }
}

/// The talker head can return bf16 logits. The pick reads them like f32 rows.
#[test]
fn a_bf16_row_is_picked_under_the_mask() {
    let mask = talker_mask();
    // The highest logit is a masked special; the best allowed id is 1234.
    let logits = row(
        |id| {
            if id >= SPECIAL_START {
                0.0
            } else {
                (id % 7) as f32
            }
        },
        &[(2500, 40.0), (1234, 20.0), (EOS as usize, -5.0)],
    )
    .astype(rmlx_mlx::Dtype::Bf16, Device::Cpu)
    .unwrap();
    assert_eq!(logits.dtype(), rmlx_mlx::Dtype::Bf16);

    let mut greedy = picker(head(0.0, 0, 1.0, None), head(0.0, 0, 1.0, None));
    assert_eq!(
        greedy
            .pick(PickHead::Talker, &logits, Some(&mask), &[])
            .unwrap(),
        1234
    );
    let mut sampled = picker(head(0.9, 50, 1.05, Some(4)), head(0.9, 50, 1.0, Some(4)));
    for id in picks(
        &mut sampled,
        PickHead::Talker,
        &logits,
        Some(&mask),
        &[100],
        64,
    ) {
        assert!(mask[id as usize], "sampled a masked special token {id}");
    }
}
