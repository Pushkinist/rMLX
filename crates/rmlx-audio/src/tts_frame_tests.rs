//! The per-frame seam: what the token pick receives for one frame, from a
//! recording `TokenPick` on synthetic rows. CPU only, no weights.

use super::{frame_step, talker_allow_mask, FrameStep, PickHead, TokenPick, TtsError, MIN_FRAMES};
use rmlx_mlx::Array;

const TALKER_VOCAB: usize = 3072;
const CP_VOCAB: usize = 2048;
const EOS: u32 = 2150;
const TALKER_TAG: f32 = -1.0;

/// One call to `pick`, as the picker saw it.
#[derive(Debug)]
struct Call {
    head: PickHead,
    allow: Option<Vec<bool>>,
    history: Vec<u32>,
    /// First value of the row the picker received.
    tag: f32,
}

/// Returns `talker_id` for the talker and 100, 101, ... for the code predictor
/// (`next_code`, then `next_code + code_step`, ...).
struct Recorder {
    calls: Vec<Call>,
    talker_id: u32,
    next_code: u32,
    code_step: u32,
}

impl Recorder {
    fn new(talker_id: u32) -> Self {
        Self {
            calls: Vec::new(),
            talker_id,
            next_code: 100,
            code_step: 1,
        }
    }
}

fn first_value(a: &Array) -> f32 {
    a.eval().unwrap();
    let b = a.to_bytes().unwrap();
    f32::from_le_bytes(b[..4].try_into().unwrap())
}

impl TokenPick for Recorder {
    fn pick(
        &mut self,
        head: PickHead,
        logits: &Array,
        allow: Option<&[bool]>,
        history: &[u32],
    ) -> Result<u32, TtsError> {
        self.calls.push(Call {
            head,
            allow: allow.map(<[bool]>::to_vec),
            history: history.to_vec(),
            tag: first_value(logits),
        });
        Ok(match head {
            PickHead::Talker => self.talker_id,
            PickHead::CodePredictor => {
                self.next_code += self.code_step;
                self.next_code - self.code_step
            }
        })
    }
}

fn filled(value: f32, shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_f32_slice(&vec![value; n as usize], shape).unwrap()
}

fn talker_row() -> Array {
    filled(TALKER_TAG, &[1, 1, TALKER_VOCAB as i32])
}

/// Group `i` of the code predictor gets a row tagged `i + 0.5`.
fn cp_tag(i: usize) -> f32 {
    i as f32 + 0.5
}

#[test]
fn a_frame_is_one_talker_pick_then_fifteen_code_predictor_picks() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut rec = Recorder::new(7);
    let mut history = vec![1, 2, 3];
    let mut cp_calls = Vec::new();
    let out = frame_step(
        &mut rec,
        &talker_row(),
        &mask,
        EOS,
        &mut history,
        |i, prev| {
            cp_calls.push((i, prev));
            Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
        },
    )
    .unwrap();

    let FrameStep::Frame(codes) = out else {
        panic!("expected a frame, got {out:?}");
    };
    let want_codes: Vec<u32> = std::iter::once(7).chain(100..115).collect();
    assert_eq!(codes, want_codes, "group 0 first, then groups 1 to 15");

    assert_eq!(
        rec.calls.len(),
        16,
        "one talker pick and 15 code-predictor picks"
    );
    let talker = &rec.calls[0];
    assert_eq!(talker.head, PickHead::Talker, "the talker picks first");
    assert_eq!(
        talker.allow.as_deref(),
        Some(mask.as_slice()),
        "the talker picks under the mask"
    );
    assert_eq!(
        talker.history,
        vec![1, 2, 3],
        "the talker sees the history before this frame"
    );
    assert_eq!(
        talker.tag, TALKER_TAG,
        "the talker picks from the talker row"
    );

    for (i, call) in rec.calls[1..].iter().enumerate() {
        assert_eq!(call.head, PickHead::CodePredictor, "pick {}", i + 1);
        assert!(
            call.allow.is_none(),
            "group {}: the code predictor has no mask",
            i + 1
        );
        assert!(
            call.history.is_empty(),
            "group {}: the code predictor has no history",
            i + 1
        );
        assert_eq!(
            call.tag,
            cp_tag(i),
            "group {} picks from cp_row({i}, ..)",
            i + 1
        );
    }

    let want_cp: Vec<(usize, u32)> = (0..15).map(|i| (i, want_codes[i])).collect();
    assert_eq!(
        cp_calls, want_cp,
        "cp_row gets the group index and the previous code"
    );

    assert_eq!(
        history,
        vec![1, 2, 3, 7],
        "the history grows by the group-0 code only"
    );
}

#[test]
fn the_end_token_ends_the_frame_and_is_not_history() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut rec = Recorder::new(EOS);
    let mut history = vec![1, 2, 3];
    let mut cp_called = false;
    let out = frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |_, _| {
        cp_called = true;
        Ok(filled(0.0, &[1, 1, CP_VOCAB as i32]))
    })
    .unwrap();
    assert!(matches!(out, FrameStep::End), "expected End, got {out:?}");
    assert_eq!(
        rec.calls.len(),
        1,
        "no code-predictor pick after the end token"
    );
    assert!(!cp_called, "no code-predictor row after the end token");
    assert_eq!(
        history,
        vec![1, 2, 3],
        "the end token is not pushed to the history"
    );
}

/// The reference sets `min_new_tokens = 2`: the end token is masked at frames 0 and 1.
#[test]
fn the_end_token_is_masked_before_min_frames() {
    assert_eq!(MIN_FRAMES, 2);
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    for step in 0..=MIN_FRAMES {
        let mut rec = Recorder::new(7);
        let mut history = vec![7; step];
        frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |i, _| {
            Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
        })
        .unwrap();
        let allow = rec.calls[0]
            .allow
            .clone()
            .expect("the talker picks under a mask");
        assert_eq!(allow.len(), TALKER_VOCAB);
        let eos_allowed = step >= MIN_FRAMES;
        for (id, (&got, &base)) in allow.iter().zip(&mask).enumerate() {
            let want = if id == EOS as usize {
                eos_allowed
            } else {
                base
            };
            assert_eq!(got, want, "step {step}, id {id}");
        }
    }
}

/// The repetition context is the whole group-0 history, not a trailing window.
#[test]
fn the_talker_sees_the_full_group0_history() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut history = Vec::new();
    let mut last_seen = Vec::new();
    for step in 0..71usize {
        let mut rec = Recorder::new(200 + step as u32);
        frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |i, _| {
            Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
        })
        .unwrap();
        last_seen = rec.calls[0].history.clone();
    }
    let want: Vec<u32> = (200..270).collect();
    assert_eq!(
        last_seen, want,
        "frame 70 must see all 70 earlier group-0 codes"
    );
    assert_eq!(history.len(), 71);
}

#[test]
fn a_talker_row_with_more_than_one_position_is_refused() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut rec = Recorder::new(7);
    let two = filled(0.0, &[1, 2, TALKER_VOCAB as i32]);
    let err = frame_step(&mut rec, &two, &mask, EOS, &mut Vec::new(), |i, _| {
        Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
    })
    .unwrap_err();
    assert!(err.to_string().contains("one position"), "got: {err}");
    assert!(
        rec.calls.is_empty(),
        "nothing may be picked from a refused row"
    );
}

#[test]
fn a_code_predictor_row_with_more_than_one_position_is_refused() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut rec = Recorder::new(7);
    let err = frame_step(
        &mut rec,
        &talker_row(),
        &mask,
        EOS,
        &mut Vec::new(),
        |_, _| Ok(filled(0.0, &[1, 2, CP_VOCAB as i32])),
    )
    .unwrap_err();
    assert!(err.to_string().contains("one position"), "got: {err}");
    assert_eq!(
        rec.calls.len(),
        1,
        "only the talker pick precedes the refused row"
    );
}

/// A pick outside the mask is a picker defect: the frame is refused, not decoded.
#[test]
fn a_talker_pick_the_mask_blocks_is_refused() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    assert!(!mask[2500], "2500 is a special token");
    let mut rec = Recorder::new(2500);
    let mut history = vec![1, 2, 3];
    let mut cp_called = false;
    let err = frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |_, _| {
        cp_called = true;
        Ok(filled(0.0, &[1, 1, CP_VOCAB as i32]))
    })
    .unwrap_err();
    assert!(matches!(err, TtsError::Pick(_)), "got: {err:?}");
    assert!(err.to_string().contains("2500"), "got: {err}");
    assert!(!cp_called, "no code-predictor row after a refused pick");
    assert_eq!(history, vec![1, 2, 3], "a refused pick is not history");
}

/// Before `MIN_FRAMES` the end token is blocked, so a picker that returns it
/// broke the mask: an error, not the end of the speech.
#[test]
fn an_end_token_before_min_frames_is_refused() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    for len in 0..MIN_FRAMES {
        let mut rec = Recorder::new(EOS);
        let mut history = vec![7; len];
        let result = frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |i, _| {
            Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
        });
        let err = match result {
            Err(err @ TtsError::Pick(_)) => err.to_string(),
            other => panic!("history of {len}: expected a pick error, got {other:?}"),
        };
        assert!(err.contains(&EOS.to_string()), "history of {len}: {err}");
        assert_eq!(history.len(), len, "a refused pick is not history");
    }
}

/// An id past the end of the talker vocabulary is outside the mask.
#[test]
fn a_talker_pick_past_the_vocabulary_is_refused() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut rec = Recorder::new(TALKER_VOCAB as u32);
    let mut history = vec![1, 2, 3];
    let result = frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |i, _| {
        Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
    });
    let err = match result {
        Err(err @ TtsError::Pick(_)) => err.to_string(),
        other => panic!("expected a pick error, got {other:?}"),
    };
    assert!(err.contains(&TALKER_VOCAB.to_string()), "got: {err}");
    assert_eq!(rec.calls.len(), 1, "no code-predictor pick follows");
}

/// A code-predictor id must be below the vocabulary of its row, or it would
/// index past an embedding table and a codebook.
#[test]
fn a_code_predictor_pick_outside_its_vocabulary_is_refused() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    for code in [CP_VOCAB as u32, 4_000_000_000] {
        let mut rec = Recorder::new(7);
        rec.next_code = code;
        let mut history = vec![1, 2, 3];
        let mut rows = 0;
        let result = frame_step(&mut rec, &talker_row(), &mask, EOS, &mut history, |i, _| {
            rows += 1;
            Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32]))
        });
        let err = match result {
            Err(err @ TtsError::Pick(_)) => err.to_string(),
            other => panic!("code {code}: expected a pick error, got {other:?}"),
        };
        assert!(err.contains(&code.to_string()), "code {code}: {err}");
        assert!(err.contains(&CP_VOCAB.to_string()), "code {code}: {err}");
        assert_eq!(
            rows, 1,
            "code {code}: the bad id must not reach the next row"
        );
        assert_eq!(
            history,
            vec![1, 2, 3],
            "code {code}: a refused frame is not history"
        );
    }
}

/// The highest id of the vocabulary is a valid code.
#[test]
fn the_last_code_of_the_vocabulary_is_a_frame() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let last = CP_VOCAB as u32 - 1;
    let mut rec = Recorder::new(7);
    rec.next_code = last;
    rec.code_step = 0;
    let out = frame_step(
        &mut rec,
        &talker_row(),
        &mask,
        EOS,
        &mut vec![1, 2, 3],
        |i, _| Ok(filled(cp_tag(i), &[1, 1, CP_VOCAB as i32])),
    )
    .unwrap();
    let FrameStep::Frame(codes) = out else {
        panic!("expected a frame, got {out:?}");
    };
    let want: Vec<u32> = std::iter::once(7).chain([last; 15]).collect();
    assert_eq!(codes, want);
}

/// The bound is the width of the row that the id was picked from, not a
/// fixed vocabulary.
#[test]
fn a_code_predictor_pick_is_bounded_by_the_width_of_its_row() {
    let mask = talker_allow_mask(TALKER_VOCAB, EOS);
    let mut rec = Recorder::new(7);
    rec.next_code = 1500;
    let result = frame_step(
        &mut rec,
        &talker_row(),
        &mask,
        EOS,
        &mut vec![1, 2, 3],
        |i, _| Ok(filled(cp_tag(i), &[1, 1, 1024])),
    );
    let err = match result {
        Err(err @ TtsError::Pick(_)) => err.to_string(),
        other => panic!("expected a pick error, got {other:?}"),
    };
    assert!(err.contains("1500") && err.contains("1024"), "got: {err}");
}
