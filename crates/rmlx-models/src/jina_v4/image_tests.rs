use super::*;

// The exact prompt tokenization (single image_pad), verified against the
// snapshot's HF fast tokenizer.
const PROMPT_IDS: [i64; 12] = [
    151644, 872, 198, 151652, 151655, 151653, 74785, 279, 2168, 13, 151645, 198,
];
const IMAGE_TOKEN: i64 = 151655;
const VSTART: i64 = 151652;
const VEND: i64 = 151653;

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn expand_image_pad_reproduces_process_images_sequence() {
    // grid_thw=[1,4,4], spatial_merge=2 -> num_merged = 1*2*2 = 4.
    let ids = expand_image_pad(&PROMPT_IDS, IMAGE_TOKEN, 4).unwrap();
    // Ground truth from the Python process_images probe.
    let expected: Vec<i64> = vec![
        151644, 872, 198, 151652, 151655, 151655, 151655, 151655, 151653, 74785, 279, 2168, 13,
        151645, 198,
    ];
    assert_eq!(ids, expected, "expanded sequence must match process_images");
    assert_eq!(ids.iter().filter(|&&t| t == IMAGE_TOKEN).count(), 4);
}

#[test]
fn expand_image_pad_rejects_bad_input() {
    assert!(expand_image_pad(&PROMPT_IDS, IMAGE_TOKEN, 0).is_err());
    // two image_pad placeholders -> error
    let two = [151652i64, 151655, 151655, 151653];
    assert!(expand_image_pad(&two, IMAGE_TOKEN, 4).is_err());
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn get_rope_index_matches_python_reference() {
    // Ground truth captured from m.model.model.get_rope_index for the
    // grid_thw=[1,4,4] synthetic image (see scripts/parity probe).
    let ids: Vec<i64> = vec![
        151644, 872, 198, 151652, 151655, 151655, 151655, 151655, 151653, 74785, 279, 2168, 13,
        151645, 198,
    ];
    let r = get_rope_index(&ids, IMAGE_TOKEN, 1, 4, 4, 2).unwrap();
    assert_eq!(r.t, vec![0, 1, 2, 3, 4, 4, 4, 4, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(r.h, vec![0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(r.w, vec![0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn get_rope_index_small_grid_consistency() {
    // grid 2x2 / merge 2 -> llm 1x1 -> 1 vision token.
    let ids: Vec<i64> = vec![151652, 151655, 151653, 74785];
    let r = get_rope_index(&ids, IMAGE_TOKEN, 1, 2, 2, 2).unwrap();
    // leading [0]=vstart -> pos 0; vision idx1 -> offset=1 -> (1,1,1)
    // trailing [2,3] -> prev_max=1 -> st_idx=2 -> [2,3]
    assert_eq!(r.t, vec![0, 1, 2, 3]);
    assert_eq!(r.h, vec![0, 1, 2, 3]);
    assert_eq!(r.w, vec![0, 1, 2, 3]);
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
fn build_mrope_tables_unit_circle_and_section_split() {
    let pos = RopeIndex {
        t: vec![0, 5, 9],
        h: vec![0, 2, 7],
        w: vec![0, 3, 8],
    };
    let head_dim = 128;
    let theta = 1_000_000.0;
    let sec = vec![16usize, 24, 24];
    let (cos, sin) = build_mrope_tables(&pos, head_dim, theta, &sec).unwrap();
    assert_eq!(cos.len(), 3 * head_dim);
    for k in 0..cos.len() {
        let m = cos[k].mul_add(cos[k], sin[k] * sin[k]);
        assert!((m - 1.0).abs() < 1e-4, "cos^2+sin^2 != 1 at {k}: {m}");
    }
    // token 0: all positions 0 -> angle 0 -> cos 1, sin 0.
    for d in 0..head_dim {
        assert!((cos[d] - 1.0).abs() < 1e-6, "tok0 cos != 1 at {d}");
        assert!(sin[d].abs() < 1e-6, "tok0 sin != 0 at {d}");
    }
    // emb = cat(freqs,freqs): channel d and d+64 share the angle.
    let half = head_dim / 2;
    for tok in 0..3 {
        let b = tok * head_dim;
        for d in 0..half {
            assert_eq!(cos[b + d], cos[b + half + d], "cat cos {tok},{d}");
            assert_eq!(sin[b + d], sin[b + half + d], "cat sin {tok},{d}");
        }
    }
    // section split: channel 0 (T, inv_freq[0]=1) at tok1 uses T pos 5.
    // channel 16 (first H channel) at tok1 uses H pos 2.
    let inv16 = 1.0f64 / theta.powf(32.0 / 128.0);
    let expect_t0 = (5.0f64).cos() as f32;
    let expect_h16 = (2.0f64 * inv16).cos() as f32;
    assert!((cos[head_dim] - expect_t0).abs() < 1e-3, "T section");
    assert!((cos[head_dim + 16] - expect_h16).abs() < 1e-3, "H section");
}

#[test]
fn build_mrope_tables_rejects_bad_section() {
    let pos = RopeIndex {
        t: vec![0],
        h: vec![0],
        w: vec![0],
    };
    assert!(build_mrope_tables(&pos, 128, 1e6, &[16, 24, 20]).is_err());
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn vision_span_locates_inclusive_bounds() {
    let ids: Vec<i64> = vec![151644, 872, 198, VSTART, 151655, 151655, VEND, 74785, 198];
    let (s, e) = vision_span(&ids, VSTART, VEND).unwrap();
    assert_eq!((s, e), (3, 6));
}
