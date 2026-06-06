use super::*;

const IMAGE_TOKEN: i64 = 151655;

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn text_only_positions_are_sequential() {
    let ids = vec![1i64, 2, 3, 4, 5];
    let r = get_rope_index(&ids, &[], IMAGE_TOKEN, 2).unwrap();
    assert_eq!(r.t, vec![0, 1, 2, 3, 4]);
    assert_eq!(r.h, vec![0, 1, 2, 3, 4]);
    assert_eq!(r.w, vec![0, 1, 2, 3, 4]);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn get_rope_index_single_image_matches_reference() {
    // grid_thw=[1,4,4], merge=2 -> llm grid 1x2x2 -> 4 vision tokens.
    // Sequence: [vstart, pad, pad, pad, pad, vend, text].
    // This is the same structure verified for Qwen2.5-VL get_rope_index
    // (the 3D index math is identical; only the *application* layout
    // differs between chunked and interleaved).
    let ids: Vec<i64> = vec![
        100,    // some text token at pos 0
        151652, // vision_start at pos 1
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN, // 4 pads
        151653,      // vision_end
        200,
        201, // trailing text
    ];
    let r = get_rope_index(&ids, &[(1, 4, 4)], IMAGE_TOKEN, 2).unwrap();
    // leading text [100, vstart] -> positions 0,1 in all dims.
    // image block: ed = index of first pad = 2. text_len=2, st_idx=0,
    // offset = 2. llm grid 1x2x2:
    // (t,h,w) for the 4 tokens: t all 0+2=2;
    // h: 0,0,1,1 + 2 = 2,2,3,3 ; w: 0,1,0,1 + 2 = 2,3,2,3
    // trailing text: max prev = 3 -> st_idx=4 -> vend,200,201 = 4,5,6
    assert_eq!(r.t, vec![0, 1, 2, 2, 2, 2, 4, 5, 6]);
    assert_eq!(r.h, vec![0, 1, 2, 2, 3, 3, 4, 5, 6]);
    assert_eq!(r.w, vec![0, 1, 2, 3, 2, 3, 4, 5, 6]);
}

#[test]
fn interleaved_section_map_layout() {
    // Toy: head_dim=24 -> half=12, mrope_section summing to 12.
    // Use [4,4,4] for a clean interleave illustration.
    let sec = interleaved_section_map(12, &[4, 4, 4]);
    // base all T; H overwrites c in {1,4,7,10} (< 4*3=12); W overwrites
    // c in {2,5,8,11} (< 12). So:
    // c: 0 1 2 3 4 5 6 7 8 9 10 11
    // T H W T H W T H W T H W
    assert_eq!(sec, vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn interleaved_section_map_target_sections() {
    // Real target: head_dim=128 -> half=64, mrope_section=[24,20,20].
    let half = 64usize;
    let sec = interleaved_section_map(half, &[24, 20, 20]);
    assert_eq!(sec.len(), half);
    // T is the default and also fills channels beyond the H/W ranges.
    // H range: offset 1, step 3, up to 60 -> {1,4,...,58} = 20 channels.
    // W range: offset 2, step 3, up to 60 -> {2,5,...,59} = 20 channels.
    let n_h = sec.iter().filter(|&&s| s == 1).count();
    let n_w = sec.iter().filter(|&&s| s == 2).count();
    let n_t = sec.iter().filter(|&&s| s == 0).count();
    assert_eq!(n_h, 20, "H channels");
    assert_eq!(n_w, 20, "W channels");
    assert_eq!(n_t, 24, "T channels (default + remainder)");
    // Spot-check the interleave: channel 1 = H, channel 2 = W, channel 0 = T.
    assert_eq!(sec[0], 0);
    assert_eq!(sec[1], 1);
    assert_eq!(sec[2], 2);
    // Channel 60 onward (>= 20*3) all T.
    assert!(sec[60..].iter().all(|&s| s == 0));
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
fn build_interleaved_tables_reference_values() {
    // Hand-computed reference for a tiny grid_thw=[1,2,2], merge=2 -> 1
    // vision token. head_dim=4 -> half=2. With section [1,1,0]? Must sum
    // to half=2. Use head_dim=6 -> half=3, section [1,1,1].
    let head_dim = 6usize;
    let theta = 1_000_000.0_f64;
    let section = vec![1usize, 1, 1];
    // positions: token 0 (t=2,h=3,w=5).
    let pos = RopeIndex3D {
        t: vec![2],
        h: vec![3],
        w: vec![5],
    };
    let (cos, sin) = build_interleaved_mrope_tables(&pos, head_dim, theta, &section).unwrap();
    // half=3. section_map: base T; H overwrites c in {1} (<1*3=3); W
    // overwrites c in {2} (<3). So sec = [T, H, W] = [0,1,2].
    // inv_freq[c] = theta^(-2c/6): c=0 ->1, c=1 ->theta^(-1/3), c=2 ->theta^(-2/3).
    let inv0 = 1.0f64;
    let inv1 = theta.powf(-1.0 / 3.0);
    let inv2 = theta.powf(-2.0 / 3.0);
    // angle[c] uses sec[c]'s position: c0->t=2, c1->h=3, c2->w=5.
    let a0 = 2.0 * inv0;
    let a1 = 3.0 * inv1;
    let a2 = 5.0 * inv2;
    let exp = [a0, a1, a2];
    for c in 0..3 {
        assert!((f64::from(cos[c]) - exp[c].cos()).abs() < 1e-4, "cos[{c}]");
        assert!((f64::from(sin[c]) - exp[c].sin()).abs() < 1e-4, "sin[{c}]");
        // cat(freqs,freqs): channel c and c+half share the angle.
        assert_eq!(cos[c], cos[c + 3], "cat cos {c}");
        assert_eq!(sin[c], sin[c + 3], "cat sin {c}");
    }
    // unit circle.
    for k in 0..cos.len() {
        let m = cos[k].mul_add(cos[k], sin[k] * sin[k]);
        assert!((m - 1.0).abs() < 1e-4, "unit circle at {k}");
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
fn build_interleaved_tables_token0_identity() {
    // All-zero positions -> cos 1, sin 0 everywhere.
    let pos = RopeIndex3D {
        t: vec![0],
        h: vec![0],
        w: vec![0],
    };
    let (cos, sin) = build_interleaved_mrope_tables(&pos, 128, 5e6, &[24, 20, 20]).unwrap();
    for d in 0..128 {
        assert!((cos[d] - 1.0).abs() < 1e-6, "cos!=1 at {d}");
        assert!(sin[d].abs() < 1e-6, "sin!=0 at {d}");
    }
}

#[test]
fn build_interleaved_tables_rejects_bad_section() {
    let pos = RopeIndex3D {
        t: vec![0],
        h: vec![0],
        w: vec![0],
    };
    // sums to 63 != 64.
    assert!(build_interleaved_mrope_tables(&pos, 128, 5e6, &[24, 20, 19]).is_err());
    // wrong arity.
    assert!(build_interleaved_mrope_tables(&pos, 128, 5e6, &[64]).is_err());
}
