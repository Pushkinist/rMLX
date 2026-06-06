use super::*;
use rmlx_mlx::Dtype;
use std::mem::size_of_val;

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn arr_from_f32(data: &[f32], shape: &[i32]) -> Array {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("from_bytes")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn materialize(a: &Array) -> Vec<f32> {
    let f = a.astype(Dtype::F32, Device::Cpu).expect("cast");
    f.eval().expect("eval");
    f.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
fn vision_scatter_shape_and_values() {
    // seq=5, hidden=4. image_token_id at positions 1,2,3 (3 vision tokens).
    let dev = Device::Cpu;
    let hidden = 4usize;
    let seq = 5usize;
    let embeds = arr_from_f32(&vec![0.0f32; seq * hidden], &[1, seq as i32, hidden as i32]);
    // 3 vision feature rows, each filled with its row index + 1.
    let mut vf = Vec::new();
    for r in 0..3 {
        for _ in 0..hidden {
            vf.push((r + 1) as f32);
        }
    }
    let vfeat = arr_from_f32(&vf, &[3, hidden as i32]);

    let input_ids: Vec<i64> = vec![100, 151655, 151655, 151655, 200];
    let positions = visual_token_positions(&input_ids, 151655);
    assert_eq!(positions, vec![1, 2, 3]);

    let out = scatter_vision_features(&embeds, &vfeat, &positions, dev).unwrap();
    out.eval().unwrap();
    assert_eq!(out.shape(), vec![1, seq as i32, hidden as i32]);

    let vals = materialize(&out);
    // pos 0: zeros; pos 1: ones; pos 2: twos; pos 3: threes; pos 4: zeros.
    assert_eq!(&vals[0..4], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&vals[4..8], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&vals[8..12], &[2.0, 2.0, 2.0, 2.0]);
    assert_eq!(&vals[12..16], &[3.0, 3.0, 3.0, 3.0]);
    assert_eq!(&vals[16..20], &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn scatter_rejects_count_mismatch() {
    let dev = Device::Cpu;
    let embeds = arr_from_f32(&[0.0f32; 5 * 4], &[1, 5, 4]);
    let vfeat = arr_from_f32(&[1.0f32; 2 * 4], &[2, 4]);
    // 3 positions but only 2 features.
    assert!(scatter_vision_features(&embeds, &vfeat, &[1, 2, 3], dev).is_err());
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
fn deepstack_inject_adds_at_positions() {
    let dev = Device::Cpu;
    let hidden = 4usize;
    let seq = 5usize;
    // base hidden: pos i filled with i.
    let mut base = Vec::new();
    for i in 0..seq {
        for _ in 0..hidden {
            base.push(i as f32);
        }
    }
    let h = arr_from_f32(&base, &[1, seq as i32, hidden as i32]);
    // deepstack embeds for positions 1,2,3 -- each row = 10.
    let de = arr_from_f32(&vec![10.0f32; 3 * hidden], &[3, hidden as i32]);
    let out = deepstack_inject(&h, &de, &[1, 2, 3], dev).unwrap();
    out.eval().unwrap();
    let vals = materialize(&out);
    // pos 0 unchanged (0); pos1 1+10=11; pos2 2+10=12; pos3 3+10=13; pos4 4.
    assert_eq!(&vals[0..4], &[0.0; 4]);
    assert_eq!(&vals[4..8], &[11.0; 4]);
    assert_eq!(&vals[8..12], &[12.0; 4]);
    assert_eq!(&vals[12..16], &[13.0; 4]);
    assert_eq!(&vals[16..20], &[4.0; 4]);
}
