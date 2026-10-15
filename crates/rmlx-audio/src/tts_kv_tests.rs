//! The key/value slot and the attention over it: the live rows a slot returns
//! are the plain concatenation of what it was given, its capacity grows only
//! when the rows pass it, and the attention over K and V at their own head
//! count equals the attention over K and V repeated for every query head. The
//! talker's RoPE agrees with the reference's table RoPE. CPU only, no weights.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::{
    plain_rope, KvSlot, Linear, Plain, RmsNormLayer, TalkerAttn, TtsError, PREDICTOR_KV_BLOCK_ROWS,
    TALKER_KV_BLOCK_ROWS as BLOCK,
};
use rmlx_mlx::{
    add, concatenate, multiply, negative, scaled_dot_product_attention, Array, Device, Dtype,
};

const D: Device = Device::Cpu;
pub(super) const HIDDEN: i32 = 32;
const HEADS: i32 = 16;
pub(super) const KV_HEADS: i32 = 8;
pub(super) const HEAD_DIM: i32 = 8;
/// Rows of the first call of a talker's sequence; every later call has one.
pub(super) const PREFILL_ROWS: i32 = 9;

/// The call of a talker's sequence whose row is the first that does not fit
/// `blocks` blocks.
pub(super) fn first_call_past(blocks: i32) -> usize {
    (blocks * BLOCK - PREFILL_ROWS + 1) as usize
}

/// A plain linear layer of weight `w`, with no bias.
pub(super) fn plain(w: &Array) -> Linear {
    Linear::Plain(Plain {
        w: w.try_clone().unwrap(),
        b: None,
    })
}

/// An attention layer of [`HEADS`] query heads over [`KV_HEADS`] key/value
/// heads. `weight` gives the array of every weight, by its shape, and
/// `as_linear` makes a projection of a weight.
pub(super) fn attention(
    hidden: i32,
    head_dim: i32,
    mut weight: impl FnMut(&[i32]) -> Array,
    as_linear: impl Fn(&Array) -> Linear,
) -> TalkerAttn {
    let mut linear = |out: i32, inp: i32| as_linear(&weight(&[out, inp]));
    let q_proj = linear(HEADS * head_dim, hidden);
    let k_proj = linear(KV_HEADS * head_dim, hidden);
    let v_proj = linear(KV_HEADS * head_dim, hidden);
    let o_proj = linear(hidden, HEADS * head_dim);
    let mut norm = || RmsNormLayer {
        w: weight(&[head_dim]),
        eps: 1e-6,
    };
    TalkerAttn {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm: norm(),
        k_norm: norm(),
        n_heads: HEADS,
        n_kv_heads: KV_HEADS,
        head_dim,
        scale: 1.0 / (head_dim as f32).sqrt(),
        rope_theta: 1e6,
    }
}

/// A fixed sequence of values in `[-1, 1)`.
pub(super) struct Values(pub(super) u64);

impl Values {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    pub(super) fn array(&mut self, shape: &[i32]) -> Array {
        let len = shape.iter().product::<i32>() as usize;
        let data: Vec<f32> = (0..len).map(|_| self.next()).collect();
        Array::from_f32_slice(&data, shape).unwrap()
    }
}

/// Give a slot of `block` rows one call per entry of `calls`, each with that
/// many new rows, and compare the live rows it returns with a plain
/// concatenation after every call. Returns the calls in which the capacity
/// grew.
fn calls_that_grew(block: i32, calls: &[i32]) -> Vec<usize> {
    let mut values = Values(0x5eed);
    let mut slot = KvSlot::new(block);
    let mut want: Option<(Array, Array)> = None;
    let mut grew = Vec::new();
    for (call, &rows) in calls.iter().enumerate() {
        let shape = [1, KV_HEADS, rows, HEAD_DIM];
        let (new_k, new_v) = (values.array(&shape), values.array(&shape));
        let capacity = slot.capacity;
        let (k, v) = slot.update_and_fetch(&new_k, &new_v, D).unwrap();
        if slot.capacity > capacity {
            grew.push(call);
        }

        let (want_k, want_v) = match want.take() {
            None => (new_k, new_v),
            Some((k_before, v_before)) => (
                concatenate(&[&k_before, &new_k], 2, D).unwrap(),
                concatenate(&[&v_before, &new_v], 2, D).unwrap(),
            ),
        };
        assert_eq!(k.shape(), want_k.shape(), "K, call {call}");
        assert_eq!(v.shape(), want_v.shape(), "V, call {call}");
        assert!(
            k.to_bytes().unwrap() == want_k.to_bytes().unwrap(),
            "K differs from the concatenation, call {call}"
        );
        assert!(
            v.to_bytes().unwrap() == want_v.to_bytes().unwrap(),
            "V differs from the concatenation, call {call}"
        );
        let live = want_k.shape()[2];
        assert_eq!(slot.rows, live, "call {call}");
        assert_eq!(
            slot.capacity,
            (live + block - 1) / block * block,
            "call {call}: the capacity is the least multiple of {block} rows that holds {live}"
        );
        want = Some((want_k, want_v));
    }
    grew
}

/// The talker's sequence: a prefill of several rows, then one row a frame,
/// past two block edges.
#[test]
fn a_slot_returns_the_concatenation_and_grows_only_past_a_block_edge() {
    let mut calls = vec![PREFILL_ROWS];
    calls.resize(1 + 2 * BLOCK as usize, 1);
    assert_eq!(
        calls_that_grew(BLOCK, &calls),
        [0, first_call_past(1), first_call_past(2)]
    );
}

#[test]
fn a_first_call_of_more_than_one_block_takes_two_blocks() {
    let mut calls = vec![BLOCK + 44];
    calls.resize(1 + BLOCK as usize, 1);
    let first_past_two_blocks = (2 * BLOCK - (BLOCK + 44) + 1) as usize;
    assert_eq!(calls_that_grew(BLOCK, &calls), [0, first_past_two_blocks]);
}

/// Calls of several rows: one that ends before an edge, one that crosses it,
/// one that fills the capacity to its last row, and the row after that.
#[test]
fn a_slot_takes_any_row_count_and_grows_only_when_the_rows_do_not_fit() {
    let calls = [BLOCK - 6, 3, 7, 1, BLOCK - 5, 1];
    assert_eq!(calls_that_grew(BLOCK, &calls), [0, 2, 5]);
}

/// The code predictor's sequence in one frame: two rows, then one row for
/// each further group. Its slot keeps no unused row, so its buffers have the
/// sizes of a plain concatenation.
#[test]
fn a_code_predictor_slot_holds_no_unused_row() {
    let mut calls = vec![2];
    calls.resize(15, 1);
    assert_eq!(
        calls_that_grew(PREDICTOR_KV_BLOCK_ROWS, &calls),
        (0..15).collect::<Vec<_>>()
    );
}

/// Rows of another dtype than the buffer are refused, for the keys and for
/// the values, and the slot keeps its rows: rows that fit the capacity
/// (`slice_update` would cast them to the dtype of the buffer) and rows that
/// grow it (the concatenation would promote the buffer to their dtype).
#[test]
fn rows_of_another_dtype_than_the_buffer_are_refused() {
    for block in [BLOCK, PREDICTOR_KV_BLOCK_ROWS] {
        refuses_rows_of_another_dtype(block);
    }
}

fn refuses_rows_of_another_dtype(block: i32) {
    let mut values = Values(0xd7e);
    let shape = [1, KV_HEADS, 1, HEAD_DIM];
    let half = |values: &mut Values| values.array(&shape).astype(Dtype::Bf16, D).unwrap();
    let mut slot = KvSlot::new(block);
    slot.update_and_fetch(&half(&mut values), &half(&mut values), D)
        .unwrap();

    let wide = values.array(&shape);
    for (k, v) in [
        (wide.try_clone().unwrap(), half(&mut values)),
        (half(&mut values), wide.try_clone().unwrap()),
    ] {
        match slot.update_and_fetch(&k, &v, D) {
            Err(TtsError::Mlx(message)) => {
                assert!(
                    message.contains("F32") && message.contains("Bf16"),
                    "the error must name both dtypes: {message}"
                );
            }
            other => panic!(
                "block of {block} rows: expected the dtype error, got {:?}",
                other.map(|_| ())
            ),
        }
        assert_eq!(slot.rows, 1);
    }
    slot.update_and_fetch(&half(&mut values), &half(&mut values), D)
        .unwrap();
    assert_eq!(slot.rows, 2);
}

/// `[1, KV_HEADS, rows, HEAD_DIM]` with every head repeated for its group of
/// query heads: `[1, HEADS, rows, HEAD_DIM]`.
fn repeated_for_the_query_heads(x: &Array) -> Array {
    let rows = x.shape()[2];
    let grouped = x.reshape(&[1, KV_HEADS, 1, rows, HEAD_DIM], D).unwrap();
    let copies = vec![&grouped; (HEADS / KV_HEADS) as usize];
    concatenate(&copies, 2, D)
        .unwrap()
        .reshape(&[1, HEADS, rows, HEAD_DIM], D)
        .unwrap()
}

/// The attention of `attn` over `x` with K and V kept by concatenation in
/// `history` and repeated for every query head, with `rope(x, offset)` applied
/// to the normed queries and keys.
fn attention_over_repeated_kv(
    attn: &TalkerAttn,
    x: &Array,
    history: &mut Option<(Array, Array)>,
    offset: i32,
    rope: impl Fn(&Array, i32) -> Array,
) -> Array {
    let rows = x.shape()[1];
    let heads = |projected: Array, heads: i32| {
        projected
            .reshape(&[1, rows, heads, HEAD_DIM], D)
            .unwrap()
            .transpose(&[0, 2, 1, 3], D)
            .unwrap()
    };
    let normed = |x: &Array, norm: &RmsNormLayer, heads: i32| {
        let flat = x.reshape(&[heads, rows, HEAD_DIM], D).unwrap();
        norm.forward(&flat, D)
            .unwrap()
            .reshape(&[1, heads, rows, HEAD_DIM], D)
            .unwrap()
    };
    let q = heads(attn.q_proj.forward(x, D).unwrap(), HEADS);
    let k = heads(attn.k_proj.forward(x, D).unwrap(), KV_HEADS);
    let v = heads(attn.v_proj.forward(x, D).unwrap(), KV_HEADS);
    let q = rope(&normed(&q, &attn.q_norm, HEADS), offset);
    let k = rope(&normed(&k, &attn.k_norm, KV_HEADS), offset);

    let (k, v) = match history.take() {
        None => (k, v),
        Some((k_before, v_before)) => (
            concatenate(&[&k_before, &k], 2, D).unwrap(),
            concatenate(&[&v_before, &v], 2, D).unwrap(),
        ),
    };
    let out = scaled_dot_product_attention(
        &q,
        &repeated_for_the_query_heads(&k),
        &repeated_for_the_query_heads(&v),
        attn.scale,
        "causal",
        None,
        D,
    )
    .unwrap()
    .transpose(&[0, 2, 1, 3], D)
    .unwrap()
    .reshape(&[1, rows, HEADS * HEAD_DIM], D)
    .unwrap();
    *history = Some((k, v));
    attn.o_proj.forward(&out, D).unwrap()
}

/// A prefill of nine query rows, then one query row a step. The live rows run
/// from 9 to 15: no multiple of a head count or of the block.
#[test]
fn attention_at_the_kv_head_count_equals_attention_over_repeated_kv() {
    let mut values = Values(0xa77e);
    let attn = attention(HIDDEN, HEAD_DIM, |shape| values.array(shape), plain);
    let mut slot = KvSlot::new(BLOCK);
    let mut history = None;
    let mut offset = 0;
    for rows in [PREFILL_ROWS, 1, 1, 1, 1, 1, 1] {
        let x = values.array(&[1, rows, HIDDEN]);
        let got = attn.forward(&x, &mut slot, offset, D).unwrap();
        let want = attention_over_repeated_kv(&attn, &x, &mut history, offset, |x, offset| {
            plain_rope(x, HEAD_DIM, attn.rope_theta, offset, D).unwrap()
        });
        offset += rows;
        assert_eq!(got.shape(), want.shape(), "{offset} live rows");
        assert!(
            got.to_bytes().unwrap() == want.to_bytes().unwrap(),
            "the outputs differ at {rows} query rows, {offset} live rows"
        );
    }
}

/// RoPE as the reference applies it to `x: [1, heads, S, head_dim]` at
/// positions `offset..offset + S`: `x * cos + rotate_half(x) * sin`, where
/// element `i` of a row turns by `position * theta^(-2 (i mod half) / head_dim)`.
/// The tables are computed on the host in f64.
fn table_rope(x: &Array, offset: i32, theta: f32) -> Array {
    let shape = x.shape();
    let (heads, rows, dim) = (shape[1], shape[2], shape[3]);
    let half = dim / 2;
    let angles: Vec<f64> = (offset..offset + rows)
        .flat_map(|position| {
            (0..dim).map(move |i| {
                let exponent = -2.0 * f64::from(i % half) / f64::from(dim);
                f64::from(position) * f64::from(theta).powf(exponent)
            })
        })
        .collect();
    let table = |f: fn(f64) -> f64| {
        let values: Vec<f32> = angles.iter().map(|&a| f(a) as f32).collect();
        Array::from_f32_slice(&values, &[1, 1, rows, dim]).unwrap()
    };
    let first = x
        .slice(&[0, 0, 0, 0], &[1, heads, rows, half], &[1; 4], D)
        .unwrap();
    let second = x
        .slice(&[0, 0, 0, half], &[1, heads, rows, dim], &[1; 4], D)
        .unwrap();
    let rotated = concatenate(&[&negative(&second, D).unwrap(), &first], 3, D).unwrap();
    add(
        &multiply(x, &table(f64::cos), D).unwrap(),
        &multiply(&rotated, &table(f64::sin), D).unwrap(),
        D,
    )
    .unwrap()
}

/// The largest absolute difference of two arrays of one shape. A value that
/// is not finite in either is a failure: the fold would drop a NaN.
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    assert_eq!(a.shape(), b.shape());
    let floats = |x: &Array| -> Vec<f32> {
        let values: Vec<f32> = x
            .astype(Dtype::F32, D)
            .unwrap()
            .to_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "an array holds a value that is not finite"
        );
        values
    };
    floats(a)
        .iter()
        .zip(floats(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A position past the last one a talker reaches: the prefill and
/// [`MAX_FRAMES`](super::MAX_FRAMES) frames.
const FAR_OFFSET: i32 = 2100;

/// The fused op computes the angle of a position in f32, which moves a value
/// by up to about 2e-4 at [`FAR_OFFSET`] and by less than 1e-6 at position 0.
/// One bf16 step at 1.0 is 7.8e-3.
const ROPE_TOLERANCE: f32 = 1e-3;

/// The shapes of the RoPE in the checkpoints: the talker and the code
/// predictor (head dim 128, base 1e6) and the codec decoder's pre-transformer
/// (16 heads, head dim 64, base 1e4), as `(heads, head_dim, theta)`.
const ROPE_SHAPES: [(i32, i32, f32); 2] = [(KV_HEADS, 128, 1e6), (16, 64, 1e4)];

/// On f32 input, the fused RoPE agrees with the reference's table RoPE: the
/// same variant (halves rotated, not traditional), base and rotated width. A
/// mismatch in any of them moves the values by far more than the tolerance.
#[test]
fn the_fused_rope_matches_the_table_rope_of_the_reference() {
    let mut values = Values(0x2073);
    for (heads, head_dim, theta) in ROPE_SHAPES {
        for offset in [0, 1, FAR_OFFSET] {
            let x = values.array(&[1, heads, PREFILL_ROWS, head_dim]);
            let diff = max_abs_diff(
                &plain_rope(&x, head_dim, theta, offset, D).unwrap(),
                &table_rope(&x, offset, theta),
            );
            assert!(
                diff < ROPE_TOLERANCE,
                "head dim {head_dim}, base {theta}, offset {offset}: the RoPEs differ by {diff}"
            );
        }
    }

    let attn = attention(HIDDEN, HEAD_DIM, |shape| values.array(shape), plain);
    let mut slot = KvSlot::new(BLOCK);
    let mut history = None;
    for (rows, offset) in [(PREFILL_ROWS, 0), (1, PREFILL_ROWS), (1, FAR_OFFSET)] {
        let x = values.array(&[1, rows, HIDDEN]);
        let got = attn.forward(&x, &mut slot, offset, D).unwrap();
        let want = attention_over_repeated_kv(&attn, &x, &mut history, offset, |x, offset| {
            table_rope(x, offset, attn.rope_theta)
        });
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < ROPE_TOLERANCE,
            "attention, {rows} rows at offset {offset}: the outputs differ by {diff}"
        );
    }
}
