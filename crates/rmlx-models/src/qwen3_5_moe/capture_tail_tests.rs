//! What a chunked capture keeps, against what it would have kept unbounded.
//!
//! The oracle is the unbounded accumulation itself: the same chunks are pushed
//! twice, once with a tail limit and once without, and the bounded answer has to
//! be the tail of the other one row for row. That is what makes a widened limit
//! visible — a bound that kept more rows, or the whole capture, still returns
//! rows the drafter can read, and only the count and the peak say so.
//!
//! Everything here runs on the CPU device and needs no model snapshot.

use rmlx_mlx::{Array, Device};

use super::CaptureTail;

/// The published DFlash 2 pair's conditioning depth: `sliding_window - 1`.
const KEEP: usize = 2047;
/// Prompt positions per prefill chunk, as both round loops pass it.
const CHUNK: usize = 1024;
/// A stand-in for `len(capture_layer_ids) * hidden_size`, narrow enough that a
/// long prompt's rows are cheap to compare byte for byte.
const WIDTH: i32 = 4;

/// A chunk whose every row holds that row's own absolute prompt position, so a
/// kept row says where it came from.
#[allow(
    clippy::expect_used,
    reason = "test helper: an array that cannot be built is the assertion failing"
)]
fn chunk_at(first_row: usize, rows: usize) -> Array {
    let data: Vec<f32> = (first_row..first_row + rows)
        .flat_map(|r| std::iter::repeat_n(r as f32, WIDTH as usize))
        .collect();
    Array::from_f32_slice(&data, &[1, rows as i32, WIDTH]).expect("build chunk")
}

/// Push `n` prompt positions through `tail` in `CHUNK`-sized chunks.
#[allow(
    clippy::expect_used,
    reason = "test helper: a chunk the accumulator refuses is the assertion failing"
)]
fn push_prompt(tail: &mut CaptureTail, n: usize) {
    let mut pos = 0usize;
    while pos < n {
        let end = (pos + CHUNK).min(n);
        tail.push(chunk_at(pos, end - pos)).expect("push chunk");
        pos = end;
    }
}

#[allow(
    clippy::expect_used,
    reason = "test assertion: an array that cannot be read back is the assertion failing"
)]
fn bytes_of(a: &Array) -> Vec<u8> {
    a.to_bytes().expect("read capture bytes")
}

/// The first element of every row — that row's prompt position.
#[allow(
    clippy::indexing_slicing,
    reason = "test assertion: the rank is the one the accumulator returns"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an array that cannot be read back is the assertion failing"
)]
fn row_positions(a: &Array) -> Vec<usize> {
    let width = a.shape()[2] as usize;
    let bytes = a.to_bytes().expect("read capture bytes");
    bytes
        .chunks_exact(4)
        .step_by(width)
        .map(|b| {
            let v = f32::from_le_bytes(b.try_into().expect("4 bytes is an f32"));
            v as usize
        })
        .collect()
}

/// A prompt far longer than the window keeps the window's rows, and they are
/// the newest ones — byte for byte what the unbounded capture ends with.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test assertion: the rank is the one the accumulator returns"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a capture that cannot be joined is the assertion failing"
)]
fn bounded_capture_is_the_tail_of_the_unbounded_one() {
    const N: usize = 5000;

    let mut unbounded = CaptureTail::new(None);
    push_prompt(&mut unbounded, N);
    assert_eq!(
        unbounded.retained_rows(),
        N,
        "an unbounded accumulation holds every prompt position"
    );
    let whole = unbounded.finish(Device::Cpu).expect("join unbounded");
    assert_eq!(whole.shape(), vec![1, N as i32, WIDTH]);
    let whole_rows = row_positions(&whole);

    let mut bounded = CaptureTail::new(Some(KEEP));
    push_prompt(&mut bounded, N);
    let peak = bounded.retained_rows();
    let kept = bounded.finish(Device::Cpu).expect("join bounded");

    assert_eq!(
        kept.shape(),
        vec![1, KEEP as i32, WIDTH],
        "the kept capture is the window's rows, not the prompt's"
    );
    assert_eq!(
        row_positions(&kept),
        whole_rows[N - KEEP..].to_vec(),
        "the kept rows are the newest ones"
    );

    let tail = whole
        .slice(
            &[0, (N - KEEP) as i32, 0],
            &[1, N as i32, WIDTH],
            &[1, 1, 1],
            Device::Cpu,
        )
        .expect("slice the unbounded tail");
    assert_eq!(
        bytes_of(&kept),
        bytes_of(&tail),
        "the kept rows differ from the tail the unbounded capture would have handed back"
    );

    // The peak is the tail plus the chunk being filled, not the prompt: a
    // bound that held every chunk to the end returns the same rows and only
    // this says so.
    assert!(
        peak <= KEEP + CHUNK,
        "held {peak} rows at the end of a {N}-position prompt, more than the \
         window {KEEP} plus one chunk {CHUNK}"
    );
}

/// A prompt shorter than the window is kept whole.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a capture that cannot be joined is the assertion failing"
)]
fn a_prompt_inside_the_window_keeps_every_row() {
    const N: usize = 1500;

    let mut bounded = CaptureTail::new(Some(KEEP));
    push_prompt(&mut bounded, N);
    let kept = bounded.finish(Device::Cpu).expect("join bounded");

    assert_eq!(kept.shape(), vec![1, N as i32, WIDTH]);
    assert_eq!(row_positions(&kept), (0..N).collect::<Vec<_>>());
}

/// A capture of another rank is refused rather than joined along whichever axis
/// happens to be second.
#[test]
fn a_chunk_of_the_wrong_rank_is_refused() {
    let mut tail = CaptureTail::new(Some(KEEP));
    let flat = match Array::from_f32_slice(&[0.0, 1.0, 2.0, 3.0], &[1, 4]) {
        Ok(a) => a,
        Err(e) => panic!("build flat chunk: {e}"),
    };
    let err = match tail.push(flat) {
        Ok(()) => panic!("a rank-2 chunk was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("[1, rows, width]"),
        "the refusal does not name the shape it wanted: {err}"
    );
}

/// Nothing pushed is an error, not an empty answer a caller would condition on.
#[test]
fn an_empty_accumulation_is_refused() {
    let tail = CaptureTail::new(None);
    match tail.finish(Device::Cpu) {
        Ok(a) => panic!("an empty accumulation returned {:?}", a.shape()),
        Err(e) => assert!(
            e.to_string().contains("no capture chunk"),
            "unexpected refusal: {e}"
        ),
    }
}
