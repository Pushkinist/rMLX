//! The windows of a long codec decode: which codes each pass gets and which of
//! its samples are kept. CPU only, no weights.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::{
    decode_in_chunks, TtsError, DECODE_CHUNK_FRAMES as CHUNK, DECODE_LEFT_CONTEXT_FRAMES as CONTEXT,
};
use rmlx_mlx::{Array, Device};

const SAMPLES_PER_FRAME: usize = 3;

/// `[1, 16, frames]` codes whose value in every group is the frame index.
fn codes(frames: usize) -> Array {
    let data: Vec<i32> = (0..16).flat_map(|_| 0..frames as i32).collect();
    Array::from_i32_slice(&data, &[1, 16, frames as i32]).unwrap()
}

/// The samples of frame `index`.
fn frame_samples(index: i32) -> impl Iterator<Item = f32> {
    (0..SAMPLES_PER_FRAME).map(move |j| (index * 10) as f32 + j as f32)
}

/// A pass that decodes each frame of its window to `SAMPLES_PER_FRAME` samples
/// derived from the frame's code, and records the window as
/// `(first code, frames)`.
fn pass(windows: &mut Vec<(i32, usize)>) -> impl FnMut(&Array) -> Result<Vec<f32>, TtsError> + '_ {
    |window| {
        let bytes = window.to_bytes().unwrap();
        let first_group: Vec<i32> = bytes
            .chunks_exact(4)
            .take(window.shape()[2] as usize)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        windows.push((first_group[0], first_group.len()));
        Ok(first_group.into_iter().flat_map(frame_samples).collect())
    }
}

fn expected(frames: usize) -> Vec<f32> {
    (0..frames as i32).flat_map(frame_samples).collect()
}

#[test]
fn a_sequence_of_at_most_one_chunk_is_one_pass_over_all_its_codes() {
    for frames in [1, CONTEXT, CHUNK - 1, CHUNK] {
        let mut windows = Vec::new();
        let samples = decode_in_chunks(&codes(frames), Device::Cpu, pass(&mut windows)).unwrap();
        assert_eq!(windows, [(0, frames)], "{frames} frames");
        assert_eq!(samples, expected(frames), "{frames} frames");
    }
}

#[test]
fn a_later_pass_starts_one_context_early_and_drops_the_context_samples() {
    for frames in [CHUNK + 1, 2 * CHUNK + 100] {
        let mut windows = Vec::new();
        let samples = decode_in_chunks(&codes(frames), Device::Cpu, pass(&mut windows)).unwrap();

        let mut want_windows = vec![(0, CHUNK)];
        let mut start = CHUNK;
        while start < frames {
            let end = (start + CHUNK).min(frames);
            want_windows.push(((start - CONTEXT) as i32, end - start + CONTEXT));
            start = end;
        }
        assert_eq!(windows, want_windows, "{frames} frames");
        assert_eq!(samples.len(), frames * SAMPLES_PER_FRAME, "{frames} frames");
        assert_eq!(samples, expected(frames), "{frames} frames");
    }
}

#[test]
fn a_pass_that_returns_a_partial_frame_is_an_error() {
    let result = decode_in_chunks(&codes(10), Device::Cpu, |_| Ok(vec![0.0; 7]));
    assert!(matches!(result, Err(TtsError::Mlx(_))), "{result:?}");
}

#[test]
fn a_sequence_of_no_frame_is_an_error() {
    let mut passes = 0;
    let result = decode_in_chunks(&codes(0), Device::Cpu, |_| {
        passes += 1;
        Ok(Vec::new())
    });
    assert!(matches!(result, Err(TtsError::Mlx(_))), "{result:?}");
    assert_eq!(passes, 0);
}
