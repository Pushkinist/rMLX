//! The memory of the frame loop and of the codec decode. The talker's key and
//! value buffers are written in place between two growths of their capacity,
//! so the MLX buffer cache stays small with no periodic release; the loop
//! releases it after a growth, when it ends and after a failed step, and the
//! buffers that its step owns are freed before that last release; the decode
//! releases it between its passes. CPU only, no weights.
//!
//! The allocator and its cache are process-global, so each check runs in a
//! child of this test binary that no other test shares.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::tts_device_tests::{run_child, started_by_parent, CHILD_DONE};
use super::tts_kv_tests::{attention, first_call_past, plain, KV_HEADS, PREFILL_ROWS};
use super::{
    decode_in_chunks, run_frames, FrameStep, KvSlot, TalkerAttn, TtsError, DECODE_CHUNK_FRAMES,
    TALKER_KV_BLOCK_ROWS as BLOCK,
};
use rmlx_mlx::{add, mlx_cache_memory_bytes, Array, Device, PeakBracket};
use std::sync::atomic::AtomicBool;

const HIDDEN: i32 = 256;
const HEAD_DIM: i32 = 128;
const LAYERS: usize = 2;
/// The live rows pass one block and two blocks inside the run.
const FRAMES: usize = (2 * BLOCK + 40) as usize;
const LAST_OWNED_FRAME: usize = 11;
const PASS_SCRATCH_ELEMENTS: usize = 1 << 20;
const MIB: u64 = 1 << 20;
/// One key or value buffer of a layer at the first capacity.
const BUFFER_BYTES: u64 = (KV_HEADS * BLOCK * HEAD_DIM * 4) as u64;
/// What the last growth of the run frees: the key buffer and the value buffer
/// of every layer, at two blocks each.
const FREED_BY_THE_LAST_GROWTH: u64 = LAYERS as u64 * 2 * 2 * BUFFER_BYTES;
/// The cache is largest in the last growth frame, before its release: it holds
/// the buffers that the growth freed and what the frame itself left, about
/// 10.5 MiB in all. The bound is four times the freed buffers. With buffers
/// one row longer in each frame and the same releases, the cache reaches about
/// 580 MiB.
const CACHE_BOUND: u64 = 4 * FREED_BY_THE_LAST_GROWTH;

/// `LAYERS` attention layers over the talker's key/value slots.
struct Talker {
    attn: TalkerAttn,
    kv: Vec<KvSlot>,
    offset: i32,
}

impl Talker {
    fn new() -> Self {
        let weight = |shape: &[i32]| {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_f32_slice(&vec![0.01; len], shape).unwrap()
        };
        Self {
            attn: attention(HIDDEN, HEAD_DIM, weight, plain),
            kv: (0..LAYERS).map(|_| KvSlot::new(BLOCK)).collect(),
            offset: 0,
        }
    }

    /// Run `rows` positions through every layer and evaluate the result.
    /// Returns whether the capacity grew.
    ///
    /// The growth read is a copy of the one in `synthesize`, which no CPU test
    /// runs to its frame loop: that needs the weights. A change of the read in
    /// `synthesize` is not seen here.
    fn step(&mut self, rows: i32) -> bool {
        let capacity = self.kv[0].capacity;
        let mut x = Array::from_f32_slice(&vec![0.5; (rows * HIDDEN) as usize], &[1, rows, HIDDEN])
            .unwrap();
        for slot in &mut self.kv {
            x = self
                .attn
                .forward(&x, slot, self.offset, Device::Cpu)
                .unwrap();
        }
        x.eval().unwrap();
        self.offset += rows;
        self.kv[0].capacity > capacity
    }
}

#[test]
fn the_buffer_cache_stays_small_with_a_release_only_after_a_growth() {
    run_child(
        "tts_memory_tests",
        "the_buffer_cache_stays_small_with_a_release_only_after_a_growth_child",
    );
}

#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn the_buffer_cache_stays_small_with_a_release_only_after_a_growth_child() {
    if !started_by_parent() {
        return;
    }
    let mut talker = Talker::new();
    let mut cache_at_entry = Vec::new();
    let mut growth_frames = Vec::new();
    let mut peak_cache = 0;
    let mut live_rise_inside_a_block = 0;

    let frames = run_frames(&AtomicBool::new(false), |index| {
        if index == FRAMES {
            return Ok((FrameStep::End, false));
        }
        cache_at_entry.push(mlx_cache_memory_bytes().unwrap());
        let bracket = PeakBracket::open();
        let grew = talker.step(if index == 0 { PREFILL_ROWS } else { 1 });
        let reading = bracket.close();
        assert!(reading.measurable(), "{reading:?}");
        if grew {
            growth_frames.push(index);
        } else {
            live_rise_inside_a_block = live_rise_inside_a_block.max(reading.headroom_bytes());
        }
        peak_cache = peak_cache.max(mlx_cache_memory_bytes().unwrap());
        Ok((FrameStep::Frame(vec![0; 16]), grew))
    })
    .unwrap();

    assert_eq!(frames.codes.len(), FRAMES);
    assert!(!frames.truncated);
    assert_eq!(growth_frames, [0, first_call_past(1), first_call_past(2)]);
    assert_eq!(cache_at_entry.len(), FRAMES);
    for (index, &cached) in cache_at_entry.iter().enumerate().skip(1) {
        let after_growth = growth_frames.contains(&(index - 1));
        assert_eq!(
            cached == 0,
            after_growth,
            "frame {index} started with {cached} B in the cache; the frame before it {}",
            if after_growth {
                "grew the capacity, so the cache must be released"
            } else {
                "grew nothing, so nothing may release the cache"
            }
        );
    }
    assert!(
        live_rise_inside_a_block < BUFFER_BYTES / 2,
        "a frame between two growths raised the live bytes by {live_rise_inside_a_block} B; \
         a copy of one buffer takes at least {BUFFER_BYTES} B"
    );
    assert!(
        peak_cache < CACHE_BOUND,
        "the buffer cache reached {} MiB over {FRAMES} frames; the bound is {} MiB",
        peak_cache / MIB,
        CACHE_BOUND / MIB
    );
    assert_eq!(
        mlx_cache_memory_bytes().unwrap(),
        0,
        "the cache must be empty once the loop has ended"
    );

    // As in `synthesize`, the step owns the key/value buffers. They must be
    // freed before the last release, or they are in the cache after it.
    for fails in [false, true] {
        let mut owned = Talker::new();
        let result = run_frames(&AtomicBool::new(false), move |index| {
            let grew = owned.step(if index == 0 { PREFILL_ROWS } else { 1 });
            match (index == LAST_OWNED_FRAME, fails) {
                (false, _) => Ok((FrameStep::Frame(vec![0; 16]), grew)),
                (true, false) => Ok((FrameStep::End, grew)),
                (true, true) => Err(TtsError::Mlx("step failed".into())),
            }
        });
        assert_eq!(result.is_err(), fails);
        assert_eq!(
            mlx_cache_memory_bytes().unwrap(),
            0,
            "the cache must be empty after a loop that owned its buffers; step failed: {fails}"
        );
    }
    println!("{CHILD_DONE}");
}

#[test]
fn an_update_inside_a_block_writes_the_buffer_in_place() {
    run_child(
        "tts_memory_tests",
        "an_update_inside_a_block_writes_the_buffer_in_place_child",
    );
}

const WIDE_HEAD_DIM: i32 = 8192;
const WIDE_BUFFER_BYTES: u64 = (KV_HEADS * BLOCK * WIDE_HEAD_DIM * 4) as u64;

/// How far the peak of live bytes rises over one single-row update of `slot`.
/// With `hold_old`, a second handle of the old key buffer lives across the
/// evaluation.
fn peak_of_an_update(slot: &mut KvSlot, hold_old: bool) -> u64 {
    let row = || {
        let values = vec![1.0; (KV_HEADS * WIDE_HEAD_DIM) as usize];
        Array::from_f32_slice(&values, &[1, KV_HEADS, 1, WIDE_HEAD_DIM]).unwrap()
    };
    let (new_k, new_v) = (row(), row());
    let held = slot
        .k
        .as_ref()
        .filter(|_| hold_old)
        .map(|old| old.try_clone().unwrap());
    let bracket = PeakBracket::open();
    let (k, v) = slot.update_and_fetch(&new_k, &new_v, Device::Cpu).unwrap();
    k.eval().unwrap();
    v.eval().unwrap();
    let reading = bracket.close();
    drop(held);
    assert!(reading.measurable(), "{reading:?}");
    reading.headroom_bytes()
}

/// The peak of live bytes over one update shows a copy of the buffer. A
/// second handle of the old buffer forces one, and that is the control.
/// Without it an update inside a block must allocate no second buffer.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn an_update_inside_a_block_writes_the_buffer_in_place_child() {
    if !started_by_parent() {
        return;
    }
    let mut slot = KvSlot::new(BLOCK);
    let growth = peak_of_an_update(&mut slot, false);
    let in_place = peak_of_an_update(&mut slot, false);
    let copied = peak_of_an_update(&mut slot, true);

    assert!(
        growth >= 2 * WIDE_BUFFER_BYTES,
        "the first update must allocate both buffers of {WIDE_BUFFER_BYTES} B; the peak rose by {growth} B"
    );
    assert!(
        copied >= WIDE_BUFFER_BYTES,
        "the control must show a copy of a {WIDE_BUFFER_BYTES} B buffer; the peak rose by {copied} B"
    );
    assert!(
        in_place < WIDE_BUFFER_BYTES / 2,
        "an update inside a block allocated {in_place} B; a buffer is {WIDE_BUFFER_BYTES} B"
    );
    println!("{CHILD_DONE}");
}

#[test]
fn the_buffer_cache_is_released_between_the_passes_of_a_long_decode() {
    run_child(
        "tts_memory_tests",
        "the_buffer_cache_is_released_between_the_passes_of_a_long_decode_child",
    );
}

#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn the_buffer_cache_is_released_between_the_passes_of_a_long_decode_child() {
    if !started_by_parent() {
        return;
    }
    let frames = 2 * DECODE_CHUNK_FRAMES + 1;
    let codes = Array::from_i32_slice(&vec![0; 16 * frames], &[1, 16, frames as i32]).unwrap();
    let mut cache_at_entry = Vec::new();

    decode_in_chunks(&codes, Device::Cpu, &AtomicBool::new(false), |window| {
        cache_at_entry.push(mlx_cache_memory_bytes().unwrap());
        let width = window.shape()[2] as usize;
        let scratch = Array::from_f32_slice(
            &vec![1.0; PASS_SCRATCH_ELEMENTS],
            &[PASS_SCRATCH_ELEMENTS as i32],
        )
        .unwrap();
        add(&scratch, &scratch, Device::Cpu)
            .unwrap()
            .eval()
            .unwrap();
        Ok(vec![0.0; width])
    })
    .unwrap();

    assert_eq!(cache_at_entry.len(), 3);
    assert!(
        cache_at_entry.iter().all(|&bytes| bytes < MIB),
        "every pass must start with an empty cache; they started with {cache_at_entry:?} B"
    );
    assert_eq!(mlx_cache_memory_bytes().unwrap(), 0);

    let mut passes = 0;
    let failed = decode_in_chunks(&codes, Device::Cpu, &AtomicBool::new(false), |window| {
        passes += 1;
        let scratch = Array::from_f32_slice(
            &vec![1.0; PASS_SCRATCH_ELEMENTS],
            &[PASS_SCRATCH_ELEMENTS as i32],
        )
        .unwrap();
        add(&scratch, &scratch, Device::Cpu)
            .unwrap()
            .eval()
            .unwrap();
        if passes == 2 {
            return Err(TtsError::Mlx("pass failed".into()));
        }
        Ok(vec![0.0; window.shape()[2] as usize])
    });
    assert!(failed.is_err());
    assert_eq!(passes, 2);
    assert_eq!(
        mlx_cache_memory_bytes().unwrap(),
        0,
        "the cache must be empty after a pass failed"
    );

    let scratch = Array::from_f32_slice(
        &vec![1.0; PASS_SCRATCH_ELEMENTS],
        &[PASS_SCRATCH_ELEMENTS as i32],
    )
    .unwrap();
    add(&scratch, &scratch, Device::Cpu)
        .unwrap()
        .eval()
        .unwrap();
    drop(scratch);
    assert!(
        mlx_cache_memory_bytes().unwrap() >= MIB,
        "no power: the buffers of the scratch did not reach the cache"
    );
    let stopped = decode_in_chunks(&codes, Device::Cpu, &AtomicBool::new(true), |_| {
        panic!("a decode whose request is gone ran a pass")
    });
    assert!(matches!(stopped, Err(TtsError::Cancelled)), "{stopped:?}");
    assert_eq!(
        mlx_cache_memory_bytes().unwrap(),
        0,
        "the cache must be empty after a stop ended the decode"
    );
    println!("{CHILD_DONE}");
}
