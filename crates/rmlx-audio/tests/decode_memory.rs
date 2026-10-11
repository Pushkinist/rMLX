//! The memory a decode to 16 kHz mono takes beside the file's bytes.
//!
//! The transcription callers hold the file's bytes, decode them and resample
//! to 16 kHz. Beside the bytes, that work may hold no more than its 16 kHz
//! output and a fixed slack: no copy of the bytes, no buffer of all the
//! channels, no buffer of all the samples at the file's own rate, and no
//! growth that holds an old and a new buffer at once.
//!
//! A hostile file may refuse, but within the same bound: a sample rate or a
//! packet length that claims more audio than the bytes hold must not turn
//! into memory.
//!
//! The tests of this binary measure one at a time, under `MEASURE`, so nothing
//! else allocates while one counts. The global allocator counts every Rust
//! allocation of the process, and a reallocation as both blocks at once,
//! whether or not the allocator moves it. It refuses any allocation that
//! would put more than 1 GiB live, so a decode that runs away fails here
//! instead of taking the machine's memory, and it keeps the largest size
//! asked for, granted or not.

#![allow(unsafe_code, reason = "a global allocator implements an unsafe trait")]
#![allow(
    clippy::expect_used,
    clippy::print_stderr,
    clippy::indexing_slicing,
    reason = "test assertions, the skip notice of a case that needs ffmpeg, and the \
              header fields the test rewrites at their offsets"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use rmlx_audio::wav::{WavDecoder, WavEncoder, WavError, LOST_CAP_SECS, MAX_AUDIO_SECS};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ASKED: AtomicUsize = AtomicUsize::new(0);
static MEASURE: Mutex<()> = Mutex::new(());

const LIVE_CAP: usize = 1 << 30;

/// Count `bytes` more live, or refuse them when that passes `LIVE_CAP`.
fn grow(bytes: usize) -> bool {
    ASKED.fetch_max(bytes, Ordering::SeqCst);
    let live = LIVE
        .fetch_add(bytes, Ordering::SeqCst)
        .saturating_add(bytes);
    if live > LIVE_CAP {
        LIVE.fetch_sub(bytes, Ordering::SeqCst);
        return false;
    }
    PEAK.fetch_max(live, Ordering::SeqCst);
    true
}

// SAFETY: every call goes to `System` unchanged, or returns null, which the
// trait allows for an allocation that fails; the counts change nothing else.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !grow(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !grow(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !grow(new_size) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract, passed on.
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        moved
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// What a decode may hold beside its bytes and its output: the reader's
/// buffer and one packet's samples.
const SLACK_BYTES: usize = 1 << 20;

/// The 16 kHz mono samples of `bytes`, as the transcription callers get them.
fn decode_16k(bytes: &[u8]) -> Vec<f32> {
    WavDecoder::decode_resampled(bytes, 16_000, MAX_AUDIO_SECS, &|| false)
        .expect("decode")
        .samples
}

/// The most bytes live at once during `decode_16k(bytes)`, above what was
/// live before it, and the length of its output.
fn peak_of_decode(bytes: &[u8]) -> (usize, usize) {
    let before = LIVE.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let samples = decode_16k(bytes);
    let peak = PEAK.load(Ordering::SeqCst) - before;
    (peak, samples.len())
}

/// A decode of `bytes` to 16 kHz under a cap of `max_secs`: its result, the
/// most bytes live at once above what was live before it, and the largest
/// allocation it asked for.
fn measure_refusal(bytes: &[u8], max_secs: u32) -> (Result<usize, WavError>, usize, usize) {
    let before = LIVE.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    ASKED.store(0, Ordering::SeqCst);
    let got = WavDecoder::decode_resampled(bytes, 16_000, max_secs, &|| false)
        .map(|audio| audio.samples.len());
    let peak = PEAK.load(Ordering::SeqCst) - before;
    (got, peak, ASKED.load(Ordering::SeqCst))
}

/// A PCM-16 WAV of `frames` frames: a different tone on each channel.
#[allow(
    clippy::cast_precision_loss,
    reason = "sample indices of a one-minute clip are exact in f32"
)]
fn wav(rate: u32, channels: u16, frames: u32) -> Vec<u8> {
    let samples: Vec<f32> = (0..frames)
        .flat_map(|i| {
            (0..channels).map(move |c| {
                let hz = 440.0 * f32::from(c + 1);
                (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin() * 0.5
            })
        })
        .collect();
    WavEncoder::encode(&samples, rate, channels).expect("encode")
}

/// `wav` with the RIFF and data lengths a writer to a pipe leaves: a
/// placeholder, not a length.
fn piped(mut wav: Vec<u8>, riff_len: u32, data_len: u32) -> Vec<u8> {
    wav.splice(4..8, riff_len.to_le_bytes());
    wav.splice(40..44, data_len.to_le_bytes());
    wav
}

/// ADTS (AAC LC, 16 kHz, mono) of `frames` silent frames: a stream that
/// declares no length.
fn adts(frames: usize) -> Vec<u8> {
    const SILENT: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    SILENT.repeat(frames)
}

/// An MP3 of 60 s with a LAME tag, made by `ffmpeg`; `None` when `ffmpeg`
/// cannot run.
fn ffmpeg_mp3(dir: &std::path::Path, rate: u32, channels: u16) -> Option<Vec<u8>> {
    let path = dir.join(format!("tone_{rate}_{channels}.mp3"));
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
        ])
        .args([
            "-i",
            &format!("sine=frequency=440:sample_rate={rate}:duration=60"),
        ])
        .args(["-ac", &channels.to_string()])
        .args(["-c:a", "libmp3lame", "-b:a", "128k"])
        .arg(&path)
        .status()
        .ok()?;
    status
        .success()
        .then(|| std::fs::read(&path).ok())
        .flatten()
}

#[test]
fn a_decode_to_16k_holds_no_more_than_its_output_beside_the_bytes() {
    let _measure = MEASURE.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let cases = [
        ("16 kHz mono WAV", wav(16_000, 1, 16_000 * 60)),
        ("48 kHz stereo WAV", wav(48_000, 2, 48_000 * 60)),
        ("8 kHz mono WAV", wav(8_000, 1, 8_000 * 60)),
        // 44_100 * 60 + 4 frames: the output runs one sample past its final
        // length before the end of the stream.
        ("44.1 kHz mono WAV", wav(44_100, 1, 44_100 * 60 + 4)),
        (
            "48 kHz stereo WAV piped by ffmpeg",
            piped(wav(48_000, 2, 48_000 * 60), 0xFFFF_FFFF, 0xFFFF_FFFF),
        ),
        (
            "48 kHz stereo WAV piped by arecord",
            piped(wav(48_000, 2, 48_000 * 60), 0x8000_0024, 0x8000_0000),
        ),
        ("16 kHz mono ADTS", adts(938)),
    ];
    let mp3s = [
        ("48 kHz stereo MP3", 48_000, 2),
        ("16 kHz mono MP3", 16_000, 1),
    ]
    .map(|(name, rate, channels)| ffmpeg_mp3(dir.path(), rate, channels).map(|mp3| (name, mp3)));
    if mp3s.iter().any(Option::is_none) {
        eprintln!(
            "SKIP a_decode_to_16k_holds_no_more_than_its_output_beside_the_bytes: \
             ffmpeg unavailable, the MP3 cases are not run"
        );
    }

    let mut over = Vec::new();
    for (name, bytes) in cases.iter().chain(mp3s.iter().flatten()) {
        let (peak, samples) = peak_of_decode(bytes);
        let output = samples * size_of::<f32>();
        let allowed = output + SLACK_BYTES;
        eprintln!(
            "{name}: file {} B, output {output} B, peak beside the bytes {peak} B",
            bytes.len()
        );
        if peak > allowed {
            over.push(format!("{name}: {peak} B > {allowed} B"));
        }
    }
    assert!(over.is_empty(), "over the output plus slack: {over:?}");
}

/// A 4 s 16 kHz mono AAC M4A made by `ffmpeg`; `None` when `ffmpeg` cannot
/// run.
fn ffmpeg_m4a(dir: &std::path::Path) -> Option<Vec<u8>> {
    let path = dir.join("tone.m4a");
    let status = std::process::Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi"])
        .args(["-i", "sine=frequency=440:sample_rate=16000:duration=4"])
        .args(["-c:a", "aac", "-b:a", "96k"])
        .arg(&path)
        .status()
        .ok()?;
    status
        .success()
        .then(|| std::fs::read(&path).ok())
        .flatten()
}

/// The offset of the first box tagged `tag`.
fn find_box(m4a: &[u8], tag: [u8; 4]) -> usize {
    m4a.windows(4)
        .position(|w| w == tag)
        .expect("the box is in the file")
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn write_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
}

/// `m4a` with the integer part of the `mp4a` sample rate set to `hz`.
fn with_rate(mut m4a: Vec<u8>, hz: u16) -> Vec<u8> {
    let at = find_box(&m4a, *b"mp4a") + 28;
    m4a[at..at + 2].copy_from_slice(&hz.to_be_bytes());
    m4a
}

/// `m4a` with its second AAC packet overwritten with zeros, which the AAC
/// decoder refuses as invalid data (0xFF reads as an end element and
/// decodes), every packet `delta` ticks long (`stts`), and `timescale` ticks
/// a second (`mdhd`).
fn damaged(mut m4a: Vec<u8>, delta: u32, timescale: u32) -> Vec<u8> {
    let stsz = find_box(&m4a, *b"stsz");
    let first_size = read_u32(&m4a, stsz + 16) as usize;
    let second_size = read_u32(&m4a, stsz + 20) as usize;
    let chunk = read_u32(&m4a, find_box(&m4a, *b"stco") + 12) as usize;
    let second = chunk + first_size;
    m4a[second..second + second_size].fill(0);
    let stts = find_box(&m4a, *b"stts");
    write_u32(&mut m4a, stts + 16, delta);
    let mdhd = find_box(&m4a, *b"mdhd");
    write_u32(&mut m4a, mdhd + 16, timescale);
    m4a
}

/// An 8 kHz 8-bit mono WAV of 4096 samples whose header declares a data
/// chunk of `0x7FFE_0000` bytes, under the placeholder bound.
fn wav_declaring_2_gib() -> Vec<u8> {
    let mut wav = WavEncoder::encode(&[0.0; 2048], 8_000, 1).expect("encode");
    // PCM-16 to 8-bit: rewrite the format fields, keep the 4096 data bytes.
    wav[32..34].copy_from_slice(&1u16.to_le_bytes());
    wav[34..36].copy_from_slice(&8u16.to_le_bytes());
    wav[28..32].copy_from_slice(&8_000u32.to_le_bytes());
    wav[4..8].copy_from_slice(&(0x7FFE_0000u32 + 36).to_le_bytes());
    wav[40..44].copy_from_slice(&0x7FFE_0000u32.to_le_bytes());
    wav
}

/// Audio the dropped packets may stand in for at 16 kHz in f32: the most any
/// stream may lose.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "LOST_CAP_SECS is a few seconds"
)]
fn lost_cap_bytes() -> usize {
    (LOST_CAP_SECS * 16_000.0) as usize * size_of::<f32>()
}

#[test]
fn a_hostile_header_is_refused_without_its_claimed_memory() {
    let _measure = MEASURE.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    // `allowed` is per case: the most the bytes can decode to at 16 kHz, the
    // silence of LOST_CAP_SECS and the slack.
    let mut cases: Vec<(&str, Vec<u8>, usize)> = vec![(
        "WAV declaring 2 GiB at 8 kHz, 8-bit",
        wav_declaring_2_gib(),
        // One frame a byte, two outputs a frame, four bytes an output.
        4096 * 2 * size_of::<f32>(),
    )];
    match ffmpeg_m4a(dir.path()) {
        Some(m4a) => {
            // An AAC frame of 1024 samples is at least 4 bytes.
            let aac_bound = m4a.len() * 256 * size_of::<f32>();
            cases.push(("M4A at 0 Hz", with_rate(m4a.clone(), 0), aac_bound));
            cases.push(("M4A at 1 Hz", with_rate(m4a.clone(), 1), aac_bound));
            cases.push((
                "M4A with a damaged packet of 2^28 ticks",
                damaged(m4a.clone(), 1 << 28, 16_000),
                aac_bound,
            ));
            cases.push((
                "M4A with a damaged packet and 1 tick a second",
                damaged(m4a, 1024, 1),
                aac_bound,
            ));
        }
        None => eprintln!(
            "SKIP a_hostile_header_is_refused_without_its_claimed_memory: \
             ffmpeg unavailable, the M4A cases are not run"
        ),
    }

    let mut wrong = Vec::new();
    for (name, bytes, bound) in &cases {
        let (got, peak, asked) = measure_refusal(bytes, MAX_AUDIO_SECS);
        let allowed = bound + lost_cap_bytes() + SLACK_BYTES;
        eprintln!(
            "{name}: file {} B, {got:?}, peak {peak} B, largest ask {asked} B",
            bytes.len()
        );
        if got.is_ok() || peak > allowed || asked > allowed {
            wrong.push(format!(
                "{name}: {got:?}, peak {peak} B, ask {asked} B, allowed {allowed} B"
            ));
        }
    }
    assert!(wrong.is_empty(), "not refused within the bound: {wrong:?}");
}

/// ADTS (AAC LC, 16 kHz, mono) of `frames` silent frames of 1024 samples:
/// a stream that declares no length.
fn silent_adts(frames: usize) -> Vec<u8> {
    const SILENT: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    SILENT.repeat(frames)
}

/// Under a 60 s cap, a WAV that declares 120 s is refused before a sample is
/// decoded, and a 120 s stream that declares no length is refused when it
/// reaches the cap: its output never holds more than 60 s.
#[test]
fn the_duration_cap_refuses_before_the_output_passes_it() {
    let _measure = MEASURE.lock().unwrap_or_else(PoisonError::into_inner);
    let cap_bytes = 60 * 16_000 * size_of::<f32>();
    let cases = [
        (
            "WAV declaring 120 s",
            wav(16_000, 1, 16_000 * 120),
            "declares 120.000 s",
            0,
        ),
        (
            "ADTS of 120 s",
            silent_adts(1875),
            "holds more than the 60 s",
            cap_bytes,
        ),
    ];
    let mut wrong = Vec::new();
    for (name, bytes, says, output) in &cases {
        let (got, peak, asked) = measure_refusal(bytes, 60);
        let allowed = output + SLACK_BYTES;
        eprintln!("{name}: {got:?}, peak {peak} B, largest ask {asked} B, allowed {allowed} B");
        let refused = got.as_ref().is_err_and(|e| e.to_string().contains(says));
        if !refused || peak > allowed {
            wrong.push(format!(
                "{name}: {got:?}, peak {peak} B, allowed {allowed} B"
            ));
        }
    }
    assert!(wrong.is_empty(), "not refused at the cap: {wrong:?}");
}
