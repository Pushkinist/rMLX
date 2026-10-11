//! Tests for WAV encode/decode round-trip, and for the decode of streams that
//! end early or hold packets that do not decode.

use std::fmt::Write as _;

use super::{StreamEnd, WavDecoder, WavEncoder, WavError};

/// Encode a short 440 Hz sine wave and decode it back; verify round-trip.
#[test]
fn round_trip_440hz() {
    let sample_rate: u32 = 16_000;
    let freq = 440.0_f32;
    let duration_secs = 0.1_f32;
    let n = (sample_rate as f32 * duration_secs) as usize;
    let original: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();

    let wav_bytes = WavEncoder::encode(&original, sample_rate, 1).unwrap();
    // Check WAV magic.
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");

    let decoded = WavDecoder::decode(&wav_bytes).unwrap();
    assert_eq!(decoded.sample_rate, sample_rate);
    assert_eq!(decoded.samples.len(), original.len());
    assert_eq!(decoded.end, StreamEnd::EndOfStream);
    assert!(decoded.declared_secs.is_some_and(|d| close(d, 0.1)));
    assert!(close(decoded.reached_secs, 0.1));
    assert_eq!(decoded.loss.dropped_packets, 0);

    // Quantisation noise should be < 1/32768 ≈ 3e-5.
    for (orig, dec) in original.iter().zip(decoded.samples.iter()) {
        let err = (orig - dec).abs();
        assert!(err < 1e-4, "sample error {err} too large");
    }
}

/// Silence encodes and decodes to zero.
#[test]
fn silence_round_trip() {
    let silence = vec![0.0_f32; 160];
    let bytes = WavEncoder::encode(&silence, 16_000, 1).unwrap();
    let decoded = WavDecoder::decode(&bytes).unwrap();
    assert_eq!(decoded.sample_rate, 16_000);
    for s in &decoded.samples {
        assert!(s.abs() < 1e-6, "non-zero silence sample: {s}");
    }
}

/// WAV header is exactly 44 bytes for PCM.
#[test]
fn header_size() {
    let samples = vec![0.0_f32; 100];
    let bytes = WavEncoder::encode(&samples, 16_000, 1).unwrap();
    assert_eq!(bytes.len(), 44 + 100 * 2);
}

/// A WAV with no samples is an error, not an empty success.
#[test]
fn an_empty_wav_is_refused() {
    let bytes = WavEncoder::encode(&[], 16_000, 1).unwrap();
    let got = WavDecoder::decode(&bytes);
    assert!(matches!(got, Err(WavError::Empty)), "{got:?}");
}

// ── Streams that end early, and packets that do not decode ───────────────────

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() < 1e-9
}

/// A 16 kHz PCM-16 WAV of `secs` seconds of a 440 Hz tone.
fn tone_wav(secs: usize) -> Vec<u8> {
    let samples: Vec<f32> = (0..16_000 * secs)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.5)
        .collect();
    WavEncoder::encode(&samples, 16_000, 1).unwrap()
}

/// Samples per MPEG-1 Layer III frame.
const MP3_FRAME_SAMPLES: usize = 1152;

/// Header byte 2 (bitrate index, sample-rate index) and frame length of the
/// MPEG-1 Layer III mono frames below.
const MP3_32K_32KBPS: (u8, usize) = (0x18, 144);
const MP3_32K_320KBPS: (u8, usize) = (0xE8, 1440);
const MP3_48K_32KBPS: (u8, usize) = (0x14, 96);

/// One MPEG-1 Layer III mono frame with no CRC whose side information and main
/// data are zero: it decodes to 1152 samples of silence.
fn mp3_frame((header_byte_2, len): (u8, usize)) -> Vec<u8> {
    let mut frame = vec![0_u8; len];
    frame.splice(..4, [0xFF, 0xFB, header_byte_2, 0xC0]);
    frame
}

fn mp3_frames(kind: (u8, usize), count: usize) -> Vec<u8> {
    (0..count).flat_map(|_| mp3_frame(kind)).collect()
}

/// An "Info" frame at 32 kHz that declares `mpeg_frames` frames, with a LAME
/// extension that declares the encoder delay and padding in samples.
fn lame_info_frame(mpeg_frames: u32, delay: u32, padding: u32) -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend_from_slice(b"Info");
    tag.extend_from_slice(&1_u32.to_be_bytes());
    tag.extend_from_slice(&mpeg_frames.to_be_bytes());
    tag.extend_from_slice(b"Lavf58.76");
    tag.extend_from_slice(&[0; 12]);
    let trim = ((delay - 529) << 12) | (padding + 529);
    tag.extend(trim.to_be_bytes().into_iter().skip(1));
    let side_info_end = 4 + 17;
    let mut frame = mp3_frame(MP3_32K_32KBPS);
    frame.splice(side_info_end..side_info_end + tag.len(), tag);
    frame
}

const LAME_DELAY: u32 = 1105;
const LAME_PADDING: u32 = 263;

/// Samples per AAC frame.
const AAC_FRAME_SAMPLES: usize = 1024;

/// ADTS header (AAC LC, 16 kHz, mono, no CRC), then one single-channel element
/// with no scale-factor bands and END: 1024 samples of silence.
const ADTS_SILENT: [u8; 11] = [
    0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
];
/// The same frame with the predictor bit set, which AAC LC does not allow.
const ADTS_BAD: [u8; 12] = [
    0xFF, 0xF1, 0x60, 0x40, 0x01, 0x9F, 0xFC, 0x00, 0xC8, 0x00, 0x40, 0x00,
];

fn adts(parts: &[(&[u8], usize)]) -> Vec<u8> {
    parts
        .iter()
        .flat_map(|&(frame, count)| frame.repeat(count))
        .collect()
}

#[test]
fn a_wav_with_no_declared_length_decodes_to_the_end_of_its_bytes() {
    let mut wav = tone_wav(2);
    wav.splice(4..8, [0xFF; 4]);
    wav.splice(40..44, [0xFF; 4]);
    let decoded = WavDecoder::decode(&wav).unwrap();
    assert_eq!(decoded.samples.len(), 2 * 16_000);
    assert_eq!(decoded.declared_secs, None);
    assert_eq!(decoded.end, StreamEnd::EndOfBytes);
}

#[test]
fn an_mp3_with_a_lame_tag_trims_the_delay_and_padding_it_declares() {
    let mut mp3 = lame_info_frame(40, LAME_DELAY, LAME_PADDING);
    mp3.extend(mp3_frames(MP3_32K_32KBPS, 40));
    let declared = 40 * MP3_FRAME_SAMPLES - (LAME_DELAY + LAME_PADDING) as usize;
    let decoded = WavDecoder::decode(&mp3).unwrap();
    assert_eq!(decoded.samples.len(), declared);
    assert!(decoded
        .declared_secs
        .is_some_and(|d| close(d, declared as f64 / 32_000.0)));
    assert_eq!(decoded.end, StreamEnd::EndOfStream);
    assert_eq!(decoded.loss.total_secs(), 0.0);
}

#[test]
fn an_mp3_far_short_of_its_lame_length_is_refused_with_the_shortfall() {
    let mp3 = lame_mp3_cut_after(10);
    let reached = (10 * MP3_FRAME_SAMPLES) as f64 - f64::from(LAME_DELAY);
    let declared = (40 * MP3_FRAME_SAMPLES) as f64 - f64::from(LAME_DELAY + LAME_PADDING);
    match WavDecoder::decode(&mp3) {
        Err(WavError::LostAudio {
            loss,
            limit_secs,
            reached_secs,
            declared_secs,
        }) => {
            assert!(close(limit_secs, 0.2), "{limit_secs}");
            assert!(close(reached_secs, reached / 32_000.0), "{reached_secs}");
            assert!(declared_secs.is_some_and(|d| close(d, declared / 32_000.0)));
            assert!(close(loss.short_secs, (declared - reached) / 32_000.0));
            assert_eq!(loss.dropped_packets, 0);
        }
        other => panic!("10 of the 40 declared frames: {}", summary(&other)),
    }
}

#[test]
fn an_mp3_with_no_length_tag_decodes_every_frame() {
    // Large frames first: a length guessed from the opening bitrate is short.
    let mut mp3 = mp3_frames(MP3_32K_320KBPS, 20);
    mp3.extend(mp3_frames(MP3_32K_32KBPS, 100));
    let decoded = WavDecoder::decode(&mp3).unwrap();
    assert_eq!(decoded.samples.len(), 120 * MP3_FRAME_SAMPLES);
    assert_eq!(decoded.declared_secs, None);
}

#[test]
fn an_mp3_whose_guessed_length_is_long_is_not_refused() {
    // Small frames first: a length guessed from the opening bitrate is long.
    let mut mp3 = mp3_frames(MP3_32K_32KBPS, 20);
    mp3.extend(mp3_frames(MP3_32K_320KBPS, 100));
    let decoded = WavDecoder::decode(&mp3).unwrap();
    assert_eq!(decoded.samples.len(), 120 * MP3_FRAME_SAMPLES);
    assert_eq!(decoded.declared_secs, None);
}

#[test]
fn an_mp3_whose_sample_rate_changes_midway_is_refused() {
    let mut mp3 = mp3_frames(MP3_32K_32KBPS, 40);
    mp3.extend(mp3_frames(MP3_48K_32KBPS, 40));
    match WavDecoder::decode(&mp3) {
        Err(WavError::LostAudio { loss, .. }) => {
            assert_eq!(loss.dropped_packets, 40);
            assert!(close(loss.dropped_secs, 1.44), "{loss:?}");
            assert!(loss.first_drop_at_secs.is_some_and(|t| close(t, 1.44)));
        }
        other => panic!("the 48 kHz half does not decode: {}", summary(&other)),
    }
}

#[test]
fn aac_frames_that_do_not_decode_between_good_ones_are_dropped_and_counted() {
    let aac = adts(&[
        (&ADTS_SILENT, 20),
        (&ADTS_BAD, 1),
        (&ADTS_SILENT, 5),
        (&ADTS_BAD, 1),
        (&ADTS_SILENT, 15),
    ]);
    let decoded = WavDecoder::decode(&aac).unwrap();
    // The dropped frames are silence in place, so the samples keep the timeline.
    assert_eq!(decoded.samples.len(), 42 * AAC_FRAME_SAMPLES);
    let loss = &decoded.loss;
    assert_eq!(loss.dropped_packets, 2);
    assert!(close(loss.dropped_secs, 0.128));
    assert!(loss.first_drop_at_secs.is_some_and(|t| close(t, 1.28)));
    assert!(loss.last_drop_at_secs.is_some_and(|t| close(t, 1.664)));
    assert!(loss
        .first_error
        .as_deref()
        .is_some_and(|e| e.contains("predictor")));
}

#[test]
fn aac_that_stops_decoding_midway_is_refused() {
    let aac = adts(&[
        (&ADTS_SILENT, 20),
        (&ADTS_BAD, 1),
        (&ADTS_SILENT, 5),
        (&ADTS_BAD, 20),
    ]);
    match WavDecoder::decode(&aac) {
        Err(WavError::LostAudio { loss, .. }) => {
            assert_eq!(loss.dropped_packets, 21);
            assert!(close(loss.dropped_secs, 1.344), "{loss:?}");
            assert!(loss.first_drop_at_secs.is_some_and(|t| close(t, 1.28)));
            assert!(loss.last_drop_at_secs.is_some_and(|t| close(t, 2.88)));
        }
        other => panic!("21 frames do not decode: {}", summary(&other)),
    }
}

#[test]
fn an_adts_stream_with_a_trailing_tag_is_not_refused() {
    // The trailing bytes make a length guessed from the file size long.
    let mut aac = adts(&[(&ADTS_SILENT, 40)]);
    aac.extend_from_slice(b"TAG");
    aac.extend_from_slice(&[0; 125]);
    let decoded = WavDecoder::decode(&aac).unwrap();
    assert_eq!(decoded.samples.len(), 40 * AAC_FRAME_SAMPLES);
    assert_eq!(decoded.declared_secs, None);
}

/// A 2 s AAC tone in an MP4 container written with `movflags`, encoded by
/// `ffmpeg`; `None` when `ffmpeg` cannot run.
fn ffmpeg_m4a(dir: &std::path::Path, movflags: &str) -> Option<Vec<u8>> {
    let out = dir.join("tone.m4a");
    let status = std::process::Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi"])
        .args(["-i", "sine=frequency=440:sample_rate=16000:duration=2"])
        .args(["-c:a", "aac", "-movflags", movflags])
        .arg(&out)
        .status()
        .ok()?;
    status.success().then(|| std::fs::read(&out).ok()).flatten()
}

#[test]
#[allow(
    clippy::print_stderr,
    reason = "the skip notice of a test that needs ffmpeg"
)]
fn an_m4a_decodes_and_a_cut_copy_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let Some(m4a) = ffmpeg_m4a(dir.path(), "+faststart") else {
        eprintln!("SKIP an_m4a_decodes_and_a_cut_copy_is_refused: ffmpeg unavailable");
        return;
    };
    let full = WavDecoder::decode(&m4a);
    assert!(
        full.as_ref()
            .is_ok_and(|d| d.samples.len() >= 2 * 16_000 && d.loss.total_secs() == 0.0),
        "{}",
        summary(&full)
    );
    let full = full.unwrap();
    assert_eq!(full.end, StreamEnd::EndOfStream);
    // The last AAC frame decodes whole; the track length ends inside it.
    assert!(
        full.declared_secs.is_some_and(|d| full.reached_secs >= d),
        "declared {:?}, reached {}",
        full.declared_secs,
        full.reached_secs
    );
    let cut = m4a.get(..m4a.len() * 3 / 4).unwrap();
    let got = WavDecoder::decode(cut);
    assert!(
        matches!(got, Err(WavError::ReadFailed { .. })),
        "an MP4 cut inside its media data: {got:?}"
    );
}

#[test]
#[allow(
    clippy::print_stderr,
    reason = "the skip notice of a test that needs ffmpeg"
)]
fn a_fragmented_m4a_declares_no_length() {
    let dir = tempfile::tempdir().unwrap();
    let Some(m4a) = ffmpeg_m4a(dir.path(), "frag_keyframe+empty_moov") else {
        eprintln!("SKIP a_fragmented_m4a_declares_no_length: ffmpeg unavailable");
        return;
    };
    let decoded = WavDecoder::decode(&m4a).unwrap();
    assert!(decoded.samples.len() >= 2 * 16_000, "{decoded:?}");
    assert_eq!(decoded.declared_secs, None);
}

// ── Lost audio: limits, placeholders, joined files ───────────────────────────

/// A decode result without its samples, for an assertion message.
pub(super) fn summary(got: &Result<super::DecodedAudio, WavError>) -> String {
    match got {
        Ok(d) => format!(
            "Ok({} samples, declared {:?}, reached {})",
            d.samples.len(),
            d.declared_secs,
            d.reached_secs
        ),
        Err(e) => format!("Err({e})"),
    }
}

/// A 2 s WAV whose RIFF and data lengths are `riff` and `data`, as a writer
/// that cannot seek back leaves them.
fn wav_with_lengths(riff: u32, data: u32) -> Vec<u8> {
    let mut wav = tone_wav(2);
    wav.splice(4..8, riff.to_le_bytes());
    wav.splice(40..44, data.to_le_bytes());
    wav
}

/// The RIFF and data lengths a writer leaves when it cannot seek back to patch
/// them, for 16-bit mono PCM.
const PLACEHOLDER_HEADERS: [(&str, u32, u32); 4] = [
    // libavformat `ff_start_tag` writes -1 for both.
    ("ffmpeg to a pipe", 0xFFFF_FFFF, 0xFFFF_FFFF),
    // sox 14.4.2 `wavwritehdr`: MS_UNSPEC 0x7ffff000, rounded down to the block.
    ("sox 14.4.2 to a pipe", 0x7FFF_F024, 0x7FFF_F000),
    // alsa-utils `arecord` to stdout: the WAVE size cap, 2147483648.
    ("arecord to stdout", 0x8000_0024, 0x8000_0000),
    // The lowest data length taken as a placeholder.
    ("placeholder bound", 0x7FFF_0024, 0x7FFF_0000),
];

#[test]
fn a_wav_whose_writer_left_a_placeholder_length_decodes_what_it_holds() {
    for (writer, riff, data) in PLACEHOLDER_HEADERS {
        let got = WavDecoder::decode(&wav_with_lengths(riff, data));
        assert!(
            got.as_ref()
                .is_ok_and(|d| d.samples.len() == 2 * 16_000 && d.declared_secs.is_none()),
            "{writer}: RIFF {riff:#x}, data {data:#x}: {}",
            summary(&got)
        );
    }
}

#[test]
fn a_placeholder_length_after_an_odd_sized_chunk_is_found() {
    // sox 14.4.2's lengths, with a 3-byte chunk and its pad byte before "data";
    // the RIFF length grows by the 12 bytes of that chunk.
    let mut wav = wav_with_lengths(0x7FFF_F030, 0x7FFF_F000);
    wav.splice(36..36, *b"junk\x03\x00\x00\x00abc\x00");
    let got = WavDecoder::decode(&wav);
    assert!(
        got.as_ref()
            .is_ok_and(|d| d.samples.len() == 2 * 16_000 && d.declared_secs.is_none()),
        "{}",
        summary(&got)
    );
}

/// The length is taken as declared: 67107 s, over the duration cap.
#[test]
fn a_wav_length_just_below_the_placeholder_bound_is_held_to() {
    let got = WavDecoder::decode(&wav_with_lengths(0x7FFE_FFFF + 36, 0x7FFE_FFFF));
    assert!(
        matches!(
            got,
            Err(WavError::TooLong {
                declared_secs: Some(_),
                ..
            })
        ),
        "{}",
        summary(&got)
    );
}

/// A 3 s 16 kHz PCM-16 tone WAV that holds `held_frames` of its 48000 frames.
fn wav_3s_cut_to(held_frames: usize) -> Vec<u8> {
    let mut wav = tone_wav(3);
    wav.truncate(44 + 2 * held_frames);
    wav
}

#[test]
fn a_wav_cut_short_of_its_header_length_by_more_than_the_limit_is_refused() {
    match WavDecoder::decode(&wav_3s_cut_to(16_000)) {
        Err(WavError::LostAudio {
            loss,
            limit_secs,
            reached_secs,
            declared_secs,
        }) => {
            assert!(close(limit_secs, 0.2), "{limit_secs}");
            assert!(close(loss.short_secs, 2.0), "{loss:?}");
            assert!(close(reached_secs, 1.0), "{reached_secs}");
            assert!(declared_secs.is_some_and(|d| close(d, 3.0)));
        }
        other => panic!("1 s of a 3 s WAV: {}", summary(&other)),
    }
}

#[test]
fn the_loss_limit_holds_on_both_sides_for_a_cut_wav() {
    // A 3 s stream may lose 0.2 s. 0.19 s short: decoded, the shortfall counted.
    let got = WavDecoder::decode(&wav_3s_cut_to(44_960));
    assert!(
        got.as_ref()
            .is_ok_and(|d| d.samples.len() == 44_960 && close(d.loss.short_secs, 0.19)),
        "{}",
        summary(&got)
    );
    // 0.21 s short: refused.
    let got = WavDecoder::decode(&wav_3s_cut_to(44_640));
    assert!(
        matches!(got, Err(WavError::LostAudio { .. })),
        "{}",
        summary(&got)
    );
}

#[test]
fn a_cut_wav_reaches_the_frames_it_holds() {
    let wav = tone_wav(2);
    let cut = wav.get(..44 + 2 * 30_400 + 1).unwrap();
    let decoded = WavDecoder::decode(cut).unwrap();
    assert_eq!(decoded.samples.len(), 30_400);
    assert!(close(decoded.reached_secs, 1.9), "{}", decoded.reached_secs);
    assert!(decoded.declared_secs.is_some_and(|d| close(d, 2.0)));
    assert!(close(decoded.loss.short_secs, 0.1), "{:?}", decoded.loss);
}

#[test]
fn audio_lost_to_frames_that_do_not_decode_over_the_limit_is_refused() {
    let aac = adts(&[(&ADTS_SILENT, 100), (&ADTS_BAD, 400), (&ADTS_SILENT, 1)]);
    let got = WavDecoder::decode(&aac);
    assert!(got.is_err(), "25.6 s of 32.1 s lost: {}", summary(&got));
}

/// ADTS of `total` 64 ms frames, `bad` of them undecodable in the middle.
fn adts_with_bad_run(total: usize, bad: usize) -> Vec<u8> {
    let before = (total - bad) / 2;
    adts(&[
        (&ADTS_SILENT, before),
        (&ADTS_BAD, bad),
        (&ADTS_SILENT, total - bad - before),
    ])
}

fn is_lost_audio(got: &Result<super::DecodedAudio, WavError>) -> bool {
    matches!(got, Err(WavError::LostAudio { .. }))
}

#[test]
fn a_short_file_may_lose_0_2_s() {
    // 50 frames, 3.2 s: the 0.2 s floor applies.
    let got = WavDecoder::decode(&adts_with_bad_run(50, 3));
    assert!(got.is_ok(), "0.192 s of 3.2 s: {}", summary(&got));
    let got = WavDecoder::decode(&adts_with_bad_run(50, 4));
    assert!(is_lost_audio(&got), "0.256 s of 3.2 s: {}", summary(&got));
}

#[test]
fn a_5_s_clip_that_lost_0_96_s_is_refused() {
    let got = WavDecoder::decode(&adts_with_bad_run(78, 15));
    assert!(is_lost_audio(&got), "0.96 s of 4.99 s: {}", summary(&got));
}

#[test]
fn a_100_s_file_may_lose_1_percent() {
    // 1563 frames, 100.03 s: the limit is 1.0003 s.
    let got = WavDecoder::decode(&adts_with_bad_run(1563, 15));
    assert!(got.is_ok(), "0.96 s of 100 s: {}", summary(&got));
    let got = WavDecoder::decode(&adts_with_bad_run(1563, 16));
    assert!(is_lost_audio(&got), "1.024 s of 100 s: {}", summary(&got));
}

#[test]
fn a_10_min_file_may_lose_5_s() {
    // 9375 frames, 600 s: 1 % is 6 s, the 5 s cap applies.
    let got = WavDecoder::decode(&adts_with_bad_run(9375, 16));
    assert!(got.is_ok(), "1.024 s of 600 s: {}", summary(&got));
    let got = WavDecoder::decode(&adts_with_bad_run(9375, 78));
    assert!(got.is_ok(), "4.992 s of 600 s: {}", summary(&got));
    let got = WavDecoder::decode(&adts_with_bad_run(9375, 79));
    assert!(is_lost_audio(&got), "5.056 s of 600 s: {}", summary(&got));
}

#[test]
fn one_frame_that_does_not_decode_at_the_end_is_dropped_not_refused() {
    let aac = adts(&[(&ADTS_SILENT, 40), (&ADTS_BAD, 1)]);
    let got = WavDecoder::decode(&aac);
    assert!(
        got.as_ref()
            .is_ok_and(|d| d.samples.len() == 41 * AAC_FRAME_SAMPLES
                && close(d.reached_secs, 2.624)
                && d.loss.dropped_packets == 1),
        "{}",
        summary(&got)
    );
}

/// The frames a LAME-tagged MP3 of 40 declared frames holds when it is cut
/// after `held` frames.
fn lame_mp3_cut_after(held: usize) -> Vec<u8> {
    let mut mp3 = lame_info_frame(40, LAME_DELAY, LAME_PADDING);
    mp3.extend(mp3_frames(MP3_32K_32KBPS, held));
    mp3
}

#[test]
fn the_loss_limit_holds_on_both_sides_for_an_mp3_short_of_its_lame_length() {
    // The LAME tag declares 1.397 s, so the 0.2 s floor applies.
    // 5 frames short, less the 263 padding samples: 5497 samples, 0.172 s.
    let got = WavDecoder::decode(&lame_mp3_cut_after(35));
    assert!(got.is_ok(), "0.172 s short: {}", summary(&got));
    // 6 frames short: 6649 samples, 0.208 s.
    let got = WavDecoder::decode(&lame_mp3_cut_after(34));
    assert!(is_lost_audio(&got), "0.208 s short: {}", summary(&got));
}

#[test]
fn two_mp3_files_joined_decode_whole() {
    let mut mp3 = lame_mp3_cut_after(40);
    mp3.extend(lame_mp3_cut_after(40));
    let got = WavDecoder::decode(&mp3);
    // The second Info frame is no packet; only the first file's delay is cut.
    let whole = 80 * MP3_FRAME_SAMPLES - LAME_DELAY as usize;
    assert!(
        got.as_ref().is_ok_and(|d| d.samples.len() == whole),
        "{whole} samples expected: {}",
        summary(&got)
    );
}

#[test]
fn a_reader_that_fails_midway_is_refused() {
    // An ADTS header with the forbidden sample-rate index 15.
    let mut bad_header = ADTS_SILENT;
    bad_header[2] = 0x7C;
    let aac = adts(&[(&ADTS_SILENT, 20), (&bad_header, 1), (&ADTS_SILENT, 20)]);
    let got = WavDecoder::decode(&aac);
    assert!(
        matches!(got, Err(WavError::ReadFailed { .. })),
        "{}",
        summary(&got)
    );
}

#[test]
fn a_frame_shorter_than_its_content_is_dropped_as_damage() {
    // The length field says 9 bytes; the element needs 4 bytes of payload, so
    // the decoder reads past the packet end (an IoError).
    let short = [0xFF, 0xF1, 0x60, 0x40, 0x01, 0x3F, 0xFC, 0x00, 0xC8];
    let aac = adts(&[(&ADTS_SILENT, 20), (&short, 1), (&ADTS_SILENT, 20)]);
    let decoded = WavDecoder::decode(&aac).unwrap();
    assert_eq!(decoded.loss.dropped_packets, 1);
    assert!(
        decoded
            .loss
            .first_error
            .as_deref()
            .is_some_and(|e| e.contains("unexpected end of bitstream")),
        "{:?}",
        decoded.loss
    );
}

#[test]
fn a_stream_of_which_no_packet_decodes_is_a_decoder_failure() {
    let aac = adts(&[(&ADTS_BAD, 3)]);
    match WavDecoder::decode(&aac) {
        Err(WavError::PacketFailed { at_secs, error }) => {
            assert!(close(at_secs, 0.0), "{at_secs}");
            assert!(error.contains("predictor"), "{error}");
        }
        other => panic!("3 frames, none decodes: {}", summary(&other)),
    }
}

thread_local! {
    /// The `warn` events of the decoder on this thread while a test captures
    /// them, each as its message and fields in text.
    static CAPTURED_WARNS: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// A process-wide subscriber that hands each `warn` event of this module to
/// the capture of its own thread. It is global, not scoped: a scoped one
/// leaves the callsite interest to the other test threads, and one of them
/// can cache it as "never" while the test runs.
struct ThreadWarns;

struct FieldText<'a>(&'a mut String);

impl tracing::field::Visit for FieldText<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        write!(self.0, "{}={value:?} ", field.name()).ok();
    }
}

impl tracing::Subscriber for ThreadWarns {
    fn register_callsite(
        &self,
        meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        if self.enabled(meta) {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::WARN)
    }
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target().starts_with("rmlx_audio::wav") && *meta.level() == tracing::Level::WARN
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut text = String::new();
        event.record(&mut FieldText(&mut text));
        CAPTURED_WARNS.with(|captured| {
            if let Some(warns) = captured.borrow_mut().as_mut() {
                warns.push(text);
            }
        });
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The decode of `bytes` and the `warn` events it emitted.
fn decode_with_warns(bytes: &[u8]) -> (Result<super::DecodedAudio, WavError>, Vec<String>) {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| tracing::subscriber::set_global_default(ThreadWarns).unwrap());
    CAPTURED_WARNS.with(|captured| *captured.borrow_mut() = Some(Vec::new()));
    let got = WavDecoder::decode(bytes);
    let warns = CAPTURED_WARNS.with(|captured| captured.borrow_mut().take());
    (got, warns.unwrap_or_default())
}

#[test]
fn a_loss_under_the_limit_is_a_warn_with_its_numbers_as_fields() {
    let (got, warns) = decode_with_warns(&adts(&[(&ADTS_SILENT, 40), (&ADTS_BAD, 1)]));
    assert!(got.is_ok(), "{}", summary(&got));
    assert!(
        warns
            .iter()
            .any(|w| w.contains("audio lost under the limit")
                && w.contains("lost_secs=0.064 ")
                && w.contains("dropped_packets=1 ")
                && w.contains("short_secs=0.0 ")),
        "{warns:?}"
    );
}

#[test]
fn bytes_that_end_inside_a_packet_of_a_stream_with_no_length_are_a_warn() {
    let mut aac = adts(&[(&ADTS_SILENT, 40)]);
    aac.extend_from_slice(ADTS_SILENT.get(..8).unwrap());
    let (got, warns) = decode_with_warns(&aac);
    assert!(
        got.as_ref()
            .is_ok_and(|d| d.end == StreamEnd::EndOfBytes && d.declared_secs.is_none()),
        "{}",
        summary(&got)
    );
    assert!(
        warns.iter().any(
            |w| w.contains("the bytes end inside a packet") && w.contains("reached_secs=2.56 ")
        ),
        "{warns:?}"
    );
}

/// The ADTS frames of `aac`, split by the length field of each header.
fn adts_frames(aac: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut at = 0;
    while let Some(head) = aac.get(at..at + 7) {
        let len = (usize::from(head[3] & 3) << 11)
            | (usize::from(head[4]) << 3)
            | usize::from(head[5] >> 5);
        frames.push(aac.get(at..at + len).unwrap());
        at += len;
    }
    frames
}

#[test]
#[allow(
    clippy::print_stderr,
    reason = "the skip notice of a test that needs ffmpeg"
)]
fn a_dropped_frame_keeps_the_later_audio_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("tone.aac");
    let made = std::process::Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi"])
        .args(["-i", "sine=frequency=440:sample_rate=16000:duration=3"])
        .args(["-c:a", "aac", "-f", "adts"])
        .arg(&out)
        .status()
        .is_ok_and(|s| s.success());
    let Some(aac) = made.then(|| std::fs::read(&out).ok()).flatten() else {
        eprintln!("SKIP a_dropped_frame_keeps_the_later_audio_in_place: ffmpeg unavailable");
        return;
    };
    // Frame 20 with a length field 100 bytes short: the decoder reads past it.
    let frames = adts_frames(&aac);
    let mut damaged = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        if i == 20 {
            let len = frame.len().saturating_sub(100).max(8);
            let mut head = frame.get(..7).unwrap().to_vec();
            head[3] = (head[3] & !3) | ((len >> 11) & 3) as u8;
            head[4] = ((len >> 3) & 0xFF) as u8;
            head[5] = (head[5] & 0x1F) | (((len & 7) as u8) << 5);
            damaged.extend_from_slice(&head);
            damaged.extend_from_slice(frame.get(7..).unwrap());
        } else {
            damaged.extend_from_slice(frame);
        }
    }
    let clean = WavDecoder::decode(&aac).unwrap();
    let got = WavDecoder::decode(&damaged).unwrap();
    assert_eq!(got.loss.dropped_packets, 1, "{:?}", got.loss);
    assert_eq!(got.samples.len(), clean.samples.len());
    // Frame 21 overlaps the gap; from frame 22 on the audio is in place.
    let after = 22 * AAC_FRAME_SAMPLES;
    assert_eq!(got.samples.get(after..), clean.samples.get(after..));
}

#[test]
fn the_limit_of_a_cut_file_scales_with_its_declared_length() {
    // A 100 s WAV that holds 99.005 s: 0.995 s short, under the 1.0 s limit of
    // 100 s and over the 0.99 s of the 99.005 s it holds.
    let mut wav = WavEncoder::encode(&vec![0.0_f32; 1_600_000], 16_000, 1).unwrap();
    wav.truncate(44 + 2 * 1_584_080);
    let got = WavDecoder::decode(&wav);
    assert!(
        got.as_ref().is_ok_and(|d| close(d.loss.short_secs, 0.995)),
        "{}",
        summary(&got)
    );
}

/// `bytes` decoded to 16 kHz under a cap of `max_secs`, never cancelled.
fn capped(bytes: &[u8], max_secs: u32) -> Result<super::DecodedAudio, WavError> {
    WavDecoder::decode_resampled(bytes, 16_000, max_secs, &|| false)
}

#[test]
fn a_file_of_exactly_the_cap_decodes_and_one_frame_more_is_refused() {
    let at_cap = WavEncoder::encode(&vec![0.25; 16_000], 16_000, 1).unwrap();
    let got = capped(&at_cap, 1);
    assert!(
        got.as_ref().is_ok_and(|d| d.samples.len() == 16_000),
        "{}",
        summary(&got)
    );
    let over = WavEncoder::encode(&vec![0.25; 16_001], 16_000, 1).unwrap();
    let got = capped(&over, 1);
    assert!(
        got.as_ref().is_err_and(|e| e.to_string()
            == "the file declares 1.000 s of audio, over the 1 s the decoder takes"),
        "{}",
        summary(&got)
    );
}

#[test]
fn a_stream_with_no_length_is_refused_when_it_passes_the_cap() {
    // 31 frames of 1024 samples are 1.984 s; 32 are 2.048 s.
    let under = adts(&[(&ADTS_SILENT, 31)]);
    let got = capped(&under, 2);
    assert!(got.is_ok(), "{}", summary(&got));
    let over = adts(&[(&ADTS_SILENT, 32)]);
    let got = capped(&over, 2);
    assert!(
        got.as_ref()
            .is_err_and(|e| e.to_string()
                == "the stream holds more than the 2 s of audio the decoder takes"),
        "{}",
        summary(&got)
    );
}

#[test]
fn a_caller_gone_before_the_first_packet_stops_the_decode_there() {
    let aac = adts(&[(&ADTS_SILENT, 40)]);
    let got = WavDecoder::decode_resampled(&aac, 16_000, super::MAX_AUDIO_SECS, &|| true);
    assert!(
        got.as_ref()
            .is_err_and(|e| e.to_string() == "audio decode cancelled at 0.00 s"),
        "{}",
        summary(&got)
    );
}

#[test]
fn a_cancel_after_three_packets_stops_the_decode_at_the_third() {
    // Three AAC packets of 1024 samples at 16 kHz end at 0.192 s; the fourth
    // question comes before the fourth packet.
    let aac = adts(&[(&ADTS_SILENT, 40)]);
    let asked = std::cell::Cell::new(0_u32);
    let gone_after_three = || {
        asked.set(asked.get() + 1);
        asked.get() > 3
    };
    let got = WavDecoder::decode_resampled(&aac, 16_000, super::MAX_AUDIO_SECS, &gone_after_three);
    assert!(
        got.as_ref()
            .is_err_and(|e| e.to_string() == "audio decode cancelled at 0.19 s"),
        "{}",
        summary(&got)
    );
}

/// `frames` silent ADTS frames whose header names `header_byte_2`'s sample
/// rate: 0x6C is 8 kHz, 0x60 16 kHz, 0x4C 48 kHz (AAC LC, mono).
fn adts_at(header_byte_2: u8, frames: usize) -> Vec<u8> {
    let mut frame = ADTS_SILENT;
    frame[2] = header_byte_2;
    adts(&[(&frame, frames)])
}

/// The cap is counted at the file's own rate, not at the output rate.
#[test]
fn the_cap_counts_the_file_rate_for_8_khz_and_48_khz_streams() {
    // 1 s at 8 kHz is 8000 frames: 7 AAC packets (7168) fit, 8 (8192) do not.
    assert!(capped(&adts_at(0x6C, 7), 1).is_ok(), "7 packets at 8 kHz");
    assert!(
        capped(&adts_at(0x6C, 8), 1).is_err_and(|e| matches!(e, WavError::TooLong { .. })),
        "8 packets at 8 kHz"
    );
    // 1 s at 48 kHz is 48000 frames: 46 packets (47104) fit, 47 (48128) do not.
    assert!(
        capped(&adts_at(0x4C, 46), 1).is_ok(),
        "46 packets at 48 kHz"
    );
    assert!(
        capped(&adts_at(0x4C, 47), 1).is_err_and(|e| matches!(e, WavError::TooLong { .. })),
        "47 packets at 48 kHz"
    );
    for hz in [8_000_u32, 48_000] {
        let at_cap = WavEncoder::encode(&vec![0.25; hz as usize], hz, 1).unwrap();
        assert!(capped(&at_cap, 1).is_ok(), "{hz} Hz WAV of 1 s");
        let over = WavEncoder::encode(&vec![0.25; hz as usize + 1], hz, 1).unwrap();
        assert!(
            capped(&over, 1).is_err(),
            "{hz} Hz WAV of 1 s and one frame"
        );
    }
}

/// A LAME MP3 that declares exactly the cap decodes: the end padding it holds
/// until the trim is not audio over the cap.
#[test]
fn an_mp3_declaring_exactly_the_cap_decodes() {
    // 29 frames of 1152 at 32 kHz, less a delay of 1105 and a padding of 303:
    // 32000 samples, 1.000 s.
    let mut mp3 = lame_info_frame(29, 1105, 303);
    mp3.extend(mp3_frames(MP3_32K_32KBPS, 29));
    let got = capped(&mp3, 1);
    assert!(
        got.as_ref().is_ok_and(|d| d.samples.len() == 16_000),
        "{}",
        summary(&got)
    );
}

/// A LAME MP3 that declares exactly the cap but ends before its end padding
/// is read holds more than the cap after the trim cuts nothing: refused.
#[test]
fn an_mp3_at_the_cap_cut_short_of_its_padding_is_refused() {
    // 31 frames of 1152 at 32 kHz, less a delay of 1105 and a padding of 2607:
    // 32000 samples, 1.000 s. The last frame dropped leaves 33455 samples.
    let mut mp3 = lame_info_frame(31, 1105, 2607);
    mp3.extend(mp3_frames(MP3_32K_32KBPS, 30));
    let got = capped(&mp3, 1);
    assert!(
        matches!(got, Err(WavError::TooLong { max_secs: 1, .. })),
        "{}",
        summary(&got)
    );
}
