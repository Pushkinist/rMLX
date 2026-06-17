//! Unit tests for the long-form transcription engine helpers (no model needed).

use super::{fmt_time, render_srt, render_vtt, resample_to_16k, OutputFormat, Segment};

#[test]
fn output_format_parse() {
    assert_eq!(OutputFormat::parse("txt").unwrap(), OutputFormat::Txt);
    assert_eq!(OutputFormat::parse("text").unwrap(), OutputFormat::Txt);
    assert_eq!(OutputFormat::parse("json").unwrap(), OutputFormat::Json);
    assert_eq!(OutputFormat::parse("srt").unwrap(), OutputFormat::Srt);
    assert_eq!(OutputFormat::parse("vtt").unwrap(), OutputFormat::Vtt);
    assert!(OutputFormat::parse("flac").is_err());
}

#[test]
fn time_formatting_srt_vtt() {
    // 1h 2m 3.456s.
    let secs = 3600.0 + 120.0 + 3.456;
    assert_eq!(fmt_time(secs, true), "01:02:03,456");
    assert_eq!(fmt_time(secs, false), "01:02:03.456");
    // Zero.
    assert_eq!(fmt_time(0.0, false), "00:00:00.000");
    // Negative clamps to zero.
    assert_eq!(fmt_time(-5.0, false), "00:00:00.000");
}

#[test]
fn srt_vtt_multi_segment() {
    let segs = vec![
        Segment {
            start: 0.0,
            end: 2.5,
            text: "hello world".to_owned(),
        },
        Segment {
            start: 2.5,
            end: 5.0,
            text: "second line".to_owned(),
        },
    ];
    let srt = render_srt(&segs);
    assert!(srt.contains("1\n00:00:00,000 --> 00:00:02,500\nhello world"));
    assert!(srt.contains("2\n00:00:02,500 --> 00:00:05,000\nsecond line"));

    let vtt = render_vtt(&segs);
    assert!(vtt.starts_with("WEBVTT\n\n"));
    assert!(vtt.contains("00:00:00.000 --> 00:00:02.500\nhello world"));
    assert!(vtt.contains("00:00:02.500 --> 00:00:05.000\nsecond line"));
}

#[test]
fn resample_identity_when_already_16k() {
    let s = vec![0.1_f32, 0.2, 0.3, 0.4];
    assert_eq!(resample_to_16k(&s, 16_000), s);
}

#[test]
fn resample_downsamples_48k_to_16k() {
    // 48k -> 16k should produce ~1/3 the samples.
    let s: Vec<f32> = (0..4800).map(|i| i as f32 / 4800.0).collect();
    let out = resample_to_16k(&s, 48_000);
    // 4800 * 16000/48000 = 1600.
    assert!((out.len() as i64 - 1600).abs() <= 1, "len = {}", out.len());
    // Monotone ramp preserved roughly.
    assert!(out.first().copied().unwrap_or(1.0) < 0.05);
    assert!(out.last().copied().unwrap_or(0.0) > 0.9);
}

#[test]
fn resample_empty() {
    assert!(resample_to_16k(&[], 48_000).is_empty());
}
