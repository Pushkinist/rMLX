//! The mono samples of a decode: channel downmix, linear resampling and the
//! cap on the stream length.

use tracing::{debug, warn};

use super::{frames_in, WavError};

/// The mono samples of a decode, built packet by packet: the channels
/// averaged, then resampled by linear interpolation when a rate other than
/// the file's own is asked for.
pub(super) struct MonoSamples {
    out: Vec<f32>,
    /// The mono frames received, before any cut.
    pub(super) received: usize,
    /// The most frames `received` may reach.
    max_frames: usize,
    max_secs: u32,
    resampler: Option<Resampler>,
}

impl MonoSamples {
    /// `hold` is the most frames [`MonoSamples::finish`] may cut from the
    /// end; `expected` is the frames the stream is likely to hold. More than
    /// `max_secs` of frames is refused before it is written.
    pub(super) fn new(
        from_rate: u32,
        to_rate: Option<u32>,
        hold: usize,
        expected: usize,
        max_secs: u32,
    ) -> Self {
        // The end padding a LAME tag declares (12 bits, at most 4095 frames)
        // is read before the trim cuts it; it is not audio over the cap.
        let max_frames = frames_in(max_secs, from_rate).saturating_add(hold);
        let expected = expected.min(max_frames);
        let resampler = to_rate.filter(|&to| to != from_rate).map(|to| Resampler {
            ratio: f64::from(to) / f64::from(from_rate),
            pending: Vec::new(),
            first: 0,
            hold,
        });
        let capacity = match &resampler {
            None => expected.saturating_add(hold),
            // Output may run one sample past the final length until `finish`.
            Some(r) => r.output_len(expected).saturating_add(1),
        };
        let mut out = Vec::new();
        if out.try_reserve_exact(capacity).is_err() {
            debug!(
                capacity,
                "the expected length cannot be reserved; the samples grow"
            );
        }
        Self {
            out,
            received: 0,
            max_frames,
            max_secs,
            resampler,
        }
    }

    /// Refuse `frames` more when they take the stream past the cap.
    fn room_for(&self, frames: usize) -> Result<(), WavError> {
        if self.received.saturating_add(frames) <= self.max_frames {
            return Ok(());
        }
        warn!(
            max_secs = self.max_secs,
            "the stream holds more audio than the cap; decode refused"
        );
        Err(WavError::TooLong {
            declared_secs: None,
            max_secs: self.max_secs,
        })
    }

    /// Refuse a stream that still holds more than the cap once `cut` frames
    /// are left out of its end.
    pub(super) fn refuse_past_cap_after(&self, cut: usize, from_rate: u32) -> Result<(), WavError> {
        if self.received.saturating_sub(cut) <= frames_in(self.max_secs, from_rate) {
            return Ok(());
        }
        warn!(
            max_secs = self.max_secs,
            "the stream holds more audio than the cap after the padding trim; decode refused"
        );
        Err(WavError::TooLong {
            declared_secs: None,
            max_secs: self.max_secs,
        })
    }

    pub(super) fn push_frames(
        &mut self,
        interleaved: &[f32],
        n_channels: usize,
    ) -> Result<(), WavError> {
        self.room_for(interleaved.len() / n_channels)?;
        let into = match &mut self.resampler {
            Some(r) => &mut r.pending,
            None => &mut self.out,
        };
        if n_channels == 1 {
            into.extend_from_slice(interleaved);
        } else {
            let inv = 1.0_f32 / n_channels as f32;
            into.extend(
                interleaved
                    .chunks_exact(n_channels)
                    .map(|frame| frame.iter().sum::<f32>() * inv),
            );
        }
        self.received += interleaved.len() / n_channels;
        if let Some(r) = &mut self.resampler {
            r.emit(&mut self.out, self.received);
        }
        Ok(())
    }

    pub(super) fn push_silence(&mut self, frames: usize) -> Result<(), WavError> {
        self.room_for(frames)?;
        let into = match &mut self.resampler {
            Some(r) => &mut r.pending,
            None => &mut self.out,
        };
        into.resize(into.len() + frames, 0.0);
        self.received += frames;
        if let Some(r) = &mut self.resampler {
            r.emit(&mut self.out, self.received);
        }
        Ok(())
    }

    /// The samples, with the last `cut` frames of the stream left out.
    pub(super) fn finish(mut self, cut: usize) -> Vec<f32> {
        let len = self.received.saturating_sub(cut);
        match self.resampler {
            None => self.out.truncate(len),
            Some(r) => r.finish(&mut self.out, len),
        }
        self.out
    }
}

/// Linear interpolation from one rate to another, fed in parts. Output `i`
/// reads input `floor(i / ratio)` and the one after it; the last output reads
/// the last input. An input is kept until no later output reads it.
struct Resampler {
    /// Output samples per input sample.
    ratio: f64,
    /// The inputs from index `first` on.
    pending: Vec<f32>,
    first: usize,
    /// The last inputs no output reads before `finish`: the frames a cut may
    /// leave out.
    hold: usize,
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    reason = "sample positions are below 2^53 and non-negative; an output reads only \
              inputs from `first` to the last one received, which `pending` holds"
)]
impl Resampler {
    /// The outputs of `inputs` input samples.
    fn output_len(&self, inputs: usize) -> usize {
        (inputs as f64 * self.ratio).round() as usize
    }

    /// The input output `i` starts at, and its exact position.
    fn source(&self, i: usize) -> (usize, f64) {
        let pos = i as f64 / self.ratio;
        (pos.floor() as usize, pos)
    }

    fn interpolate(&self, idx: usize, pos: f64) -> f32 {
        let a = self.pending[idx - self.first];
        let b = self.pending[idx + 1 - self.first];
        let frac = (pos - idx as f64) as f32;
        a + (b - a) * frac
    }

    /// Write every output whose two inputs are received and not held, then
    /// drop the inputs no later output reads.
    fn emit(&mut self, out: &mut Vec<f32>, received: usize) {
        let ready = received.saturating_sub(self.hold);
        loop {
            let (idx, pos) = self.source(out.len());
            if idx + 1 >= ready {
                break;
            }
            out.push(self.interpolate(idx, pos));
        }
        let keep_from = self.source(out.len()).0.min(ready.saturating_sub(1));
        self.pending.drain(..keep_from - self.first);
        self.first = keep_from;
    }

    /// Write the outputs of a stream of `len` inputs that `emit` left.
    fn finish(mut self, out: &mut Vec<f32>, len: usize) {
        if len == 0 {
            out.clear();
            return;
        }
        self.pending.truncate(len - self.first);
        let out_len = self.output_len(len);
        out.truncate(out_len);
        let last = len - 1;
        for i in out.len()..out_len {
            let (idx, pos) = self.source(i);
            if idx >= last {
                out.push(self.pending[last - self.first]);
            } else {
                out.push(self.interpolate(idx, pos));
            }
        }
    }
}

#[cfg(test)]
#[path = "samples_tests.rs"]
mod samples_tests;
