// LOC-exempt: STFT, encoder, LSTM, and output Conv are tightly coupled inference
// stages; splitting would scatter the forward-pass arithmetic across files without
// meaningful cohesion gain.

//! Silero VAD v4 Voice Activity Detection (16 kHz, internal use only).
//!
//! Architecture (16 kHz path) from https://github.com/snakers4/silero-vad (MIT):
//! - STFT via learned conv1d basis (256-sample window, 128-sample hop)
//! - Magnitude spectrogram: sqrt(real^2 + imag^2)
//! - Encoder: 4x Conv1d with ReLU
//! - Decoder: 1-layer LSTM (hidden=128) + 1x1 Conv + Sigmoid
//!
//! Weights are vendored in crates/rmlx-audio/assets/silero_vad_16k.safetensors.
//! Convert script: scripts/convert_silero_vad.py (run once at asset-prep time).

use rmlx_mlx::{
    add, conv1d, matmul, maximum, multiply, scalar_f32, sigmoid, sqrt, tanh, Array, Device, Dtype,
};
use thiserror::Error;
use tracing::{debug, info, instrument};

/// Element-wise ReLU: max(x, 0).
fn relu(x: &Array, device: Device) -> Result<Array, VadError> {
    maximum(x, &scalar_f32(0.0), device).map_err(VadError::from)
}

/// STFT hop length in samples.
pub const HOP_LENGTH: usize = 128;
/// STFT window length in samples.
pub const WIN_LENGTH: usize = 256;
/// Number of magnitude bins (win_length/2 + 1).
pub const N_FREQ: usize = WIN_LENGTH / 2 + 1;
/// Silence/pad samples on each side before STFT.
pub const STFT_PAD: usize = 64;
/// LSTM hidden size.
pub const HIDDEN: usize = 128;

/// Vendored Silero VAD safetensors asset (embedded at compile time).
pub const ASSET_BYTES: &[u8] = include_bytes!("../assets/silero_vad_16k.safetensors");

/// VAD errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum VadError {
    /// Missing or malformed weight.
    #[error("weight: {0}")]
    Weight(String),
    /// MLX operation failed.
    #[error("mlx: {0}")]
    Mlx(String),
}

impl From<rmlx_core::error::Error> for VadError {
    fn from(e: rmlx_core::error::Error) -> Self {
        Self::Mlx(e.to_string())
    }
}

// ── Safetensors loader (header-only, no mmap for 1.2 MB file) ────────────────

fn parse_safetensors_f32(
    data: &[u8],
) -> Result<std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>, VadError> {
    if data.len() < 8 {
        return Err(VadError::Weight("file too short".into()));
    }
    // bounds checked: data.len() >= 8
    #[allow(clippy::indexing_slicing, reason = "data.len() >= 8 checked above")]
    let header_len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;
    let hend = 8 + header_len;
    if hend > data.len() {
        return Err(VadError::Weight("header truncated".into()));
    }
    #[allow(clippy::indexing_slicing, reason = "hend <= data.len() checked")]
    let hdr: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&data[8..hend])
            .map_err(|_| VadError::Weight("header not UTF-8".into()))?,
    )
    .map_err(|e| VadError::Weight(format!("header JSON: {e}")))?;

    let obj = hdr
        .as_object()
        .ok_or_else(|| VadError::Weight("header not object".into()))?;
    let mut out = std::collections::HashMap::new();

    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        if meta["dtype"].as_str().unwrap_or("") != "F32" {
            return Err(VadError::Weight(format!("{name}: expected F32")));
        }
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or_else(|| VadError::Weight(format!("{name}: missing shape")))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offsets = meta["data_offsets"]
            .as_array()
            .ok_or_else(|| VadError::Weight(format!("{name}: missing offsets")))?;
        // offsets array has exactly 2 elements per safetensors spec
        #[allow(
            clippy::indexing_slicing,
            reason = "safetensors data_offsets always has 2 elements"
        )]
        let a = offsets[0].as_u64().unwrap_or(0) as usize + hend;
        #[allow(
            clippy::indexing_slicing,
            reason = "safetensors data_offsets always has 2 elements"
        )]
        let b = offsets[1].as_u64().unwrap_or(0) as usize + hend;
        if b > data.len() || a > b {
            return Err(VadError::Weight(format!("{name}: offsets OOB")));
        }
        let n: usize = shape.iter().product::<usize>().max(1);
        #[allow(
            clippy::indexing_slicing,
            reason = "a..b checked: a <= b <= data.len()"
        )]
        let bytes = &data[a..b];
        if bytes.len() != n * 4 {
            return Err(VadError::Weight(format!(
                "{name}: expected {} bytes, got {}",
                n * 4,
                bytes.len()
            )));
        }
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| {
                #[allow(clippy::indexing_slicing, reason = "chunks_exact(4) guarantees len==4")]
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            })
            .collect();
        out.insert(name.clone(), (shape, floats));
    }
    Ok(out)
}

type WMap = std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>;

fn get(map: &WMap, name: &str) -> Result<(Vec<usize>, Array), VadError> {
    let (shape, floats) = map
        .get(name)
        .ok_or_else(|| VadError::Weight(format!("missing '{name}'")))?;
    let s32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    let arr = Array::from_f32_slice(floats, &s32)
        .map_err(|e| VadError::Weight(format!("{name}: {e}")))?;
    Ok((shape.clone(), arr))
}

// ── Encoder Conv1d layer ──────────────────────────────────────────────────────

struct EncLayer {
    w: Array,
    b: Array,
}

impl EncLayer {
    fn forward(&self, x: &Array, device: Device) -> Result<Array, VadError> {
        let y = conv1d(x, &self.w, 1, 1, 1, 1, device)?;
        let y = add(&y, &self.b, device)?;
        relu(&y, device)
    }
}

// ── VadState ──────────────────────────────────────────────────────────────────

/// LSTM hidden and cell state. Carry across calls for streaming audio.
#[non_exhaustive]
#[allow(missing_debug_implementations)]
pub struct VadState {
    /// h_n [1, 1, HIDDEN].
    pub h: Array,
    /// c_n [1, 1, HIDDEN].
    pub c: Array,
}

impl VadState {
    /// Zeroed state for a new stream.
    pub fn new_zeroed(device: Device) -> Result<Self, VadError> {
        let z = vec![0.0_f32; HIDDEN];
        let s = &[1, 1, HIDDEN as i32];
        let h = Array::from_f32_slice(&z, s)
            .map_err(|e| VadError::Mlx(e.to_string()))?
            .astype(Dtype::F32, device)?;
        let c = Array::from_f32_slice(&z, s)
            .map_err(|e| VadError::Mlx(e.to_string()))?
            .astype(Dtype::F32, device)?;
        Ok(Self { h, c })
    }
}

// ── SileroVad ─────────────────────────────────────────────────────────────────

/// Loaded Silero VAD model.
#[allow(missing_debug_implementations)]
pub struct SileroVad {
    stft_basis: Array,
    enc: [EncLayer; 4],
    w_ih: Array,
    w_hh: Array,
    b_ih: Array,
    b_hh: Array,
    out_w: Array,
    out_b: Array,
}

impl SileroVad {
    /// Load from the embedded asset bytes.
    #[instrument(level = "info", name = "silero_vad_load")]
    pub fn load(device: Device) -> Result<Self, VadError> {
        info!("loading Silero VAD");
        let map = parse_safetensors_f32(ASSET_BYTES)?;

        // PyTorch conv1d weights are [out, in, k]; MLX conv1d expects [out, k, in].
        // Transpose all conv weights: axes [0, 2, 1].
        let (_, stft_basis_raw) = get(&map, "stft.forward_basis_buffer")?;
        // stft_basis_raw: [258, 1, 256] PyTorch → [258, 256, 1] MLX
        let stft_basis = stft_basis_raw
            .transpose(&[0, 2, 1], device)
            .map_err(|e| VadError::Mlx(e.to_string()))?;

        let make_enc = |i: usize| -> Result<EncLayer, VadError> {
            let (_, w_raw) = get(&map, &format!("encoder.{i}.reparam_conv.weight"))?;
            // e.g. [128, 129, 3] PyTorch → [128, 3, 129] MLX
            let w = w_raw
                .transpose(&[0, 2, 1], device)
                .map_err(|e| VadError::Mlx(e.to_string()))?;
            let (bs, b) = get(&map, &format!("encoder.{i}.reparam_conv.bias"))?;
            // bias shape is [out_channels]; safetensors guarantees at least 1 element.
            #[allow(
                clippy::indexing_slicing,
                reason = "bias shape has ≥1 element; checked by parse_safetensors_f32"
            )]
            let out_ch = bs[0] as i32;
            let b = b
                .reshape(&[1, 1, out_ch], device)
                .map_err(|e| VadError::Mlx(e.to_string()))?;
            Ok(EncLayer { w, b })
        };

        let enc = [make_enc(0)?, make_enc(1)?, make_enc(2)?, make_enc(3)?];

        let (_, w_ih) = get(&map, "decoder.rnn.weight_ih")?;
        let (_, w_hh) = get(&map, "decoder.rnn.weight_hh")?;
        let (_, b_ih) = get(&map, "decoder.rnn.bias_ih")?;
        let (_, b_hh) = get(&map, "decoder.rnn.bias_hh")?;

        let (_, out_w_raw) = get(&map, "decoder.decoder.2.weight")?;
        // out_w_raw: [1, 128, 1] PyTorch [out=1, in=128, k=1] → [1, 1, 128] MLX [out, k, in]
        let out_w = out_w_raw
            .transpose(&[0, 2, 1], device)
            .map_err(|e| VadError::Mlx(e.to_string()))?;
        let (_, out_b) = get(&map, "decoder.decoder.2.bias")?;
        // out_b: [1] -> [1, 1, 1] for broadcast
        let out_b = out_b
            .reshape(&[1, 1, 1], device)
            .map_err(|e| VadError::Mlx(e.to_string()))?;

        info!("Silero VAD loaded");
        Ok(Self {
            stft_basis,
            enc,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            out_w,
            out_b,
        })
    }

    /// Run on a 16kHz mono f32 PCM slice.
    ///
    /// Returns (per_frame_voice_probs, new_state). Each frame = 128 samples = 8ms.
    #[instrument(skip(self, samples, state), level = "debug")]
    pub fn forward(
        &self,
        samples: &[f32],
        state: VadState,
        device: Device,
    ) -> Result<(Vec<f32>, VadState), VadError> {
        if samples.is_empty() {
            return Ok((vec![], state));
        }

        // 1. Zero-pad: 64 on each side.
        let mut padded = vec![0.0_f32; STFT_PAD + samples.len() + STFT_PAD];
        // Range is exactly [STFT_PAD, STFT_PAD + samples.len()), which fits inside padded.
        #[allow(
            clippy::indexing_slicing,
            reason = "padded is sized to hold STFT_PAD + samples + STFT_PAD"
        )]
        padded[STFT_PAD..STFT_PAD + samples.len()].copy_from_slice(samples);

        let n = padded.len() as i32;
        let x =
            Array::from_f32_slice(&padded, &[1, n, 1]).map_err(|e| VadError::Mlx(e.to_string()))?;

        // 2. STFT via Conv1d(stft_basis, stride=128, pad=0).
        let stft = conv1d(&x, &self.stft_basis, HOP_LENGTH as i32, 0, 1, 1, device)?;
        // stft: [1, T, 258] — shape has 3 dims, [1] is T.
        #[allow(clippy::indexing_slicing, reason = "conv1d output is always rank-3")]
        let t = stft.shape()[1];

        // 3. Magnitude: first 129 = real, next 129 = imag.
        let real = stft.slice(&[0, 0, 0], &[1, t, N_FREQ as i32], &[1, 1, 1], device)?;
        let imag = stft.slice(
            &[0, 0, N_FREQ as i32],
            &[1, t, 2 * N_FREQ as i32],
            &[1, 1, 1],
            device,
        )?;
        let mag = sqrt(
            &add(
                &multiply(&real, &real, device)?,
                &multiply(&imag, &imag, device)?,
                device,
            )?,
            device,
        )?;
        // mag: [1, T, 129]

        // 4. Encoder.
        let mut x = mag;
        for layer in &self.enc {
            x = layer.forward(&x, device)?;
        }
        // x: [1, T, 128]

        // 5. LSTM (frame-by-frame).
        let (lstm_out, (new_h, new_c)) = self.lstm_seq(&x, state.h, state.c, device)?;

        // 6. ReLU + 1x1 Conv + Sigmoid.
        let x = relu(&lstm_out, device)?;
        let y = conv1d(&x, &self.out_w, 1, 0, 1, 1, device)?;
        let y = add(&y, &self.out_b, device)?;
        let probs = sigmoid(&y, device)?;
        // probs: [1, T, 1]

        // 7. Materialise — synchronous eval required before reading bytes.
        probs.eval().map_err(VadError::from)?;
        let raw = probs.to_bytes().map_err(VadError::from)?;
        let n_f = raw.len() / 4;
        let mut out = Vec::with_capacity(n_f);
        for i in 0..n_f {
            #[allow(
                clippy::indexing_slicing,
                reason = "i*4..i*4+4 bounded by n_f = raw.len()/4"
            )]
            out.push(f32::from_le_bytes([
                raw[i * 4],
                raw[i * 4 + 1],
                raw[i * 4 + 2],
                raw[i * 4 + 3],
            ]));
        }

        debug!(n_frames = out.len(), "VAD forward done");
        Ok((out, VadState { h: new_h, c: new_c }))
    }

    /// Frame-by-frame LSTM over input `[1, T, 128]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "LSTM slice bounds derived from HIDDEN constant; all offsets provably bounded"
    )]
    fn lstm_seq(
        &self,
        x: &Array,
        mut h: Array,
        mut c: Array,
        device: Device,
    ) -> Result<(Array, (Array, Array)), VadError> {
        let t_frames = x.shape()[1] as usize;
        let hsz = HIDDEN as i32;
        let mut outs: Vec<Array> = Vec::with_capacity(t_frames);

        // Pre-compute weight transposes once.
        let wih_t = self.w_ih.transpose(&[1, 0], device)?;
        let whh_t = self.w_hh.transpose(&[1, 0], device)?;

        for t in 0..t_frames {
            let xt = x
                .slice(
                    &[0, t as i32, 0],
                    &[1, t as i32 + 1, hsz],
                    &[1, 1, 1],
                    device,
                )?
                .reshape(&[1, hsz], device)?;
            let ht = h.reshape(&[1, hsz], device)?;
            let ct = c.reshape(&[1, hsz], device)?;

            let b_ih = self.b_ih.reshape(&[1, 4 * hsz], device)?;
            let b_hh = self.b_hh.reshape(&[1, 4 * hsz], device)?;

            let gates = add(
                &add(
                    &add(
                        &matmul(&xt, &wih_t, device)?,
                        &matmul(&ht, &whh_t, device)?,
                        device,
                    )?,
                    &b_ih,
                    device,
                )?,
                &b_hh,
                device,
            )?;
            // gates: [1, 512] = [i | f | g | o] each of size 128.

            let ig = sigmoid(&gates.slice(&[0, 0], &[1, hsz], &[1, 1], device)?, device)?;
            let fg = sigmoid(
                &gates.slice(&[0, hsz], &[1, 2 * hsz], &[1, 1], device)?,
                device,
            )?;
            let gg = tanh(
                &gates.slice(&[0, 2 * hsz], &[1, 3 * hsz], &[1, 1], device)?,
                device,
            )?;
            let og = sigmoid(
                &gates.slice(&[0, 3 * hsz], &[1, 4 * hsz], &[1, 1], device)?,
                device,
            )?;

            let c_new = add(
                &multiply(&fg, &ct, device)?,
                &multiply(&ig, &gg, device)?,
                device,
            )?;
            let h_new = multiply(&og, &tanh(&c_new, device)?, device)?;

            h = h_new.reshape(&[1, 1, hsz], device)?;
            c = c_new.reshape(&[1, 1, hsz], device)?;
            outs.push(h.try_clone().map_err(VadError::from)?);
        }

        if outs.is_empty() {
            let empty_data = vec![0.0_f32; 0];
            let empty = Array::from_f32_slice(&empty_data, &[1, 0, HIDDEN as i32])
                .map_err(|e| VadError::Mlx(e.to_string()))?;
            return Ok((empty, (h, c)));
        }

        let refs: Vec<&Array> = outs.iter().collect();
        let out = rmlx_mlx::concatenate(&refs, 1, device)?;
        Ok((out, (h, c)))
    }
}

// ── Segment extraction ────────────────────────────────────────────────────────

/// Extract voiced segments from per-frame probabilities.
///
/// Returns `(start_sample, end_sample)` pairs.
///
/// Parameters:
/// - `threshold`: speech/silence boundary (0.5 typical).
/// - `min_speech_frames`: minimum consecutive voiced frames.
/// - `min_silence_frames`: silence gap needed to split a segment.
pub fn voiced_segments(
    probs: &[f32],
    threshold: f32,
    min_speech_frames: usize,
    min_silence_frames: usize,
) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut speech_start: Option<usize> = None;
    let mut silence_run = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        if p >= threshold {
            if speech_start.is_none() {
                speech_start = Some(i);
            }
            silence_run = 0;
        } else if speech_start.is_some() {
            silence_run += 1;
            if silence_run >= min_silence_frames {
                let end_frame = i - silence_run;
                let start = speech_start.take().unwrap_or(0);
                if end_frame > start && end_frame - start >= min_speech_frames {
                    segments.push((start * HOP_LENGTH, end_frame * HOP_LENGTH));
                }
                silence_run = 0;
            }
        }
    }

    // Flush trailing segment.
    if let Some(start) = speech_start {
        let end_frame = probs.len().saturating_sub(silence_run);
        if end_frame > start && end_frame - start >= min_speech_frames {
            segments.push((start * HOP_LENGTH, end_frame * HOP_LENGTH));
        }
    }

    segments
}

#[cfg(test)]
#[path = "vad_tests.rs"]
mod tests;
