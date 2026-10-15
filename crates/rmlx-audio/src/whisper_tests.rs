//! Whisper model unit tests.
//!
//! Full model tests (load + transcribe) are integration tests under
//! `crates/rmlx-audio/tests/transcribe.rs`. They resolve the Whisper snapshot
//! from `RMLX_O_MODELS_ROOT` auto-discovery (the same convention as
//! `make model-check-full`) and skip gracefully when the model or fixtures are
//! absent — no bespoke env var. The tests here verify config parsing, NPZ
//! parsing logic, and weight-map helpers.

use super::snapshot_fixture::{config_json, decoder_tensors, tiny_config_json, weight_map, Tensor};
use super::{DecodeFilters, TextDecoder, WhisperConfig, WhisperError};
use crate::npz::extract_npy_shape;
use crate::tokenizer::{
    N_VOCAB, TOK_EN, TOK_EOT, TOK_NO_TIMESTAMPS, TOK_SOT, TOK_TIMESTAMP_BEGIN, TOK_TRANSCRIBE,
};
use rmlx_mlx::{Array, Device};
use std::collections::HashMap;

fn fresh_logits() -> Vec<f32> {
    // All zeros so we can see exactly which entries get masked to -inf.
    vec![0.0_f32; N_VOCAB]
}

/// SuppressBlank masks EOT + blank ids at the first step only.
#[test]
fn suppress_blank_first_step_only() {
    let blank = vec![TOK_EOT, 220];
    let f = DecodeFilters::new(vec![], blank, false);
    let mut l = fresh_logits();
    f.apply(&mut l, &[], true);
    assert!(l[TOK_EOT as usize].is_infinite());
    assert!(l[220].is_infinite());

    // Not the first step: blank ids are no longer suppressed (EOT allowed).
    let mut l2 = fresh_logits();
    f.apply(&mut l2, &[5_u32], false);
    assert!(l2[TOK_EOT as usize].is_finite());
}

/// no_timestamps mode masks every timestamp token and notimestamps.
#[test]
fn no_timestamps_mode_masks_all_timestamps() {
    let f = DecodeFilters::new(vec![], vec![TOK_EOT], false);
    let mut l = fresh_logits();
    f.apply(&mut l, &[5_u32], false);
    assert!(l[TOK_TIMESTAMP_BEGIN as usize].is_infinite());
    assert!(l[N_VOCAB - 1].is_infinite());
    // A plain text token stays finite.
    assert!(l[100].is_finite());
}

/// Timestamp mode: the first sampled token must be a timestamp (BOS rule).
#[test]
fn timestamp_mode_bos_forces_timestamp() {
    let f = DecodeFilters::new(vec![], vec![TOK_EOT], true);
    let mut l = fresh_logits();
    f.apply(&mut l, &[], true);
    // All text < timestamp_begin masked.
    assert!(l[100].is_infinite());
    assert!(l[TOK_NO_TIMESTAMPS as usize].is_infinite());
    // Timestamps remain available.
    assert!(l[TOK_TIMESTAMP_BEGIN as usize].is_finite());
}

/// Timestamp mode: right after a single opening timestamp the model must be
/// able to emit TEXT (the penultimate<2 ⇒ treated-as-timestamp branch forces a
/// non-timestamp next). This is the bug that produced empty transcripts.
#[test]
fn timestamp_mode_after_open_ts_allows_text() {
    let f = DecodeFilters::new(vec![], vec![TOK_EOT], true);
    let mut l = fresh_logits();
    // One opening timestamp sampled.
    f.apply(&mut l, &[TOK_TIMESTAMP_BEGIN], false);
    // Text tokens must remain selectable (NOT all masked → not forced to EOT).
    assert!(
        l[100].is_finite(),
        "after a single opening timestamp the decoder must be able to emit text"
    );
    // And further timestamps are masked (must pair with non-timestamp first).
    assert!(l[TOK_TIMESTAMP_BEGIN as usize].is_infinite());
}

/// Config JSON parses correctly for the large-v3 snapshot.
#[test]
fn parse_config_large_v3() {
    let json = r#"{
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": 51866,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 32,
        "model_type": "whisper"
    }"#;
    let cfg = WhisperConfig::from_json(json).unwrap();
    assert_eq!(cfg.n_mels, 128);
    assert_eq!(cfg.n_audio_ctx, 1500);
    assert_eq!(cfg.n_audio_state, 1280);
    assert_eq!(cfg.n_vocab, 51866);
    assert_eq!(cfg.n_text_ctx, 448);
}

fn config_refusal(json: &str) -> String {
    let err = WhisperConfig::from_json(json)
        .expect_err("the config must be refused")
        .to_string();
    assert!(
        err.contains("config.json"),
        "the error names no file: {err}"
    );
    err
}

fn assert_names(err: &str, words: &[&str]) {
    for word in words {
        assert!(err.contains(word), "the error does not name {word}: {err}");
    }
}

/// large-v3, large-v3-turbo (4 decoder layers) and distil-large-v3 (2) share
/// the large-v3 vocabulary (mlx-community `config.json` of each).
#[test]
fn the_large_v3_family_configs_load() {
    for n_text_layer in [32, 4, 2] {
        WhisperConfig::from_json(&config_json(128, 51_866, 1500, n_text_layer).to_string())
            .unwrap_or_else(|e| panic!("n_text_layer {n_text_layer}: {e}"));
    }
}

/// tiny, base, small, medium, large-v1 and large-v2 state `n_vocab` 51865 and
/// 80 mel bins; their `.en` models state 51864 (mlx-community `config.json`).
#[test]
fn a_config_of_another_vocabulary_is_refused_naming_n_vocab() {
    for n_vocab in [51_865, 51_864] {
        let err = config_refusal(&config_json(80, n_vocab, 1500, 4).to_string());
        assert_names(&err, &["n_vocab", &n_vocab.to_string(), "51866"]);
    }
}

/// Whisper has an 80-mel and a 128-mel filterbank only.
#[test]
fn an_n_mels_other_than_80_or_128_is_refused_naming_n_mels() {
    let err = config_refusal(&config_json(64, 51_866, 1500, 32).to_string());
    assert_eq!(
        err,
        "config.json: n_mels is 64, the decode supports 80 or 128 only \
         (Whisper large-v3, large-v3-turbo and distil-large-v3)"
    );
}

/// The decode's 30 s window is 3000 mel frames, which the encoder halves to
/// 1500 positions.
#[test]
fn an_audio_context_other_than_1500_is_refused() {
    let err = config_refusal(&config_json(128, 51_866, 750, 32).to_string());
    assert_names(&err, &["n_audio_ctx", "750", "1500"]);
}

/// A quantized snapshot stores packed weights that the load would read as
/// plain matrices, and the first window would fail in a matmul: the load
/// refuses the `quantization` key and names it.
#[test]
fn a_quantized_config_is_refused_naming_the_key() {
    let mut config = config_json(128, 51_866, 1500, 4);
    config["quantization"] = serde_json::json!({ "group_size": 64, "bits": 4 });
    let err = config_refusal(&config.to_string());
    assert_names(&err, &["quantization"]);
}

/// A `quantization` key of `null` states no quantization.
#[test]
fn a_null_quantization_key_loads() {
    let mut config = config_json(128, 51_866, 1500, 4);
    config["quantization"] = serde_json::Value::Null;
    WhisperConfig::from_json(&config.to_string()).expect("a null key states no quantization");
}

/// The refusal comes before the weight file is read: a directory with a
/// large-v2, a 64-mel or a quantized `config.json` and no weights fails on the
/// config, and one with a large-v3 `config.json` and no weights fails on the
/// weight file.
#[test]
fn load_refuses_the_config_before_it_reads_the_weights() {
    let dir = tempfile::tempdir().unwrap();
    let load_error = |json: String| {
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        match super::WhisperModel::load(dir.path()) {
            Ok(_) => panic!("a directory with no weights must not load"),
            Err(e) => e.to_string(),
        }
    };
    let refused = load_error(config_json(80, 51_865, 1500, 32).to_string());
    assert_names(&refused, &["config.json", "n_vocab", "51865"]);
    let refused = load_error(config_json(64, 51_866, 1500, 32).to_string());
    assert_names(&refused, &["config.json", "n_mels", "64"]);
    let mut quantized = config_json(128, 51_866, 1500, 32);
    quantized["quantization"] = serde_json::json!({ "group_size": 64, "bits": 8 });
    assert_names(&load_error(quantized.to_string()), &["quantization"]);
    let no_weights = load_error(config_json(128, 51_866, 1500, 32).to_string());
    assert_names(&no_weights, &["weight file error"]);
}

/// The server sends a load error to the client: it names the file, never the
/// directory the operator configured.
#[test]
fn a_load_error_names_the_file_and_no_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    let load_error = || match super::WhisperModel::load(dir.path()) {
        Ok(_) => panic!("a directory with no weights must not load"),
        Err(e) => e.to_string(),
    };
    let no_config = load_error();
    assert_names(&no_config, &["config.json", "No such file"]);
    assert!(!no_config.contains(&path), "{no_config}");
    std::fs::write(
        dir.path().join("config.json"),
        config_json(128, 51_866, 1500, 32).to_string(),
    )
    .unwrap();
    let no_weights = load_error();
    assert_names(&no_weights, &["weights.safetensors", "weights.npz"]);
    assert!(!no_weights.contains(&path), "{no_weights}");
}

/// The local large-v3 `config.json` passes the check. Runs on the CPU when
/// `RMLX_O_MODELS_ROOT` holds `mlx-community__whisper-large-v3-mlx`; no weight
/// is read.
#[test]
fn the_local_large_v3_config_loads() {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!("SKIP the_local_large_v3_config_loads: RMLX_O_MODELS_ROOT is unset");
        return;
    };
    let path = std::path::Path::new(&root)
        .join("mlx-community__whisper-large-v3-mlx")
        .join("config.json");
    let Ok(json) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP the_local_large_v3_config_loads: no mlx-community__whisper-large-v3-mlx config.json under the root"
        );
        return;
    };
    let cfg = WhisperConfig::from_json(&json).expect("the large-v3 config must load");
    assert_eq!(cfg.n_vocab, 51_866);
}

/// NPY shape extraction for 2D array.
#[test]
fn npy_shape_2d() {
    let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280, 1280), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, vec![1280, 1280]);
}

/// NPY shape extraction for 1D array.
#[test]
fn npy_shape_1d() {
    let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280,), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, vec![1280]);
}

/// NPY shape extraction for 3D array.
#[test]
fn npy_shape_3d() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (32, 400, 1280), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, vec![32, 400, 1280]);
}

/// NPY shape of a scalar (0-dimensional) is empty vec.
#[test]
fn npy_shape_scalar() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, Vec::<usize>::new());
}

// NOTE: Full smoke-probe + long-form regression tests (WhisperModel::load →
// transcribe) live in crates/rmlx-audio/tests/transcribe.rs and resolve the
// snapshot from RMLX_O_MODELS_ROOT auto-discovery (skip-if-absent).

// ── The decoder of one window, on a decoder of two small blocks ──────────────

const TINY_STATE: usize = 4;
const TINY_BLOCKS: usize = 2;

/// `len` small values in [-0.5, 0.5) from a hash (the murmur3 finaliser) of
/// each index and `seed`. A multiplicative sequence would make each row of
/// four a shifted copy of the next, which a layer norm maps to one row.
fn host_values(len: usize, seed: u32) -> Vec<f32> {
    (0..len as u32)
        .map(|i| {
            let mut h = i ^ seed.wrapping_mul(0x9E37_79B9);
            h = (h ^ (h >> 16)).wrapping_mul(0x85EB_CA6B);
            h = (h ^ (h >> 13)).wrapping_mul(0xC2B2_AE35);
            h ^= h >> 16;
            (h >> 20) as f32 / 4096.0 - 0.5
        })
        .collect()
}

/// An array of `shape` holding [`host_values`].
fn values(shape: &[i32], seed: u32) -> Result<Array, WhisperError> {
    let len: i32 = shape.iter().product();
    Ok(Array::from_f32_slice(
        &host_values(len as usize, seed),
        shape,
    )?)
}

/// The tensors of [`tiny_decoder`]: the names, shapes and biases of the shared
/// fixture, with [`host_values`] in place of its values. The fixture fills each
/// tensor with a linear ramp, and a layer norm maps most token rows of a ramp
/// to the same row; the hash fill gives each token its own scores.
fn tiny_decoder_tensors() -> Vec<Tensor> {
    let mut tensors = decoder_tensors(TINY_BLOCKS);
    for (seed, tensor) in (1..).zip(&mut tensors) {
        tensor.bytes = host_values(tensor.bytes.len() / 4, seed)
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
    }
    tensors
}

/// A text decoder of two blocks with a state of four, the large-v3 vocabulary
/// and a context of 16 positions.
fn tiny_decoder() -> Result<TextDecoder, WhisperError> {
    let cfg = WhisperConfig::from_json(&tiny_config_json(0, TINY_BLOCKS).to_string())?;
    TextDecoder::load(&weight_map(&tiny_decoder_tensors()), &cfg)
}

fn first_largest(row: &[f32]) -> u32 {
    let mut best = 0;
    for (id, logit) in row.iter().enumerate() {
        if row.get(best).is_some_and(|largest| logit > largest) {
            best = id;
        }
    }
    best as u32
}

/// The cross-attention keys and values of a window are projected one time,
/// by the forward that reads the start of the window: one projection for each
/// decoder block. Each decode of the window reads them and projects nothing,
/// whatever the number of decodes. A second window projects its own.
#[test]
fn the_cross_attention_of_a_window_is_projected_one_time() -> Result<(), WhisperError> {
    let decoder = tiny_decoder()?;
    let encoder_out = values(&[1, 3, TINY_STATE as i32], 99)?;
    let filters = DecodeFilters::new(Vec::new(), Vec::new(), true);
    let prefix = [TOK_SOT, TOK_EN, TOK_TRANSCRIBE];

    assert_eq!(decoder.cross_projections(), 0);
    let (_, window) = decoder.open_window(&encoder_out, Device::Cpu)?;
    assert_eq!(decoder.cross_projections(), TINY_BLOCKS);
    let mut decodes = Vec::new();
    for decode in 1..=3 {
        decodes.push(window.decode(&prefix, 4, &filters, |row: &[f32]| Ok(first_largest(row)))?);
        assert_eq!(
            decoder.cross_projections(),
            TINY_BLOCKS,
            "after decode {decode} of the window"
        );
    }
    assert!(decodes.iter().all(|decoded| !decoded.tokens.is_empty()));
    assert!(decodes.windows(2).all(|pair| pair[0] == pair[1]));

    decoder.open_window(&encoder_out, Device::Cpu)?;
    assert_eq!(decoder.cross_projections(), 2 * TINY_BLOCKS);
    Ok(())
}

/// The tensors of [`tiny_decoder`] by name, as `f64`.
fn tiny_tensors() -> HashMap<String, Vec<f64>> {
    tiny_decoder_tensors()
        .into_iter()
        .map(|tensor| {
            let values = tensor
                .bytes
                .chunks_exact(4)
                .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect();
            (tensor.name, values)
        })
        .collect()
}

/// `x @ wᵀ + b` for the linear `name` of [`tiny_tensors`], a weight `[out][in]`
/// of four by four and a bias when the tensors hold one.
fn linear(tensors: &HashMap<String, Vec<f64>>, name: &str, x: &[f64]) -> Vec<f64> {
    let weight = &tensors[&format!("{name}.weight")];
    let bias = tensors.get(&format!("{name}.bias"));
    weight
        .chunks_exact(TINY_STATE)
        .enumerate()
        .map(|(out, row)| {
            let sum: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum();
            sum + bias.map_or(0.0, |bias| bias[out])
        })
        .collect()
}

/// Layer norm `name` of [`tiny_tensors`] over the four values of `x`,
/// population variance, eps 1e-5.
fn layer_norm(tensors: &HashMap<String, Vec<f64>>, name: &str, x: &[f64]) -> Vec<f64> {
    let (weight, bias) = (
        &tensors[&format!("{name}.weight")],
        &tensors[&format!("{name}.bias")],
    );
    let mean = x.iter().sum::<f64>() / TINY_STATE as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / TINY_STATE as f64;
    x.iter()
        .zip(weight.iter().zip(bias))
        .map(|(v, (w, b))| (v - mean) / (var + 1e-5).sqrt() * w + b)
        .collect()
}

/// The cross-attention scores of block 0 before the softmax, worked out on the
/// host from the tensors of [`tiny_decoder`]. The key projections have no bias.
/// Two heads of two dimensions: the scores of head `h` are `q_h · k_hᵀ`, with q
/// and k each scaled by `2^-0.25`, so the product is scaled by `2^-0.5`.
/// Returns `[token][frame]` for `head`.
fn tiny_block_0_cross_scores(tokens: &[u32], encoder: &[f64], head: usize) -> Vec<Vec<f64>> {
    let tensors = tiny_tensors();
    let embedding = &tensors["decoder.token_embedding.weight"];
    let position = &tensors["decoder.positional_embedding"];
    let x0: Vec<Vec<f64>> = tokens
        .iter()
        .enumerate()
        .map(|(at, &token)| {
            (0..TINY_STATE)
                .map(|i| embedding[token as usize * TINY_STATE + i] + position[at * TINY_STATE + i])
                .collect()
        })
        .collect();
    let scale = 2.0_f64.powf(-0.25);
    let block = "decoder.blocks.0";
    let n1: Vec<Vec<f64>> = x0
        .iter()
        .map(|x| layer_norm(&tensors, &format!("{block}.attn_ln"), x))
        .collect();
    let project = |rows: &[Vec<f64>], name: &str| -> Vec<Vec<f64>> {
        let name = format!("{block}.{name}");
        rows.iter().map(|x| linear(&tensors, &name, x)).collect()
    };
    let (q, k, v) = (
        project(&n1, "attn.query"),
        project(&n1, "attn.key"),
        project(&n1, "attn.value"),
    );
    let mut attended = vec![vec![0.0; TINY_STATE]; tokens.len()];
    for h in 0..2 {
        let dims = 2 * h..2 * h + 2;
        for t in 0..tokens.len() {
            let scores: Vec<f64> = (0..=t)
                .map(|u| {
                    dims.clone()
                        .map(|d| q[t][d] * scale * k[u][d] * scale)
                        .sum::<f64>()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
            for (u, score) in scores.iter().enumerate() {
                let weight = (score - max).exp() / total;
                for d in dims.clone() {
                    attended[t][d] += weight * v[u][d];
                }
            }
        }
    }
    let attention = project(&attended, "attn.out");
    let x1: Vec<Vec<f64>> = x0
        .iter()
        .zip(&attention)
        .map(|(x, a)| x.iter().zip(a).map(|(x, a)| x + a).collect())
        .collect();
    let n2: Vec<Vec<f64>> = x1
        .iter()
        .map(|x| layer_norm(&tensors, &format!("{block}.cross_attn_ln"), x))
        .collect();
    let cross_q = project(&n2, "cross_attn.query");
    let frames: Vec<Vec<f64>> = encoder
        .chunks_exact(TINY_STATE)
        .map(<[f64]>::to_vec)
        .collect();
    let cross_k = project(&frames, "cross_attn.key");
    let dims = 2 * head..2 * head + 2;
    cross_q
        .iter()
        .map(|q| {
            cross_k
                .iter()
                .map(|k| dims.clone().map(|d| q[d] * k[d]).sum::<f64>() * scale * scale)
                .collect()
        })
        .collect()
}

/// The alignment pass of a window reads the window's cross-attention keys and
/// values and projects none. It gives heads x tokens x frames scores, those of
/// head 1 of block 0 are the hand-worked `q · kᵀ · head_dim^-0.5`, and a decode
/// of the window after it gives the tokens of a decode of a twin window that
/// was not aligned. Every two token rows of the expected scores differ by more
/// than 10 times the bound, so a mix-up of any two rows fails the compare.
#[test]
fn the_alignment_pass_reads_the_cached_cross_attention() -> Result<(), WhisperError> {
    const BOUND: f64 = 1e-4;
    let decoder = tiny_decoder()?;
    let encoder = host_values(3 * TINY_STATE, 99);
    let encoder_out = Array::from_f32_slice(&encoder, &[1, 3, TINY_STATE as i32])?;
    let filters = DecodeFilters::new(Vec::new(), Vec::new(), true);
    let prefix = [TOK_SOT, TOK_EN, TOK_TRANSCRIBE];
    let tokens = [
        TOK_SOT,
        TOK_EN,
        TOK_TRANSCRIBE,
        TOK_NO_TIMESTAMPS,
        440,
        TOK_EOT,
    ];
    let heads = [(0, 1), (1, 0)];
    let frames = 2;

    let (_, window) = decoder.open_window(&encoder_out, Device::Cpu)?;
    let projected = decoder.cross_projections();
    let scores = window.align(&tokens, &heads, frames)?;
    assert_eq!(decoder.cross_projections(), projected);
    assert_eq!(
        (
            scores.heads,
            scores.tokens,
            scores.frames,
            scores.values.len()
        ),
        (2, 6, 2, 2 * 6 * 2)
    );

    let encoder: Vec<f64> = encoder.into_iter().map(f64::from).collect();
    let expected = tiny_block_0_cross_scores(&tokens, &encoder, 1);
    for (token, row) in expected.iter().enumerate() {
        for (other, other_row) in expected.iter().enumerate().skip(token + 1) {
            let apart = row
                .iter()
                .zip(other_row)
                .take(frames)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                apart > 10.0 * BOUND,
                "the expected rows of tokens {token} and {other} are {apart} apart"
            );
        }
        for (frame, value) in row.iter().take(frames).enumerate() {
            let found = f64::from(scores.values[token * frames + frame]);
            assert!(
                (found - value).abs() < BOUND,
                "token {token}, frame {frame}: {found} against {value}"
            );
        }
    }

    let pick = |row: &[f32]| Ok(first_largest(row));
    let after = window.decode(&prefix, 4, &filters, pick)?;
    let (_, twin) = decoder.open_window(&encoder_out, Device::Cpu)?;
    let plain = twin.decode(&prefix, 4, &filters, pick)?;
    assert_eq!(after.tokens, plain.tokens);
    Ok(())
}

/// Alignment scores of another count than heads x tokens x frames are an
/// error, never filled.
#[test]
fn alignment_scores_of_the_wrong_count_are_an_error() {
    assert!(super::HeadScores::new(2, 3, 4, vec![0.0; 24]).is_ok());
    for len in [23, 25, 0] {
        let error = super::HeadScores::new(2, 3, 4, vec![0.0; len]).map(|_| ());
        assert!(
            matches!(&error, Err(WhisperError::Mlx(message)) if message.contains("2 heads x 3 tokens x 4 frames = 24")),
            "{len}: {error:?}"
        );
    }
}
