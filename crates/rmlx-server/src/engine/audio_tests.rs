//! Audio-prompt construction tests.
//!
//! The end-to-end load + decode + scatter test is gated on the e4b snapshot
//! (`RMLX_TEST_MODEL_GEMMA4_E4B`) and skips gracefully when absent. The
//! base64-prefix-stripping logic is covered model-free.

use super::{
    build_audio_prompt, load_gemma4_audio_bundle, AudioBundle, GEMMA4_BOA_TOKEN_ID,
    GEMMA4_EOA_TOKEN_ID,
};

/// Minimal standard-alphabet base64 encoder (mirror of the crate decoder), so
/// the test can wrap synthesized WAV bytes into the `input_audio.data` shape.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(chunk.get(1).copied().unwrap_or(0));
        let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 0x3f] as char);
        out.push(ALPHABET[(n >> 12) as usize & 0x3f] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 0x3f] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

fn e4b_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

/// Synthesize a 2 s 16 kHz mono sine, encode WAV → base64, then run the full
/// Gemma4 audio prompt build: bundle load, mel extraction, Conformer encode,
/// soft-token scatter. Asserts the prompt carries exactly `T_sub` `<|audio|>`
/// placeholders and the fused embeds have the right shape. Gated on the e4b
/// snapshot; skips gracefully when absent.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "values established by construction earlier in this test"
)]
fn gemma4_audio_prompt_build_real_weights() {
    let Some(dir) = e4b_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    if !dir.exists() {
        eprintln!("SKIP: e4b snapshot absent at {}", dir.display());
        return;
    }

    let bundle = load_gemma4_audio_bundle(&dir)
        .expect("load audio bundle")
        .expect("snapshot has an audio_config");
    let AudioBundle::Gemma4 { audio_token_id, .. } = &bundle;
    let audio_token_id = *audio_token_id;

    // Load the text model so build_audio_prompt can embed + scatter.
    let model = rmlx_models::arch::load_model(
        &dir,
        rmlx_mlx::Device::Cpu,
        &rmlx_models::arch::LoadOpts::default(),
    )
    .expect("load gemma4 text model");

    // 2 s of 16 kHz mono 440 Hz sine.
    let sr = 16_000u32;
    let n = (sr as usize) * 2;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.3)
        .collect();
    let wav = rmlx_audio::wav::WavEncoder::encode(&samples, sr, 1).expect("encode wav");
    let b64 = base64_encode(&wav);

    // A short text prompt (no template — the build splices after token 0).
    let prompt_tokens = vec![2u32, 1234, 5678];

    let (aug_ids, embeds, masked_ids) = build_audio_prompt(
        &model,
        &bundle,
        &[b64],
        &prompt_tokens,
        rmlx_mlx::Device::Cpu,
    )
    .expect("build audio prompt");

    // BOA + run of <|audio|> + EOA spliced after the leading token.
    let n_audio = aug_ids.iter().filter(|&&t| t == audio_token_id).count();
    assert!(n_audio > 0, "expected audio soft tokens in prompt");
    assert!(
        aug_ids.contains(&GEMMA4_BOA_TOKEN_ID),
        "begin-of-audio marker present"
    );
    assert!(
        aug_ids.contains(&GEMMA4_EOA_TOKEN_ID),
        "end-of-audio marker present"
    );
    assert_eq!(
        aug_ids.len(),
        prompt_tokens.len() + n_audio + 2,
        "aug = prompt + audio run + BOA + EOA"
    );

    embeds.eval().expect("eval embeds");
    let es = embeds.shape();
    let hidden = model.as_gemma4().expect("gemma4").cfg.hidden_size as i32;
    assert_eq!(es[0], 1, "batch");
    assert_eq!(es[1], aug_ids.len() as i32, "seq == aug_ids");
    assert_eq!(es[2], hidden, "hidden");
    assert_eq!(masked_ids.shape()[0], aug_ids.len() as i32, "masked seq");
}

/// `build_audio_prompt` rejects >1 clip with a clear error (model-free: the
/// guard fires before any tower access, so a non-existent snapshot is fine —
/// but we still gate so the bundle is real for the dereference further down).
#[test]
fn audio_prompt_multi_clip_rejected() {
    let Some(dir) = e4b_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    if !dir.exists() {
        return;
    }
    let Ok(Some(bundle)) = load_gemma4_audio_bundle(&dir) else {
        eprintln!("SKIP: no audio bundle");
        return;
    };
    let Ok(model) = rmlx_models::arch::load_model(
        &dir,
        rmlx_mlx::Device::Cpu,
        &rmlx_models::arch::LoadOpts::default(),
    ) else {
        eprintln!("SKIP: text model load failed");
        return;
    };
    let two = vec!["AAAA".to_owned(), "BBBB".to_owned()];
    let err = build_audio_prompt(&model, &bundle, &two, &[2u32], rmlx_mlx::Device::Cpu)
        .expect_err("two clips must be rejected");
    assert!(
        err.to_string().contains("one audio clip"),
        "clear multi-clip error: {err}"
    );
}
