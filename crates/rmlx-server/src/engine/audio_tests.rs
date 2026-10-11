//! Audio-prompt construction tests.
//!
//! The end-to-end load + decode + scatter test is gated on the e4b snapshot
//! (`RMLX_TEST_MODEL_GEMMA4_E4B`) and skips gracefully when absent. The
//! base64-prefix-stripping logic is covered model-free.

use super::{
    audio_secs_budget, build_audio_prompt, decode_input_audio, load_gemma4_audio_bundle,
    AudioBundle, GEMMA4_BOA_TOKEN_ID, GEMMA4_EOA_TOKEN_ID,
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
        eprintln!(
            "SKIP gemma4_audio_prompt_build_real_weights: RMLX_TEST_MODEL_GEMMA4_E4B not set"
        );
        return;
    };
    if !dir.exists() {
        eprintln!(
            "SKIP gemma4_audio_prompt_build_real_weights: e4b snapshot absent at {}",
            dir.display()
        );
        return;
    }

    let bundle = load_gemma4_audio_bundle(&dir)
        .expect("load audio bundle")
        .expect("snapshot has an audio_config");
    // e4b ships the Conformer audio tower (not the encoder-free unified arch).
    let AudioBundle::Gemma4 { audio_token_id, .. } = &bundle else {
        eprintln!(
            "SKIP gemma4_audio_prompt_build_real_weights: e4b loaded a non-Conformer audio bundle (unexpected)"
        );
        return;
    };
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

    // The 2 s clip yields 50 audio tokens (25 a second): the prompt, the two
    // markers and the tokens fill a 55-token context, and one token less
    // leaves a budget of 1 s.
    let context_tokens = prompt_tokens.len() + 2 + 50;
    let too_small = build_audio_prompt(
        &model,
        &bundle,
        std::slice::from_ref(&b64),
        &prompt_tokens,
        context_tokens - 1,
        rmlx_mlx::Device::Cpu,
        &|| false,
    )
    .expect_err("2 s of audio over a 1 s budget");
    assert!(
        too_small.to_string().contains("more than the 1 s"),
        "budget error names the 1 s: {too_small}"
    );

    let (aug_ids, embeds, masked_ids) = build_audio_prompt(
        &model,
        &bundle,
        &[b64],
        &prompt_tokens,
        context_tokens,
        rmlx_mlx::Device::Cpu,
        &|| false,
    )
    .expect("build audio prompt");
    assert!(
        aug_ids.len() <= context_tokens,
        "the prompt fits the context the budget was cut for: {} > {context_tokens}",
        aug_ids.len()
    );

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
    let err = build_audio_prompt(
        &model,
        &bundle,
        &two,
        &[2u32],
        8192,
        rmlx_mlx::Device::Cpu,
        &|| false,
    )
    .expect_err("two clips must be rejected");
    assert!(
        err.to_string().contains("one audio clip"),
        "clear multi-clip error: {err}"
    );
}

/// A WAV header declaring `data_bytes` of 16 kHz mono PCM-16, followed by
/// 100 silent samples.
fn wav_declaring(data_bytes: u32) -> Vec<u8> {
    let mut wav = rmlx_audio::wav::WavEncoder::encode(&[0.0; 100], 16_000, 1).expect("encode wav");
    wav.splice(4..8, (data_bytes + 36).to_le_bytes());
    wav.splice(40..44, data_bytes.to_le_bytes());
    wav
}

/// An `input_audio` clip that declares more than `MAX_AUDIO_SECS` is refused
/// before a packet is decoded, naming the cap and its declared length.
#[test]
fn an_input_audio_clip_over_the_duration_cap_is_refused_up_front() {
    // 10801 s of 16 kHz PCM-16.
    let b64 = base64_encode(&wav_declaring(10_801 * 16_000 * 2));
    let err = decode_input_audio(&b64, rmlx_audio::wav::MAX_AUDIO_SECS, 640, &|| false)
        .expect_err("over the cap");
    assert_eq!(
        err.to_string(),
        "other: input_audio decode failed: the file declares 10801.000 s of audio, \
         over the 10800 s the decoder takes"
    );
}

/// The chat path's sign that the client is gone is a closed token channel; a
/// decode asked through it stops before the first packet.
#[test]
fn an_input_audio_decode_stops_when_the_token_channel_is_closed() {
    let b64 = base64_encode(&wav_declaring(200));
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let (samples, _) = decode_input_audio(&b64, rmlx_audio::wav::MAX_AUDIO_SECS, 640, &|| {
        tx.is_closed()
    })
    .expect("open channel");
    assert_eq!(samples.len(), 100);
    drop(rx);
    let err = decode_input_audio(&b64, rmlx_audio::wav::MAX_AUDIO_SECS, 640, &|| {
        tx.is_closed()
    })
    .expect_err("closed channel");
    assert_eq!(
        err.to_string(),
        "other: input_audio decode failed: audio decode cancelled at 0.00 s"
    );
}

/// The audio-token budget in seconds: 25 tokens a second at 640 samples a
/// token (the Conformer's 160-sample hop times four, and the unified model's
/// 640-sample frame), after the prompt and the two markers, at most 3 hours.
#[test]
fn the_audio_budget_is_the_context_left_at_25_tokens_a_second() {
    // (8192 - 100 - 2) tokens / 25 = 323.6 s.
    assert_eq!(audio_secs_budget(8192, 100, 640), 323);
    // Each input moved alone, so a swapped or dropped input reads another
    // figure: the context, the prompt, the samples a token, the two markers.
    assert_eq!(audio_secs_budget(4096, 100, 640), 159);
    assert_eq!(audio_secs_budget(8192, 1000, 640), 287);
    assert_eq!(audio_secs_budget(8192, 100, 320), 161);
    assert_eq!(audio_secs_budget(127, 0, 16_000), 125);
    assert_eq!(audio_secs_budget(131_072, 0, 640), 5242);
    assert_eq!(
        audio_secs_budget(1 << 20, 0, 640),
        rmlx_audio::wav::MAX_AUDIO_SECS
    );
    assert_eq!(audio_secs_budget(100, 100, 640), 0);
    assert_eq!(audio_secs_budget(100, 200, 640), 0);
}

/// A clip longer than the context budget is refused before it decodes:
/// 400 s declared against the 323 s an 8192-token context leaves after a
/// 100-token prompt. The up-front refusal reads no packet (its memory is
/// measured in rmlx-audio's `decode_memory`).
#[test]
fn an_input_audio_clip_over_the_context_budget_is_refused_up_front() {
    let b64 = base64_encode(&wav_declaring(400 * 16_000 * 2));
    let err = decode_input_audio(&b64, audio_secs_budget(8192, 100, 640), 640, &|| false)
        .expect_err("over the budget");
    assert_eq!(
        err.to_string(),
        "other: input_audio holds more than the 323 s of audio the context leaves \
         after the prompt (25 tokens a second)"
    );
}

/// A stream that passes the context budget as it decodes (no declared
/// length) is refused with the same context message, and the rate follows the
/// samples a token.
#[test]
fn an_input_audio_stream_over_the_context_budget_names_the_context() {
    let mut wav =
        rmlx_audio::wav::WavEncoder::encode(&vec![0.0; 3 * 16_000], 16_000, 1).expect("encode wav");
    wav.splice(4..8, [0xFF; 4]);
    wav.splice(40..44, [0xFF; 4]);
    let b64 = base64_encode(&wav);
    let err = decode_input_audio(&b64, 2, 320, &|| false).expect_err("over the budget");
    assert_eq!(
        err.to_string(),
        "other: input_audio holds more than the 2 s of audio the context leaves \
         after the prompt (50 tokens a second)"
    );
}
