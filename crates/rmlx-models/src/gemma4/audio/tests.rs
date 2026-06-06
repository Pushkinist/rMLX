//! Audio encoder unit tests.

use super::*;

/// SSCP frequency-axis input size (log-mel feature dim) — drives the
/// synthetic `input_proj_linear` shape (upstream `INPUT_FEAT_SIZE`).
const SSCP_INPUT_FEAT_SIZE: usize = 128;

fn e4b_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn materialize_f32(arr: &Array) -> Vec<f32> {
    let f = arr.astype(Dtype::F32, Device::Cpu).expect("cast f32");
    f.eval().expect("materialize");
    f.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn small_cfg() -> Gemma4AudioConfig {
    Gemma4AudioConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        subsampling_conv_channels: vec![16, 8],
        conv_kernel_size: 5,
        residual_weight: 0.5,
        attention_chunk_size: 4,
        attention_context_left: 5,
        attention_context_right: 0,
        attention_logit_cap: 50.0,
        attention_invalid_logits_value: -1e9,
        use_clipped_linears: true,
        rms_norm_eps: 1e-6,
        gradient_clipping: 1e10,
        output_proj_dims: Some(48),
        audio_token_id: 258881,
    }
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn zeros(shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_bytes(f32_bytes(&vec![0.0f32; n as usize]), shape, Dtype::F32).unwrap()
}
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn ones(shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_bytes(f32_bytes(&vec![1.0f32; n as usize]), shape, Dtype::F32).unwrap()
}

/// Zero/random-init tower from a synthetic config — no snapshot needed.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn synth_tower(cfg: &Gemma4AudioConfig) -> AudioEncoder {
    let hidden = cfg.hidden_size as i32;
    let heads = cfg.num_attention_heads as i32;
    let hd = hidden / heads;
    let clip_lin = |out: i32, inn: i32| ClippableLinear {
        weight: zeros(&[out, inn]),
        clip: None,
    };
    let rms = |dim: i32| AudioRmsNorm {
        weight: ones(&[dim]),
        eps: cfg.rms_norm_eps,
    };
    let ln = |dim: i32| ChannelLayerNorm {
        weight: ones(&[dim]),
        eps: cfg.rms_norm_eps,
    };
    let ff = || ConformerFeedForward {
        pre_layer_norm: rms(hidden),
        ffw_layer_1: clip_lin(hidden * 4, hidden),
        ffw_layer_2: clip_lin(hidden, hidden * 4),
        post_layer_norm: rms(hidden),
        gradient_clipping: cfg.gradient_clipping,
        residual_weight: cfg.residual_weight,
    };
    let c0 = cfg.subsampling_conv_channels[0] as i32;
    let c1 = cfg.subsampling_conv_channels[1] as i32;
    let mut freq = SSCP_INPUT_FEAT_SIZE as i32;
    for _ in 0..2 {
        freq = (freq + 2 - 3) / 2 + 1;
    }
    let subsample = SubSampleConvProjection {
        layer0: SscpConvBlock {
            conv_w: zeros(&[c0, 3, 3, 1]),
            norm: ln(c0),
            time_stride: 2,
        },
        layer1: SscpConvBlock {
            conv_w: zeros(&[c1, 3, 3, c0]),
            norm: ln(c1),
            time_stride: 2,
        },
        input_proj_w: zeros(&[hidden, freq * c1]),
    };
    let mut layers = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        let attn = build_attention(
            cfg,
            clip_lin(heads * hd, hidden),
            clip_lin(heads * hd, hidden),
            clip_lin(heads * hd, hidden),
            clip_lin(hidden, hidden),
            zeros(&[heads * hd, hidden]),
            zeros(&[hd]),
        );
        layers.push(ConformerBlock {
            feed_forward1: ff(),
            self_attn: attn,
            lconv1d: ConformerLightConv1d {
                pre_layer_norm: rms(hidden),
                linear_start: clip_lin(hidden * 2, hidden),
                depthwise_conv_w: zeros(&[hidden, cfg.conv_kernel_size as i32, 1]),
                conv_norm: rms(hidden),
                linear_end: clip_lin(hidden, hidden),
                gradient_clipping: cfg.gradient_clipping,
                causal_padding: (cfg.conv_kernel_size - 1) as i32,
                hidden_size: hidden,
            },
            feed_forward2: ff(),
            norm_pre_attn: rms(hidden),
            norm_post_attn: rms(hidden),
            norm_out: rms(hidden),
            gradient_clipping: cfg.gradient_clipping,
        });
    }
    let opd = cfg.output_proj_dims.unwrap() as i32;
    AudioEncoder {
        cfg: cfg.clone(),
        subsample,
        layers,
        output_proj: Some((zeros(&[opd, hidden]), zeros(&[opd]))),
    }
}

/// Primary DoD test: zero-init forward asserts output `[1, T_sub,
/// output_proj_dims]` (T_sub = SSCP subsampling factor) and finiteness.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gemma4_audio_encoder_forward_shape() {
    let device = Device::Cpu;
    let cfg = small_cfg();
    let enc = synth_tower(&cfg);

    let t_frames = 100i32;
    let mel = {
        let n = (t_frames as usize) * SSCP_INPUT_FEAT_SIZE;
        let mut v = vec![0.0f32; n];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i % 97) as f32 / 97.0) - 0.5;
        }
        Array::from_bytes(
            f32_bytes(&v),
            &[1, t_frames, SSCP_INPUT_FEAT_SIZE as i32],
            Dtype::F32,
        )
        .unwrap()
    };
    let mask = zeros(&[1, t_frames]); // all-valid (no padding)

    let out = enc.forward(&mel, &mask, device).expect("audio forward");
    out.eval().expect("eval");

    // SSCP halves T twice: T_sub = ((T+2-3)/2+1) applied twice.
    let mut t_sub = t_frames;
    for _ in 0..2 {
        t_sub = (t_sub + 2 - 3) / 2 + 1;
    }
    let shp = out.shape();
    assert_eq!(shp[0], 1, "batch");
    assert_eq!(
        shp[2],
        cfg.output_proj_dims.unwrap() as i32,
        "output_proj_dims (text_hidden-equivalent audio dim)"
    );
    assert_eq!(shp[1], t_sub, "T_sub subsampling factor");

    let vals = materialize_f32(&out);
    assert_eq!(
        vals.iter().filter(|v| !v.is_finite()).count(),
        0,
        "non-finite values in audio embeddings"
    );
}

/// Real-weights forward (gated on the e4b snapshot, `--ignored`).
#[test]
#[ignore = "requires e4b snapshot; run with --ignored"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gemma4_audio_forward_real_weights() {
    let Some(dir_buf) = e4b_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let dir = dir_buf.as_path();
    if !dir.exists() {
        eprintln!("SKIP: e4b snapshot absent at {}", dir.display());
        return;
    }
    let device = Device::Cpu;
    let cfg = Gemma4AudioConfig::from_model_dir(dir)
        .expect("read audio_config")
        .expect("audio_config present");
    let enc = load_audio_tower(dir, &cfg).expect("load audio tower");

    let t_frames = 200i32;
    let n = (t_frames as usize) * SSCP_INPUT_FEAT_SIZE;
    let mut v = vec![0.0f32; n];
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i % 131) as f32 / 131.0) - 0.5;
    }
    let mel = Array::from_bytes(
        f32_bytes(&v),
        &[1, t_frames, SSCP_INPUT_FEAT_SIZE as i32],
        Dtype::F32,
    )
    .unwrap();
    let mask = zeros(&[1, t_frames]);

    let out = enc.forward(&mel, &mask, device).expect("audio forward");
    out.eval().expect("eval");
    let shp = out.shape();
    assert_eq!(shp[0], 1);
    assert_eq!(shp[2], cfg.output_proj_dims.unwrap() as i32);
    let vals = materialize_f32(&out);
    assert_eq!(
        vals.iter().filter(|v| !v.is_finite()).count(),
        0,
        "non-finite real-weight audio embeddings"
    );
    let max_abs = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(max_abs > 0.0, "real audio embeddings all zero");
    eprintln!("gemma4 audio (real): shape={shp:?} max_abs={max_abs:.4}");
}
