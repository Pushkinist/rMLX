use super::*;

fn medgemma_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_MEDGEMMA").map(std::path::PathBuf::from)
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

/// Small synthetic config: 32x32 image, patch 8 -> 4x4=16 patches,
/// pool kernel 2 -> 2x2=4 soft tokens. Exercises every op end-to-end.
fn synth_cfg() -> Gemma3VisionConfig {
    Gemma3VisionConfig {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        head_dim: 8,
        patch_size: 8,
        image_size: 32,
        num_channels: 3,
        layer_norm_eps: 1e-6,
        mm_tokens_per_image: 4,
        text_hidden_size: 24,
        mm_norm_eps: 1e-6,
    }
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn synth_tower(cfg: &Gemma3VisionConfig) -> (VisionModel, MultiModalProjector) {
    let zeros = |shape: &[i32]| -> Array {
        let n: i32 = shape.iter().product();
        Array::from_bytes(f32_bytes(&vec![0.0_f32; n as usize]), shape, Dtype::F32).unwrap()
    };
    let ones = |shape: &[i32]| -> Array {
        let n: i32 = shape.iter().product();
        Array::from_bytes(f32_bytes(&vec![1.0_f32; n as usize]), shape, Dtype::F32).unwrap()
    };
    let hidden = cfg.hidden_size as i32;
    let inter = cfg.intermediate_size as i32;
    let pps = cfg.patches_per_side() as i32;
    let lin = |out: i32, inn: i32| Linear {
        weight: zeros(&[out, inn]),
        bias: zeros(&[out]),
    };
    let ln = |dim: i32| LayerNorm {
        weight: ones(&[dim]),
        bias: zeros(&[dim]),
        eps: cfg.layer_norm_eps,
    };
    let scale = (cfg.head_dim as f32).powf(-0.5);
    let mut encoder = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        encoder.push(EncoderLayer {
            layer_norm1: ln(hidden),
            layer_norm2: ln(hidden),
            attn: Attention {
                q_proj: lin(hidden, hidden),
                k_proj: lin(hidden, hidden),
                v_proj: lin(hidden, hidden),
                out_proj: lin(hidden, hidden),
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim,
                scale,
            },
            mlp: Mlp {
                fc1: lin(inter, hidden),
                fc2: lin(hidden, inter),
            },
        });
    }
    let vision = VisionModel {
        cfg: cfg.clone(),
        patch_embedding_w: zeros(&[hidden, cfg.patch_size as i32, cfg.patch_size as i32, 3]),
        patch_embedding_b: zeros(&[hidden]),
        position_embedding: zeros(&[pps * pps, hidden]),
        encoder,
        post_layernorm: ln(hidden),
    };
    let projector = MultiModalProjector {
        soft_emb_norm_w: ones(&[hidden]),
        input_projection_w: zeros(&[hidden, cfg.text_hidden_size as i32]),
        norm_eps: cfg.mm_norm_eps,
        patches_per_side: cfg.patches_per_side(),
        tokens_per_side: cfg.tokens_per_side(),
        pool_kernel: cfg.pool_kernel(),
    };
    (vision, projector)
}

fn synth_pixels(cfg: &Gemma3VisionConfig) -> Gemma3PixelValues {
    let s = cfg.image_size;
    let n = 3 * s * s;
    let mut pv = vec![0.0_f32; n];
    for (i, x) in pv.iter_mut().enumerate() {
        *x = ((i % 255) as f32) / 255.0 - 0.5;
    }
    Gemma3PixelValues {
        pixel_values: pv,
        height: s,
        width: s,
        num_soft_tokens: cfg.mm_tokens_per_image,
    }
}

/// Primary DoD test: synthetic forward asserts vision -> projector ->
/// [1, mm_tokens_per_image, text_hidden] is finite. Runs without snapshot.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma3_vision_forward_shape() {
    let device = Device::Cpu;
    let cfg = synth_cfg();
    let (vision, projector) = synth_tower(&cfg);
    let pv = synth_pixels(&cfg);

    let vout = vision.forward(&pv, device).expect("vision forward");
    vout.eval().expect("eval vision");
    assert_eq!(
        vout.shape(),
        vec![
            1,
            (cfg.patches_per_side() * cfg.patches_per_side()) as i32,
            cfg.hidden_size as i32
        ],
        "vision output [1, num_patches, hidden]"
    );

    let feats = projector.forward(&vout, device).expect("projector");
    feats.eval().expect("eval feats");
    assert_eq!(
        feats.shape(),
        vec![
            1,
            cfg.mm_tokens_per_image as i32,
            cfg.text_hidden_size as i32
        ],
        "projected image features [1, mm_tokens_per_image, text_hidden]"
    );
    let vals = materialize_f32(&feats);
    assert_eq!(
        vals.iter().filter(|v| !v.is_finite()).count(),
        0,
        "non-finite values in image features"
    );
}

/// Real-weights forward (gated on the medgemma snapshot, `--ignored`).
#[test]
#[ignore = "requires medgemma snapshot; run with --ignored"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma3_vision_forward_real_weights() {
    let Some(dir_buf) = medgemma_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_MEDGEMMA not set");
        return;
    };
    let dir = dir_buf.as_path();
    if !dir.exists() {
        eprintln!("SKIP: medgemma snapshot absent at {}", dir.display());
        return;
    }
    let device = Device::Cpu;
    let cfg = Gemma3VisionConfig::from_model_dir(dir)
        .expect("read vision_config")
        .expect("vision_config present");
    let (vision, projector) = load_vision_tower(dir, &cfg).expect("load vision tower");
    let proc = Gemma3ImageProcessor::from_model_dir(dir).expect("processor");

    let (iw, ih) = (640u32, 640u32);
    let img = image::RgbImage::from_pixel(iw, ih, image::Rgb([200u8, 60u8, 60u8]));
    let mut png = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), iw, ih, image::ExtendedColorType::Rgb8)
            .expect("encode");
    }
    let pv = proc.preprocess(&png).expect("preprocess");
    assert_eq!(pv.num_soft_tokens, cfg.mm_tokens_per_image);

    let vout = vision.forward(&pv, device).expect("vision forward");
    vout.eval().expect("eval");
    assert_eq!(
        vout.shape(),
        vec![
            1,
            (cfg.patches_per_side() * cfg.patches_per_side()) as i32,
            cfg.hidden_size as i32
        ]
    );

    let feats = projector.forward(&vout, device).expect("projector");
    feats.eval().expect("eval");
    assert_eq!(
        feats.shape(),
        vec![
            1,
            cfg.mm_tokens_per_image as i32,
            cfg.text_hidden_size as i32
        ]
    );
    let vals = materialize_f32(&feats);
    assert_eq!(vals.iter().filter(|v| !v.is_finite()).count(), 0);
    let max_abs = vals.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max_abs > 0.0, "real vision features all zero");
    eprintln!(
        "gemma3 vision (real): shape={:?} soft={} max_abs={:.4}",
        feats.shape(),
        cfg.mm_tokens_per_image,
        max_abs
    );
}
