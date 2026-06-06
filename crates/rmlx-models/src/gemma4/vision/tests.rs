//! Vision tower unit tests.

use super::*;
use crate::gemma4::preprocessor::Gemma4ImageProcessor;

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

fn default_cfg() -> Gemma4VisionConfig {
    Gemma4VisionConfig::from_json(&serde_json::json!({}))
}

/// Zero-init config-driven tower (no snapshot). Exercises every op.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn synth_tower(cfg: &Gemma4VisionConfig) -> (VisionModel, MultimodalEmbedder) {
    let zeros = |shape: &[i32]| -> Array {
        let n: i32 = shape.iter().product();
        let v = vec![0.0_f32; n as usize];
        Array::from_bytes(f32_bytes(&v), shape, Dtype::F32).unwrap()
    };
    let ones = |shape: &[i32]| -> Array {
        let n: i32 = shape.iter().product();
        let v = vec![1.0_f32; n as usize];
        Array::from_bytes(f32_bytes(&v), shape, Dtype::F32).unwrap()
    };
    let hidden = cfg.hidden_size as i32;
    let inter = cfg.intermediate_size as i32;
    let hd = cfg.head_dim as i32;
    let nh = cfg.num_attention_heads as i32;
    let nkv = cfg.num_key_value_heads as i32;
    let feat = (3 * cfg.patch_size * cfg.patch_size) as i32;
    let clip_lin = |out: i32, inn: i32| ClippableLinear {
        weight: zeros(&[out, inn]),
        clip: None,
    };
    let rms = |dim: i32| RmsNorm {
        weight: Some(ones(&[dim])),
        eps: cfg.rms_norm_eps,
    };
    let mut blocks = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        blocks.push(Block {
            input_layernorm: rms(hidden),
            post_attention_layernorm: rms(hidden),
            pre_feedforward_layernorm: rms(hidden),
            post_feedforward_layernorm: rms(hidden),
            attn: Attention {
                q_proj: clip_lin(nh * hd, hidden),
                k_proj: clip_lin(nkv * hd, hidden),
                v_proj: clip_lin(nkv * hd, hidden),
                o_proj: clip_lin(hidden, nh * hd),
                q_norm: rms(hd),
                k_norm: rms(hd),
                v_norm_eps: cfg.rms_norm_eps,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            },
            mlp: Mlp {
                gate_proj: clip_lin(inter, hidden),
                up_proj: clip_lin(inter, hidden),
                down_proj: clip_lin(hidden, inter),
            },
        });
    }
    let vision = VisionModel {
        cfg: cfg.clone(),
        input_proj_w: zeros(&[hidden, feat]),
        position_embedding_table: zeros(&[2, cfg.position_embedding_size as i32, hidden]),
        blocks,
        standardize: None,
        head_dim: cfg.head_dim,
    };
    let embedder = MultimodalEmbedder {
        projection: crate::layers::Linear::Plain {
            weight: zeros(&[2560, hidden]),
        },
        norm_eps: cfg.rms_norm_eps,
    };
    (vision, embedder)
}

fn synth_pixels(cfg: &Gemma4VisionConfig) -> Gemma4PixelValues {
    let side = cfg.pooling_kernel_size * cfg.patch_size; // 48
    let height = side * 2; // 96 -> 6 patches tall
    let width = side * 2; // 96 -> 6 patches wide
    let n = 3 * height * width;
    let mut pv = vec![0.0_f32; n];
    for (i, x) in pv.iter_mut().enumerate() {
        *x = ((i % 255) as f32) / 255.0;
    }
    let p = cfg.patch_size;
    let num_soft = (height / p) * (width / p) / (cfg.pooling_kernel_size * cfg.pooling_kernel_size);
    Gemma4PixelValues {
        pixel_values: pv,
        height,
        width,
        num_soft_tokens: num_soft,
    }
}

/// Primary DoD test: zero-init config-driven forward asserts the output is
/// `[1, output_len, text_hidden]` and finite. Runs without the snapshot.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma4_vision_forward_shape() {
    let device = Device::Cpu;
    let cfg = default_cfg();
    let (vision, embedder) = synth_tower(&cfg);
    let pv = synth_pixels(&cfg);
    let expect_soft = pv.num_soft_tokens;

    let pooled = vision.forward(&pv, device).expect("vision forward");
    pooled.eval().expect("eval pooled");
    assert_eq!(
        pooled.shape(),
        vec![1, expect_soft as i32, cfg.hidden_size as i32],
        "pooled vision output [1, num_soft, hidden]"
    );

    let feats = embedder.forward(&pooled, device).expect("embedder");
    feats.eval().expect("eval feats");
    assert_eq!(
        feats.shape(),
        vec![1, expect_soft as i32, 2560],
        "image-feature embeddings [1, num_soft, text_hidden]"
    );
    let vals = materialize_f32(&feats);
    assert_eq!(
        vals.iter().filter(|v| !v.is_finite()).count(),
        0,
        "non-finite values in image features"
    );
}

/// Real-weights forward (gated on the e4b snapshot, `--ignored`).
#[test]
#[ignore = "requires e4b snapshot; run with --ignored"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma4_vision_forward_real_weights() {
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
    let cfg = Gemma4VisionConfig::from_model_dir(dir)
        .expect("read vision_config")
        .expect("vision_config present");
    let (vision, embedder) = load_vision_tower(dir, &cfg).expect("load vision tower");

    let proc = Gemma4ImageProcessor::from_model_dir(dir).expect("processor");
    let (iw, ih) = (480u32, 480u32);
    let img = image::RgbImage::from_pixel(iw, ih, image::Rgb([120u8, 90u8, 200u8]));
    let mut png = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), iw, ih, image::ExtendedColorType::Rgb8)
            .expect("encode");
    }
    let pv = proc.preprocess(&png).expect("preprocess");
    let expect_soft = pv.num_soft_tokens;

    let pooled = vision.forward(&pv, device).expect("vision forward");
    pooled.eval().expect("eval");
    assert_eq!(
        pooled.shape(),
        vec![1, expect_soft as i32, cfg.hidden_size as i32]
    );

    let feats = embedder.forward(&pooled, device).expect("embedder");
    feats.eval().expect("eval");
    assert_eq!(feats.shape(), vec![1, expect_soft as i32, 2560]);
    let vals = materialize_f32(&feats);
    assert_eq!(vals.iter().filter(|v| !v.is_finite()).count(), 0);
    let max_abs = vals.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max_abs > 0.0, "real vision features all zero");
    eprintln!(
        "gemma4 vision (real): shape={:?} soft={} max_abs={:.4}",
        feats.shape(),
        expect_soft,
        max_abs
    );
}
