use super::*;
use crate::jina_v4::config::JinaV4Config;
use crate::jina_v4::preprocess::{preprocess_image_bytes, ImagePreprocessConfig};

fn jina_v4_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from)
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn synth_png(w: u32, h: u32) -> Vec<u8> {
    let mut img = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(
                x,
                y,
                image::Rgb([
                    (x * 251 % 256) as u8,
                    (y * 193 % 256) as u8,
                    ((x ^ y) % 256) as u8,
                ]),
            );
        }
    }
    let mut png = Vec::new();
    {
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .expect("encode test PNG");
    }
    png
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

fn nonzero_bias_count(b: &Option<Array>) -> usize {
    let Some(arr) = b else {
        return 0;
    };
    materialize_f32(arr)
        .iter()
        .filter(|&&v| v.abs() > 0.0)
        .count()
}

// ----- pure host geometry (no model needed) ---------------------------

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
fn geometry_window_index_is_a_permutation() {
    let cfg = JinaV4Config::from_json(&serde_json::json!({
        "text_config": {}, "vision_config": {}
    }))
    .expect("default config");
    let vc = &cfg.vision_config;
    let head_dim = vc.hidden_size / vc.num_heads; // 80
                                                  // 8x12 patch grid (grid divisible by merge_size=2).
    let geo = compute_geometry(vc, 1, 8, 12, head_dim).expect("geometry");

    let mu = vc.spatial_merge_size * vc.spatial_merge_size;
    let n_groups = (8 * 12) / mu;
    assert_eq!(geo.window_index.len(), n_groups);
    assert_eq!(geo.reverse_index.len(), n_groups);

    // window_index is a permutation of 0..n_groups.
    let mut seen = vec![false; n_groups];
    for &i in &geo.window_index {
        assert!((0..n_groups as i32).contains(&i), "oob window idx {i}");
        assert!(!seen[i as usize], "duplicate window idx {i}");
        seen[i as usize] = true;
    }
    assert!(seen.iter().all(|&b| b), "window_index not surjective");

    // reverse_index o window_index == identity.
    for g in 0..n_groups {
        let rev = geo.reverse_index[g] as usize;
        assert_eq!(
            geo.window_index[rev] as usize, g,
            "reverse_index is not argsort(window_index)"
        );
    }

    // cu_seqlens (full) covers the whole sequence.
    assert_eq!(*geo.cu_seqlens.first().unwrap(), 0);
    assert_eq!(*geo.cu_seqlens.last().unwrap(), 8 * 12);
    // cu_window_seqlens is monotone, dedup'd, ends at seq.
    assert_eq!(*geo.cu_window_seqlens.first().unwrap(), 0);
    assert_eq!(*geo.cu_window_seqlens.last().unwrap(), 8 * 12);
    for w in geo.cu_window_seqlens.windows(2) {
        assert!(
            w[1] > w[0],
            "cu_window_seqlens not strictly increasing after dedup"
        );
    }

    // cos/sin tables have unit-circle magnitude (cos^2+sin^2 == 1).
    let n = (8 * 12) * head_dim;
    assert_eq!(geo.cos.len(), n);
    assert_eq!(geo.sin.len(), n);
    for k in 0..n {
        let m = geo.cos[k].mul_add(geo.cos[k], geo.sin[k] * geo.sin[k]);
        assert!((m - 1.0).abs() < 1e-4, "cos^2+sin^2 != 1 at {k}: {m}");
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn block_mask_is_block_diagonal() {
    let cu = [0usize, 3, 5];
    let m = build_block_mask(&cu, 5, Device::Cpu).expect("mask");
    let v = materialize_f32(&m);
    assert_eq!(v.len(), 25);
    let at = |r: usize, c: usize| v[r * 5 + c];
    // in-block (0..3)x(0..3) and (3..5)x(3..5) == 0; cross == very negative
    assert_eq!(at(0, 0), 0.0);
    assert_eq!(at(2, 2), 0.0);
    assert_eq!(at(3, 4), 0.0);
    assert!(at(0, 3) < -1.0e6, "cross-block must be masked");
    assert!(at(4, 0) < -1.0e6, "cross-block must be masked");
}

// ----- end-to-end (gated on the on-disk snapshot) ---------------------

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn load_vision() -> Option<(JinaV4Vision, Device)> {
    let Some(dir_buf) = jina_v4_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_JINA_V4 not set");
        return None;
    };
    let dir = dir_buf.as_path();
    if !dir.exists() {
        eprintln!("SKIP: jina-v4 snapshot absent at {}", dir.display());
        return None;
    }
    let cfg = JinaV4Config::from_file(&dir.join("config.json")).expect("config");
    let v = load_vision_tower(dir, &cfg.vision_config).expect("load vision tower");
    Some((v, Device::Cpu))
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn forward_shape_finite_deterministic() {
    let Some((vis, dev)) = load_vision() else {
        return;
    };
    let pcfg = ImagePreprocessConfig::default();
    // 84x56 image -> smart_resize(56,84,28,..) = 56x84 -> grid 4x6, t=1.
    let png = synth_png(84, 56);
    let pv = preprocess_image_bytes(&png, &pcfg).expect("preprocess");

    let merge = vis.config().spatial_merge_size;
    let expect_merged = pv.num_patches / (merge * merge);

    let out = vis.forward(&pv, dev).expect("vision forward");
    out.eval().expect("eval");
    assert_eq!(
        out.shape(),
        vec![expect_merged as i32, vis.config().out_hidden_size as i32],
        "merged image embedding must be [num_patches/merge^2, out_hidden(2048)]"
    );

    let vals = materialize_f32(&out);
    assert_eq!(vals.len(), expect_merged * vis.config().out_hidden_size);
    assert_eq!(
        vals.iter().filter(|v| !v.is_finite()).count(),
        0,
        "non-finite values in vision output"
    );
    let max_abs = vals.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max_abs > 0.0, "vision output all zero");

    // Deterministic: identical input -> bit-identical output.
    let out2 = vis.forward(&pv, dev).expect("vision forward 2");
    out2.eval().expect("eval 2");
    assert_eq!(
        vals,
        materialize_f32(&out2),
        "vision forward not deterministic"
    );

    eprintln!(
        "jina-v4 vision: shape={:?} merged={} max_abs={:.4}",
        out.shape(),
        expect_merged,
        max_abs
    );
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn bias_trap_guard_biases_present_and_nonzero() {
    let Some((vis, _)) = load_vision() else {
        return;
    };
    // Every block must carry non-zero biases on mlp.{gate,up,down}_proj
    // and attn.proj (the jina bias trap — stock mlx_vlm has bias=False).
    for (i, blk) in vis.blocks.iter().enumerate() {
        assert!(
            blk.attn.proj.bias.is_some(),
            "block {i}: attn.proj.bias missing (bias trap not avoided)"
        );
        assert!(
            blk.attn.qkv.bias.is_some(),
            "block {i}: attn.qkv.bias missing"
        );
        assert!(
            blk.mlp.gate_proj.bias.is_some()
                && blk.mlp.up_proj.bias.is_some()
                && blk.mlp.down_proj.bias.is_some(),
            "block {i}: mlp.*_proj.bias missing (bias trap not avoided)"
        );
        // Non-zero check on a representative subset (every 8th block to
        // keep the test fast — proves the tensors are real, not zeros).
        if i % 8 == 0 {
            assert!(
                nonzero_bias_count(&blk.attn.proj.bias) > 0,
                "block {i}: attn.proj.bias is all-zero"
            );
            assert!(
                nonzero_bias_count(&blk.mlp.down_proj.bias) > 0,
                "block {i}: mlp.down_proj.bias is all-zero"
            );
        }
    }
    // PatchMerger biases (mlp.0 / mlp.2) present + non-zero.
    assert!(
        nonzero_bias_count(&vis.merger.mlp0.bias) > 0,
        "merger.mlp.0.bias missing/zero"
    );
    assert!(
        nonzero_bias_count(&vis.merger.mlp2.bias) > 0,
        "merger.mlp.2.bias missing/zero"
    );
    // PatchEmbed proj has NO bias (the only Linear without one).
    assert!(
        vis.patch_embed_w.shape()[0] == vis.config().hidden_size as i32,
        "patch_embed flat weight first dim must be hidden_size"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn window_and_full_blocks_both_exercised() {
    let Some((vis, dev)) = load_vision() else {
        return;
    };
    // fullatt_block_indexes must intersect AND not cover all blocks, so
    // both the window-mask and full-mask paths run in one forward.
    let fa = &vis.config().fullatt_block_indexes;
    assert!(!fa.is_empty(), "no full-attention blocks configured");
    assert!(
        fa.len() < vis.blocks.len(),
        "all blocks are full-attention — window path never exercised"
    );
    assert!(
        fa.iter().any(|&i| i < vis.blocks.len()),
        "fullatt index out of range"
    );
    // A forward over a multi-window grid runs both dispatch arms without
    // panicking and yields finite output.
    let pcfg = ImagePreprocessConfig::default();
    let png = synth_png(140, 112); // larger grid -> multiple windows
    let pv = preprocess_image_bytes(&png, &pcfg).expect("preprocess");
    let out = vis.forward(&pv, dev).expect("forward (window+full)");
    out.eval().expect("eval");
    assert_eq!(out.shape()[1], vis.config().out_hidden_size as i32);
    assert_eq!(
        materialize_f32(&out)
            .iter()
            .filter(|v| !v.is_finite())
            .count(),
        0
    );
}
