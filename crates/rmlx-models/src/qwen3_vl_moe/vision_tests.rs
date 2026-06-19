use super::*;

/// Synthetic-weight ViT forward: build a tiny tower with deterministic
/// weights and check the output shape + finiteness. No model needed.
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
fn qwen3_vl_moe_vision_tower_shape() {
    let device = Device::Cpu;
    let cfg = Qwen3VlMoeVisionConfig {
        depth: 2,
        hidden_size: 8,
        intermediate_size: 16,
        out_hidden_size: 8,
        num_heads: 2,
        in_channels: 3,
        patch_size: 16,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        num_position_embeddings: 16,
        deepstack_visual_indexes: vec![0],
        layer_norm_eps: 1e-6,
    };
    let hidden = cfg.hidden_size;
    let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let ngps = 4usize;
    let feat_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;

    let ones = |n: usize| -> Array { f32_arr(&vec![0.01f32; n], &[n as i32]).unwrap() };
    let mat = |out: usize, inn: usize| -> Array {
        let mut d = vec![0.0f32; out * inn];
        for (k, v) in d.iter_mut().enumerate() {
            *v = ((k % 7) as f32 - 3.0) * 0.01;
        }
        f32_arr(&d, &[out as i32, inn as i32]).unwrap()
    };
    let lin = |out: usize, inn: usize| -> Linear {
        Linear {
            weight: mat(out, inn),
            bias: Some(ones(out)),
        }
    };
    let ln = |dim: usize| -> LayerNorm {
        LayerNorm {
            weight: f32_arr(&vec![1.0f32; dim], &[dim as i32]).unwrap(),
            bias: f32_arr(&vec![0.0f32; dim], &[dim as i32]).unwrap(),
            eps: 1e-6,
        }
    };

    let merged_dim = hidden * merge_unit;
    let block = || -> Block {
        Block {
            norm1: ln(hidden),
            norm2: ln(hidden),
            attn: Attention {
                qkv: lin(3 * hidden, hidden),
                proj: lin(hidden, hidden),
                num_heads: cfg.num_heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
            },
            mlp: Mlp {
                linear_fc1: lin(cfg.intermediate_size, hidden),
                linear_fc2: lin(hidden, cfg.intermediate_size),
            },
        }
    };

    let vis = Qwen3VlMoeVision {
        cfg: cfg.clone(),
        patch_embed_w: mat(hidden, feat_len),
        patch_embed_b: ones(hidden),
        pos_embed: mat(ngps * ngps, hidden),
        blocks: vec![block(), block()],
        merger: PatchMerger {
            norm: ln(hidden),
            linear_fc1: lin(merged_dim, merged_dim),
            linear_fc2: lin(cfg.out_hidden_size, merged_dim),
            merged_dim,
            use_postshuffle_norm: false,
        },
        deepstack_mergers: vec![PatchMerger {
            norm: ln(merged_dim),
            linear_fc1: lin(merged_dim, merged_dim),
            linear_fc2: lin(cfg.out_hidden_size, merged_dim),
            merged_dim,
            use_postshuffle_norm: true,
        }],
        head_dim,
        num_grid_per_side: ngps,
    };

    let (gt, gh, gw) = (1usize, 4usize, 4usize);
    let num_patches = gt * gh * gw;
    let pv = vec![0.05f32; num_patches * feat_len];
    let out = vis
        .forward(&pv, (gt, gh, gw), device)
        .expect("vision forward");
    Array::eval(&out.image_embeds).expect("eval");

    let expect_merged = num_patches / merge_unit;
    assert_eq!(
        out.image_embeds.shape(),
        vec![expect_merged as i32, cfg.out_hidden_size as i32],
        "merged embeds must be [num_patches/merge^2, out_hidden]"
    );
    assert_eq!(out.deepstack_embeds.len(), 1, "one deepstack embed");
    assert_eq!(
        out.deepstack_embeds[0].shape(),
        vec![expect_merged as i32, cfg.out_hidden_size as i32]
    );

    let vals: Vec<f32> = out
        .image_embeds
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals.iter().filter(|v| !v.is_finite()).count(), 0);
}

/// Query-tiled ViT attention must equal one SDPA over all queries: each query
/// attends to every key in both, so tiling the query dim only changes the
/// command-buffer boundaries, not the math. Drive the tiling path with a small
/// `budget` so `seq` stays tiny, then compare against a single SDPA.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: all unwraps are on values constructed locally in this fn"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "test fixture: indices bounded by locally-constructed shapes"
)]
fn qwen3_vl_moe_vision_attn_tiling_matches_single_sdpa() {
    let device = Device::Cpu;
    let (h, seq, d) = (2i32, 9i32, 4i32);
    let scale = (d as f32).powf(-0.5);

    // Deterministic q/k/v in [1, H, seq, D].
    let n = (h * seq * d) as usize;
    let fill = |off: usize| -> Array {
        let data: Vec<f32> = (0..n)
            .map(|i| (((i + off) % 13) as f32 - 6.0) * 0.05)
            .collect();
        f32_arr(&data, &[1, h, seq, d]).unwrap()
    };
    let q = fill(0);
    let k = fill(3);
    let v = fill(7);

    // qkv/proj are unused by attend_tiled (it takes q/k/v directly); 1x1 dummies.
    let dummy_w = f32_arr(&[0.0f32], &[1, 1]).unwrap();
    let attn = Attention {
        qkv: Linear {
            weight: dummy_w.try_clone().unwrap(),
            bias: None,
        },
        proj: Linear {
            weight: dummy_w,
            bias: None,
        },
        num_heads: h as usize,
        head_dim: d as usize,
        scale,
    };

    // Reference: one SDPA over all queries.
    let reference = scaled_dot_product_attention(&q, &k, &v, scale, "", None, device).unwrap();
    Array::eval(&reference).unwrap();

    // Tiled: budget below seq*seq (=81) forces tiling (tile = budget/seq rows).
    // budget=12, seq=9 -> tile=1, so every query is its own command buffer.
    let tiled = attn
        .attend_tiled(&q, &k, &v, seq, h, d, 12, device)
        .unwrap();
    Array::eval(&tiled).unwrap();

    assert_eq!(tiled.shape(), reference.shape());
    let rv: Vec<f32> = reference
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let tv: Vec<f32> = tiled
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(rv.len(), tv.len());
    for (a, b) in rv.iter().zip(tv.iter()) {
        assert!(
            (a - b).abs() <= 1e-5,
            "tiled attention diverged from single SDPA: {a} vs {b}"
        );
    }

    // Budget above seq*seq must collapse to the single-SDPA path (bit-identical
    // small-image behavior).
    let untiled = attn
        .attend_tiled(&q, &k, &v, seq, h, d, 10_000, device)
        .unwrap();
    Array::eval(&untiled).unwrap();
    let uv: Vec<f32> = untiled
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (a, b) in rv.iter().zip(uv.iter()) {
        assert!((a - b).abs() <= 1e-6, "untiled path must match reference");
    }
}
