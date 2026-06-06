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
