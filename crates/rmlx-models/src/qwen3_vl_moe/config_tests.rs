use super::*;

fn target_config() -> serde_json::Value {
    // The exact nested shape of the target snapshot's config.json
    // (mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit).
    serde_json::json!({
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "text_config": {
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_theta": 5000000,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            },
            "head_dim": 128,
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
            "norm_topk_prob": true,
            "mlp_only_layers": [],
            "tie_word_embeddings": false
        },
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 2048,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [8, 16, 24]
        }
    })
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn parses_target_snapshot_config() {
    let raw = target_config();
    let cfg = Qwen3VlMoeConfig::from_raw(
        raw.as_object().unwrap(),
        (64, 4, "affine".to_string(), HashMap::new()),
    )
    .expect("parse config");

    // Text decoder — plain Qwen3-MoE.
    assert_eq!(cfg.text.num_hidden_layers, 48);
    assert_eq!(cfg.text.hidden_size, 2048);
    assert_eq!(cfg.text.num_attention_heads, 32);
    assert_eq!(cfg.text.num_key_value_heads, 4);
    assert_eq!(cfg.text.head_dim, 128);
    assert_eq!(cfg.text.vocab_size, 151936);
    assert_eq!(cfg.text.num_experts, 128);
    assert_eq!(cfg.text.num_experts_per_tok, 8);
    assert_eq!(cfg.text.moe_intermediate_size, 768);
    assert_eq!(cfg.text.decoder_sparse_step, 1);
    assert!(cfg.text.mlp_only_layers.is_empty());
    assert_eq!(cfg.text.rope_theta, 5_000_000.0);
    assert_eq!(cfg.text.mrope_section, vec![24, 20, 20]);
    assert!(cfg.text.mrope_interleaved);
    // mrope_section sums to head_dim/2.
    assert_eq!(
        cfg.text.mrope_section.iter().sum::<usize>(),
        cfg.text.head_dim / 2
    );

    // Vision tower — Qwen3-VL ViT.
    assert_eq!(cfg.vision.depth, 27);
    assert_eq!(cfg.vision.hidden_size, 1152);
    assert_eq!(cfg.vision.out_hidden_size, 2048);
    assert_eq!(cfg.vision.patch_size, 16);
    assert_eq!(cfg.vision.spatial_merge_size, 2);
    assert_eq!(cfg.vision.deepstack_visual_indexes, vec![8, 16, 24]);

    // Token ids.
    assert_eq!(cfg.image_token_id, 151655);
    assert_eq!(cfg.vision_start_token_id, 151652);
    assert_eq!(cfg.vision_end_token_id, 151653);

    // Vision out_hidden must match text hidden (scatter target).
    assert_eq!(cfg.vision.out_hidden_size, cfg.text.hidden_size);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rejects_missing_text_config() {
    let raw = serde_json::json!({ "vision_config": {} });
    assert!(Qwen3VlMoeConfig::from_raw(
        raw.as_object().unwrap(),
        (64, 4, "affine".to_string(), HashMap::new())
    )
    .is_err());
}
