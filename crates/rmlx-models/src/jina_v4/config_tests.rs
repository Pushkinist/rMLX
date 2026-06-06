use super::*;

fn jina_v4_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from)
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
fn parses_real_config_json() {
    let Some(dir_buf) = jina_v4_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let config_path = dir_buf.join("config.json");
    if !config_path.exists() {
        eprintln!(
            "SKIP: model not present at {}; skipping live config parse test",
            dir_buf.display()
        );
        return;
    }

    let cfg = JinaV4Config::from_file(&config_path).expect("failed to parse jina-v4 config.json");

    // text_config assertions
    let tc = &cfg.text_config;
    assert_eq!(tc.hidden_size, 2048, "text hidden_size");
    assert_eq!(tc.num_hidden_layers, 36, "num_hidden_layers");
    assert_eq!(tc.num_attention_heads, 16, "num_attention_heads");
    assert_eq!(tc.num_key_value_heads, 2, "num_key_value_heads (GQA 8:1)");
    assert_eq!(tc.intermediate_size, 11008, "intermediate_size");
    assert!((tc.rms_norm_eps - 1e-6_f32).abs() < 1e-10, "rms_norm_eps");
    assert_eq!(tc.rope_theta as u64, 1_000_000, "rope_theta");
    // head_dim: inferred as 2048/16 = 128
    assert_eq!(tc.head_dim, 128, "head_dim (inferred)");
    assert_eq!(tc.vocab_size, 151936, "vocab_size");
    assert!(!tc.use_sliding_window, "use_sliding_window should be false");
    assert_eq!(
        tc.max_position_embeddings, 128000,
        "max_position_embeddings"
    );

    // vision_config assertions
    let vc = &cfg.vision_config;
    assert_eq!(vc.depth, 32, "vision depth");
    assert_eq!(vc.hidden_size, 1280, "vision hidden_size");
    assert_eq!(vc.intermediate_size, 3420, "vision intermediate_size");
    assert_eq!(
        vc.fullatt_block_indexes,
        vec![7, 15, 23, 31],
        "fullatt_block_indexes"
    );
    assert_eq!(vc.window_size, 112, "vision window_size");
    assert_eq!(vc.patch_size, 14, "patch_size");
    assert_eq!(vc.spatial_merge_size, 2, "spatial_merge_size");

    // top-level metadata
    assert_eq!(cfg.single_vector_pool_strategy, "mean");
    assert_eq!(cfg.multi_vector_projector_dim, 128);
    assert_eq!(
        cfg.matryoshka_dims,
        vec![128, 256, 512, 1024, 2048],
        "matryoshka_dims"
    );
    assert_eq!(cfg.task_names.len(), 3, "task_names len");
    assert_eq!(cfg.task_names[0], "retrieval");
    assert_eq!(cfg.task_names[1], "text-matching");
    assert_eq!(cfg.task_names[2], "code");
    // 3D M-RoPE section (sums to head_dim/2 = 64).
    assert_eq!(
        cfg.mrope_section,
        Some(vec![16, 24, 24]),
        "rope_scaling.mrope_section"
    );

    // special token ids
    assert_eq!(cfg.vision_start_token_id, 151652);
    assert_eq!(cfg.vision_end_token_id, 151653);
    assert_eq!(cfg.vision_token_id, 151654);
    assert_eq!(cfg.bos_token_id, 151643);
    assert_eq!(cfg.eos_token_id, 151645);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn parses_minimal_inline_json() {
    // Verifies defaults + forward-compat (unknown keys silently ignored).
    let json = serde_json::json!({
        "architectures": ["JinaEmbeddingsV4Model"],
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "vocab_size": 151936,
            "use_sliding_window": false,
            "max_position_embeddings": 128000,
            "rope_scaling": {"mrope_section": [16, 24, 24], "rope_type": "default"},
            "future_key_ignored": true
        },
        "vision_config": {
            "depth": 32,
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "num_heads": 16,
            "out_hidden_size": 2048,
            "fullatt_block_indexes": [7, 15, 23, 31],
            "window_size": 112,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3
        },
        "single_vector_pool_strategy": "mean",
        "multi_vector_projector_dim": 128,
        "matryoshka_dims": [128, 256, 512, 1024, 2048],
        "task_names": ["retrieval", "text-matching", "code"],
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "vision_token_id": 151654,
        "image_token_id": 151655,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "another_future_key": 42
    });

    let cfg = JinaV4Config::from_json(&json).expect("inline parse failed");
    assert_eq!(cfg.text_config.hidden_size, 2048);
    assert_eq!(cfg.text_config.head_dim, 128); // 2048/16
    assert_eq!(cfg.matryoshka_dims, vec![128, 256, 512, 1024, 2048]);
    assert_eq!(cfg.task_names.len(), 3);
    assert_eq!(cfg.vision_config.depth, 32);
    assert_eq!(cfg.multi_vector_projector_dim, 128);
    assert_eq!(cfg.vision_start_token_id, 151652);
    assert_eq!(cfg.image_token_id, 151655);
    assert_eq!(cfg.mrope_section, Some(vec![16, 24, 24]));
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn mrope_section_absent_is_none() {
    // Text-only config with no rope_scaling -> mrope_section None.
    let json = serde_json::json!({
        "text_config": {"hidden_size": 2048, "num_attention_heads": 16},
        "vision_config": {}
    });
    let cfg = JinaV4Config::from_json(&json).expect("parse");
    assert_eq!(cfg.mrope_section, None);
}
