//! BitNet unit tests.

#[test]
fn test_bitnet_config_from_json() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate the actual BitNet config.json content.
    let raw_json = serde_json::json!({
        "architectures": ["BitNetForCausalLM"],
        "hidden_size": 2560,
        "num_hidden_layers": 30,
        "num_attention_heads": 20,
        "num_key_value_heads": 5,
        "intermediate_size": 6912,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": true,
        "max_position_embeddings": 4096,
        "model_type": "bitnet"
    });

    let cfg_raw: rmlx_loader::ModelConfig = serde_json::from_value(raw_json)?;
    let cfg = super::config::BitNetConfig::from_model_config(&cfg_raw)?;

    assert_eq!(cfg.num_hidden_layers, 30);
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_attention_heads, 20);
    assert_eq!(cfg.num_key_value_heads, 5);
    assert_eq!(cfg.head_dim, 128); // 2560 / 20
    assert_eq!(cfg.vocab_size, 128256);
    assert!((cfg.rms_norm_eps - 1e-5_f32).abs() < 1e-8_f32);
    assert!((cfg.rope_theta - 500_000.0_f32).abs() < 1.0);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.max_position_embeddings, 4096);
    Ok(())
}

#[test]
fn test_bf16_encoding() {
    // BF16(1.0) = 0x3F80 = [0x80, 0x3F] in little-endian.
    let bits = 1.0_f32.to_bits();
    let bf16 = (bits >> 16) as u16;
    let bytes = bf16.to_le_bytes();
    assert_eq!(bytes, [0x80, 0x3F], "bf16(1.0) LE encoding");

    // Round-trip: decode back.
    let decoded = f32::from_bits(u32::from(u16::from_le_bytes(bytes)) << 16);
    assert!((decoded - 1.0_f32).abs() < 1e-5_f32, "bf16 round-trip");
}

#[test]
fn test_trit_packing_values() {
    // Verify the trit encoding documented in loader.rs: value = raw - 1
    // (raw 0 → -1, 1 → 0, 2 → +1, 3 → +2). Matches HF `unpack_weights` and
    // the mlx-lm `bitlinear_layers.py` kernel `(w & 3) - 1`.
    // 0x55 = 0101_0101 — 2-bit pairs (LSB first): raw [1,1,1,1] → [0,0,0,0]
    // 0x00 = 0000_0000 — raw [0,0,0,0] → [-1,-1,-1,-1]
    // 0xAA = 1010_1010 — raw [2,2,2,2] → [+1,+1,+1,+1]
    let decode_byte = |b: u8| -> Vec<i8> {
        (0..4)
            .map(|shift| {
                let raw = ((b >> (shift * 2)) & 0x3) as i8;
                raw - 1
            })
            .collect()
    };

    assert_eq!(decode_byte(0x55), vec![0, 0, 0, 0]);
    assert_eq!(decode_byte(0x00), vec![-1, -1, -1, -1]);
    assert_eq!(decode_byte(0xAA), vec![1, 1, 1, 1]);
    // 0x41 = 0100_0001 → raw [1, 0, 0, 1] → [0, -1, -1, 0]
    assert_eq!(decode_byte(0x41), vec![0, -1, -1, 0]);
    // 0x94 = 1001_0100 → bits [1:0]=00, [3:2]=01, [5:4]=01, [7:6]=10 → raw [0,1,1,2] → [-1,0,0,1]
    assert_eq!(decode_byte(0x94), vec![-1, 0, 0, 1]);
}
