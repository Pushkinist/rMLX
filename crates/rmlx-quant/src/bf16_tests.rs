use super::*;

#[test]
fn zero_round_trips() {
    // 0.0f32 as bf16: bits = 0x0000
    let result = bf16_to_f32([0x00, 0x00]);
    assert_eq!(result, 0.0_f32);
}

#[test]
fn one_round_trips() {
    // 1.0f32 = 0x3F80_0000; bf16 = 0x3F80
    // le_bytes = [0x80, 0x3F]
    let result = bf16_to_f32([0x80, 0x3F]);
    assert_eq!(result, 1.0_f32);
}

#[test]
fn known_fraction() {
    // 0.5f32 = 0x3F00_0000; bf16 = 0x3F00
    // le_bytes = [0x00, 0x3F]
    let result = bf16_to_f32([0x00, 0x3F]);
    assert_eq!(result, 0.5_f32);
}

#[test]
fn bulk_decode_basic() {
    // Encode [0.0, 1.0, 0.5] as bf16 LE
    let input: Vec<u8> = vec![
        0x00, 0x00, // 0.0
        0x80, 0x3F, // 1.0
        0x00, 0x3F, // 0.5
    ];
    let mut out = vec![0.0_f32; 3];
    bf16_decode_into(&input, &mut out).unwrap();
    assert_eq!(out[0], 0.0_f32);
    assert_eq!(out[1], 1.0_f32);
    assert_eq!(out[2], 0.5_f32);
}

#[test]
fn length_mismatch_is_err() {
    let input = vec![0u8; 5]; // not a multiple of 2 matching out
    let mut out = vec![0.0_f32; 3];
    assert!(bf16_decode_into(&input, &mut out).is_err());
}

#[test]
fn bf16_truncation_of_f32() {
    // 0x3F80_3FFF truncated to bf16 = 0x3F80 (lower 16 bits lost)
    // Verify: f32::from_bits(0x3F80_3FFF) truncated to bf16 = 0x3F80 = 1.0
    let f = f32::from_bits(0x3F80_3FFF_u32);
    // bf16 of f: upper 16 bits = 0x3F80 → 1.0
    let bf16_bits = (f.to_bits() >> 16) as u16;
    assert_eq!(bf16_bits, 0x3F80);
    let reconstructed = bf16_to_f32(bf16_bits.to_le_bytes());
    assert_eq!(reconstructed, 1.0_f32);
}
