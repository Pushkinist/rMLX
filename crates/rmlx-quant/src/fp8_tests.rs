use super::*;

// ── e8m0_decode ──────────────────────────────────────────────────────────

#[test]
fn e8m0_one() {
    // 0x7F: e = 127 → 2^(127-127) = 2^0 = 1.0
    assert_eq!(e8m0_decode(0x7F), 1.0_f32);
}

#[test]
fn e8m0_half() {
    // 0x7E: e = 126 → 2^(126-127) = 2^-1 = 0.5
    assert_eq!(e8m0_decode(0x7E), 0.5_f32);
}

#[test]
fn e8m0_two() {
    // 0x80: e = 128 → 2^(128-127) = 2^1 = 2.0
    assert_eq!(e8m0_decode(0x80), 2.0_f32);
}

#[test]
fn e8m0_nan() {
    // 0xFF: reserved NaN
    assert!(e8m0_decode(0xFF).is_nan());
}

#[test]
fn e8m0_zero_byte() {
    // 0x00: e = 0 → 2^(0-127) = 2^-127 (very small subnormal)
    let v = e8m0_decode(0x00);
    assert!(v > 0.0_f32, "2^-127 should be positive, got {v}");
    assert!(v < 1e-37_f32, "2^-127 should be very small, got {v}");
    // 2^-127 ≈ 5.88e-39; f32 subnormal territory
    let expected = 2.0_f32.powi(-127);
    assert!(
        (v - expected).abs() < 1e-50_f32 || (v / expected - 1.0).abs() < 1e-5_f32,
        "e8m0_decode(0x00) = {v}, expected ~{expected}"
    );
}

// ── e4m3_decode ──────────────────────────────────────────────────────────

#[test]
fn e4m3_positive_zero() {
    // 0x00: s=0, e=0, m=0 → subnormal +0
    assert_eq!(e4m3_decode(0x00), 0.0_f32);
}

#[test]
fn e4m3_negative_zero() {
    // 0x80: s=1, e=0, m=0 → subnormal -0
    // Note: -0.0 == 0.0 in IEEE 754 floating point
    let v = e4m3_decode(0x80);
    assert_eq!(v, 0.0_f32, "e4m3(0x80) should be -0.0 (== 0.0), got {v}");
    // Verify it is actually the negative zero
    assert!(v.is_sign_negative(), "e4m3(0x80) should be negative zero");
}

#[test]
fn e4m3_positive_one() {
    // 0x38: s=0, e=0b0111=7, m=0b000=0
    // Normal: 2^(7-7) * (1 + 0/8) = 1.0 * 1.0 = 1.0
    assert_eq!(e4m3_decode(0x38), 1.0_f32);
}

#[test]
fn e4m3_negative_one() {
    // 0xB8: s=1, e=0b0111=7, m=0b000=0
    // Normal: -2^(7-7) * (1 + 0/8) = -1.0
    assert_eq!(e4m3_decode(0xB8), -1.0_f32);
}

#[test]
fn e4m3_nan_positive() {
    // 0x7F: s=0, e=0xF, m=0x7 → NaN
    assert!(e4m3_decode(0x7F).is_nan());
}

#[test]
fn e4m3_nan_negative() {
    // 0xFF: s=1, e=0xF, m=0x7 → NaN
    assert!(e4m3_decode(0xFF).is_nan());
}

#[test]
fn e4m3_subnormal() {
    // 0x01: s=0, e=0, m=1
    // Subnormal: +2^-6 * (1/8) = +2^-9
    // 2^-9 = 1/512 ≈ 0.001953125
    let v = e4m3_decode(0x01);
    let expected = 2.0_f32.powi(-9);
    assert!(
        (v - expected).abs() < 1e-10_f32,
        "e4m3(0x01) expected 2^-9 = {expected}, got {v}"
    );
}

// ── ue4m3_decode ─────────────────────────────────────────────────────────

#[test]
fn ue4m3_positive_zero() {
    // 0x00: e=0, m=0 → subnormal 0.0
    assert_eq!(ue4m3_decode(0x00), 0.0_f32);
}

#[test]
fn ue4m3_one() {
    // 0x38: e = (0x38 >> 3) & 0xF = 7, m = 0x38 & 0x7 = 0
    // Normal: 2^(7-7) * (1 + 0/8) = 1.0
    // bit-pattern: 0b00111000 → e=7, m=0 ✓
    assert_eq!(ue4m3_decode(0x38), 1.0_f32);
}

#[test]
fn ue4m3_e15_m0() {
    // 0x78 = 0b01111000: e = (0x78 >> 3) & 0xF = 15, m = 0
    // Normal: 2^(15-7) * (1 + 0/8) = 2^8 = 256.0
    // Note: the task spec marked this as "?? verify" — 256.0 is the correct value.
    assert_eq!(ue4m3_decode(0x78), 256.0_f32);
}

#[test]
fn ue4m3_largest_finite() {
    // 0xFF = 0b11111111: e = (0xFF >> 3) & 0xF = (0x1F) & 0xF = 15, m = 7
    // Normal: 2^(15-7) * (1 + 7/8) = 256 * 15/8 = 480.0
    // No NaN reservation in UE4M3 — 0xFF is finite.
    assert_eq!(ue4m3_decode(0xFF), 480.0_f32);
}
