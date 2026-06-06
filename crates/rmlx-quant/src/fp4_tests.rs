use super::*;

#[test]
fn e2m1_all_16_entries() {
    // Exhaustive table check — bit-pattern derivation in table comment above.
    let expected: [f32; 16] = [
        0.0_f32,  // 0x0
        0.5_f32,  // 0x1
        1.0_f32,  // 0x2
        1.5_f32,  // 0x3
        2.0_f32,  // 0x4
        3.0_f32,  // 0x5
        4.0_f32,  // 0x6
        6.0_f32,  // 0x7
        -0.0_f32, // 0x8
        -0.5_f32, // 0x9
        -1.0_f32, // 0xA
        -1.5_f32, // 0xB
        -2.0_f32, // 0xC
        -3.0_f32, // 0xD
        -4.0_f32, // 0xE
        -6.0_f32, // 0xF
    ];

    for nibble in 0u8..16 {
        let got = e2m1_decode(nibble);
        let exp = expected[nibble as usize];
        // Use to_bits for exact comparison (handles -0.0 vs 0.0).
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "nibble=0x{nibble:X}: expected {exp}, got {got}"
        );
    }
}

#[test]
fn e2m1_high_nibble_ignored() {
    // Upper nibble must be masked away: 0xF2 & 0xF = 0x2 = +1.0
    assert_eq!(e2m1_decode(0xF2), 1.0_f32);
}

#[test]
fn e2m1_positive_zero_and_negative_zero_differ_in_bits() {
    // +0 and -0 are distinct bit patterns but equal under ==.
    let pos_zero = e2m1_decode(0x0);
    let neg_zero = e2m1_decode(0x8);
    assert_eq!(pos_zero, neg_zero, "+0 == -0 as float");
    assert_ne!(
        pos_zero.to_bits(),
        neg_zero.to_bits(),
        "+0 and -0 differ in bits"
    );
}
