use super::*;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Build a scales byte slice where every group has `scale_byte`.
fn const_scales(rows: usize, groups_per_row: usize, scale_byte: u8) -> Vec<u8> {
    vec![scale_byte; rows * groups_per_row]
}

// ── mxfp8 round-trip: scale=1.0, every element = 0x38 (+1.0) ────────────

#[test]
fn mxfp8_roundtrip_all_ones() {
    // scale byte 0x7F → E8M0 decode → 2^(127-127) = 1.0
    // element byte 0x38 → E4M3 decode → +1.0
    // expected: all 1.0
    let rows = 1;
    let cols = 32; // one group
    let packed: Vec<u8> = vec![0x38; rows * cols];
    let scales = const_scales(rows, cols / 32, 0x7F);

    let params = MxParams {
        family: MxFamily::Mxfp8,
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "mxfp8 idx={i}: expected 1.0, got {v}");
    }
}

// ── mxfp4 round-trip: scale=1.0, every nibble = +1.0 ────────────────────

#[test]
fn mxfp4_roundtrip_all_ones() {
    // scale byte 0x7F → E8M0 → 1.0
    // nibble 0x2 = E2M1 +1.0, pack two 0x2 nibbles → byte 0x22 (lo=0x2, hi=0x2)
    let rows = 1;
    let cols = 32; // one group
                   // packed has cols/2 = 16 bytes; each byte has two +1.0 nibbles.
    let packed: Vec<u8> = vec![0x22; rows * (cols / 2)];
    let scales = const_scales(rows, cols / 32, 0x7F);

    let params = MxParams {
        family: MxFamily::Mxfp4,
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "mxfp4 idx={i}: expected 1.0, got {v}");
    }
}

// ── nvfp4 (UE4M3, default): scale=1.0, every nibble = +1.0 ─────────────

#[test]
fn nvfp4_unsigned_scale_all_ones() {
    // scale byte 0x38 → UE4M3: e=(0x38>>3)&0xF=7, m=0 → 2^0 * 1 = 1.0
    // nibble 0x2 = E2M1 +1.0
    let rows = 1;
    let cols = 16; // one group of 16
    let packed: Vec<u8> = vec![0x22; rows * (cols / 2)];
    let scales = const_scales(rows, cols / 16, 0x38);

    let params = MxParams {
        family: MxFamily::Nvfp4 {
            compat_mlx_signed_scale: false,
        },
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "nvfp4-ue4m3 idx={i}: expected 1.0, got {v}");
    }
}

// ── nvfp4 compat mode: same scale byte 0x38 also gives 1.0 ─────────────

#[test]
fn nvfp4_compat_scale_all_ones() {
    // For byte 0x38: E4M3 signed → s=0, e=7, m=0 → 2^(7-7)*1 = 1.0 (same)
    // So both modes agree on 0x38.
    let rows = 1;
    let cols = 16;
    let packed: Vec<u8> = vec![0x22; rows * (cols / 2)];
    let scales = const_scales(rows, cols / 16, 0x38);

    let params = MxParams {
        family: MxFamily::Nvfp4 {
            compat_mlx_signed_scale: true,
        },
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "nvfp4-compat idx={i}: expected 1.0, got {v}");
    }
}

// ── nvfp4: scale byte where signed and unsigned diverge ──────────────────

#[test]
fn nvfp4_scale_divergence_at_0xb8() {
    // 0xB8 = 0b10111000
    // Signed E4M3: s=1, e=7, m=0 → -2^(7-7)*1 = -1.0
    // UE4M3: e=(0xB8>>3)&0xF=7, m=0 → +2^(7-7)*1 = +1.0
    // Both modes must produce different outputs.
    let rows = 1;
    let cols = 16;
    // Use nibble 0x2 (+1.0) for elements so the sign difference is clear.
    let packed: Vec<u8> = vec![0x22; rows * (cols / 2)];
    let scale_byte: u8 = 0xB8;
    let scales = vec![scale_byte; rows * (cols / 16)];

    let params_unsigned = MxParams {
        family: MxFamily::Nvfp4 {
            compat_mlx_signed_scale: false,
        },
        rows,
        cols,
    };
    let params_signed = MxParams {
        family: MxFamily::Nvfp4 {
            compat_mlx_signed_scale: true,
        },
        rows,
        cols,
    };

    let out_unsigned = dequant_vec(&params_unsigned, &packed, &scales).unwrap();
    let out_signed = dequant_vec(&params_signed, &packed, &scales).unwrap();

    // Unsigned: 1.0 * 1.0 = +1.0 for each element
    for (i, &v) in out_unsigned.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "unsigned mode idx={i}: expected +1.0, got {v}");
    }
    // Signed compat: -1.0 * 1.0 = -1.0 for each element
    for (i, &v) in out_signed.iter().enumerate() {
        assert_eq!(
            v, -1.0_f32,
            "signed compat mode idx={i}: expected -1.0, got {v}"
        );
    }

    // Confirm they differ
    assert_ne!(
        out_unsigned[0].to_bits(),
        out_signed[0].to_bits(),
        "unsigned and signed modes should produce different results for 0xB8 scale"
    );
}

// ── Shape error: scales too short ────────────────────────────────────────

#[test]
fn mxfp8_err_scales_too_short() {
    // rows=1, cols=64 → needs 2 scale bytes (2 groups of 32).
    let params = MxParams {
        family: MxFamily::Mxfp8,
        rows: 1,
        cols: 64,
    };
    let packed = vec![0x38u8; 64];
    let scales = vec![0x7Fu8; 1]; // should be 2 — too short
    let result = dequant_vec(&params, &packed, &scales);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for too-short scales, got {result:?}"
    );
}

// ── NaN scale propagates to output group ─────────────────────────────────

#[test]
fn mxfp8_nan_scale_propagates() {
    // One 0xFF scale byte → E8M0 NaN → entire 32-element group becomes NaN.
    let rows = 1;
    let cols = 32;
    let packed = vec![0x38u8; rows * cols]; // all +1.0 elements
    let scales = vec![0xFFu8; 1]; // NaN scale

    let params = MxParams {
        family: MxFamily::Mxfp8,
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        nan_count, 32,
        "all 32 elements of the NaN-scale group should be NaN"
    );
}

// ── NaN element propagates to that cell ──────────────────────────────────

#[test]
fn mxfp8_nan_element_propagates() {
    // One 0x7F element (E4M3 NaN) in position 5 → that cell is NaN.
    let rows = 1;
    let cols = 32;
    let mut packed = vec![0x38u8; rows * cols]; // all +1.0
    packed[5] = 0x7F; // NaN element
    let scales = vec![0x7Fu8; 1]; // scale = 1.0

    let params = MxParams {
        family: MxFamily::Mxfp8,
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    assert!(out[5].is_nan(), "element at idx 5 should be NaN");
    // All other elements should be 1.0
    for (i, &v) in out.iter().enumerate() {
        if i != 5 {
            assert_eq!(v, 1.0_f32, "non-NaN element idx={i} should be 1.0, got {v}");
        }
    }
}

// ── mxfp8: two-row, two-group sanity check ───────────────────────────────

#[test]
fn mxfp8_two_rows_two_groups() {
    // rows=2, cols=64 → 4 groups total; each with scale=0x7F(1.0) and
    // element=0x38(+1.0) → all output = 1.0.
    let rows = 2;
    let cols = 64;
    let packed = vec![0x38u8; rows * cols];
    let scales = vec![0x7Fu8; rows * (cols / 32)];

    let params = MxParams {
        family: MxFamily::Mxfp8,
        rows,
        cols,
    };
    let out = dequant_vec(&params, &packed, &scales).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "idx={i}: expected 1.0, got {v}");
    }
}
