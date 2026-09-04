//! Tests for the dense code plane.

use super::{pack_row_into, read_code, render_msl_code_plane, row_words};

/// Deterministic code stream in `[0, 2^bits)`.
fn codes(n: usize, bits: u8) -> Vec<u8> {
    let hi = 1_u32 << bits;
    let mut s = 0x9E37_79B9_u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 13) % hi) as u8
        })
        .collect()
}

#[test]
fn a_row_occupies_the_bits_its_codes_need_rounded_to_a_word() {
    // The point of the plane: no word padding per group, one partial word per
    // row at most.
    assert_eq!(row_words(128, 3), 12); // iso3, head_dim 128: 384 bits
    assert_eq!(row_words(128, 4), 16); // iso4
    assert_eq!(row_words(129, 3), 13); // rotor3, 43 groups x 3 codes
    assert_eq!(row_words(129, 4), 17); // rotor4
    for (n_codes, bits) in [(128_usize, 3_u8), (128, 4), (129, 3), (129, 4), (1, 3)] {
        let words = row_words(n_codes, bits);
        let needed = n_codes * bits as usize;
        assert!(words * 32 >= needed, "{n_codes}x{bits} does not fit");
        assert!(
            (words * 32) - needed < 32,
            "{n_codes}x{bits} wastes a whole word"
        );
    }
}

#[test]
fn every_code_survives_a_round_trip_including_the_straddling_ones() {
    for bits in [3_u8, 4] {
        for n_codes in [1_usize, 4, 31, 32, 33, 128, 129, 384] {
            let src = codes(n_codes, bits);
            let mut plane = Vec::new();
            pack_row_into(&src, bits, &mut plane);
            assert_eq!(plane.len(), row_words(n_codes, bits));
            let back: Vec<u8> = (0..n_codes).map(|i| read_code(&plane, i, bits)).collect();
            assert_eq!(back, src, "bits={bits} n_codes={n_codes}");
        }
    }
}

#[test]
fn rows_pack_back_to_back_and_stay_independent() {
    let bits = 3_u8;
    let n_codes = 129;
    let rows: Vec<Vec<u8>> = (0..4).map(|_| codes(n_codes, bits)).collect();
    let mut plane = Vec::new();
    for row in &rows {
        pack_row_into(row, bits, &mut plane);
    }
    let stride = row_words(n_codes, bits);
    assert_eq!(plane.len(), stride * rows.len());
    for (r, row) in rows.iter().enumerate() {
        // A row read through its own slice must not see its neighbours' bits:
        // the row alignment is what makes that true.
        #[allow(
            clippy::indexing_slicing,
            reason = "plane.len() == stride * rows.len()"
        )]
        let slice = &plane[r * stride..(r + 1) * stride];
        let back: Vec<u8> = (0..n_codes).map(|i| read_code(slice, i, bits)).collect();
        assert_eq!(&back, row, "row {r}");
    }
}

#[test]
fn a_code_wider_than_the_field_is_truncated_not_smeared() {
    // Masking is what keeps an out-of-range index out of its neighbour's bits.
    let mut plane = Vec::new();
    pack_row_into(&[0xFF, 0, 0, 0], 3, &mut plane);
    assert_eq!(read_code(&plane, 0, 3), 0x7);
    assert_eq!(read_code(&plane, 1, 3), 0);
}

#[test]
fn the_msl_reader_and_the_rust_reader_state_the_same_parameters() {
    // The kernels read the plane through the emitted MSL; a width or group
    // count that disagreed with the Rust side would decode another codec's
    // bits with nothing to notice.
    let src = render_msl_code_plane(3, 4);
    assert!(src.contains("#define CP_BITS 3u"));
    assert!(src.contains("#define CP_MASK 0x7u"));
    assert!(src.contains("#define CP_CODES_PER_GROUP 4u"));
    let src4 = render_msl_code_plane(4, 3);
    assert!(src4.contains("#define CP_BITS 4u"));
    assert!(src4.contains("#define CP_MASK 0xFu"));
    assert!(src4.contains("#define CP_CODES_PER_GROUP 3u"));
}
