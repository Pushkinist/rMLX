//! Unit tests for the NPZ / ZIP parser.

use super::{extract_npy_shape, parse_npy_array, parse_npz};
use crate::test_events::fields_recorded_by;
use rmlx_mlx::Dtype;

// ── NPY header helpers ────────────────────────────────────────────────────────

/// Each dtype that loads gives its stored dtype, its shape and its bytes.
#[test]
fn the_dtypes_that_load_give_their_stored_dtype() {
    for (descr, dtype, width) in [
        ("<f2", Dtype::F16, 2),
        ("<f4", Dtype::F32, 4),
        ("|b1", Dtype::U8, 1),
        ("|u1", Dtype::U8, 1),
        ("|i1", Dtype::U8, 1),
        ("<u4", Dtype::U32, 4),
        ("<i4", Dtype::I32, 4),
    ] {
        let payload = vec![1_u8; 2 * width];
        let array = parse_npy_array("w", &npy_bytes(descr, "(2,)", &payload)).unwrap();
        assert_eq!(array.dtype(), dtype, "{descr}");
        assert_eq!(array.shape(), vec![2], "{descr}");
        assert_eq!(array.to_bytes().unwrap(), payload, "{descr}");
    }
}

#[test]
fn npy_shape_1d() {
    let hdr = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280,), }";
    assert_eq!(extract_npy_shape(hdr).unwrap(), vec![1280]);
}

#[test]
fn npy_shape_2d() {
    let hdr = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280, 1280), }";
    assert_eq!(extract_npy_shape(hdr).unwrap(), vec![1280, 1280]);
}

#[test]
fn npy_shape_3d() {
    let hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (32, 400, 1280), }";
    assert_eq!(extract_npy_shape(hdr).unwrap(), vec![32, 400, 1280]);
}

#[test]
fn npy_shape_trailing_comma() {
    let hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3,), }";
    assert_eq!(extract_npy_shape(hdr).unwrap(), vec![2, 3]);
}

/// A dimension that does not parse is an error of the entry, not a dimension
/// that drops out of the shape.
#[test]
fn a_dimension_that_does_not_parse_is_an_error_naming_the_entry() {
    for shape in [
        "(99999999999999999999, 2)",
        "(-1, 2)",
        "(2,,3)",
        "(abc,)",
        "(2, , 3)",
    ] {
        let hdr = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        assert_eq!(extract_npy_shape(&hdr), None, "{shape}");
        let err = parse_npy_array("bad_dim", &npy_bytes("<f4", shape, &[0; 16]))
            .expect_err("the shape does not parse")
            .to_string();
        assert!(err.contains("bad_dim"), "{shape}: {err}");
        assert!(err.contains("shape"), "{shape}: {err}");
    }
}

#[test]
fn npy_shape_scalar() {
    let hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
    assert_eq!(extract_npy_shape(hdr).unwrap(), Vec::<usize>::new());
}

// ── ZIP archive fixture builders ──────────────────────────────────────────────

/// Build a minimal ZIP32 archive containing a single stored `.npy` entry.
///
/// The entry contains a tiny f32 scalar (value 1.0).
fn build_zip32_stored_npy(entry_name: &str, payload: &[u8]) -> Vec<u8> {
    use std::io::Write;

    let fname = format!("{entry_name}.npy").into_bytes();
    let fname_len = fname.len() as u16;

    let mut out = Vec::new();

    // ── Local file header ─────────────────────────────────────────────────
    let lh_offset = out.len() as u32;
    out.write_all(&0x0403_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&20_u16.to_le_bytes()).unwrap(); // version needed
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // flags
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // compression (stored)
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod time
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod date
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // crc32 (unchecked)
    out.write_all(&(payload.len() as u32).to_le_bytes())
        .unwrap(); // comp size
    out.write_all(&(payload.len() as u32).to_le_bytes())
        .unwrap(); // uncomp size
    out.write_all(&fname_len.to_le_bytes()).unwrap();
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // extra len
    out.write_all(&fname).unwrap();
    out.write_all(payload).unwrap();

    // ── Central directory ─────────────────────────────────────────────────
    let cd_offset = out.len() as u32;
    out.write_all(&0x0201_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&20_u16.to_le_bytes()).unwrap(); // version made by
    out.write_all(&20_u16.to_le_bytes()).unwrap(); // version needed
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // flags
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // compression
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod time
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod date
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // crc32
    out.write_all(&(payload.len() as u32).to_le_bytes())
        .unwrap();
    out.write_all(&(payload.len() as u32).to_le_bytes())
        .unwrap();
    out.write_all(&fname_len.to_le_bytes()).unwrap();
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // extra len
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // comment len
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // disk start
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // internal attr
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // external attr
    out.write_all(&lh_offset.to_le_bytes()).unwrap(); // local header offset
    out.write_all(&fname).unwrap();

    let cd_size = out.len() as u32 - cd_offset;

    // ── End-of-central-directory ──────────────────────────────────────────
    out.write_all(&0x0605_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // disk number
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // disk with cd
    out.write_all(&1_u16.to_le_bytes()).unwrap(); // entries on disk
    out.write_all(&1_u16.to_le_bytes()).unwrap(); // total entries
    out.write_all(&cd_size.to_le_bytes()).unwrap(); // cd size
    out.write_all(&cd_offset.to_le_bytes()).unwrap(); // cd offset
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // comment len

    out
}

/// Build a ZIP64 archive where the local-header sizes are `0xFFFFFFFF`
/// (ZIP64 sentinel) but the actual sizes live in the ZIP64 extra field.
/// This matches the format produced by NumPy's `savez` for large files.
///
/// The actual payload is small; only the *structure* is ZIP64.
fn build_zip64_npy(entry_name: &str, payload: &[u8]) -> Vec<u8> {
    use std::io::Write;

    let fname = format!("{entry_name}.npy").into_bytes();
    let fname_len = fname.len() as u16;
    let payload_len = payload.len() as u64;

    let mut out = Vec::new();

    // ── ZIP64 extra field for local header ────────────────────────────────
    let zip64_extra_local = {
        let mut ex = Vec::new();
        ex.write_all(&0x0001_u16.to_le_bytes()).unwrap(); // id
        ex.write_all(&16_u16.to_le_bytes()).unwrap(); // size (uncomp + comp)
        ex.write_all(&payload_len.to_le_bytes()).unwrap(); // uncomp size
        ex.write_all(&payload_len.to_le_bytes()).unwrap(); // comp size
        ex
    };

    // ── Local file header (ZIP64: size fields = 0xFFFFFFFF) ───────────────
    let lh_offset = out.len() as u64;
    out.write_all(&0x0403_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&45_u16.to_le_bytes()).unwrap(); // version needed (4.5 = ZIP64)
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // flags
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // compression (stored)
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod time
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod date
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // crc32
    out.write_all(&0xFFFF_FFFFu32.to_le_bytes()).unwrap(); // comp size = sentinel
    out.write_all(&0xFFFF_FFFFu32.to_le_bytes()).unwrap(); // uncomp size = sentinel
    out.write_all(&fname_len.to_le_bytes()).unwrap();
    out.write_all(&(zip64_extra_local.len() as u16).to_le_bytes())
        .unwrap();
    out.write_all(&fname).unwrap();
    out.write_all(&zip64_extra_local).unwrap();
    out.write_all(payload).unwrap();

    // ── ZIP64 extra field for central directory ───────────────────────────
    let zip64_extra_cd = {
        let mut ex = Vec::new();
        ex.write_all(&0x0001_u16.to_le_bytes()).unwrap(); // id
        ex.write_all(&24_u16.to_le_bytes()).unwrap(); // size (uncomp + comp + offset)
        ex.write_all(&payload_len.to_le_bytes()).unwrap(); // uncomp size
        ex.write_all(&payload_len.to_le_bytes()).unwrap(); // comp size
        ex.write_all(&lh_offset.to_le_bytes()).unwrap(); // local header offset
        ex
    };

    // ── Central directory ─────────────────────────────────────────────────
    let cd_offset = out.len() as u64;
    out.write_all(&0x0201_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&45_u16.to_le_bytes()).unwrap(); // version made by
    out.write_all(&45_u16.to_le_bytes()).unwrap(); // version needed
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // flags
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // compression
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod time
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // last mod date
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // crc32
    out.write_all(&0xFFFF_FFFFu32.to_le_bytes()).unwrap(); // comp = sentinel
    out.write_all(&0xFFFF_FFFFu32.to_le_bytes()).unwrap(); // uncomp = sentinel
    out.write_all(&fname_len.to_le_bytes()).unwrap();
    out.write_all(&(zip64_extra_cd.len() as u16).to_le_bytes())
        .unwrap();
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // comment len
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // disk start
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // internal attr
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // external attr
    out.write_all(&0xFFFF_FFFFu32.to_le_bytes()).unwrap(); // local hdr offset = sentinel
    out.write_all(&fname).unwrap();
    out.write_all(&zip64_extra_cd).unwrap();

    let cd_size = out.len() as u64 - cd_offset;

    // ── ZIP64 end-of-central-directory record ─────────────────────────────
    let eocd64_offset = out.len() as u64;
    out.write_all(&0x0606_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&44_u64.to_le_bytes()).unwrap(); // size of record (fixed 44)
    out.write_all(&45_u16.to_le_bytes()).unwrap(); // version made by
    out.write_all(&45_u16.to_le_bytes()).unwrap(); // version needed
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // disk number
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // disk with cd
    out.write_all(&1_u64.to_le_bytes()).unwrap(); // entries on disk
    out.write_all(&1_u64.to_le_bytes()).unwrap(); // total entries
    out.write_all(&cd_size.to_le_bytes()).unwrap(); // cd size
    out.write_all(&cd_offset.to_le_bytes()).unwrap(); // cd offset

    // ── ZIP64 EOCD locator ────────────────────────────────────────────────
    out.write_all(&0x0706_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&0_u32.to_le_bytes()).unwrap(); // disk with eocd64
    out.write_all(&eocd64_offset.to_le_bytes()).unwrap(); // offset of eocd64
    out.write_all(&1_u32.to_le_bytes()).unwrap(); // total disks

    // ── Regular EOCD (with sentinels pointing at ZIP64 records) ──────────
    out.write_all(&0x0605_4b50_u32.to_le_bytes()).unwrap(); // sig
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // disk number
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // disk with cd
    out.write_all(&0xFFFF_u16.to_le_bytes()).unwrap(); // entries on disk = sentinel
    out.write_all(&0xFFFF_u16.to_le_bytes()).unwrap(); // total entries = sentinel
    out.write_all(&0xFFFF_FFFF_u32.to_le_bytes()).unwrap(); // cd size = sentinel
    out.write_all(&0xFFFF_FFFF_u32.to_le_bytes()).unwrap(); // cd offset = sentinel
    out.write_all(&0_u16.to_le_bytes()).unwrap(); // comment len

    out
}

/// Build a minimal f32 scalar `.npy` payload.
///
/// NPY v1.0 format: the header block (after the 10-byte fixed prefix) must be
/// padded with spaces and terminated with `\n` to a multiple of 64 bytes total.
fn npy_f32_scalar(value: f32) -> Vec<u8> {
    let hdr = b"{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
    // Fixed prefix: 6 (magic) + 1 (major) + 1 (minor) + 2 (hdr_len u16) = 10 bytes.
    // Header content must pad to the next multiple of 64 total.
    let hdr_base = hdr.len(); // e.g. 56
                              // Total = 10 + hdr_len. We need total % 64 == 0.
                              // Minimum total ≥ 10 + hdr_base + 1 (newline).
    let min_total = 10 + hdr_base + 1;
    let total = ((min_total + 63) / 64) * 64; // round up to next multiple of 64
    let header_len = total - 10;

    let mut h = hdr.to_vec();
    // Pad with spaces up to `header_len - 1`, then append newline.
    while h.len() < header_len - 1 {
        h.push(b' ');
    }
    h.push(b'\n');
    assert_eq!(h.len(), header_len, "NPY header builder: length mismatch");

    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1); // major
    out.push(0); // minor
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&value.to_le_bytes());
    out
}

// ── Parse tests ──────────────────────────────────────────────────────────────

/// ZIP32 stored entry round-trips through the parser.
#[test]
fn zip32_stored_roundtrip() {
    let payload = npy_f32_scalar(3.14);
    let zip = build_zip32_stored_npy("my_weight", &payload);
    let map = parse_npz(&zip).expect("zip32 parse should succeed");
    assert!(map.contains_key("my_weight"), "key 'my_weight' missing");
    let arr = map.get("my_weight").unwrap();
    // scalar shape → ndim 0
    assert_eq!(arr.ndim(), 0);
}

/// ZIP64 stored entry round-trips through the parser (sentinel sizes in headers).
#[test]
fn zip64_stored_roundtrip() {
    let payload = npy_f32_scalar(2.71);
    let zip = build_zip64_npy("w64", &payload);
    let map = parse_npz(&zip).expect("zip64 parse should succeed");
    assert!(map.contains_key("w64"), "key 'w64' missing; map = {map:?}");
    let arr = map.get("w64").unwrap();
    assert_eq!(arr.ndim(), 0);
}

/// Deflate-compressed entry round-trips.
/// A ZIP32 archive of one deflate entry whose directory states `declared_size`
/// as the size after inflate.
fn build_zip32_deflated_npy(entry_name: &str, payload: &[u8], declared_size: u32) -> Vec<u8> {
    use std::io::Write;

    let compressed = miniz_oxide::deflate::compress_to_vec(payload, 6);
    let fname = format!("{entry_name}.npy").into_bytes();
    let fname_len = fname.len() as u16;
    let comp_size = compressed.len() as u32;
    let uncomp_size = declared_size;

    let mut zip = Vec::new();

    let lh_offset = 0u32;
    // Local header.
    zip.write_all(&0x0403_4b50_u32.to_le_bytes()).unwrap();
    zip.write_all(&20_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap(); // flags
    zip.write_all(&8_u16.to_le_bytes()).unwrap(); // method: deflate
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u32.to_le_bytes()).unwrap(); // crc
    zip.write_all(&comp_size.to_le_bytes()).unwrap();
    zip.write_all(&uncomp_size.to_le_bytes()).unwrap();
    zip.write_all(&fname_len.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap(); // extra
    zip.write_all(&fname).unwrap();
    zip.write_all(&compressed).unwrap();

    let cd_offset = zip.len() as u32;
    // Central directory.
    zip.write_all(&0x0201_4b50_u32.to_le_bytes()).unwrap();
    zip.write_all(&20_u16.to_le_bytes()).unwrap();
    zip.write_all(&20_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&8_u16.to_le_bytes()).unwrap(); // deflate
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u32.to_le_bytes()).unwrap();
    zip.write_all(&comp_size.to_le_bytes()).unwrap();
    zip.write_all(&uncomp_size.to_le_bytes()).unwrap();
    zip.write_all(&fname_len.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u32.to_le_bytes()).unwrap();
    zip.write_all(&lh_offset.to_le_bytes()).unwrap();
    zip.write_all(&fname).unwrap();

    let cd_size = zip.len() as u32 - cd_offset;

    // EOCD.
    zip.write_all(&0x0605_4b50_u32.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip.write_all(&1_u16.to_le_bytes()).unwrap();
    zip.write_all(&1_u16.to_le_bytes()).unwrap();
    zip.write_all(&cd_size.to_le_bytes()).unwrap();
    zip.write_all(&cd_offset.to_le_bytes()).unwrap();
    zip.write_all(&0_u16.to_le_bytes()).unwrap();
    zip
}

#[test]
fn deflate_entry_roundtrip() {
    let payload = npy_f32_scalar(1.0);
    let zip = build_zip32_deflated_npy("deflated", &payload, payload.len() as u32);

    let map = parse_npz(&zip).expect("deflate parse should succeed");
    let key = "deflated";
    assert!(map.contains_key(key), "key '{key}' missing");
    assert_eq!(map[key].ndim(), 0);
}

/// An entry that inflates to more than the directory says is an error that
/// names the entry.
#[test]
fn an_entry_that_inflates_beyond_its_declared_size_is_an_error() {
    let payload = npy_bytes("|u1", "(10000,)", &[0; 10_000]);
    let zip = build_zip32_deflated_npy("bomb", &payload, 100);
    let err = parse_npz(&zip).unwrap_err().to_string();
    assert!(err.contains("bomb"), "{err}");
    assert!(err.contains("deflate"), "{err}");
}

/// Empty data returns an empty map (not an error).
#[test]
fn empty_npz_returns_error() {
    let result = parse_npz(&[]);
    assert!(result.is_err(), "empty bytes should error (no EOCD)");
}

/// Truncated EOCD returns an error, not a panic.
#[test]
fn truncated_eocd_returns_error() {
    // Just the EOCD magic with only 10 bytes (needs 22).
    let data = b"PK\x05\x06\x00\x00\x00\x00\x00\x00";
    let result = parse_npz(data);
    assert!(result.is_err());
}

// ── Integer entries ───────────────────────────────────────────────────────────

/// Small synthetic (layer, head) pairs.
const PAIRS: [(i64, i64); 4] = [(1, 0), (2, 3), (5, 7), (9, 11)];

/// An NPY v1.0 file of `descr`, `shape` (as written in the header) and `payload`.
fn npy_bytes(descr: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
    npy_bytes_with_order(descr, "False", shape, payload)
}

fn npy_bytes_with_order(descr: &str, fortran: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}")
            .into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(payload);
    out
}

fn int64_pairs_npy(pairs: &[(i64, i64)]) -> Vec<u8> {
    let payload: Vec<u8> = pairs
        .iter()
        .flat_map(|&(layer, head)| [layer, head])
        .flat_map(i64::to_le_bytes)
        .collect();
    npy_bytes("<i8", &format!("({}, 2)", pairs.len()), &payload)
}

fn i32_values(array: &rmlx_mlx::Array) -> Vec<i32> {
    array
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A little-endian `<i8` entry of shape (4, 2) gives its values.
#[test]
fn an_int64_entry_gives_its_values() {
    let array = parse_npy_array("alignment_heads", &int64_pairs_npy(&PAIRS)).unwrap();
    assert_eq!(array.shape(), vec![4, 2]);
    let expected: Vec<i32> = PAIRS
        .iter()
        .flat_map(|&(layer, head)| [layer as i32, head as i32])
        .collect();
    assert_eq!(i32_values(&array), expected);
}

/// A value that int32 cannot hold is an error that names the entry.
#[test]
fn an_int64_value_beyond_int32_is_an_error_naming_the_entry() {
    let npy = int64_pairs_npy(&[(1 << 40, 0)]);
    let err = parse_npy_array("alignment_heads", &npy)
        .unwrap_err()
        .to_string();
    assert!(err.contains("alignment_heads"), "{err}");
    assert!(err.contains("does not fit in int32"), "{err}");
}

/// An archive with an int64 entry loads it.
#[test]
fn an_archive_with_an_int64_entry_loads_it() {
    let zip = build_zip32_stored_npy("alignment_heads", &int64_pairs_npy(&PAIRS));
    let map = parse_npz(&zip).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key("alignment_heads"));
}

// ── An entry that cannot be read is an error of the archive ──────────────────

/// The error of an archive of one entry named `bad_entry` that holds `npy`.
fn error_of_entry(npy: &[u8]) -> String {
    parse_npz(&build_zip32_stored_npy("bad_entry", npy))
        .expect_err("the archive must be refused")
        .to_string()
}

fn assert_names(err: &str, cause: &str) {
    assert!(err.contains("bad_entry"), "no entry name: {err}");
    assert!(err.contains(cause), "no cause {cause:?}: {err}");
}

/// An unknown dtype names the entry and the dtype.
#[test]
fn an_unknown_dtype_is_an_error_naming_the_entry_and_the_dtype() {
    for descr in ["<f8", "<u8", "<c8", "|S4"] {
        let err = error_of_entry(&npy_bytes(descr, "(1,)", &[0; 8]));
        assert_names(&err, &format!("unknown dtype '{descr}'"));
    }
}

/// A multi-byte entry with the `>` mark names the entry and the byte order.
#[test]
fn a_big_endian_entry_is_an_error_naming_the_entry_and_the_byte_order() {
    for descr in [">i8", ">f4", ">f2", ">i4", ">u4"] {
        let err = error_of_entry(&npy_bytes(descr, "(1,)", &[0; 8]));
        assert_names(&err, "byte order");
        assert_names(&err, descr);
    }
}

/// Column-major data is refused: the reader takes rows.
#[test]
fn fortran_order_is_an_error_naming_the_entry() {
    let npy = npy_bytes_with_order("<f4", "True", "(2, 2)", &[0; 16]);
    assert_names(&error_of_entry(&npy), "fortran_order");
}

/// Data shorter than the header says is an error naming the entry.
#[test]
fn truncated_data_is_an_error_naming_the_entry() {
    let err = error_of_entry(&npy_bytes("<f4", "(4,)", &[0; 8]));
    assert_names(&err, "data truncated (need 16, have 8)");
}

/// A compression method other than stored and deflate names the entry.
#[test]
fn an_unsupported_compression_method_is_an_error_naming_the_entry() {
    let mut zip = build_zip32_stored_npy("bad_entry", &npy_bytes("<f4", "(1,)", &[0; 4]));
    // Method 12 (bzip2) in the local header (offset 8) and the central one.
    zip[8..10].copy_from_slice(&12_u16.to_le_bytes());
    let central = zip.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    zip[central + 10..central + 12].copy_from_slice(&12_u16.to_le_bytes());
    let err = parse_npz(&zip).unwrap_err().to_string();
    assert_names(&err, "unsupported ZIP compression method 12");
}

// ── Byte order that loads ─────────────────────────────────────────────────────

/// The `<` and `=` marks read as little-endian.
#[test]
fn little_endian_and_native_int64_entries_load() {
    for descr in ["<i8", "=i8"] {
        let payload: Vec<u8> = [7_i64, 0].into_iter().flat_map(i64::to_le_bytes).collect();
        let array = parse_npy_array("heads", &npy_bytes(descr, "(1, 2)", &payload)).unwrap();
        assert_eq!(i32_values(&array), [7, 0], "{descr}");
    }
}

/// One byte has no byte order: `|`, `>` and `<` all load.
#[test]
fn single_byte_entries_load_with_any_mark() {
    for descr in ["|u1", ">u1", "<u1", "|b1", ">b1", "|i1", ">i1"] {
        let array = parse_npy_array("bytes", &npy_bytes(descr, "(2,)", &[1, 2])).unwrap();
        assert_eq!(array.dtype(), Dtype::U8, "{descr}");
        assert_eq!(array.to_bytes().unwrap(), [1, 2], "{descr}");
    }
}

// ── Sizes and offsets that come from the archive ─────────────────────────────

/// A header that states more elements than a `usize` of bytes can hold is an
/// error that names the entry and says the byte count overflows, not a panic
/// or an allocation.
#[test]
fn a_shape_whose_byte_count_overflows_is_an_error_naming_the_entry() {
    for (descr, shape) in [
        ("<i8", "(2305843009213693952,)"),
        ("<f4", "(4294967296, 4294967296)"),
        ("<f4", "(4611686018427387904,)"),
        ("|u1", "(2147483647, 2147483647, 2147483647)"),
    ] {
        let npy = npy_bytes(descr, shape, &[]);
        let err = parse_npy_array("lying", &npy)
            .expect_err("the header lies")
            .to_string();
        assert!(err.contains("lying"), "{descr} {shape}: {err}");
        assert!(err.contains("overflows"), "{descr} {shape}: {err}");
    }
}

/// A lying shape on an unknown dtype is an error that names the entry.
#[test]
fn a_lying_shape_of_an_unknown_dtype_is_an_error_naming_the_entry() {
    let npy = npy_bytes("<f8", "(4294967296, 4294967296)", &[]);
    let err = parse_npy_array("lying", &npy).unwrap_err().to_string();
    assert!(err.contains("lying"), "{err}");
}

/// A dimension that does not fit in int32 is an error, also with no data.
#[test]
fn a_dimension_beyond_int32_is_an_error_naming_the_entry() {
    let npy = npy_bytes("|u1", "(0, 2147483648)", &[]);
    let err = parse_npy_array("wide", &npy).unwrap_err().to_string();
    assert!(err.contains("wide"), "{err}");
    assert!(err.contains("int32"), "{err}");
}

/// Patch `bytes` of a ZIP64 archive at `at` with `value` as 8 little-endian bytes.
fn put_u64(zip: &mut [u8], at: usize, value: u64) {
    zip[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn find(zip: &[u8], signature: [u8; 4]) -> usize {
    zip.windows(4).position(|w| w == signature).unwrap()
}

/// A ZIP64 archive whose offsets and sizes read from the archive are near the
/// top of `u64` is an error, not a panic or a wrapped offset.
#[test]
fn offsets_and_sizes_that_overflow_are_errors() {
    let near_max = u64::MAX - 10;
    let archive = build_zip64_npy("w64", &npy_f32_scalar(1.0));
    // Within the central-directory entry: fixed part 46 bytes, the file name
    // `w64.npy`, then the ZIP64 extra (id, size, uncompressed, compressed, offset).
    let central = find(&archive, *b"PK\x01\x02");
    let extra = central + 46 + "w64.npy".len();

    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    let mut zip = archive.clone();
    let locator = find(&zip, *b"PK\x06\x07");
    put_u64(&mut zip, locator + 8, near_max);
    cases.push(("offset of the ZIP64 EOCD", zip));

    let mut zip = archive.clone();
    let eocd64 = find(&zip, *b"PK\x06\x06");
    put_u64(&mut zip, eocd64 + 48, near_max);
    cases.push(("offset of the central directory", zip));

    let mut zip = archive.clone();
    put_u64(&mut zip, extra + 20, near_max);
    cases.push(("offset of the local header", zip));

    let mut zip = archive;
    put_u64(&mut zip, extra + 12, near_max);
    cases.push(("compressed size", zip));

    for (what, zip) in &cases {
        assert!(
            parse_npz(zip).is_err(),
            "{what}: the archive must be refused"
        );
    }
    // One prefix: the error says "ZIP structure" once, then the entry, then the cause.
    let (_, local_header_offset) = &cases[2];
    assert_eq!(
        parse_npz(local_header_offset).unwrap_err().to_string(),
        "ZIP structure: entry w64: local header: offset 0xfffffffffffffff5 + 30 overflows"
    );
}

// ── The parts of the archive ──────────────────────────────────────────────────

/// A ZIP32 archive of stored members, in order. A member whose name ends in `/`
/// is a directory member and holds no data.
fn build_zip32_members(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut offsets = Vec::new();
    for (name, data) in members {
        offsets.push(out.len() as u32);
        out.extend_from_slice(&0x0403_4b50_u32.to_le_bytes());
        out.extend_from_slice(&20_u16.to_le_bytes());
        out.extend_from_slice(&[0; 8]); // flags, method (stored), time, date
        out.extend_from_slice(&0_u32.to_le_bytes()); // crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
    }
    let cd_offset = out.len() as u32;
    for ((name, data), offset) in members.iter().zip(&offsets) {
        out.extend_from_slice(&0x0201_4b50_u32.to_le_bytes());
        out.extend_from_slice(&20_u16.to_le_bytes());
        out.extend_from_slice(&20_u16.to_le_bytes());
        out.extend_from_slice(&[0; 8]); // flags, method (stored), time, date
        out.extend_from_slice(&0_u32.to_le_bytes()); // crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0; 12]); // extra, comment, disk, attributes
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
    }
    let cd_size = out.len() as u32 - cd_offset;
    out.extend_from_slice(&0x0605_4b50_u32.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out
}

fn f32_npy() -> Vec<u8> {
    npy_f32_scalar(1.0)
}

/// Two good members load, each under its name without the `.npy` suffix.
#[test]
fn two_members_load() {
    let npy = f32_npy();
    let zip = build_zip32_members(&[("a.npy", &npy), ("b.npy", &npy)]);
    let map = parse_npz(&zip).unwrap();
    let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["a", "b"]);
}

/// A bad signature in the central directory, before the last entry, is an
/// error and not a shorter archive.
#[test]
fn a_bad_signature_before_the_last_entry_is_an_error() {
    let npy = f32_npy();
    let mut zip = build_zip32_members(&[("a.npy", &npy), ("b.npy", &npy)]);
    let first = find(&zip, *b"PK\x01\x02");
    let second = first
        + zip[first + 4..]
            .windows(4)
            .position(|w| w == b"PK\x01\x02")
            .unwrap()
        + 4;
    zip[second..second + 4].copy_from_slice(b"XXXX");
    let err = parse_npz(&zip).unwrap_err().to_string();
    assert!(err.contains("central directory entry 1"), "{err}");
    assert!(err.contains("bad signature"), "{err}");
}

/// One good member and one that cannot be read is an error: the archive loads
/// whole or not at all.
#[test]
fn a_good_member_and_an_unreadable_one_are_an_error() {
    let good = f32_npy();
    let bad = npy_bytes("<f8", "(1,)", &[0; 8]);
    for members in [
        [("good.npy", &good[..]), ("bad_entry.npy", &bad[..])],
        [("bad_entry.npy", &bad[..]), ("good.npy", &good[..])],
    ] {
        let err = parse_npz(&build_zip32_members(&members))
            .unwrap_err()
            .to_string();
        assert!(err.contains("bad_entry"), "{err}");
    }
}

/// A member of no bytes that is not a directory is read, and it is not an
/// `.npy` file.
#[test]
fn an_empty_npy_member_is_an_error_naming_the_entry() {
    let err = parse_npz(&build_zip32_members(&[("empty.npy", &[])]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty"), "{err}");
    assert!(err.contains("bad magic"), "{err}");
}

/// A directory member is skipped with no event and the rest loads.
#[test]
fn a_directory_member_is_skipped_and_the_rest_loads() {
    let npy = f32_npy();
    let zip = build_zip32_members(&[("sub/", &[]), ("w.npy", &npy)]);
    let mut loaded = None;
    let fields = fields_recorded_by(|| loaded = Some(parse_npz(&zip).unwrap()));
    assert_eq!(loaded.unwrap().keys().collect::<Vec<_>>(), ["w"]);
    assert!(
        fields.iter().all(|(name, _)| name != "member"),
        "{fields:?}"
    );
}

/// A member that is not an `.npy` file is not a weight: it is skipped with one
/// debug event that names it, and it has no key.
#[test]
fn a_member_that_is_not_an_npy_file_is_skipped_with_one_event() {
    let npy = f32_npy();
    let zip = build_zip32_members(&[("w.npy", &npy), ("__MACOSX/notes.txt", b"not an array")]);
    let mut loaded = None;
    let fields = fields_recorded_by(|| loaded = Some(parse_npz(&zip).unwrap()));
    assert_eq!(loaded.unwrap().keys().collect::<Vec<_>>(), ["w"]);
    assert_eq!(
        fields
            .iter()
            .filter(|(name, value)| name == "member" && value == "__MACOSX/notes.txt")
            .count(),
        1,
        "{fields:?}"
    );
    assert!(
        fields
            .iter()
            .any(|(name, value)| name == "level" && value == "DEBUG"),
        "{fields:?}"
    );
}

/// A deflate entry that fails after a large declared size gives a short error
/// that names the entry: the text holds the status, not the output so far.
#[test]
fn a_failed_deflate_entry_gives_a_short_error_naming_the_entry() {
    let payload = npy_bytes("|u1", "(200000,)", &vec![0; 200_000]);
    let zip = build_zip32_deflated_npy("bomb", &payload, 100_000);
    let err = parse_npz(&zip).unwrap_err().to_string();
    assert!(err.contains("bomb"), "{err}");
    assert!(err.len() < 300, "{} bytes: {}", err.len(), &err[..100]);
}
