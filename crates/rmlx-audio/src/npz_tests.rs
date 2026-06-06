//! Unit tests for the NPZ / ZIP parser.

use super::{extract_npy_dtype, extract_npy_shape, parse_npz};
use rmlx_mlx::Dtype;

// ── NPY header helpers ────────────────────────────────────────────────────────

#[test]
fn npy_dtype_f2() {
    let hdr = "{'descr': '<f2', 'fortran_order': False, 'shape': (128,), }";
    assert_eq!(extract_npy_dtype(hdr), Some(Dtype::F16));
}

#[test]
fn npy_dtype_f4() {
    let hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (100,), }";
    assert_eq!(extract_npy_dtype(hdr), Some(Dtype::F32));
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
#[test]
fn deflate_entry_roundtrip() {
    use std::io::Write;

    let payload = npy_f32_scalar(1.0);
    let compressed = miniz_oxide::deflate::compress_to_vec(&payload, 6);
    let fname = b"deflated.npy";
    let fname_len = fname.len() as u16;
    let comp_size = compressed.len() as u32;
    let uncomp_size = payload.len() as u32;

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
    zip.write_all(fname).unwrap();
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
    zip.write_all(fname).unwrap();

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

    let map = parse_npz(&zip).expect("deflate parse should succeed");
    let key = "deflated";
    assert!(map.contains_key(key), "key '{key}' missing");
    assert_eq!(map[key].ndim(), 0);
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
