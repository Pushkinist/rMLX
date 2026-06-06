// LOC-exempt: ZIP format parsing is a single cohesive unit; splitting the
// EOCD/ZIP64-locator/central-directory/local-header parse sequence across
// submodules would scatter the offset arithmetic without cohesion gain.

//! ZIP/NPZ archive parser with full ZIP64 central-directory support.
//!
//! ## Why central-directory parsing?
//!
//! NumPy's `numpy.save` and `numpy.savez` emit ZIP archives. Large NPZ files
//! (≥ 4 GiB total or individual entries ≥ 4 GiB) use ZIP64 extensions where
//! local-file-header size fields are `0xFFFFFFFF` and the real 64-bit sizes
//! live in a ZIP64 extra field (header id `0x0001`). The previous
//! local-header scanner hard-errored on these entries.
//!
//! The central directory always carries correct 64-bit sizes for ZIP64 entries
//! (via the same extra-field mechanism). Walking it instead of the local
//! headers is both more correct and more robust.
//!
//! ## Format overview
//!
//! ```text
//! [local headers + file data ...]
//! [central directory records ...]
//! [ZIP64 end-of-central-directory record]      -- only present in ZIP64
//! [ZIP64 end-of-central-directory locator]     -- only present in ZIP64
//! [end-of-central-directory record]
//! ```
//!
//! To locate the central directory:
//! 1. Scan backwards for the EOCD magic (`PK\x05\x06`).
//! 2. If EOCD's `cd_offset` or `total_entries` are the sentinel `0xFFFFFFFF` /
//!    `0xFFFF`, the ZIP64 EOCD locator (`PK\x06\x07`) immediately precedes the
//!    EOCD. Read the ZIP64 EOCD (`PK\x06\x06`) at the 64-bit offset it
//!    contains.
//! 3. Walk central-directory records (`PK\x01\x02`). For each, read the
//!    local-header offset (ZIP64 extra field if the 32-bit value is the
//!    sentinel).
//! 4. Seek to each local header, skip past its filename + extra fields, and
//!    read the compressed data. Use the sizes from the central directory —
//!    local headers in ZIP64 archives have sentinel sizes.
//!
//! ## Compression
//!
//! - Method 0 (Stored): verbatim copy.
//! - Method 8 (Deflate): decompress with `miniz_oxide`.
//! - Other methods: skip with a warning.

use std::collections::HashMap;
use std::path::Path;

use rmlx_mlx::{Array, Dtype};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

// ── Error type ────────────────────────────────────────────────────────────────

/// NPZ / ZIP parse errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum NpzError {
    /// File I/O error.
    #[error("file I/O: {0}")]
    Io(#[from] std::io::Error),
    /// ZIP structural error (bad magic, truncation, etc.).
    #[error("ZIP structure: {0}")]
    Zip(String),
    /// NPY array parse error.
    #[error("NPY parse ({name}): {msg}")]
    Npy {
        /// Entry name that failed to parse.
        name: String,
        /// Description of the parse failure.
        msg: String,
    },
    /// MLX array construction error.
    #[error("MLX: {0}")]
    Mlx(String),
    /// EOCD not found.
    #[error("no end-of-central-directory record found; file may be corrupt or not a ZIP")]
    NoEocd,
}

/// A map of weight name → `Array`.
pub type WeightMap = HashMap<String, Array>;

// ── Magic constants ───────────────────────────────────────────────────────────

/// Local file header signature.
const SIG_LOCAL: u32 = 0x0403_4b50;
/// Central directory file header signature.
const SIG_CENTRAL: u32 = 0x0201_4b50;
/// ZIP64 end-of-central-directory record signature.
const SIG_EOCD64: u32 = 0x0606_4b50;
/// ZIP64 end-of-central-directory locator signature.
const SIG_EOCD64_LOC: u32 = 0x0706_4b50;
/// ZIP64 extra field header id.
const ZIP64_EXTRA_ID: u16 = 0x0001;

/// Sentinel value in 32-bit fields indicating "read from ZIP64 extra field".
const ZIP64_SENTINEL_U32: u32 = 0xFFFF_FFFF;
/// Sentinel value in 16-bit fields indicating "read from ZIP64 extra field".
const ZIP64_SENTINEL_U16: u16 = 0xFFFF;

// ── Public entry point ────────────────────────────────────────────────────────

/// Load all `.npy` arrays from a `.npz` archive at `path`.
///
/// Supports ZIP32 and ZIP64 archives. Deflate-compressed and stored entries
/// are both handled. The `.npy` suffix is stripped from key names.
#[instrument(skip(path), fields(path = %path.as_ref().display()), level = "info")]
pub fn load_npz(path: impl AsRef<Path>) -> Result<WeightMap, NpzError> {
    let data = std::fs::read(path.as_ref())?;
    info!(bytes = data.len(), "reading NPZ");
    parse_npz(&data)
}

// ── Central-directory walker ──────────────────────────────────────────────────

/// Parse a `.npz` byte slice via the central directory.
///
/// This is the authoritative parser. It locates the EOCD (with ZIP64 fallback),
/// walks every central-directory entry, and for each entry reads the
/// compressed data from the local-header offset.
#[allow(
    clippy::too_many_lines,
    reason = "central-directory parse is one sequential algorithm; factoring each step into a helper would require passing many offset/slice pairs without clarity gain"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "ZIP central-directory walk is inherently branchy: ZIP64 sentinel checks, extra-field parsing, and per-entry routing are all inline for cache locality; factoring branches out would not reduce real complexity"
)]
pub fn parse_npz(data: &[u8]) -> Result<WeightMap, NpzError> {
    // 1. Find EOCD.
    let eocd_pos = find_eocd(data)?;
    let (cd_offset, total_entries) = read_eocd(data, eocd_pos)?;

    debug!(cd_offset, total_entries, "central directory located");

    // 2. Walk central directory.
    let mut pos = cd_offset;
    let mut map = WeightMap::new();

    for entry_idx in 0..total_entries {
        if pos + 46 > data.len() {
            return Err(NpzError::Zip(format!(
                "central directory entry {entry_idx}: truncated at offset {pos}"
            )));
        }

        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above (pos + 46 <= data.len())"
        )]
        let sig = u32_le(&data[pos..pos + 4]);
        if sig != SIG_CENTRAL {
            // Hit end-of-central-directory or padding; stop.
            break;
        }

        // All offsets pos+10..pos+46 are within bounds: checked pos + 46 <= data.len() above.
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let comp_method = u16_le(&data[pos + 10..pos + 12]);
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let comp_size_raw = u32_le(&data[pos + 20..pos + 24]);
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let uncomp_size_raw = u32_le(&data[pos + 24..pos + 28]);
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let fname_len = u16_le(&data[pos + 28..pos + 30]) as usize;
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let extra_len = u16_le(&data[pos + 30..pos + 32]) as usize;
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let comment_len = u16_le(&data[pos + 32..pos + 34]) as usize;
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds checked above: pos + 46 <= data.len()"
        )]
        let local_hdr_offset_raw = u32_le(&data[pos + 42..pos + 46]);

        let fname_start = pos + 46;
        let extra_start = fname_start + fname_len;
        let entry_end = extra_start + extra_len + comment_len;

        if entry_end > data.len() {
            return Err(NpzError::Zip(format!(
                "central directory entry {entry_idx}: variable-length fields exceed data"
            )));
        }

        #[allow(
            clippy::indexing_slicing,
            reason = "fname_start + fname_len = extra_start, both bounded by entry_end <= data.len()"
        )]
        let fname_bytes = &data[fname_start..extra_start];
        let fname = std::str::from_utf8(fname_bytes)
            .map_err(|_| NpzError::Zip(format!("entry {entry_idx}: filename is not valid UTF-8")))?
            .trim_end_matches(".npy")
            .to_owned();

        // 3. Parse ZIP64 extra field (if present) from central directory.
        #[allow(
            clippy::indexing_slicing,
            reason = "extra_start + extra_len = end of extra, bounded by entry_end <= data.len()"
        )]
        let extra = &data[extra_start..extra_start + extra_len];
        let z64 = parse_zip64_extra(extra, comp_size_raw, uncomp_size_raw, local_hdr_offset_raw);

        let comp_size = z64.comp_size;
        let uncomp_size = z64.uncomp_size;
        let local_hdr_offset = z64.local_hdr_offset;

        pos = entry_end;

        // Skip directory entries and empty placeholder entries.
        if comp_size == 0 && uncomp_size == 0 {
            continue;
        }
        if fname.ends_with('/') {
            continue;
        }

        // 4. Read compressed data from local header.
        let raw = match read_local_entry(data, local_hdr_offset, comp_size, comp_method) {
            Ok(r) => r,
            Err(e) => {
                warn!(entry = fname, error = %e, "skipping unreadable NPZ entry");
                continue;
            }
        };

        if fname.is_empty() {
            continue;
        }

        // 5. Parse the .npy payload.
        match parse_npy_array(&fname, &raw) {
            Ok(arr) => {
                map.insert(fname, arr);
            }
            Err(e) => {
                warn!(error = %e, "skipping npy entry");
            }
        }
    }

    info!(n_weights = map.len(), "NPZ loaded via central directory");
    Ok(map)
}

// ── EOCD locator ─────────────────────────────────────────────────────────────

/// Scan backwards for the EOCD signature. Returns the byte offset of the
/// `PK\x05\x06` magic.
///
/// The EOCD may have a variable-length comment (0–65535 bytes) appended after
/// the fixed 22-byte record. We search the last 65535 + 22 bytes of the file.
fn find_eocd(data: &[u8]) -> Result<usize, NpzError> {
    let eocd_magic = [0x50, 0x4b, 0x05, 0x06];
    let search_start = data.len().saturating_sub(65536 + 22);
    // Scan from the end.
    for i in (search_start..data.len().saturating_sub(21)).rev() {
        #[allow(
            clippy::indexing_slicing,
            reason = "i >= search_start, i + 4 <= data.len() (loop bound is data.len()-22+1)"
        )]
        if data[i..i + 4] == eocd_magic {
            return Ok(i);
        }
    }
    Err(NpzError::NoEocd)
}

/// Read the central-directory offset and entry count from the EOCD at `eocd_pos`.
///
/// If the EOCD contains sentinel values, locates and reads the ZIP64 EOCD record
/// instead.
fn read_eocd(data: &[u8], eocd_pos: usize) -> Result<(usize, usize), NpzError> {
    if eocd_pos + 22 > data.len() {
        return Err(NpzError::Zip("EOCD record truncated".to_owned()));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "eocd_pos + 22 <= data.len() checked above"
    )]
    let total_entries_raw = u16_le(&data[eocd_pos + 10..eocd_pos + 12]);
    #[allow(
        clippy::indexing_slicing,
        reason = "eocd_pos + 22 <= data.len() checked above"
    )]
    let cd_offset_raw = u32_le(&data[eocd_pos + 16..eocd_pos + 20]);

    if cd_offset_raw != ZIP64_SENTINEL_U32 && total_entries_raw != ZIP64_SENTINEL_U16 {
        // Standard ZIP32 EOCD is fully valid.
        return Ok((cd_offset_raw as usize, total_entries_raw as usize));
    }

    // ZIP64: find the locator immediately before the EOCD.
    read_eocd64(data, eocd_pos)
}

/// Locate the ZIP64 EOCD via the ZIP64 EOCD locator record.
fn read_eocd64(data: &[u8], eocd_pos: usize) -> Result<(usize, usize), NpzError> {
    // The ZIP64 EOCD locator is 20 bytes and sits immediately before the EOCD.
    if eocd_pos < 20 {
        return Err(NpzError::Zip(
            "ZIP64 EOCD locator: not enough space before EOCD".to_owned(),
        ));
    }
    let loc_pos = eocd_pos - 20;

    #[allow(
        clippy::indexing_slicing,
        reason = "loc_pos + 20 = eocd_pos <= data.len()"
    )]
    let loc_sig = u32_le(&data[loc_pos..loc_pos + 4]);
    if loc_sig != SIG_EOCD64_LOC {
        return Err(NpzError::Zip(format!(
            "expected ZIP64 EOCD locator at {loc_pos:#x}, got {loc_sig:#010x}"
        )));
    }

    // Offset of the ZIP64 EOCD record (8 bytes at locator offset 8).
    #[allow(
        clippy::indexing_slicing,
        reason = "loc_pos + 20 <= data.len(), so loc_pos + 8 + 8 <= data.len()"
    )]
    let eocd64_offset = u64_le(&data[loc_pos + 8..loc_pos + 16]) as usize;

    if eocd64_offset + 56 > data.len() {
        return Err(NpzError::Zip(format!(
            "ZIP64 EOCD64 at {eocd64_offset:#x}: truncated"
        )));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "eocd64_offset + 56 <= data.len() checked above"
    )]
    let eocd64_sig = u32_le(&data[eocd64_offset..eocd64_offset + 4]);
    if eocd64_sig != SIG_EOCD64 {
        return Err(NpzError::Zip(format!(
            "expected ZIP64 EOCD64 at {eocd64_offset:#x}, got {eocd64_sig:#010x}"
        )));
    }

    // total_entries is at offset 32 (8 bytes).
    #[allow(
        clippy::indexing_slicing,
        reason = "eocd64_offset + 56 <= data.len() checked above"
    )]
    let total_entries = u64_le(&data[eocd64_offset + 32..eocd64_offset + 40]) as usize;
    // cd_offset is at offset 48 (8 bytes).
    #[allow(
        clippy::indexing_slicing,
        reason = "eocd64_offset + 56 <= data.len() checked above"
    )]
    let cd_offset = u64_le(&data[eocd64_offset + 48..eocd64_offset + 56]) as usize;

    debug!(eocd64_offset, total_entries, cd_offset, "ZIP64 EOCD64 read");

    Ok((cd_offset, total_entries))
}

// ── ZIP64 extra field parser ──────────────────────────────────────────────────

struct Zip64Fields {
    comp_size: usize,
    uncomp_size: usize,
    local_hdr_offset: usize,
}

/// Extract ZIP64 fields from the extra field block.
///
/// The ZIP64 extra field (header id `0x0001`) contains fields only for those
/// that hold the sentinel `0xFFFFFFFF` (or `0xFFFF`) in the base record.
/// The order is: uncompressed size, compressed size, local header offset —
/// each present only if the corresponding base field was sentinel.
fn parse_zip64_extra(
    extra: &[u8],
    comp_size_raw: u32,
    uncomp_size_raw: u32,
    local_hdr_offset_raw: u32,
) -> Zip64Fields {
    let mut comp_size = comp_size_raw as usize;
    let mut uncomp_size = uncomp_size_raw as usize;
    let mut local_hdr_offset = local_hdr_offset_raw as usize;

    // Fast path: no ZIP64 sentinels.
    if comp_size_raw != ZIP64_SENTINEL_U32
        && uncomp_size_raw != ZIP64_SENTINEL_U32
        && local_hdr_offset_raw != ZIP64_SENTINEL_U32
    {
        return Zip64Fields {
            comp_size,
            uncomp_size,
            local_hdr_offset,
        };
    }

    // Walk extra field headers looking for id 0x0001.
    let mut i = 0usize;
    while i + 4 <= extra.len() {
        #[allow(
            clippy::indexing_slicing,
            reason = "i + 4 <= extra.len() checked by while condition"
        )]
        let hdr_id = u16_le(&extra[i..i + 2]);
        #[allow(
            clippy::indexing_slicing,
            reason = "i + 4 <= extra.len() checked by while condition"
        )]
        let hdr_size = u16_le(&extra[i + 2..i + 4]) as usize;
        let field_start = i + 4;
        let field_end = field_start + hdr_size;

        if field_end > extra.len() {
            break; // Truncated extra field; best-effort stop.
        }

        if hdr_id == ZIP64_EXTRA_ID {
            // Fields are present in order: uncomp_size, comp_size, local_hdr_offset.
            // Each 8 bytes, present only when the corresponding base field was sentinel.
            let mut cursor = field_start;

            if uncomp_size_raw == ZIP64_SENTINEL_U32 && cursor + 8 <= field_end {
                #[allow(
                    clippy::indexing_slicing,
                    reason = "cursor + 8 <= field_end <= extra.len()"
                )]
                {
                    uncomp_size = u64_le(&extra[cursor..cursor + 8]) as usize;
                    cursor += 8;
                }
            }
            if comp_size_raw == ZIP64_SENTINEL_U32 && cursor + 8 <= field_end {
                #[allow(
                    clippy::indexing_slicing,
                    reason = "cursor + 8 <= field_end <= extra.len()"
                )]
                {
                    comp_size = u64_le(&extra[cursor..cursor + 8]) as usize;
                    cursor += 8;
                }
            }
            if local_hdr_offset_raw == ZIP64_SENTINEL_U32 && cursor + 8 <= field_end {
                #[allow(
                    clippy::indexing_slicing,
                    reason = "cursor + 8 <= field_end <= extra.len()"
                )]
                {
                    local_hdr_offset = u64_le(&extra[cursor..cursor + 8]) as usize;
                }
            }
            break;
        }

        i = field_end;
    }

    Zip64Fields {
        comp_size,
        uncomp_size,
        local_hdr_offset,
    }
}

// ── Local-header entry reader ─────────────────────────────────────────────────

/// Read and decompress a single entry from the local header at `offset`.
///
/// Uses `comp_size` from the **central directory** (more reliable than the local
/// header in ZIP64 archives where local sizes may be `0xFFFFFFFF`).
fn read_local_entry(
    data: &[u8],
    offset: usize,
    comp_size: usize,
    comp_method: u16,
) -> Result<Vec<u8>, NpzError> {
    if offset + 30 > data.len() {
        return Err(NpzError::Zip(format!(
            "local header at {offset:#x}: truncated"
        )));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "offset + 30 <= data.len() checked above"
    )]
    let sig = u32_le(&data[offset..offset + 4]);
    if sig != SIG_LOCAL {
        return Err(NpzError::Zip(format!(
            "local header at {offset:#x}: bad signature {sig:#010x}"
        )));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "offset + 30 <= data.len() checked above"
    )]
    let fname_len = u16_le(&data[offset + 26..offset + 28]) as usize;
    #[allow(
        clippy::indexing_slicing,
        reason = "offset + 30 <= data.len() checked above"
    )]
    let extra_len = u16_le(&data[offset + 28..offset + 30]) as usize;
    let data_start = offset + 30 + fname_len + extra_len;

    let data_end = data_start + comp_size;
    if data_end > data.len() {
        return Err(NpzError::Zip(format!(
            "local entry data at {data_start:#x}+{comp_size}: exceeds file size"
        )));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "data_start and data_end both bounded by data.len() check above"
    )]
    let entry_data = &data[data_start..data_end];

    match comp_method {
        0 => Ok(entry_data.to_vec()),
        8 => miniz_oxide::inflate::decompress_to_vec(entry_data)
            .map_err(|e| NpzError::Zip(format!("deflate decompress failed: {e:?}"))),
        other => {
            warn!(
                method = other,
                "unsupported ZIP compression method; skipping"
            );
            Ok(Vec::new())
        }
    }
}

// ── NPY parser ───────────────────────────────────────────────────────────────

/// Parse a `.npy` byte slice into an [`Array`].
///
/// Supports NPY format version 1.0 (2-byte header length) and 2.0+ (4-byte
/// header length). Only `f2` (float16) and `f4` (float32) dtypes are accepted.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds explicitly checked via offset arithmetic before each access"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "Dtype has many variants; only F16/F32 are valid for NPZ weights; others are an error"
)]
#[allow(
    clippy::items_after_statements,
    reason = "local declarations after guards follow the sequential parse flow"
)]
pub fn parse_npy_array(name: &str, data: &[u8]) -> Result<Array, NpzError> {
    let npy_err = |msg: &str| NpzError::Npy {
        name: name.to_owned(),
        msg: msg.to_owned(),
    };

    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err(npy_err("not a valid .npy file (bad magic)"));
    }
    let major = data[6];
    let (header_len, header_start) = if major >= 2 {
        if data.len() < 12 {
            return Err(npy_err("too short for NPY v2 header"));
        }
        (
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
            12usize,
        )
    } else {
        (u16::from_le_bytes([data[8], data[9]]) as usize, 10usize)
    };

    let header_end = header_start + header_len;
    if header_end > data.len() {
        return Err(npy_err("header truncated"));
    }

    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| npy_err("header is not valid UTF-8"))?;

    let dtype = extract_npy_dtype(header_str).ok_or_else(|| npy_err("cannot parse dtype"))?;
    let shape = extract_npy_shape(header_str).ok_or_else(|| npy_err("cannot parse shape"))?;

    let raw = &data[header_end..];
    let n_elems: usize = shape.iter().product();
    let elem_bytes = match dtype {
        Dtype::F16 => 2,
        Dtype::F32 => 4,
        _ => {
            return Err(npy_err(&format!("unsupported dtype {dtype:?}")));
        }
    };
    let needed = n_elems * elem_bytes;
    if raw.len() < needed {
        return Err(npy_err(&format!(
            "data truncated (need {needed}, have {})",
            raw.len()
        )));
    }

    let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    Array::from_bytes(&raw[..needed], &shape_i32, dtype).map_err(|e| NpzError::Mlx(e.to_string()))
}

// ── NPY header field extractors (pub for tests) ───────────────────────────────

/// Extract the numpy dtype descriptor from a `.npy` header string.
pub fn extract_npy_dtype(header: &str) -> Option<Dtype> {
    let start = header.find("'descr'")?;
    let rest = &header[start + 7..];
    let rest = rest.trim_start_matches([' ', ':'].as_ref());
    let rest = rest.trim_start_matches(['\'', '"'].as_ref());
    let end = rest.find(['\'', '"']).unwrap_or(rest.len());
    let s = rest[..end].trim_start_matches(['<', '>', '=', '|'].as_ref());
    match s {
        "f2" => Some(Dtype::F16),
        "f4" => Some(Dtype::F32),
        _ => None,
    }
}

/// Extract the shape tuple from a `.npy` header string.
pub fn extract_npy_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape'")?;
    let rest = &header[start + 7..];
    let p0 = rest.find('(')?;
    let p1 = rest.find(')')?;
    let inner = &rest[p0 + 1..p1];
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    Some(dims)
}

// ── Little-endian read helpers ────────────────────────────────────────────────

#[inline]
fn u16_le(b: &[u8]) -> u16 {
    #[allow(
        clippy::indexing_slicing,
        reason = "callers verify slice is at least 2 bytes before calling"
    )]
    u16::from_le_bytes([b[0], b[1]])
}

#[inline]
fn u32_le(b: &[u8]) -> u32 {
    #[allow(
        clippy::indexing_slicing,
        reason = "callers verify slice is at least 4 bytes before calling"
    )]
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn u64_le(b: &[u8]) -> u64 {
    #[allow(
        clippy::indexing_slicing,
        reason = "callers verify slice is at least 8 bytes before calling"
    )]
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "npz_tests.rs"]
mod npz_tests;
