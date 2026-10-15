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
//! - Other methods: an error that names the entry.

use std::collections::HashMap;
use std::path::Path;

use rmlx_mlx::{Array, Dtype};
use thiserror::Error;
use tracing::{debug, info, instrument};

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
        if end_of(pos, 46, "central directory entry").map_err(NpzError::Zip)? > data.len() {
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
            return Err(NpzError::Zip(format!(
                "central directory entry {entry_idx}: bad signature {sig:#010x} at offset {pos}"
            )));
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
        let member = std::str::from_utf8(fname_bytes).map_err(|_| {
            NpzError::Zip(format!("entry {entry_idx}: filename is not valid UTF-8"))
        })?;

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

        if member.ends_with('/') {
            continue;
        }
        let Some(fname) = member.strip_suffix(".npy") else {
            debug!(member, "skipping a member that is not an .npy file");
            continue;
        };

        // 4. Read compressed data from local header.
        let raw = read_local_entry(data, local_hdr_offset, comp_size, uncomp_size, comp_method)
            .map_err(|cause| NpzError::Zip(format!("entry {fname}: {cause}")))?;

        // 5. Parse the .npy payload.
        let array = parse_npy_array(fname, &raw)?;
        map.insert(fname.to_owned(), array);
    }

    info!(n_weights = map.len(), "NPZ loaded via central directory");
    Ok(map)
}

/// `base + len` for an offset or a size that the archive states; an overflow
/// is a corrupt archive.
fn end_of(base: usize, len: usize, what: &str) -> Result<usize, String> {
    base.checked_add(len)
        .ok_or_else(|| format!("{what}: offset {base:#x} + {len} overflows"))
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

    if end_of(eocd64_offset, 56, "ZIP64 EOCD record").map_err(NpzError::Zip)? > data.len() {
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
/// header in ZIP64 archives where local sizes may be `0xFFFFFFFF`). A deflate
/// stream that inflates to more than `uncomp_size` bytes is an error.
fn read_local_entry(
    data: &[u8],
    offset: usize,
    comp_size: usize,
    uncomp_size: usize,
    comp_method: u16,
) -> Result<Vec<u8>, String> {
    let header_end = end_of(offset, 30, "local header")?;
    if header_end > data.len() {
        return Err(format!("local header at {offset:#x}: truncated"));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "offset + 30 <= data.len() checked above"
    )]
    let sig = u32_le(&data[offset..offset + 4]);
    if sig != SIG_LOCAL {
        return Err(format!(
            "local header at {offset:#x}: bad signature {sig:#010x}"
        ));
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
    let data_start = header_end + fname_len + extra_len;

    let data_end = end_of(data_start, comp_size, "local entry data")?;
    if data_end > data.len() {
        return Err(format!(
            "local entry data at {data_start:#x}+{comp_size}: exceeds file size"
        ));
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "data_start and data_end both bounded by data.len() check above"
    )]
    let entry_data = &data[data_start..data_end];

    match comp_method {
        0 => Ok(entry_data.to_vec()),
        8 => miniz_oxide::inflate::decompress_to_vec_with_limit(entry_data, uncomp_size).map_err(
            |e| format!("deflate decompress failed (declared size {uncomp_size} bytes): {e}"),
        ),
        other => Err(format!("unsupported ZIP compression method {other}")),
    }
}

// ── NPY parser ───────────────────────────────────────────────────────────────

/// Parse a `.npy` byte slice into an [`Array`].
///
/// Supports NPY format version 1.0 (2-byte header length) and 2.0+ (4-byte
/// header length). The dtypes are `f2`, `f4`, `b1`, `u1`, `i1` (as `U8`), `u4`,
/// `i4`, and `i8` (int64), which is stored as `I32`. Everything else is an
/// error that names the entry: another dtype, a multi-byte dtype with the `>`
/// mark (the marks `<`, `=` and `|` read as little-endian; `=` is native order,
/// little-endian on Apple Silicon), `'fortran_order': True`, a shape that does
/// not parse or whose byte count overflows, a dimension beyond int32, data
/// shorter than the shape, and an `i8` value that int32 cannot hold.
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

    let shape = extract_npy_shape(header_str).ok_or_else(|| npy_err("cannot parse shape"))?;
    let (mark, code) = npy_descr(header_str).ok_or_else(|| npy_err("cannot parse dtype"))?;
    if is_big_endian(mark, code) {
        return Err(npy_err(&format!(
            "big-endian dtype '>{code}' is not supported (byte order)"
        )));
    }
    if is_fortran_order(header_str) {
        return Err(npy_err(
            "'fortran_order': True is not supported (column-major data)",
        ));
    }
    let is_int64 = code == "i8";
    let (dtype, elem_bytes) = if is_int64 {
        (Dtype::I32, 8)
    } else {
        let dtype = dtype_of(code).ok_or_else(|| {
            npy_err(&format!(
                "unknown dtype '{}{code}'",
                mark.map_or(String::new(), String::from)
            ))
        })?;
        let width = match dtype {
            Dtype::U8 => 1,
            Dtype::F16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::Bf16 => return Err(npy_err(&format!("unsupported dtype {dtype:?}"))),
        };
        (dtype, width)
    };

    let needed = shape
        .iter()
        .try_fold(elem_bytes, |bytes: usize, &dim| bytes.checked_mul(dim))
        .ok_or_else(|| npy_err("shape is too large: its byte count overflows"))?;
    let raw = &data[header_end..];
    let bytes = raw.get(..needed).ok_or_else(|| {
        npy_err(&format!(
            "data truncated (need {needed}, have {})",
            raw.len()
        ))
    })?;
    let dims = shape
        .iter()
        .map(|&dim| i32::try_from(dim))
        .collect::<Result<Vec<i32>, _>>()
        .map_err(|_| npy_err("a dimension does not fit in int32"))?;

    let array = if is_int64 {
        let narrowed = int64_to_i32_bytes(bytes)
            .map_err(|value| npy_err(&format!("int64 value {value} does not fit in int32")))?;
        Array::from_bytes(&narrowed, &dims, dtype)
    } else {
        Array::from_bytes(bytes, &dims, dtype)
    };
    array.map_err(|e| NpzError::Mlx(e.to_string()))
}

// ── NPY header field extractors (pub for tests) ───────────────────────────────

/// The byte-order mark (`<`, `>`, `=` or `|`, if the descriptor has one) and
/// the type code of the dtype descriptor of a `.npy` header, as in
/// `(Some('<'), "i8")` for `'<i8'`.
fn npy_descr(header: &str) -> Option<(Option<char>, &str)> {
    let start = header.find("'descr'")?;
    let rest = &header[start + 7..];
    let rest = rest.trim_start_matches([' ', ':'].as_ref());
    let rest = rest.trim_start_matches(['\'', '"'].as_ref());
    let end = rest.find(['\'', '"']).unwrap_or(rest.len());
    let descr = &rest[..end];
    let mark = descr
        .chars()
        .next()
        .filter(|c| matches!(c, '<' | '>' | '=' | '|'));
    Some((
        mark,
        descr.trim_start_matches(['<', '>', '=', '|'].as_ref()),
    ))
}

/// Whether the header says the data is stored column-major.
fn is_fortran_order(header: &str) -> bool {
    header.find("'fortran_order'").is_some_and(|start| {
        header[start + 15..]
            .trim_start_matches([' ', ':'].as_ref())
            .starts_with("True")
    })
}

/// A multi-byte dtype with the `>` mark. Byte order has no meaning for one
/// byte (`b1`, `u1`, `i1`).
fn is_big_endian(mark: Option<char>, code: &str) -> bool {
    mark == Some('>') && !matches!(code, "b1" | "u1" | "i1")
}

/// The dtype that a type code stores as: `f2`, `f4`, the booleans and 1-byte
/// ints (`b1`, `u1`, `i1`) as `U8`, and `u4` and `i4`. `i8` has no MLX dtype;
/// [`parse_npy_array`] reads it as `I32`.
fn dtype_of(code: &str) -> Option<Dtype> {
    match code {
        "f2" => Some(Dtype::F16),
        "f4" => Some(Dtype::F32),
        "b1" | "u1" | "i1" => Some(Dtype::U8),
        "u4" => Some(Dtype::U32),
        "i4" => Some(Dtype::I32),
        _ => None,
    }
}

/// Little-endian int64 `bytes` as little-endian int32. The error is the first
/// value that int32 cannot hold.
pub(crate) fn int64_to_i32_bytes(bytes: &[u8]) -> Result<Vec<u8>, i64> {
    let mut narrowed = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(8) {
        let mut word = [0_u8; 8];
        word.copy_from_slice(chunk);
        let value = i64::from_le_bytes(word);
        let small = i32::try_from(value).map_err(|_| value)?;
        narrowed.extend_from_slice(&small.to_le_bytes());
    }
    Ok(narrowed)
}

/// Extract the shape tuple from a `.npy` header string. `None` when a
/// dimension is not a non-negative integer that fits in `usize`.
pub fn extract_npy_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape'")?;
    let rest = &header[start + 7..];
    let p0 = rest.find('(')?;
    let p1 = rest.find(')')?;
    let inner = &rest[p0 + 1..p1];
    let fields: Vec<&str> = inner.split(',').map(str::trim).collect();
    // `(n,)` ends in an empty field; any other empty field is a bad dimension.
    let (last, before) = fields.split_last()?;
    let dims = if last.is_empty() { before } else { &fields };
    dims.iter().map(|field| field.parse().ok()).collect()
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
pub(crate) mod npz_tests;
