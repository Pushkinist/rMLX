//! Dense code plane — the shared bit-packing of the packed K/V codecs.
//!
//! # Layout
//!
//! A codec's codes for one `(token, KV head)` row are packed **densely across
//! the row's groups**, LSB-first, `bits` per code, and the row is padded to a
//! whole `u32`. Code `idx` of a row therefore starts at bit `idx * bits` of the
//! row and may straddle two words.
//!
//! Row-aligned rather than packed continuously across the whole buffer because
//! the ring appends whole tokens: an append is a `slice_update` at a word
//! offset, and every reader indexes by `(token, head)`. A row starting at a bit
//! offset that is not a multiple of 32 has neither. The padding costs less than
//! one word per row — 0.25 bits per value at `head_dim = 128`.
//!
//! # Why one packer
//!
//! The codecs differ only in how many codes a group carries — four for the iso
//! quaternion block, three for the rotor grade-1 block — so both reduce to
//! "`n_codes` codes of `bits` width per row". A second packer would be a twin,
//! and the two would drift in exactly the way the store's rate figures did.

use std::fmt::Write as _;

/// `u32` words one row of `n_codes` codes occupies at `bits` per code.
///
/// The padding to a whole word is the row-alignment described in the module
/// docs.
#[must_use]
pub fn row_words(n_codes: usize, bits: u8) -> usize {
    (n_codes * bits as usize).div_ceil(32)
}

/// Pack one row's codes and append the row's words to `out`.
///
/// Appends exactly [`row_words`]`(codes.len(), bits)` words. Each code is
/// masked to `bits`, so a caller that hands over an out-of-range index writes a
/// truncated code rather than corrupting its neighbour.
pub fn pack_row_into(codes: &[u8], bits: u8, out: &mut Vec<u32>) {
    let mask = code_mask(bits);
    let base = out.len();
    out.resize(base + row_words(codes.len(), bits), 0);
    for (idx, &code) in codes.iter().enumerate() {
        let bit = idx * bits as usize;
        let word = base + bit / 32;
        let off = bit % 32;
        let value = u32::from(code) & mask;
        // `word` and `word + 1` are both inside the region just resized:
        // the last code ends at bit `n_codes * bits <= row_words * 32`.
        #[allow(
            clippy::indexing_slicing,
            reason = "word < base + row_words(codes.len(), bits) = out.len() by the resize above"
        )]
        {
            out[word] |= value << off;
            if off + bits as usize > 32 {
                out[word + 1] |= value >> (32 - off);
            }
        }
    }
}

/// Read code `idx` out of one row's words.
///
/// `row` is the row's own word slice — [`row_words`] long. A shorter slice is a
/// caller contract violation and panics rather than reading a neighbouring
/// row's bits as this row's code.
#[must_use]
#[inline]
pub fn read_code(row: &[u32], idx: usize, bits: u8) -> u8 {
    let bit = idx * bits as usize;
    let word = bit / 32;
    let off = bit % 32;
    // Both indexes are inside a row sized by `row_words`, which is what every
    // caller slices; see the doc comment.
    #[allow(
        clippy::indexing_slicing,
        reason = "row is row_words(n_codes, bits) long by the caller contract; code idx < n_codes \
                  ends inside it"
    )]
    let mut v = row[word] >> off;
    #[allow(
        clippy::indexing_slicing,
        reason = "a straddling code has bits left in the next word, which the row contains"
    )]
    if off + bits as usize > 32 {
        v |= row[word + 1] << (32 - off);
    }
    (v & code_mask(bits)) as u8
}

/// Low `bits` set.
#[inline]
fn code_mask(bits: u8) -> u32 {
    (1_u32 << bits) - 1
}

/// MSL declarations for the dense code plane: the `CP_*` parameters, the
/// per-group reader and the atomic writer.
///
/// Emitted into each codec's generated header so the kernels read the plane
/// through one definition, the way the Rust side reads it through
/// [`read_code`]. `codes_per_group` is the codec's per-group code count — four
/// for iso, three for rotor — which is both the reader's unit and what turns a
/// group index into a code index inside the kernels.
pub(crate) fn render_msl_code_plane(bits: u8, codes_per_group: usize) -> String {
    let mask = code_mask(bits);
    // `uint3` for rotor's grade-1 triple, `uint4` for iso's quaternion block.
    let vec = format!("uint{codes_per_group}");
    let mut extract = String::new();
    for e in 0..codes_per_group {
        let lane = ["x", "y", "z", "w"].get(e).copied().unwrap_or("x");
        let _ = writeln!(extract, "    out.{lane} = (v >> {e}u * CP_BITS) & CP_MASK;");
    }
    format!(
        "
// Dense code plane — codes packed LSB-first across a row's groups, the row
// padded to a whole u32. Mirrors crate::code_plane.
#define CP_BITS {bits}u
#define CP_MASK 0x{mask:X}u
#define CP_CODES_PER_GROUP {codes_per_group}u

// u32 words one row of `n_groups` groups occupies.
inline uint cp_row_words(uint n_groups) {{
    return (n_groups * CP_CODES_PER_GROUP * CP_BITS + 31u) / 32u;
}}

// One group's codes, returned by value: a vector lives in registers where a
// `thread uint[]` out-parameter is addressable and can be spilled to thread
// memory, which on the decode path costs more than the loads it saves.
typedef {vec} cp_group_t;

// Read all CP_CODES_PER_GROUP codes of group `group_id`.
//
// Templated over the address space because MLX binds a small input buffer as
// `constant` and a large one as `device`, and MSL will not convert between the
// two: a single-address-space reader compiles for one dispatch shape and fails
// the other at JIT time.
//
// A whole group is CP_CODES_PER_GROUP * CP_BITS bits — at most 32, and never
// spanning more than a word pair — so this is a word pair, where a code at a
// time would be one load per code. That matters: every decode
// lane needs its group's whole code set (the quaternion product and the rotor
// sandwich both mix all of them), so a per-code reader multiplies the loads on
// the hottest path in the kernel by the group size.
template <typename P>
inline cp_group_t cp_read_group(P codes, uint row_base, uint group_id) {{
    uint span = CP_CODES_PER_GROUP * CP_BITS;
    uint bit  = group_id * span;
    uint word = row_base + (bit >> 5u);
    uint off  = bit & 31u;
    // A group's span never exceeds 32 bits, so its bits fit one word once
    // shifted down — no 64-bit arithmetic, which Apple GPUs emulate.
    //
    // The straddle is a select, not a branch. Whether a group crosses a word
    // boundary depends on its index, so within one simdgroup some lanes cross
    // and some do not: a branch there is executed by every lane anyway, and
    // costs the divergence on top. The second word is addressed with a select
    // (`word` when there is nothing to fetch, so never out of bounds) and its
    // contribution is masked to zero the same way.
    bool crosses = off + span > 32u;
    uint v       = codes[word] >> off;
    uint hi      = codes[word + (crosses ? 1u : 0u)];
    v |= crosses ? (hi << (32u - off)) : 0u;
    cp_group_t out;
{extract}
    return out;
}}

// OR code `idx` into the row whose first word is `codes[row_base]`. The plane
// is zero-initialised at dispatch, so an OR is a write; a code that straddles
// two words ORs into both.
inline void cp_write_code(device uint* codes, uint row_base, uint idx, uint code) {{
    uint bit  = idx * CP_BITS;
    uint word = row_base + (bit >> 5u);
    uint off  = bit & 31u;
    uint v    = code & CP_MASK;
    atomic_fetch_or_explicit((device atomic_uint *)&codes[word], v << off,
                             memory_order_relaxed);
    if (off + CP_BITS > 32u) {{
        atomic_fetch_or_explicit((device atomic_uint *)&codes[word + 1u],
                                 v >> (32u - off), memory_order_relaxed);
    }}
}}
"
    )
}

#[cfg(test)]
#[path = "code_plane_tests.rs"]
mod code_plane_tests;
