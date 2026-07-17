//! Byte-accounting primitives shared by every KV store's `byte_size`.
//!
//! # Why these exist
//!
//! KV byte totals are the number the quantized codecs are accepted or rejected
//! on. The only way that number stays true as stores gain buffers is if it is
//! *derived* from the allocations themselves — never restated as a per-codec
//! formula that a reader has to remember to update.
//!
//! So each store's `byte_size` is built from these two primitives only:
//!
//! - [`array_bytes`] / [`opt_array_bytes`] — an `Array`'s real shape × dtype
//!   item size. Picks up a dtype or capacity change for free.
//! - [`vec_bytes`] / [`opt_vec_bytes`] — a `Vec`'s real length × element size.
//!
//! # The drift guard
//!
//! Every `byte_size` opens with an exhaustive `let Self { .. } = self`
//! destructure that names **all** fields and binds non-payload ones to `_`.
//! Adding a buffer to a store is then a hard compile error (E0027, "pattern
//! does not mention field") until the author classifies it as payload or
//! metadata. That is deliberate: the alternative — a `..` rest-pattern — is
//! what let a GPU ring live in these stores while the byte total stayed
//! silently blind to it.

use rmlx_mlx::Array;

/// Bytes an `Array` occupies: element count × dtype item size.
///
/// Reads the live shape, so a buffer that grew (or changed dtype) reports its
/// new size with no formula to update. No FFI eval — safe to call anywhere.
pub(crate) fn array_bytes(a: &Array) -> u64 {
    let n: u64 = a.shape().iter().map(|&d| d as u64).product();
    n * a.dtype().itemsize() as u64
}

/// [`array_bytes`] for an optional buffer; unallocated (`None`) is 0 bytes.
pub(crate) fn opt_array_bytes(a: Option<&Array>) -> u64 {
    a.map_or(0, array_bytes)
}

/// Bytes a slice occupies: length × element size.
pub(crate) fn vec_bytes<T>(v: &[T]) -> u64 {
    v.len() as u64 * size_of::<T>() as u64
}

/// [`vec_bytes`] for an optional payload; absent (`None`) is 0 bytes.
pub(crate) fn opt_vec_bytes<T>(v: Option<&Vec<T>>) -> u64 {
    v.map_or(0, |v| vec_bytes(v))
}

/// Bytes of the *filled* prefix of a `[B, kv_h, seq, D]` per-position buffer.
///
/// The bf16 decode mirrors and the SWA ring are pre-allocated to a ceiling and
/// only compacted later, so the same configuration would report different
/// totals depending on when the metric is read. Only `filled` positions ever
/// hold live K/V. Scaling by the filled length — using the buffer's own
/// per-position size, so per-layer head_dim and dtype are picked up — keeps the
/// figure comparable across contexts.
///
/// A buffer that is not the 4-D per-position shape is counted whole: it has no
/// sequence axis to take a prefix of.
#[allow(
    clippy::indexing_slicing,
    reason = "indices 0..=3 are bounds-checked by the `s.len() != 4` early return above"
)]
pub(crate) fn filled_seq_bytes(a: &Array, filled: u64) -> u64 {
    let s = a.shape();
    if s.len() != 4 {
        return array_bytes(a);
    }
    let per_pos = s[0] as u64 * s[1] as u64 * s[3] as u64 * a.dtype().itemsize() as u64;
    let cap = s[2] as u64;
    per_pos * filled.min(cap)
}

/// [`filled_seq_bytes`] for an optional buffer; unallocated (`None`) is 0 bytes.
pub(crate) fn opt_filled_seq_bytes(a: Option<&Array>, filled: u64) -> u64 {
    a.map_or(0, |a| filled_seq_bytes(a, filled))
}
