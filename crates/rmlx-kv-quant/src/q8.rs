// promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill) can still reach them across the crate
// boundary. Doc/visibility warnings on the promoted surface are silenced; the
// API is otherwise unchanged.
#![allow(missing_docs, missing_debug_implementations, unreachable_pub)]
//! CPU-side q8_0 quantization helpers for KV-cache encoding.
//!
//! Implements symmetric 8-bit quantization (`q8_0`) at group size 128.
//! Each group of 128 `f32` values is encoded to 128 `u8` codes plus one
//! `f32` scale: `scale = max(|x|) / 127`, `code = clamp(round(x / scale), -128, 127)`.
//! Reconstruction: `x ≈ scale × (code as f32)`.
//!
//! These are the CPU reference paths. The GPU counterparts live in
//! [`crate::q8_msl`].
//!
//! # Public API
//!
//! - [`Q8_GROUP_SIZE`] — group size constant (128).
//! - [`q8_quantize`] — encode a flat `f32` slice to `(codes, scales)`.
//! - [`q8_dequantize`] — decode `(codes, scales)` back to `Vec<f32>`.

// ── q8_0 helpers (inline — no new crate dep) ─────────────────────────────────
//
// q8_0: symmetric 8-bit, group_size=128.
// scale = max(|x|) / 127
// code = clamp(round(x / scale), -128, 127) as i8, stored as u8
// recon = scale * (code as f32)
//
// Stored as (Vec<u8> codes, Vec<f32> scales). Each group is 128 f32 elements
// producing 128 bytes (i8 reinterpreted as u8) and one f32 scale.

pub const Q8_GROUP_SIZE: usize = 128;

/// Quantize a flat f32 slice to q8_0. `x.len()` must be a multiple of `Q8_GROUP_SIZE`.
pub fn q8_quantize(x: &[f32]) -> (Vec<u8>, Vec<f32>) {
    assert!(
        x.len().is_multiple_of(Q8_GROUP_SIZE),
        "q8_quantize: len={} not a multiple of Q8_GROUP_SIZE={}",
        x.len(),
        Q8_GROUP_SIZE
    );
    let n_groups = x.len() / Q8_GROUP_SIZE;
    let mut codes = vec![0u8; x.len()];
    let mut scales = vec![0.0_f32; n_groups];

    // chunks_exact eliminates the explicit `start = g * Q8_GROUP_SIZE` offset
    // arithmetic and lets LLVM prove group slice bounds statically.
    for (group, (code_group, scale_slot)) in x
        .chunks_exact(Q8_GROUP_SIZE)
        .zip(codes.chunks_exact_mut(Q8_GROUP_SIZE).zip(scales.iter_mut()))
    {
        let abs_max = group
            .iter()
            .copied()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };
        *scale_slot = scale;

        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (&v, slot) in group.iter().zip(code_group.iter_mut()) {
            let code = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            // Store i8 as u8 (bit-identical reinterpretation; reversed on decode).
            *slot = code as u8;
        }
    }
    (codes, scales)
}

/// Dequantize q8_0 back to f32. `codes` and `scales` must be consistent.
pub fn q8_dequantize(codes: &[u8], scales: &[f32]) -> Vec<f32> {
    let n = codes.len();
    debug_assert_eq!(n % Q8_GROUP_SIZE, 0);
    debug_assert_eq!(n / Q8_GROUP_SIZE, scales.len());

    let mut out = vec![0.0_f32; n];

    // zip of chunks_exact pairs: LLVM proves group_codes.len() == Q8_GROUP_SIZE
    // on every iteration, eliding per-element bounds checks on group_codes[i].
    for ((&scale, group_codes), out_group) in scales
        .iter()
        .zip(codes.chunks_exact(Q8_GROUP_SIZE))
        .zip(out.chunks_exact_mut(Q8_GROUP_SIZE))
    {
        for (&code_byte, slot) in group_codes.iter().zip(out_group.iter_mut()) {
            // Reinterpret u8 as i8.
            *slot = scale * f32::from(code_byte as i8);
        }
    }
    out
}

#[cfg(test)]
#[path = "q8_tests.rs"]
mod tests;
