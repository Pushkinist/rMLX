//! OCP microscaling formats: mxfp8 (E8M0 + E4M3), mxfp4 (E8M0 + E2M1), nvfp4
//! (UE4M3 or signed-E4M3 + E2M1).
//!
//! Stage 1. Primary test path is mxfp8 g32 against
//! `mlx-community__gemma-4-e4b-it-mxfp8`. Smoke probe rejects `!!!!!!`
//! generation.
//!
//! # NaN policy
//!
//! - E8M0 scale byte `0xFF`: produce NaN in the entire 32-element group.
//!   `tracing::warn!` once per `dequant_to_f32` call (not per element).
//! - mxfp8 element byte `0x7F` or `0xFF` (E4M3 NaN): produce `f32::NAN` for
//!   that element. `tracing::warn!` once per call.
//! - mxfp4 / nvfp4 nibbles: no NaN values exist in E2M1; these never produce
//!   NaN.

use rmlx_core::{Error, Result};
use tracing::warn;

use crate::fp4::e2m1_decode;
use crate::fp8::{e4m3_decode, e8m0_decode, ue4m3_decode};

// ── MxFamily ─────────────────────────────────────────────────────────────────

/// Microscaling family selector.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — three MX quantization families (Mxfp8/Mxfp4/Nvfp4); adding a family requires updating dequant, group_size, and all MxFamily match arms in the loader"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxFamily {
    /// E8M0 scale (g32) + E4M3 elements (1 byte each, unpacked).
    Mxfp8,
    /// E8M0 scale (g32) + E2M1 elements (2 nibbles per byte, low first).
    Mxfp4,
    /// E2M1 elements (same packing as Mxfp4) + E4M3 or UE4M3 scale (g16).
    ///
    /// `compat_mlx_signed_scale = false` (default): use Blackwell-correct
    /// unsigned UE4M3 for scales.
    ///
    /// `compat_mlx_signed_scale = true`: use signed E4M3 (same as the MLX
    /// bug ml-explore/mlx#2962) for parity with MLX-produced snapshots.
    Nvfp4 {
        /// Use signed E4M3 for scale decoding (MLX bug parity mode).
        compat_mlx_signed_scale: bool,
    },
}

// ── MxParams ─────────────────────────────────────────────────────────────────

/// Parameters that describe one microscaling-quantized weight tensor.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed params struct — three fields are the complete MX quant descriptor contract; adding a field requires updating MxParams construction in the loader and all dequant callers"
)]
#[derive(Debug, Clone)]
pub struct MxParams {
    /// Which microscaling family this tensor uses (mxfp8, mxfp4, or nvfp4).
    pub family: MxFamily,
    /// Number of rows (output channels).
    pub rows: usize,
    /// Number of columns (input channels, logical, unpacked).
    pub cols: usize,
}

impl MxParams {
    /// Group size: 32 for mxfp8 / mxfp4, 16 for nvfp4.
    pub fn group_size(&self) -> u32 {
        match self.family {
            MxFamily::Mxfp8 | MxFamily::Mxfp4 => 32,
            MxFamily::Nvfp4 { .. } => 16,
        }
    }
}

// ── Shape validation ─────────────────────────────────────────────────────────

/// Cold helper: "rows/cols must be > 0" error.
#[cold]
fn err_zero_dim() -> Error {
    Error::Quant("mxfp: rows and cols must be > 0".to_owned())
}

/// Cold helper: "cols not multiple of group_size" error.
#[cold]
fn err_cols_not_multiple(cols: usize, gs: usize) -> Error {
    Error::Quant(format!(
        "mxfp: cols={cols} is not a multiple of group_size={gs}"
    ))
}

/// Cold helper: "cols must be even for mxfp4/nvfp4" error.
#[cold]
fn err_cols_odd(cols: usize) -> Error {
    Error::Quant(format!(
        "mxfp4/nvfp4: cols={cols} must be even (2 nibbles per byte)"
    ))
}

/// Cold helper: "packed length mismatch" error.
#[cold]
fn err_packed_len(got: usize, exp: usize, rows: usize, cols: usize, family: MxFamily) -> Error {
    Error::Quant(format!(
        "mxfp: packed length {got} != expected {exp} \
         (rows={rows}, cols={cols}, family={family:?})"
    ))
}

/// Cold helper: "scales length mismatch" error.
#[cold]
fn err_scales_len(got: usize, exp: usize, rows: usize, cols: usize, gs: usize) -> Error {
    Error::Quant(format!(
        "mxfp: scales length {got} != expected {exp} \
         (rows={rows}, cols={cols}, group_size={gs})"
    ))
}

/// Cold helper: "out length mismatch" error.
#[cold]
fn err_out_len(got: usize, exp: usize, rows: usize, cols: usize) -> Error {
    Error::Quant(format!(
        "mxfp: out length {got} != expected {exp} (rows={rows}, cols={cols})"
    ))
}

fn validate(params: &MxParams, packed: &[u8], scales: &[u8], out: &[f32]) -> Result<()> {
    let gs = params.group_size() as usize;

    if params.cols == 0 || params.rows == 0 {
        return Err(err_zero_dim());
    }
    if !params.cols.is_multiple_of(gs) {
        return Err(err_cols_not_multiple(params.cols, gs));
    }

    let groups_per_row = params.cols / gs;

    // Expected packed byte length
    let exp_packed = match params.family {
        MxFamily::Mxfp8 => params.rows * params.cols, // 1 byte per element
        MxFamily::Mxfp4 | MxFamily::Nvfp4 { .. } => {
            if !params.cols.is_multiple_of(2) {
                return Err(err_cols_odd(params.cols));
            }
            params.rows * (params.cols / 2)
        }
    };
    if packed.len() != exp_packed {
        return Err(err_packed_len(
            packed.len(),
            exp_packed,
            params.rows,
            params.cols,
            params.family,
        ));
    }

    // Expected scales byte length: rows * groups_per_row (1 byte per group)
    let exp_scales = params.rows * groups_per_row;
    if scales.len() != exp_scales {
        return Err(err_scales_len(
            scales.len(),
            exp_scales,
            params.rows,
            params.cols,
            gs,
        ));
    }

    let exp_out = params.rows * params.cols;
    if out.len() != exp_out {
        return Err(err_out_len(out.len(), exp_out, params.rows, params.cols));
    }

    Ok(())
}

// ── Core dequant ─────────────────────────────────────────────────────────────

/// Dequantize the entire packed weight into a row-major f32 buffer.
///
/// # Input shapes
/// - `packed`:
/// - mxfp8 → `rows * cols` bytes (1 byte per element)
/// - mxfp4 → `rows * (cols/2)` bytes (2 nibbles per byte, low first)
/// - nvfp4 → `rows * (cols/2)` bytes (same as mxfp4)
/// - `scales`:
/// - mxfp8 / mxfp4 → `rows * (cols/32)` bytes (E8M0, 1 byte per group)
/// - nvfp4 → `rows * (cols/16)` bytes (UE4M3 or signed-E4M3)
///
/// # Output
/// `out`: `rows * cols` f32 elements, row-major.
///
/// # NaN handling
/// - E8M0 scale `0xFF`: the whole 32-element group gets `f32::NAN`. One
///   `warn!` per call.
/// - mxfp8 element `0x7F` / `0xFF`: that element becomes `f32::NAN`. One
///   `warn!` per call.
/// - mxfp4 / nvfp4: no NaN possible from elements; only from scales.
///
/// # Errors
/// Returns `Error::Quant(_)` on shape mismatch or unsupported params.
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "dequant kernel — splitting fragments the hot loop; per-block element dispatch (mxfp8/mxfp4/nvfp4) and scale logic must stay co-located for compiler vectorisation"
)]
pub fn dequant_to_f32(
    params: &MxParams,
    packed: &[u8],
    scales: &[u8],
    out: &mut [f32],
) -> Result<()> {
    validate(params, packed, scales, out)?;

    // Warn once on first nvfp4 decode (default mode).
    if let MxFamily::Nvfp4 {
        compat_mlx_signed_scale: false,
    } = params.family
    {
        warn!(
            "nvfp4 dequant using unsigned UE4M3 scales (Blackwell-correct). \
             MLX-produced nvfp4 snapshots used signed E4M3 (ml-explore/mlx#2962 — \
             137× less dynamic range). Pass compat_mlx_signed_scale=true for parity."
        );
    }

    let gs = params.group_size() as usize;
    let groups_per_row = params.cols / gs;

    // NaN-warn flags (fire at most once per call).
    let mut scale_nan_warned = false;
    let mut elem_nan_warned = false;

    match params.family {
        MxFamily::Mxfp8 => {
            // Walk rows via chunks_exact_mut to elide per-element `r * cols + c`
            // index arithmetic and let LLVM see sequential pointer streams.
            // `validate` above asserts cols % gs == 0, so chunks_exact_mut(gs) is
            // safe on both the packed and out row slices.
            for (r, (out_row, packed_row)) in out
                .chunks_exact_mut(params.cols)
                .zip(packed.chunks_exact(params.cols))
                .enumerate()
            {
                // scales_row length == groups_per_row; r < rows from enumerate on chunks_exact_mut.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "r*groups_per_row..(r+1)*groups_per_row <= scales.len(): scales.len()==rows*groups_per_row; r < rows from chunks_exact_mut enumerate"
                )]
                let scales_row = &scales[r * groups_per_row..(r + 1) * groups_per_row];
                for (g, (out_group, packed_group)) in out_row
                    .chunks_exact_mut(gs)
                    .zip(packed_row.chunks_exact(gs))
                    .enumerate()
                {
                    // g < groups_per_row from chunks_exact_mut(gs) producing cols/gs chunks.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "g < groups_per_row == scales_row.len(): g from chunks_exact_mut(gs) on cols-wide row, cols/gs == groups_per_row"
                    )]
                    let scale_byte = scales_row[g];
                    let scale_f32 = e8m0_decode(scale_byte);

                    if scale_f32.is_nan() && !scale_nan_warned {
                        warn!(
                            row = r,
                            group = g,
                            scale_byte,
                            "mxfp8: NaN scale (0xFF E8M0) — broken snapshot detected"
                        );
                        scale_nan_warned = true;
                    }

                    for (slot, &elem_byte) in out_group.iter_mut().zip(packed_group.iter()) {
                        let elem_f32 = e4m3_decode(elem_byte);

                        if elem_f32.is_nan() && !elem_nan_warned {
                            warn!(elem_byte, "mxfp8: NaN element byte (E4M3 NaN)");
                            elem_nan_warned = true;
                        }

                        // NaN * anything = NaN — propagate naturally.
                        *slot = elem_f32 * scale_f32;
                    }
                }
            }
        }

        MxFamily::Mxfp4 => {
            // packed layout: rows × (cols/2) bytes, two nibbles per byte (lo first).
            // Walk paired (out_row, packed_row) to avoid per-element `r * cols + c`
            // recomputation. Within each group walk (out_pair, &packed_byte) where
            // each byte expands to 2 elements — elides the `c.is_multiple_of(2)` branch.
            let packed_cols = params.cols / 2;
            for (r, (out_row, packed_row)) in out
                .chunks_exact_mut(params.cols)
                .zip(packed.chunks_exact(packed_cols))
                .enumerate()
            {
                // scales_row length == groups_per_row; r < rows from enumerate on chunks_exact_mut.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "r*groups_per_row..(r+1)*groups_per_row <= scales.len(): scales.len()==rows*groups_per_row; r < rows from chunks_exact_mut enumerate"
                )]
                let scales_row = &scales[r * groups_per_row..(r + 1) * groups_per_row];
                // Each group is gs output elements = gs/2 packed bytes.
                let packed_gs = gs / 2;
                for (g, (out_group, packed_group)) in out_row
                    .chunks_exact_mut(gs)
                    .zip(packed_row.chunks_exact(packed_gs))
                    .enumerate()
                {
                    // g < groups_per_row from chunks_exact_mut(gs) producing cols/gs chunks.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "g < groups_per_row == scales_row.len(): g from chunks_exact_mut(gs) on cols-wide row, cols/gs == groups_per_row"
                    )]
                    let scale_byte = scales_row[g];
                    let scale_f32 = e8m0_decode(scale_byte);

                    if scale_f32.is_nan() && !scale_nan_warned {
                        warn!(
                            row = r,
                            group = g,
                            scale_byte,
                            "mxfp4: NaN scale (0xFF E8M0) — broken snapshot detected"
                        );
                        scale_nan_warned = true;
                    }

                    // Expand one packed byte → two output slots (lo nibble, hi nibble).
                    // out_pair comes from chunks_exact_mut(2) — always length 2.
                    // out_pair[0] and out_pair[1] are always valid: chunks_exact_mut(2) guarantees len==2.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "out_pair.len()==2 from chunks_exact_mut(2); indices 0 and 1 always valid"
                    )]
                    for (out_pair, &byte) in out_group.chunks_exact_mut(2).zip(packed_group.iter())
                    {
                        out_pair[0] = e2m1_decode(byte & 0xF) * scale_f32;
                        out_pair[1] = e2m1_decode((byte >> 4) & 0xF) * scale_f32;
                    }
                }
            }
        }

        MxFamily::Nvfp4 {
            compat_mlx_signed_scale,
        } => {
            // Same nibble-pair expansion as Mxfp4; only scale decoding differs.
            // nvfp4 group size = 16, so packed_gs = gs/2 = 8 bytes per group.
            let packed_cols = params.cols / 2;
            let packed_gs = gs / 2;
            for (r, (out_row, packed_row)) in out
                .chunks_exact_mut(params.cols)
                .zip(packed.chunks_exact(packed_cols))
                .enumerate()
            {
                // scales_row length == groups_per_row; r < rows from enumerate on chunks_exact_mut.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "r*groups_per_row..(r+1)*groups_per_row <= scales.len(): scales.len()==rows*groups_per_row; r < rows from chunks_exact_mut enumerate"
                )]
                let scales_row = &scales[r * groups_per_row..(r + 1) * groups_per_row];
                for (g, (out_group, packed_group)) in out_row
                    .chunks_exact_mut(gs)
                    .zip(packed_row.chunks_exact(packed_gs))
                    .enumerate()
                {
                    // g < groups_per_row from chunks_exact_mut(gs) producing cols/gs chunks.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "g < groups_per_row == scales_row.len(): g from chunks_exact_mut(gs) on cols-wide row, cols/gs == groups_per_row"
                    )]
                    let scale_byte = scales_row[g];
                    let scale_f32 = if compat_mlx_signed_scale {
                        // MLX bug: treat scale as signed E4M3 (incorrect per Blackwell spec).
                        e4m3_decode(scale_byte)
                    } else {
                        // Blackwell-correct: unsigned UE4M3.
                        ue4m3_decode(scale_byte)
                    };

                    // Note: E4M3 NaN bytes (0x7F, 0xFF) can occur in compat mode.
                    if scale_f32.is_nan() && !scale_nan_warned {
                        warn!(
                            row = r,
                            group = g,
                            scale_byte,
                            compat_mlx_signed_scale,
                            "nvfp4: NaN scale byte — broken snapshot"
                        );
                        scale_nan_warned = true;
                    }

                    // out_pair comes from chunks_exact_mut(2) — always length 2.
                    // out_pair[0] and out_pair[1] are always valid: chunks_exact_mut(2) guarantees len==2.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "out_pair.len()==2 from chunks_exact_mut(2); indices 0 and 1 always valid"
                    )]
                    for (out_pair, &byte) in out_group.chunks_exact_mut(2).zip(packed_group.iter())
                    {
                        out_pair[0] = e2m1_decode(byte & 0xF) * scale_f32;
                        out_pair[1] = e2m1_decode((byte >> 4) & 0xF) * scale_f32;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Convenience wrapper: allocates once and returns a fresh `Vec<f32>`.
pub fn dequant_vec(params: &MxParams, packed: &[u8], scales: &[u8]) -> Result<Vec<f32>> {
    let mut out = vec![0.0_f32; params.rows * params.cols];
    dequant_to_f32(params, packed, scales, &mut out)?;
    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "mxfp_tests.rs"]
mod tests;
