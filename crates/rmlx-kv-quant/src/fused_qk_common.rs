//! Shared scaffold for the fused-QK MSL dispatchers (q8, turbo-k3, turbo-k4,
//! iso). Validates geometry, prepares the Q/mask/scale/dims device arrays,
//! and computes the launch grid. Codec-specific parts (codes/sideband
//! reshapes, kernel handle, input order beyond the common prefix) stay in
//! each `*_fused_qk_msl.rs`.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

/// Validated, device-ready common inputs for one fused-QK dispatch.
pub(crate) struct FusedQkSetup {
    /// `[b * n_q_heads * head_dim]` f32, eval'd.
    pub q_f32: Array,
    /// `[b * n_q_heads * kv_seq]` f32 mask, or a 1-element dummy. Eval'd.
    pub mask_flat: Array,
    /// `[1]` f32 attention scale. Eval'd.
    pub scale_arr: Array,
    /// `[5]` u32: `[head_dim, kv_seq, kv_h, heads_per_kv, has_mask]`. Eval'd.
    pub dims_arr: Array,
    pub n_q_heads: i32,
    /// `b * n_q_heads * kv_seq` — the flat output length.
    pub out_total: i64,
    pub grid_x: i32,
    pub grid_y: i32,
    /// 1 when an additive mask was supplied, 0 for the dummy mask — surfaced
    /// so each dispatcher can log it uniformly on its trace event.
    pub has_mask: u32,
}

/// Validate geometry and build the codec-independent device arrays.
/// `label` names the calling dispatcher in error messages.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the dispatcher signatures 1:1"
)]
pub(crate) fn build_fused_qk_setup(
    label: &str,
    query: &Array,
    additive_mask: Option<&Array>,
    b: i32,
    kv_h: i32,
    kv_seq: i32,
    head_dim: i32,
    heads_per_kv: i32,
    scale: f32,
    device: Device,
) -> Result<FusedQkSetup> {
    if head_dim != 128 && head_dim != 256 {
        return Err(Error::Quant(format!(
            "{label}: head_dim={head_dim} not supported \
             (only 128 and 256 are wired; legacy dequant+SDPA path handles other dims)"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "{label}: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "{label}: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }
    let n_q_heads = kv_h * heads_per_kv;

    let q_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    let (mask_flat, has_mask) = if let Some(m) = additive_mask {
        let mask_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(kv_seq);
        let m_f = if m.dtype() == Dtype::F32 {
            m.reshape(&[mask_total as i32], device)?
        } else {
            m.astype(Dtype::F32, device)?
                .reshape(&[mask_total as i32], device)?
        };
        (m_f, 1u32)
    } else {
        let zero_bytes = [0u8; 4];
        let dummy = Array::from_bytes(&zero_bytes, &[1], Dtype::F32)
            .map_err(|e| Error::Mlx(format!("{label} dummy mask: {e}")))?;
        (dummy, 0u32)
    };

    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };

    let dims_vals: [u32; 5] = [
        head_dim as u32,
        kv_seq as u32,
        kv_h as u32,
        heads_per_kv as u32,
        has_mask,
    ];
    // LE byte assembly — no pointer reinterpret needed, drops the historic
    // 4x-cloned unsafe block.
    let mut dims_bytes = [0u8; 20];
    for (chunk, v) in dims_bytes.chunks_exact_mut(4).zip(dims_vals.iter()) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    let dims_arr = Array::from_bytes(&dims_bytes, &[5], Dtype::U32)?;

    // Inputs stay lazy: `MetalKernel::apply` enqueues a graph node, so MLX
    // materialises them — and applies `ensure_row_contiguous` — inside the
    // kernel's own `eval_gpu`. A blocking eval here would only stall the host
    // once per layer per decode step. See `crate::flash_decode_common` docs.

    let out_total: i64 = i64::from(b) * i64::from(n_q_heads) * i64::from(kv_seq);
    let grid_x: i64 = i64::from(kv_seq) * i64::from(head_dim);
    let grid_y: i64 = i64::from(b) * i64::from(n_q_heads);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "{label}: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }

    Ok(FusedQkSetup {
        q_f32,
        mask_flat,
        scale_arr,
        dims_arr,
        n_q_heads,
        out_total,
        grid_x: grid_x as i32,
        grid_y: grid_y as i32,
        has_mask,
    })
}
