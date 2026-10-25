// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn counted_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(query.dtype(), device)
}

/// Codes only: the f32 named in the comment is not declared.
pub fn codes_only_gpu(x: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n_words], Dtype::U32)?; // was Dtype::F32
    let codes = kernel.apply(invoke, device)?.remove(0);
    Ok(codes)
}
