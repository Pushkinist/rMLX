// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn counted_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(query.dtype(), device)
}

/// Codes only: the f32 named in each comment is not declared.
pub fn codes_a_gpu(x: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::U32)?;
    let c = '"'; // invoke.add_output_shape(&[n], Dtype::F32)?
    let codes = kernel.apply(invoke, device)?.remove(0);
    Ok(codes)
}

pub fn codes_b_gpu(x: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::U32)?;
    let d = '\"'; // invoke.add_output_shape(&[n], Dtype::F32)?
    let codes = kernel.apply(invoke, device)?.remove(0);
    Ok(codes)
}
