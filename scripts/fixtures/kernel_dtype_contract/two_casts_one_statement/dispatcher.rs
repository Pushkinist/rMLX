// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn pair_sdpa(query: &Array, device: Device) -> Result<(Array, Array)> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let mut o = kernel.apply(invoke, device)?;
    Ok((o.remove(0).astype(query.dtype(), device)?, o.remove(0).astype(query.dtype(), device)?))
}
