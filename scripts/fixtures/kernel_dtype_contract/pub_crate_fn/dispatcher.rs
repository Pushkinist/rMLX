// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn ok_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(query.dtype(), device)
}

pub(crate) fn leaky_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.reshape(&[n], device)
}
