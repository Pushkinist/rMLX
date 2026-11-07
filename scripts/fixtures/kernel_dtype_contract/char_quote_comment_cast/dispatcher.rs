// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn leaky_char_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    let c = '"'; // dst.astype(query.dtype(), device)
    let d = '\"'; // dst.astype(query.dtype(), device)
    dst.reshape(&[n], device)
}
