// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn substring_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&mid)?;
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let mid_buf = kernel.apply(invoke, device)?.remove(0);
    mid_buf.reshape(&[n], device)
}
