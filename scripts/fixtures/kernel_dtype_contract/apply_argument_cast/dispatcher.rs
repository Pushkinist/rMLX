// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn arg_cast_sdpa(query: &Array, k: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let dst = kernel.apply(with_input(invoke, k.astype(query.dtype(), device)?), device)?.remove(0);
    dst.reshape(&[n], device)
}
