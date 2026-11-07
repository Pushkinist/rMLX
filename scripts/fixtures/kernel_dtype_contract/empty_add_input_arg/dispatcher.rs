// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn empty_arg_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut first = MetalKernelInvoke::new();
    first.add_output_shape(&[n], Dtype::F32)?;
    let one = k.apply(first, device)?.remove(0);
    let two = k.apply(second, device)?.remove(0);
    second.add_input(&*query)?;
    one.reshape(&[n], device)
}
