// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

pub fn tagged_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    let tag = "x//y"; invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    Ok(dst)
}
