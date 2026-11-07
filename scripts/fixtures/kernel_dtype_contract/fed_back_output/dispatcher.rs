// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// The first kernel's f32 output goes straight into the second kernel and never leaves.
pub fn two_stage_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut first = MetalKernelInvoke::new();
    first.add_output_shape(&[n], Dtype::F32)?;
    let mid = kernel_a.apply(first, device)?.remove(0);
    let mut second = MetalKernelInvoke::new();
    second.add_input(&mid)?;
    second.add_output_shape(&[n], Dtype::F32)?;
    let out = kernel_b.apply(second, device)?.remove(0);
    out.astype(query.dtype(), device)
}
