// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

// f32-out-ok: read back by an MSL kernel that declares its own buffer type.

pub fn blank_marker_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.reshape(&[n], device)
}
