// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// Restores the caller's dtype.
pub fn clean_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(query.dtype(), device)
}
