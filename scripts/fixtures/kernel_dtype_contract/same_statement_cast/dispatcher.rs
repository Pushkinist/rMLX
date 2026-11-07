// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// The cast sits in the statement of the dispatch.
pub fn same_stmt_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n], Dtype::F32)?;
    kernel.apply(invoke, device)?.remove(0).astype(query.dtype(), device)
}
