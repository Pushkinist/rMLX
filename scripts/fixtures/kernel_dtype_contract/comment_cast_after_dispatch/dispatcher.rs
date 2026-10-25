// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// The cast that would restore the dtype is only written in a comment.
pub fn restore_in_comment_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    // restore: dst.astype(query.dtype(), device)
    Ok(dst)
}
