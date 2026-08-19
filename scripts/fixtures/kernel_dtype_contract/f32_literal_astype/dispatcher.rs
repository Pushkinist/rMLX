// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// The only cast is the upcast on the way IN — not a guard.
pub fn upcast_only_sdpa(query: &Array, device: Device) -> Result<Array> {
    let q = query.astype(Dtype::F32, device)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&q)?;
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst_flat = kernel.apply(invoke, device)?.remove(0);
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}
