// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// The only derived cast is on the way IN; the f32 output leaves uncast.
pub fn input_cast_sdpa(query: &Array, k: &Array, device: Device) -> Result<Array> {
    let kk = k.astype(query.dtype(), device)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&kk)?;
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst_flat = kernel.apply(invoke, device)?.remove(0);
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}
