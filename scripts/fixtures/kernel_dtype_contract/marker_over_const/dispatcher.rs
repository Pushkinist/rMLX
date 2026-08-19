// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

// f32-out-ok: read back by another MSL kernel that declares its buffer type.
const TG_SIZE: i32 = 64;

/// The marker above belongs to the const, not to this function.
pub fn leaky_after_const_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst_flat = kernel.apply(invoke, device)?.remove(0);
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}
