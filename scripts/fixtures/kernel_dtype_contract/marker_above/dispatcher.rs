// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// Scores, not an activation.
// f32-out-ok: read back by another MSL kernel that declares its buffer type.
pub fn marked_scores(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let out = kernel.apply(invoke, device)?.remove(0);
    out.reshape(&[b, n_q_heads, 1, kv_seq], device)
}
