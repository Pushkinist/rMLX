// Synthetic fixture for scripts/check/kernel_dispatch.sh dtype-contract — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// The declaration rustfmt split across lines; the return is uncast.
pub fn split_decl_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(
        &[dst_len as i32],
        Dtype::F32,
    )?;
    let dst_flat = kernel.apply(invoke, device)?.remove(0);
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}
