// Synthetic fixture for scripts/check_kernel_dtype_contract.sh - not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// Pins f32 by the bare variant name - not a guard.
pub fn bare_f32_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(F32, device)
}

/// Pins bf16 by the bare variant name - not a guard.
pub fn bare_bf16_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(Bf16, device)
}

/// Pins f16 by the bare variant name - not a guard.
pub fn bare_f16_sdpa(query: &Array, device: Device) -> Result<Array> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[dst_len], Dtype::F32)?;
    let dst = kernel.apply(invoke, device)?.remove(0);
    dst.astype(F16)
}
