// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// No MetalKernelInvoke here, so this file is out of scope even though it
/// names the same dtype.
pub fn pick_dtype() -> Dtype {
    Dtype::F32
}
