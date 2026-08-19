// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// Quantizer handing f32 scales back inside a tuple — no array is obviously
/// "the output", which is exactly how this one hides.
pub fn quantize_gpu(x: &Array, device: Device) -> Result<(Array, Array)> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n_words], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups], Dtype::F32)?;
    let mut outputs = kernel.apply(invoke, device)?;
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    Ok((codes, scales))
}
