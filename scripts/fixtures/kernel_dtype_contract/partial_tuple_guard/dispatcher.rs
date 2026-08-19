// Synthetic fixture for scripts/check_kernel_dtype_contract.sh — not compiled.
use rmlx_mlx::metal_kernel::MetalKernelInvoke;

/// Two f32 buffers escape; only one of them is narrowed.
pub fn quantize_partial_guard_gpu(k: &Array, device: Device) -> Result<(Array, Array, Array)> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_output_shape(&[n_words], Dtype::U32)?;
    invoke.add_output_shape(&[n_groups], Dtype::F32)?;
    invoke.add_output_shape(&[n_groups], Dtype::F32)?;
    let mut outputs = kernel.apply(invoke, device)?;
    let biases = outputs.remove(2);
    let scales = outputs.remove(1);
    let codes = outputs.remove(0);
    let scales_r = scales.reshape(&sg_shape, device)?.astype(k.dtype(), device)?;
    let biases_r = biases.reshape(&sg_shape, device)?;
    Ok((codes, scales_r, biases_r))
}
