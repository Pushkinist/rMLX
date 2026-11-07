// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f(x: &Array, d: Device) -> Result<()> {
    let a = scalar_f32(1.0).astype(Dtype::F32, d)?;
    let b = scalar_f32(1.0).astype(F32, d)?;
    let c = scalar_f32(1.0).astype(rmlx_mlx::Dtype::F32, d)?;
    Ok(())
}
