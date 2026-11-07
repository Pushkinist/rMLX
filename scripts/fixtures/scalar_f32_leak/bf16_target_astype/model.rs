// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f(x: &Array, d: Device) -> Result<Array> {
    let a = scalar_f32(1.0).astype(Dtype::Bf16, d)?;
    let b = scalar_f32(1.0).astype(Dtype::I32, d)?;
    x.add(&a, d)
}
