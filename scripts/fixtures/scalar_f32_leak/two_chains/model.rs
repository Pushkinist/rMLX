// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f(x: &Array, d: Device) -> Result<()> {
    let a = scalar_f32(1.0)
        .reshape(&[1], d)?;
    let b = scalar_f32(2.0)
        .reshape(&[1], d)?;
    Ok(())
}
