// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f(x: &Array, d: Device) -> Result<Array> {
    let s = scalar_f32(1.0)
        .reshape(&[1], d)?
        .multiply(&x, d)?;
    Ok(s)
}
