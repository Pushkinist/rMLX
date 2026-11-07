// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f(x: &Array, d: Device) -> Result<Array> {
    let s = scalar_f32(1.0);
    let t = x.astype(x.dtype(), d)?;
    Ok(t)
}
