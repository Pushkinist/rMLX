// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn scale(x: &Array, d: Device) -> Result<Array> {
    let s = scalar_f32(0.5).astype(x.dtype(), d)?;
    x.multiply(&s, d)
}
