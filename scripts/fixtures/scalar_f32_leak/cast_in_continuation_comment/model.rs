// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() {
    let t = scalar_f32(1.0)
        .mul(y)?; // .astype(x.dtype(), d)
}
