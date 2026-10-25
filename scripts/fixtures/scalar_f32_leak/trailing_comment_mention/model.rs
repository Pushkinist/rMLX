// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f(x: Array) -> Array {
    let n = 3; // never call scalar_f32(x) bare here
    x
}
