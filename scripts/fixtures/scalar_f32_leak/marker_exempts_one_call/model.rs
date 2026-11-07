// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() -> Array {
    // f32-ok: the tower runs in f32
    let a = scalar_f32(1.0);
    let b = scalar_f32(2.0);
    a.add(&b)
}
