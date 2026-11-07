// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() -> Array {
    // f32-ok: the tower runs in f32
    let n = 3;
    let s = scalar_f32(1.0);
    s
}
