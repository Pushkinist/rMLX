// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() -> Array {
    let s = scalar_f32(1.0); // f32-ok: the tower runs in f32
    s
}
