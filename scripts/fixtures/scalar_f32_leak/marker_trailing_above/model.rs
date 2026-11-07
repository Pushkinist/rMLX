// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() {
    let u = 1; // f32-ok: the tower runs in f32
    let t = scalar_f32(1.0);
}
