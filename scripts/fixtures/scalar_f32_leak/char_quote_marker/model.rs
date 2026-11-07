// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() {
    let c = '"'; let t = scalar_f32(1.0); // f32-ok: the tower runs in f32
    let d = '\"'; let u = scalar_f32(2.0); // f32-ok: the tower runs in f32
}
