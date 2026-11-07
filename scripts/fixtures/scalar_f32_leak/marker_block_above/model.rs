// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
fn f() -> Array {
    // f32-ok: the whole tower runs in f32,
    // so the scalar never meets a bf16 tensor.
    let s = scalar_f32(1.0);
    s
}
