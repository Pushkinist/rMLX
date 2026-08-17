// Crate-local helpers that merely share a name with an mlx-c symbol. They are
// not FFI calls, so they must not be flagged — that is what the `sys::` anchor
// on the banned pattern buys, and this fixture is what makes dropping the
// anchor fail the corpus instead of merely widening it unnoticed.
fn mlx_eval(v: u32) -> u32 {
    v + 1
}

fn mlx_array_tostring(a: u32) -> String {
    format!("{a}")
}

pub fn use_them() -> u32 {
    let a = mlx_eval(7);
    let _s = mlx_array_tostring(a);
    a
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
