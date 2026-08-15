// Fixture: one marker must exempt exactly one call — the loop below it is
// still a violation. Expected exit 1.
fn dispatch(q: &Array, b: &Array, c: &Array) -> Result<Vec<Array>> {
    // eval-ok: host readback follows
    q.eval()?;
    for arr in [b, c] {
        arr.eval()?;
    }
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
