// Fixture: a same-line marker exempts its own call, not the call after it. Expected exit 1.
fn dispatch(q: &Array, b: &Array) -> Result<Vec<Array>> {
    q.eval()?; // eval-ok: host readback follows
    b.eval()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
