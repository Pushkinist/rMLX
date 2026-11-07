// Fixture: a marker, a plain statement, then an eval - the marker is gone by then. Expected exit 1.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    // eval-ok: host readback follows
    let n = q.shape();
    q.eval()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
