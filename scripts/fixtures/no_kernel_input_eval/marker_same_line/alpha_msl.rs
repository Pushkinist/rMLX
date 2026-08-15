// Fixture: marker on the call's own line. Expected exit 0.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    q.eval()?; // eval-ok: host readback follows
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
