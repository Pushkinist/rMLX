// Fixture: marker on the comment lines directly above the call. Expected
// exit 0.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    // eval-ok: host readback follows — `to_bytes` copies to CPU, so the
    // array has to be materialised first.
    q.eval()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
