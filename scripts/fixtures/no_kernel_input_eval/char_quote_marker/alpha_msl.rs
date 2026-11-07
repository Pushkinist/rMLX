// Fixture: a character literal holding a quote before a real eval-ok marker on the same line.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    let c = '"'; q.eval()?; // eval-ok: host readback
    let d = '\"'; q.eval()?; // eval-ok: host readback
    q.eval()?; let e = '"'// eval-ok: host readback
    q.eval()?; let f = '\"'// eval-ok: host readback
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
