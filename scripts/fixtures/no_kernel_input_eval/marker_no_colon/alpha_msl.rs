// Fixture: a marker without its colon.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    q.eval()?; // eval-ok
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
