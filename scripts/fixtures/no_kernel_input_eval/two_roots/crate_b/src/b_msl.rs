// Fixture: a dispatcher of the second crate that evaluates.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    q.eval()?;
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
