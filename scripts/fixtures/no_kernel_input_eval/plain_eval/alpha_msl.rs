// Fixture: the plain `x.eval()?;` spelling. Expected exit 1.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    q.eval()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
