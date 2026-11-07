// Fixture: a space between eval and its parenthesis.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    q.eval ()?;
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
