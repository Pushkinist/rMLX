// Fixture: a dispatcher with no eval.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
