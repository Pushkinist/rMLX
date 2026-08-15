// Fixture: a dispatcher with no eval at all. Expected exit 0.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
