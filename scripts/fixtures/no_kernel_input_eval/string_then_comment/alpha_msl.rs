// Fixture: a string literal holding two slashes on the line of the call.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    let url = "a//b"; // q.eval() would block here
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
