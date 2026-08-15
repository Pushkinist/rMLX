// Fixture: a dispatcher one directory down. Expected exit 1 (the scan must
// recurse).
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    q.eval()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
