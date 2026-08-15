// Fixture: a trailing comment that merely names the call is not a call.
// Expected exit 0.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let k = 1; // never call .eval() on kernel inputs
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
