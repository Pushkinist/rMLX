// Fixture: a character literal holding a quote must not hide a call behind a later string with two slashes.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    let c = '"'; let s = "a//b"; q.eval()?;
    let d = '\"'; let s = "a//b"; q.eval()?;
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
