// Fixture: the dispatcher itself is clean; the eval hides in the shared
// scaffold next to it. Expected exit 1 (reported against the scaffold).
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    materialise(q)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
