// Fixture: the UFCS spelling `Array::eval(&q)?;`. Expected exit 1.
fn dispatch(q: &Array) -> Result<Vec<Array>> {
    Array::eval(q)?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    kernel().apply(invoke, device)
}
