fn dispatch_test(q: &Array) -> Result<Vec<Array>> {
    let mut invoke = MetalKernelInvoke::new();
    q.eval()?;
    kernel().apply(invoke, device)
}
