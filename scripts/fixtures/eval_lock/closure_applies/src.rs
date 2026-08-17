fn build() -> Result<Closure> {
    // The "fuse two fused kernels" refactor. `Closure::apply` takes the eval
    // lock internally, so applying one from inside a closure body deadlocks —
    // even though nothing here calls .eval() directly.
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        let inner = swiglu_compiled()?;
        let outs = inner.apply(&[&x, &x])?;
        Ok(outs)
    });
    Ok(raw)
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
