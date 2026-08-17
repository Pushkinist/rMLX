fn build() -> Result<Closure> {
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        // An unbalanced brace inside a string literal must not end the body
        // scan early and hide the evaluation that follows it.
        let msg = "trailing brace: }";
        let y = add(&x, &x, device)?;
        y.eval()?;
        Ok(vec![y])
    });
    Ok(raw)
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
