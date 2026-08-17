fn build() -> Result<Closure> {
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        let mut iter = inputs.into_iter();
        let x = iter.next().expect("x");
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
