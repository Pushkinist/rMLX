fn build() -> Result<Closure> {
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        let mut iter = inputs.into_iter();
        let x = iter.next().expect("x");
        // A brace inside a string must not desync the body scan: "{}"
        let y = add(&x, &x, device)?;
        let y = if y.dtype() == in_dtype { y } else { y.astype(in_dtype, device)? };
        Ok(vec![y])
    });
    with_eval_lock(|| unsafe { sys::mlx_closure_apply(&raw mut out, cls, vin) });
    Ok(raw)
}
