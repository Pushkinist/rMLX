fn build() -> Result<Closure> {
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        let mut iter = inputs.into_iter();
        let x = iter.next().expect("x");
        let y = add(&x, &x, device)?;
        let y = if y.dtype() == in_dtype { y } else { y.astype(in_dtype, device)? };
        let outs = kernel.apply(invoke, device)?;
        Ok(vec![y])
    });
    let status = with_eval_lock(|| unsafe { sys::mlx_closure_apply(&raw mut o, c, i) });
    Ok(raw)
}
