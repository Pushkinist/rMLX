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
