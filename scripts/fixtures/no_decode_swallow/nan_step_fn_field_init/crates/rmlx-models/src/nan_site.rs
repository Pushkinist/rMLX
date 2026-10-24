fn prefill() -> Result<(), E> {
    let ctx = DecodeCtx { step_fn, other: 1 };
    let n = count_nan_in_bytes(&row);
    if n > 0 {
        return Err(nan_error(n));
    }
    Ok(())
}
