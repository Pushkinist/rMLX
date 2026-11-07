fn prefill() -> Result<(), E> {
    let n = count_nan_in_bytes(&row);
    let ctx = DecodeCtx { step_fn, other: 1 };
    if n > 0 {
        return Err(nan_error(n));
    }
    Ok(())
}
