fn prefill() -> Result<(), E> {
    let n = count_nan_in_bytes(&row);
    if n > 0 {
        return Err(nan_error(n));
    }
    let tok = step_fn(&row);
    Ok(())
}
