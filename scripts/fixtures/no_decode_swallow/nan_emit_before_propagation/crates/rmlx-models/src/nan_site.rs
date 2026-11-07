fn prefill() -> Result<(), E> {
    let n = count_nan_in_bytes(&row);
    let tok = step_fn(&row);
    if n > 0 {
        return Err(nan_error(n));
    }
    Ok(())
}
