fn prefill() -> Result<(), E> {
    let n = count_nan_in_bytes(&row);
    let _ = n;
    let tok = step_fn(&row);
    Ok(())
}
