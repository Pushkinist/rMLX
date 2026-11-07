fn prefill() -> Result<(), E> {
    let n = count_nan_in_bytes(&row);
    if n > 0 {
        return Err(nan_error(n));
    }
    let m = count_nan_in_bytes(&other);
    if m > 0 {
        return Err(nan_error(m));
    }
    Ok(())
}
