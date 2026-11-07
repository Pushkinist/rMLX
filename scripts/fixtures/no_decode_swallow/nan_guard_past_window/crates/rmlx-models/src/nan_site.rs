fn prefill() -> Result<(), E> {
    let n = count_nan_in_bytes(&row);
    let pad_1 = 0;
    let pad_2 = 0;
    let pad_3 = 0;
    let pad_4 = 0;
    let pad_5 = 0;
    let pad_6 = 0;
    let pad_7 = 0;
    let pad_8 = 0;
    let pad_9 = 0;
    let pad_10 = 0;
    let pad_11 = 0;
    let pad_12 = 0;
    let pad_13 = 0;
    let pad_14 = 0;
    let pad_15 = 0;
    let pad_16 = 0;
    if n > 0 { return Err(nan_error(n)); }
    Ok(())
}
