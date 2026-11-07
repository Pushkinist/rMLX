pub fn chunked_prefill() -> Result<(), E> {
    for c in chunks {
        if let Err(e) = run(c) {
            tracing::error!(error = %e, "prefill chunk failed");
            return Err(e);
        }
    }
    exit_prefill(&mut cache);
    Ok(())
}
