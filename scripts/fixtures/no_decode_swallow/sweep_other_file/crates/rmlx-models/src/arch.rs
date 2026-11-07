pub fn chunked_prefill() -> Result<(), E> {
    if let Err(e) = run(c) {
        tracing::error!(error = %e, "prefill chunk failed");
        return Err(e);
    }
    Ok(())
}
