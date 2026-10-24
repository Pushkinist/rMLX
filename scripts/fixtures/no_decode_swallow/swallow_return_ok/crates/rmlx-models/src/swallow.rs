pub fn run() -> Result<(), E> {
    for c in chunks {
        if let Err(e) = step(c) {
            tracing::error!(error = %e, "chunk failed");
            return Ok(());
        }
    }
    Ok(())
}
