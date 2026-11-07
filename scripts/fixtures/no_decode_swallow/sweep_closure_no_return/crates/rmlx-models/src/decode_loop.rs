pub fn chunked_prefill() -> Result<(), E> {
    let mut run_chunks = |cache: &mut C| -> Result<(), E> {
        for c in chunks {
            if let Err(e) = run(c) {
                tracing::error!(error = %e, "prefill chunk failed");
                continue;
            }
        }
        Ok(())
    };
    let result = run_chunks(&mut cache);
    exit_prefill(&mut cache);
    result?;
    Ok(())
}
