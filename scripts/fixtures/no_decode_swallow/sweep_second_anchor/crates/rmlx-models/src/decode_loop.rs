pub fn chunked_prefill() -> Result<(), E> {
    let mut run_chunks = |cache: &mut C| -> Result<(), E> {
        Ok(())
    };
    if let Err(e) = eval(c) {
        tracing::error!(error = %e, "prefill chunk cache eval failed");
        return Err(e);
    }
    let result = run_chunks(&mut cache);
    exit_prefill(&mut cache);
    result?;
    Ok(())
}
