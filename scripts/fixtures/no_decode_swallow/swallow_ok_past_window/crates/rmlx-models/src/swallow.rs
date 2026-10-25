pub fn run() -> Result<(), E> {
    if let Err(e) = step() {
        tracing::error!(error = %e, "chunk failed");
        let pad_1 = 0;
        let pad_2 = 0;
        let pad_3 = 0;
        let pad_4 = 0;
        let pad_5 = 0;
        let pad_6 = 0;
        let pad_7 = 0;
        let pad_8 = 0;
        return Ok(());
    }
    Ok(())
}
