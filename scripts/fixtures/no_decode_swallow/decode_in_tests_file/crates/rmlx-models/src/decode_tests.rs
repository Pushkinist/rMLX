pub fn decode() -> Result<Vec<u32>, E> {
    loop {
        match step() {
            Ok(t) => out.push(t),
            Err(e) => {
                tracing::error!(error = %e, "decode step failed");
                break;
            }
        }
    }
}
