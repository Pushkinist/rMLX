pub fn decode() -> Result<Vec<u32>, E> {
    loop {
        match step() {
            Ok(t) => out.push(t),
            Err(e) => {
                tracing::error!(error = %e, "decode step failed");
                let pad_1 = 0;
                let pad_2 = 0;
                let pad_3 = 0;
                let pad_4 = 0;
                let pad_5 = 0;
                let pad_6 = 0;
                let pad_7 = 0;
                return Err(e.into());
            }
        }
    }
}
