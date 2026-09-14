pub struct CodecAlpha4 {
    rows: Vec<u8>,
}

impl CodecAlpha4 {
    pub fn append(&mut self, block: &[u8]) {
        for byte in block {
            self.rows.push(*byte);
        }
    }

    pub fn byte_size(&self) -> usize {
        self.rows.len()
    }
}
