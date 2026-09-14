pub struct CodecGamma3 {
    rows: Vec<u8>,
    scales: Vec<f32>,
}

impl CodecGamma3 {
    pub fn append(&mut self, block: &[u8], scale: f32) {
        for byte in block {
            self.rows.push(*byte);
        }
        self.scales.push(scale);
    }

    pub fn byte_size(&self) -> usize {
        self.rows.len()
    }

    pub fn reset(&mut self) {
        self.rows.clear();
        self.scales.clear();
    }
}
