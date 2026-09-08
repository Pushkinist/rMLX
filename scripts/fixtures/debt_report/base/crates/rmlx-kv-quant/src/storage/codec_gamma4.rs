pub struct CodecGamma4 {
    rows: Vec<u8>,
    scales: Vec<f32>,
}

impl CodecGamma4 {
    pub fn dequantize(&self, index: usize) -> f32 {
        let scale = self.scales[index];
        self.rows[index] as f32 * scale
    }

    pub fn truncate_to(&mut self, n: usize) {
        self.rows.truncate(n);
        self.scales.truncate(n);
    }

    pub fn high_watermark(&self) -> usize {
        self.rows.len().max(self.scales.len())
    }
}
