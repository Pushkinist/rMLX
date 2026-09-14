pub struct CodecBeta3 {
    rows: Vec<u8>,
    scales: Vec<f32>,
    zero_points: Vec<i8>,
    ring_cursor: usize,
    high_watermark: usize,
}

impl CodecBeta3 {
    pub fn append_row(&mut self, block: &[u8], scale: f32, zero_point: i8) {
        for byte in block {
            self.rows.push(*byte);
        }
        self.scales.push(scale);
        self.zero_points.push(zero_point);
        self.ring_cursor = (self.ring_cursor + 1) % self.rows.len().max(1);
        if self.ring_cursor > self.high_watermark {
            self.high_watermark = self.ring_cursor;
        }
    }

    pub fn dequantize_row(&self, index: usize) -> Vec<f32> {
        let scale = self.scales[index];
        let zero_point = self.zero_points[index] as f32;
        self.rows
            .iter()
            .map(|b| (*b as f32 - zero_point) * scale)
            .collect()
    }

    pub fn reset(&mut self) {
        self.rows.clear();
        self.scales.clear();
        self.zero_points.clear();
        self.ring_cursor = 0;
        self.high_watermark = 0;
    }
}
