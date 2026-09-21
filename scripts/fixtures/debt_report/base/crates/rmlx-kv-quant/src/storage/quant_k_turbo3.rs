const TURBO3_K_BITS: u8 = 3;

pub struct QuantKTurbo3 {
    codes: Vec<u32>,
    scales: Vec<f32>,
}

impl QuantKTurbo3 {
    pub fn new(capacity: usize) -> Self {
        Self {
            codes: Vec::with_capacity(capacity),
            scales: Vec::with_capacity(capacity),
        }
    }

    pub fn words_per_step(&self, d: usize) -> usize {
        d * TURBO3_K_BITS as usize / 32
    }

    pub fn byte_size(&self) -> usize {
        self.codes.len() * 4 + self.scales.len() * 4
    }
}
