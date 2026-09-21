const TURBO4_K_BITS: u8 = 4;

pub struct QuantKTurbo4 {
    codes: Vec<u32>,
    scales: Vec<f32>,
}

impl QuantKTurbo4 {
    pub fn words_per_step(&self, d: usize) -> usize {
        d * TURBO4_K_BITS as usize / 32
    }

    pub fn byte_size(&self) -> usize {
        self.codes.len() * 4 + self.scales.len() * 4
    }
}
