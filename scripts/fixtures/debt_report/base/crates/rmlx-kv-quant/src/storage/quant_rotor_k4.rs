pub struct QuantRotorK4 {
    packed: Vec<u8>,
}

impl QuantRotorK4 {
    pub fn byte_size(&self) -> usize {
        self.packed.len() * WIDTH_4 / 8
    }
}
