pub struct QuantRotorK3 {
    packed: Vec<u8>,
}

impl QuantRotorK3 {
    pub fn byte_size(&self) -> usize {
        self.packed.len() * WIDTH_3 / 8
    }
}
