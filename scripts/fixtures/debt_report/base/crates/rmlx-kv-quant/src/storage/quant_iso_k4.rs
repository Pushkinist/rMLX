pub struct QuantIsoK4 {
    packed: Vec<u8>,
}

impl QuantIsoK4 {
    pub fn byte_size(&self) -> usize {
        self.packed.len() * ISO_WIDTH_4 / 8
    }
}
