pub struct QuantIsoK3 {
    packed: Vec<u8>,
}

impl QuantIsoK3 {
    pub fn byte_size(&self) -> usize {
        self.packed.len() * ISO_WIDTH_3 / 8
    }
}
