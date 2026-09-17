const BITS_3: usize = 3;

pub struct QuantRotorV3 {
    blocks: Vec<u8>,
}

impl QuantRotorV3 {
    pub fn push_block(&mut self, block: &[u8]) {
        self.blocks.extend_from_slice(block);
    }

    pub fn rows(&self) -> usize {
        self.blocks.len() / BITS_3
    }
}
