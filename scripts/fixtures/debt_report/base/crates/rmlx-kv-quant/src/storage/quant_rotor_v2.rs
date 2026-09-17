const BITS_2: usize = 2;

pub struct QuantRotorV2 {
    blocks: Vec<u8>,
}

impl QuantRotorV2 {
    pub fn push_block(&mut self, block: &[u8]) {
        self.blocks.extend_from_slice(block);
    }

    pub fn rows(&self) -> usize {
        self.blocks.len() / BITS_2
    }
}
