const BITS_4: usize = 4;

pub struct QuantRotorV4 {
    blocks: Vec<u8>,
}

impl QuantRotorV4 {
    pub fn push_block(&mut self, block: &[u8]) {
        self.blocks.extend_from_slice(block);
    }

    pub fn rows(&self) -> usize {
        self.blocks.len() / BITS_4
    }
}
