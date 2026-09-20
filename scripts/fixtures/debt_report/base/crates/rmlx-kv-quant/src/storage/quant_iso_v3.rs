const ISO_BITS_3: usize = 3;

pub struct QuantIsoV3 {
    codes: Vec<u32>,
}

impl QuantIsoV3 {
    pub fn append(&mut self, row: &[u32]) {
        self.codes.extend_from_slice(row);
    }

    pub fn rows(&self) -> usize {
        self.codes.len() / ISO_BITS_3
    }
}
