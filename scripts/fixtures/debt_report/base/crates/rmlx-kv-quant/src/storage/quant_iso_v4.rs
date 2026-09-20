const ISO_BITS_4: usize = 4;

pub struct QuantIsoV4 {
    codes: Vec<u32>,
}

impl QuantIsoV4 {
    pub fn append(&mut self, row: &[u32]) {
        self.codes.extend_from_slice(row);
    }

    pub fn rows(&self) -> usize {
        self.codes.len() / ISO_BITS_4
    }
}
