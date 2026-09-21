pub struct Blocks {
    codes: Vec<u8>,
    scales: Vec<f32>,
}

pub struct Reader {
    blocks: Vec<Blocks>,
}

impl Reader {
    fn k_turbo3_shape(&self, n: usize) -> usize {
        n * 3
    }

    fn k_turbo4_shape(&self, n: usize) -> usize {
        n * 4
    }

    fn write_quant_k_turbo3(&self, n: usize) -> usize {
        let words = n * 3 / 32;
        let scales = n / 32;
        words + scales
    }

    fn write_quant_k_turbo4(&self, n: usize) -> usize {
        let words = n * 4 / 32;
        let scales = n / 32;
        words + scales
    }

    fn read_tsym3(&self, n: usize) -> usize {
        self.k_turbo3_shape(n) + self.blocks.len()
    }

    fn read_tsym4(&self, n: usize) -> usize {
        self.k_turbo4_shape(n) + self.blocks.len()
    }

    fn read_quant_k_turbo3(&self, n: usize) -> usize {
        let shape = self.k_turbo3_shape(n);
        shape + self.blocks.len() * 3
    }

    fn read_quant_k_turbo4(&self, n: usize) -> usize {
        let shape = self.k_turbo4_shape(n);
        shape + self.blocks.len() * 4
    }

    fn read_quant_k(&self, n: usize) -> usize {
        n + self.blocks.len()
    }
}
