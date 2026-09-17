pub struct Cache {
    seq: usize,
}

impl Cache {
    fn update_rotor3(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq * 3
    }

    fn update_rotor4(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq * 4
    }

    fn update_rotor3_sym(&mut self, n: usize) -> usize {
        let scaled = n * 3;
        self.seq += scaled;
        self.seq
    }

    fn update_rotor4_sym(&mut self, n: usize) -> usize {
        let scaled = n * 4;
        self.seq += scaled;
        self.seq
    }

    fn update_rotor_5_sym(&mut self, n: usize) -> usize {
        let scaled = n * 5;
        self.seq += scaled;
        self.seq
    }

    fn update_affine(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq
    }
}
