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

    fn update_iso3(&mut self, n: usize) -> usize {
        let groups = n / 4;
        self.seq += groups * 3;
        self.seq
    }

    fn update_iso4(&mut self, n: usize) -> usize {
        let groups = n / 4;
        self.seq += groups * 4;
        self.seq
    }

    fn update_iso_k_only_3(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq * 3
    }

    fn update_iso_k_only_4(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq * 4
    }

    fn iso_v_update(&mut self, n: usize) -> usize {
        let groups = n / 4;
        self.seq += groups;
        self.seq
    }

    fn iso_sym_update(&mut self, n: usize) -> usize {
        let groups = n / 4;
        self.seq += groups;
        self.seq * 2
    }

    fn update_tsym3(&mut self, n: usize) -> usize {
        let scaled = n * 3;
        self.seq += scaled;
        self.seq
    }

    fn update_tsym4(&mut self, n: usize) -> usize {
        let scaled = n * 4;
        self.seq += scaled;
        self.seq
    }

    fn update_k8vturbo3(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq * 3
    }

    fn update_k8vturbo2(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq * 2
    }

    fn update_affine(&mut self, n: usize) -> usize {
        self.seq += n;
        self.seq
    }
}
