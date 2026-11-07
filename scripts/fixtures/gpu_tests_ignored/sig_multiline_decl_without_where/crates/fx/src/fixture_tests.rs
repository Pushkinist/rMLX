trait L {
    fn load(
        &self,
        x: u32,
    ) -> u32;
    fn other(&self);
}

#[test]
fn gpu_j() {
    let device = Device::Gpu;
    if true {
        run(device);
    }
}
