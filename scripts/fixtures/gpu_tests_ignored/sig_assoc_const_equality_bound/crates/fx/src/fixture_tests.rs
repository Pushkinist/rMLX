trait L {
    fn load<T>(&self, x: T) -> u32
    where
        T: Tr<N = 4>;
}

#[test]
fn gpu_u() {
    let device = Device::Gpu;
    if true {
        run(device);
    }
}
