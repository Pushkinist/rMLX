trait L {
    fn load<T>(&self, x: T) -> u32
    where
        T: Into<[u8; { 4 }]>;
}

#[test]
fn gpu_o() {
    let device = Device::Gpu;
    if true {
        run(device);
    }
}
