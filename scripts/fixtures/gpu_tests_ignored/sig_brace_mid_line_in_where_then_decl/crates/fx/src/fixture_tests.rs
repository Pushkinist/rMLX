trait L {
    fn load<T>(&self, x: T) -> u32
    where
        T: Into<Foo<{ 4 }>> + Copy,
        T: Clone;
}

#[test]
fn gpu_s() {
    let device = Device::Gpu;
    if true {
        run(device);
    }
}
