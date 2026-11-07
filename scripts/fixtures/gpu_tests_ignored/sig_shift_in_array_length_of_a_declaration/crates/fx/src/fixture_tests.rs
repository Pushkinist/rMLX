trait Cells {
    fn f(x: [u8; 1 << 4]);
    fn g(&self);
}

#[test]
fn gpu_z1() {
    if true {
        let d = Device::Gpu;
    }
}
