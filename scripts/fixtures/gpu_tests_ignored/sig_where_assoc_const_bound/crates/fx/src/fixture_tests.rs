trait C {
    fn f<const N: usize>(
        x: [u8; N],
    ) where
        [(); N * 2]: Sized;
}

#[test]
fn gpu_g() {
    let device = Device::Gpu;
    if true {
        run(device);
    }
}
