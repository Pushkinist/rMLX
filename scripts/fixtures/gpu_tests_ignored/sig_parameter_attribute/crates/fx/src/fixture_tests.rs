#[test]
fn gpu_t(
    #[allow(unused)] x: u32,
) {
    let device = Device::Gpu;
    run(device, x);
}
