#[test]
fn gpu_n<const N: usize = { 4 }>(
    x: u32,
) {
    let device = Device::Gpu;
    run(device, x);
}
