#[test]
fn gpu_b(
    x: u32,
) -> u32 { // comment {
    let device = Device::Gpu;
    run(device, x)
}
