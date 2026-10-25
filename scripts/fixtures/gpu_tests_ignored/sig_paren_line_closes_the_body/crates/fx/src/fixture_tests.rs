fn helper(
    x: u32,
) -> u32 { x }

#[test]
fn gpu_e() {
    let device = Device::Gpu;
    run(device);
}
