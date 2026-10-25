#[test]
fn gpu_a(
    x: u32,
) { let device = Device::Gpu;
    run(device, x);
}
