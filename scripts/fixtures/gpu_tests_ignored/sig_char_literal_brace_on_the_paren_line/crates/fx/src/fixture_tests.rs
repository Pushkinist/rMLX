#[test]
fn gpu_ch(
    x: u32,
) { let c = '}';
    let device = Device::Gpu;
    run(device, c, x);
}
