#[test]
fn gpu_c2(
    x: u32,
) { /* drives the GPU */
    let device = Device::Gpu;
    run(device, x);
}
