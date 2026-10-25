#[test]
fn gpu_f(
    buf: [u8;
        4],
) {
    let device = Device::Gpu;
    run(device, buf);
}
