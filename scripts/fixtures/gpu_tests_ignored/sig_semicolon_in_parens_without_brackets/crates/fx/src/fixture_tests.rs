#[test]
fn gpu_par($(
    $p: u32;
)*) {
    let device = Device::Gpu;
    run(device);
}
