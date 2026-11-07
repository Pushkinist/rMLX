#[ignore = "GPU Metal context"]
#[test]
fn gpu_same() {
    let device = Device::Gpu;
    run(device);
}
