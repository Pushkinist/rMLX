#[ignore = "GPU Metal context"]
#[test]
fn gpu_fy() {
    let device = Device::Gpu;
    run(device);
}
