#[ignore = "GPU Metal context"]
#[test]
fn fb_gpu() {
    let device = Device::Gpu;
    run(device);
}
