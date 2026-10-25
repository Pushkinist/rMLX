#[ignore = "GPU Metal context"]
#[test]
fn gpu_fx() {
    let device = Device::Gpu;
    run(device);
}
