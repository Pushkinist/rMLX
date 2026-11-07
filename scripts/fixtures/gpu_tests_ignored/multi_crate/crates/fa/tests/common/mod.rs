pub fn gpu_helper(test: &str) {
    let device = Device::Gpu;
    run(test, device);
}
