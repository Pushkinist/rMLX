// Regression guard: the compliant non-macro shape stays green.

#[ignore = "GPU Metal context"]
#[test]
fn plain_gpu_test() {
    let device = Device::Gpu;
    run(device);
}
