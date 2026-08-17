// Regression guard: the original (non-macro) detection must keep working.

#[test]
fn plain_gpu_test() {
    let device = Device::Gpu;
    run(device);
}
