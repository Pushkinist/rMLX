// A compliant test in the one listed member: with the member floor removed this
// tree reads clean, so only the floor makes the case fail.

#[ignore = "GPU Metal context"]
#[test]
fn plain_gpu_test() {
    let device = Device::Gpu;
    run(device);
}
