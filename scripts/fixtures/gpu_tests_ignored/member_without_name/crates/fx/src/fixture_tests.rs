// A compliant test: with the package-name refusal removed this tree lists a test
// under an empty crate name, so only the refusal makes the case fail.

#[ignore = "GPU Metal context"]
#[test]
fn plain_gpu_test() {
    let device = Device::Gpu;
    run(device);
}
