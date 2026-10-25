#[ignore = "GPU Metal context"]
#[test]
fn sub_tests_rs_gpu() {
    let device = Device::Gpu;
    run(device);
}

#[ignore = "GPU Metal context"]
#[tokio::test(flavor = "multi_thread")]
async fn tokio_flavored_gpu() {
    let device = Device::Gpu;
    run(device);
}
