#[test]
#[ignore = "needs a Metal context"]
fn gpu_ok() {
    #[allow(unused)] let d = Device::Gpu;
}

#[test]
fn gpu_t05() {
    let d = Device::Gpu;
}
