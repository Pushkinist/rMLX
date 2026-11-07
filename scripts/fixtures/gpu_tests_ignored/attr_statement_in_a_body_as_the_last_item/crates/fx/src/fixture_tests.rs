#[test]
fn gpu_t14() {
    let d = Device::Gpu;
}

#[test]
#[ignore = "needs a Metal context"]
fn gpu_ok() {
    #[allow(unused)] let d = Device::Gpu;
}
