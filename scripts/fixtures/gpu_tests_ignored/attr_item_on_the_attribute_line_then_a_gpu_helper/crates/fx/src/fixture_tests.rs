#[derive(Clone)] struct A;

fn helper() -> Device {
    Device::Gpu
}

#[test]
fn gpu_t04() {
    let d = helper();
}
