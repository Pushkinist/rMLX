#[cfg(x)] const D: Device = Device::Gpu;

#[test]
fn gpu_through_alias() {
    let d = D;
}
