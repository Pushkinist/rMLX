pub struct P {
    #[serde(skip)] cb: fn(u8),
}

#[test]
fn gpu_after_fn_pointer() {
    let d = Device::Gpu;
}
