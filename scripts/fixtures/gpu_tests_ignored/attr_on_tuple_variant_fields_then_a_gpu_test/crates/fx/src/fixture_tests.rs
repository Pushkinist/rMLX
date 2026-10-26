pub enum E {
    A,
    B(
        #[serde(default)] u8,
        u16,
    ),
}

#[test]
fn gpu_after_variant_fields() {
    let d = Device::Gpu;
}
