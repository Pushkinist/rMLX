pub struct P(
    #[serde(default)] pub u8,
    #[serde(default)] pub u16,
);

#[test]
fn gpu_after_tuple_fields() {
    let d = Device::Gpu;
}
