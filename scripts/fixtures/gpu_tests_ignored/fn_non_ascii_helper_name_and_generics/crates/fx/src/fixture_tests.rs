#[doc = "é ] ü"]
#[cfg_attr(test, allow(non_snake_case))]
fn h_ü<Tä: Copy>(_x: [Tä; 2], _y: Option<Vec<Tä>>) -> Device
where
    Tä: Clone,
{
    Device::Gpu
}

#[test]
fn gpu_uses_helper() {
    let _d = h_ü::<u8>([0, 1], None);
}

#[test]
fn gpu_after() {
    let _d = Device::Gpu;
}
