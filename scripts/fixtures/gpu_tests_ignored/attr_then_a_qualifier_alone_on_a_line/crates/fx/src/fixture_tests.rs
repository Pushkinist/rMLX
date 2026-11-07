#[tokio::test]
async
fn gpu_split_after_async() {
    let _ = Device::Gpu;
}

#[inline]
pub(crate)
fn gpu_helper() -> Device {
    Device::Gpu
}

#[test]
fn calls_split_helper() {
    let _ = gpu_helper();
}

#[inline]
unsafe extern "C"
fn gpu_extern_split() -> Device {
    Device::Gpu
}

#[inline]
const
fn gpu_const_split() -> Device {
    Device::Gpu
}
