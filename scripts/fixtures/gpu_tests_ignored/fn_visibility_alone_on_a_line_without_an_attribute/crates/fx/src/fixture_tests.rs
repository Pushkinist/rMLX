pub(crate)
fn gpu_helper() -> Device {
    Device::Gpu
}

unsafe extern "C"
fn gpu_extern_helper() -> Device {
    Device::Gpu
}

#[test]
fn calls_vis_helper() {
    let _ = gpu_helper();
}

#[test]
fn calls_extern_helper() {
    let _ = unsafe { gpu_extern_helper() };
}
