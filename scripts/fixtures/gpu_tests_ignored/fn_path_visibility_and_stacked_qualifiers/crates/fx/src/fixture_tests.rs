pub(in crate::a) fn h_path() -> Device {
    Device::Gpu
}

async unsafe fn h_async_unsafe() -> Device {
    Device::Gpu
}

unsafe extern "C" fn h_unsafe_extern() -> Device {
    Device::Gpu
}

pub const fn h_pub_const() -> Device {
    Device::Gpu
}

#[test]
fn calls_path() {
    let _d = h_path();
}

#[test]
fn calls_async_unsafe() {
    let _d = unsafe { h_async_unsafe() };
}

#[test]
fn calls_unsafe_extern() {
    let _d = unsafe { h_unsafe_extern() };
}

#[test]
fn calls_pub_const() {
    let _d = h_pub_const();
}
