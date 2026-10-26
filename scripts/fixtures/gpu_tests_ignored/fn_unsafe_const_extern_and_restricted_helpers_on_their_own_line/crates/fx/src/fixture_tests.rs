#[inline]
unsafe fn h_unsafe() -> Device {
    Device::Gpu
}

#[inline]
const fn h_const() -> Device {
    Device::Gpu
}

#[inline]
extern "C" fn h_extern() -> Device {
    Device::Gpu
}

#[inline]
pub(in crate) fn h_pubin() -> Device {
    Device::Gpu
}

#[inline]
pub(crate) const unsafe fn h_pubcrate() -> Device {
    Device::Gpu
}

#[test]
fn calls_unsafe() {
    let _d = unsafe { h_unsafe() };
}

#[test]
fn calls_const() {
    let _d = h_const();
}

#[test]
fn calls_extern() {
    let _d = h_extern();
}

#[test]
fn calls_pubin() {
    let _d = h_pubin();
}

#[test]
fn calls_pubcrate() {
    let _d = unsafe { h_pubcrate() };
}
