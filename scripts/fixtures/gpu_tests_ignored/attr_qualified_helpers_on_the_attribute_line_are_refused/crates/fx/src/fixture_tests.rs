#[inline] unsafe fn h_unsafe() -> Device { Device::Gpu }
#[inline] const fn h_const() -> Device { Device::Gpu }
#[inline] extern "C" fn h_extern() -> Device { Device::Gpu }
#[inline] pub(in crate) fn h_pubin() -> Device { Device::Gpu }
#[inline] pub(crate) const unsafe fn h_pcu() -> Device { Device::Gpu }

#[test]
fn t_unsafe() {
    let _d = unsafe { h_unsafe() };
}

#[test]
fn t_const() {
    let _d = h_const();
}

#[test]
fn t_extern() {
    let _d = h_extern();
}

#[test]
fn t_pubin() {
    let _d = h_pubin();
}

#[test]
fn t_pcu() {
    let _d = unsafe { h_pcu() };
}
