struct S;
#[cfg(test)] impl S { fn h_impl() -> Device { Device::Gpu } }

#[test]
fn t_impl() {
    let _d = S::h_impl();
}

#[cfg(test)] mod m { #[test] fn t_mod() { let _d = Device::Gpu; } }
