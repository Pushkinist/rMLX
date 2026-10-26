pub struct S<
    #[cfg(x)] T,
    #[cfg(y)] U,
>(T, U);

#[test]
fn gpu_after_generics() {
    let d = Device::Gpu;
}
