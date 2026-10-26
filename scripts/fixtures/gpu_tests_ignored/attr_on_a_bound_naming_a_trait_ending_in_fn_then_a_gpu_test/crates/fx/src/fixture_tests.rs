pub struct S<
    #[cfg(x)] T: Handler_fn + Send,
    U,
>(T, U);

#[test]
fn gpu_after_bound() {
    let d = Device::Gpu;
}
