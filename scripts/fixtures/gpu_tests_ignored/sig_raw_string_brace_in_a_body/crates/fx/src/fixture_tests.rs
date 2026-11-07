#[test]
fn gpu_raw<T>(x: T)
where
    T: Copy,
{
    let s = r#""}""#;
    let device = Device::Gpu;
    run(device, s, x);
}
