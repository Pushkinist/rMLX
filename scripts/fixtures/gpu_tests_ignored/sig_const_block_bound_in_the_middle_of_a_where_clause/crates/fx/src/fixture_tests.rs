#[test]
fn gpu_wb<T>(x: T)
where
    T: Tr<{ 4 }>,
    T: Copy,
{
    let d = Device::Gpu;
}
