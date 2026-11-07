fn gpu_helper<T>(x: T)
where
    T: Into<[u8; 4]>,
{
    let device = Device::Gpu;
    run(x, device);
}

#[test]
fn calls_helper() {
    gpu_helper(1);
}
