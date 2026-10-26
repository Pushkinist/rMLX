#[test]
fn gpu_hc<T>(a: T) -> u32 /* a note that
    goes on;
    and on */
where
    T: Copy,
{
    let d = Device::Gpu;
    0
}
