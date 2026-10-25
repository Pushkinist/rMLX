#[test]
fn gpu_g<const N: usize>(
    x: [u8; N],
) where
    [(); N * 2]: Sized,
{
    let device = Device::Gpu;
    run(device, x);
}
