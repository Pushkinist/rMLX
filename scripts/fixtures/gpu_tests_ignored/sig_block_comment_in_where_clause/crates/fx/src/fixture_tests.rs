#[test]
fn gpu_l2<T>(x: T)
where
    /* the device
     * to use;
     */
    T: Copy,
{
    let device = Device::Gpu;
    run(device, x);
}
