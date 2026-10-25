#[test]
fn gpu_l(
    /* the device
     * to use;
     */
    x: u32,
) {
    let device = Device::Gpu;
    run(device, x);
}
