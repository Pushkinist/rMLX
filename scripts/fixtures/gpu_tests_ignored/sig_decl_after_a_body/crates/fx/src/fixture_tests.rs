fn first() {
    run();
}

trait T {
    fn wrapped(
        x: u32,
    ) -> u32;
}

#[test]
fn gpu_after() {
    let device = Device::Gpu;
    run(device);
}
