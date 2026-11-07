fn gpu_helper(
    x: u32,
) -> u32 { run(x, Device::Gpu) }

#[test]
fn calls_the_inline_helper() {
    gpu_helper(1);
}
