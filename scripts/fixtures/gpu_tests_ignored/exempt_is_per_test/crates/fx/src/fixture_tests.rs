// gpu-test-gate: exempt
#[test]
fn exempt_policy_value() {
    pick(Device::Gpu);
}

#[test]
fn real_gpu() {
    let device = Device::Gpu;
    run(device);
}
