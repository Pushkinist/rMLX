// A marker whose claim the scanner can now check for itself. The test binds
// Device::Gpu right here, so nothing needs declaring — and leaving the marker
// would hold a perfectly listable test out of `--list` forever. Stale markers
// are the failure mode of every opt-out, so this one fails closed.

// gpu-test-gate: metal-unscanned  the handler picks the device
#[ignore = "GPU Metal context"]
#[test]
fn declared_but_visible() {
    let device = Device::Gpu;
    run(device);
}
