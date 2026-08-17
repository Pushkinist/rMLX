// Negative control for the declared-route marker, plus the list/enforce split
// it introduces. The declared test is ENFORCED (its #[ignore] is required) and
// must stay OUT of `--list`; the plain GPU test beside it must still go in, so
// a change that starts listing declared tests — or that drops plain ones —
// shows up as a different stdout rather than as a still-green run.

// gpu-test-gate: metal-unscanned  the handler picks the device
#[ignore = "GPU Metal context"]
#[test]
fn declared_metal_test() {
    post("/v1/embeddings");
}

#[ignore = "GPU Metal context"]
#[test]
fn plain_gpu_test() {
    let device = Device::Gpu;
    run(device);
}
