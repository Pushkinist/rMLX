// Cutting this line at the first `//` would end it mid-string, so its last
// significant character would no longer be the `}` that closes it — the fn
// would latch and swallow the violation below. A URL in a one-line fn is an
// ordinary thing to write, so the comment scan is quote-aware.

#[ignore = "GPU Metal context"]
#[test]
fn probe() { let d = Device::Gpu; fetch(d, "http://localhost:8080/v1/models"); }

#[test]
fn later_plain_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device);
}
