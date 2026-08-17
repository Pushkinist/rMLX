// Negative control for the exemption marker. It only exempts from a fn's
// ATTRIBUTE block; a copy inside the body must not, or any test could opt
// itself out by dropping one comment line among its statements.
//
// BOTH fns below must be flagged, and the second one is the load-bearing
// half. The first is protected by the attribute snapshot regardless of the
// marker rule's guard — its attributes were already captured before the body
// was read. What the guard actually prevents is the marker LEAKING out of one
// body into the pending attribute block of the next fn, silently exempting a
// test that never mentioned it.

#[test]
fn gpu_marker_inside_body() {
    // gpu-test-gate: exempt
    let device = Device::Gpu;
    run(device);
}

#[test]
fn gpu_after_marker_in_body() {
    let device = Device::Gpu;
    run(device);
}
