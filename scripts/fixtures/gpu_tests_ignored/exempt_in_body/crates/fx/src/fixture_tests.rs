// Negative control for the exemption marker. It only exempts from a fn's
// ATTRIBUTE block; a copy inside the body must not, or any test could opt
// itself out by dropping one comment line among its statements.
//
// BOTH fns below must be flagged. Neither is protected by the marker rule's
// `!in_fn` guard: the first fn's attributes were already snapshotted before
// its body was read, and the marker cannot carry to the second because the
// fn-close rule clears the pending attribute block. Deleting that guard was
// measured to change no output on the fixture set or the real tree — the two
// mechanisms are mutually redundant, which is why no SINGLE mutation kills
// this fixture (deleting the guard AND the fn-close reset does).
//
// So what this fixture pins is the BEHAVIOUR, not the guard: a marker sitting
// among a fn's statements exempts nothing, and does not reach the next fn. If
// the mechanism is ever reworked, this is the property that must survive.

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
