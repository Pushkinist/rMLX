mod common;

// gpu-test-gate: metal-unscanned  spawns a child
#[ignore = "GPU Metal context"]
#[test]
fn declared_child() {
    spawn("rmlx serve");
}

#[ignore = "GPU Metal context"]
#[test]
fn integ_via_helper() {
    common::gpu_helper("integ_via_helper");
}
