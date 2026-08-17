// The declared-route marker is scoped exactly like the exemption: it must lead
// its line INSIDE the fn's attribute block. A copy among a fn's statements
// declares nothing — otherwise a marker could be smuggled in as prose — and it
// must not carry to the next fn either. Both tests below stay undeclared, so
// both are reported.

#[ignore = "GPU Metal context"]
#[test]
fn marker_inside_body() {
    // gpu-test-gate: metal-unscanned
    post("/v1/embeddings");
}

#[ignore = "GPU Metal context"]
#[test]
fn test_after_body_marker() {
    post("/v1/embeddings");
}
