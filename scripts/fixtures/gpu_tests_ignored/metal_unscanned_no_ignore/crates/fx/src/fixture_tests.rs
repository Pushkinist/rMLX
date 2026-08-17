// The marker's enforcement half: a declared route makes the test GPU-touching,
// so a missing #[ignore] is a violation. Before the marker existed this exact
// shape was invisible in both directions — never flagged, never listed — and
// deleting the attribute was caught by nothing.

// gpu-test-gate: metal-unscanned  the handler picks the device
#[test]
fn declared_without_ignore() {
    post("/v1/embeddings");
}
