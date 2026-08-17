// Both markers on one test: it drives Metal (declared) and it does not
// (exempt). There is no right answer, so the gate fails closed instead of
// picking one — silently preferring either half is how a marker rots into a
// permanent, unreviewed exemption.

// gpu-test-gate: metal-unscanned  the handler picks the device
// gpu-test-gate: exempt
#[ignore = "GPU Metal context"]
#[test]
fn declared_and_exempt() {
    post("/v1/embeddings");
}
