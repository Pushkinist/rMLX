// The fail-closed backstop for the attribute capture, and the reason the
// close-test is not the whole fix. Recognising more spellings makes the latch
// rarer; it cannot make it impossible, because the scan has documented limits
// (raw-string hashes, block comments) and any shape past them looks exactly
// like an attribute that has not closed yet.
//
// So an attribute still open at the end of a file is REPORTED. Without that,
// the swallowed test below is unclassified and the gate exits 0 saying every
// GPU test carries #[ignore] — fail-open, which is the one direction this gate
// must never take. Nothing after the opener bares to a `]`, so nothing closes
// the capture and the file ends inside it.

#[ignore = "GPU Metal context — this string's closing quote never arrives
fn swallowed_gpu_test() {
    let device = Device::Gpu;
    run(device);
}
