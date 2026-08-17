// The other half of the attribute latch, reached by an easier-to-write shape
// than a wrapped string: a trailing comment after the closing bracket. Read
// from the raw line, `#[test] // why` does not end in `]`, so the capture
// latches and swallows the rest of the file — the test below is never
// classified and the gate reports a clean scan over it. The close-test reads
// the line's significant text, so the comment is not part of the decision.

#[test] // an ordinary explanatory comment, not part of the attribute
fn gpu_after_comment_attr() {
    let device = Device::Gpu;
    run(device);
}
