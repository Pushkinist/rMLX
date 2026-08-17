// The closing brace is indented differently from the `fn` keyword, so the
// capture's close marker never matches and the parser loses the file from
// here. That must be reported, not skipped: everything after an unterminated
// capture is unclassified, which looks exactly like a clean scan.

#[test]
fn plain_gpu() {
    let device = Device::Gpu;
    run(device);
  }
