// The `}` here lives in a comment, so the line still OPENS the body. A brace
// count reads it as balanced, declines to latch, and the Device::Gpu below
// never reaches the fn's body — the test goes unclassified and un-flagged.
// That direction is a recall regression, not just a missed extension.

#[test]
fn plain_gpu_no_ignore() { // the body closes with a } below
    let device = Device::Gpu;
    run(device);
}
