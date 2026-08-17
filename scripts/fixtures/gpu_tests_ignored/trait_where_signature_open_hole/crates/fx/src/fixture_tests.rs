// THIS FIXTURE PINS A KNOWN OPEN HOLE. It asserts the gate's CURRENT, WRONG
// answer, so that the hole is visible in the corpus instead of living only in
// a comment — and so that closing it is a deliberate act that updates this
// file, not a surprise.
//
// The shape: a signature-only `fn` is recognised by its trailing `;`, but a
// `where` clause pushes that `;` onto a later line. The declaration therefore
// latches the multi-line capture, and the capture terminates at the first
// later line that bares to the fn's indent. In a Rust test file that is `    }`
// — the close of any nested block — which is one of the most common lines
// there is. The difference from `trait_where_signature/` is exactly the `if`
// block below.
//
// When the latch closes that way, no `U` is emitted: the swallowed `#[test]`
// was never registered, so nothing looks unterminated. The gate reports
// `OK` and exits 0 over an un-ignored Device::Gpu test.
//
// This is FAIL-OPEN, not fail-closed. It is unreachable in the tree today (no
// scanned file has a where-split signature) and the class is tracked for
// reconciliation against the compiled `cargo test -- --list`, which is what
// actually closes it.

trait Loader {
    fn load<T>(&self, x: T) -> u32
    where
        T: Copy;
}

#[test]
fn after_where_gpu_no_ignore() {
    let device = Device::Gpu;
    if device.is_gpu() {
        run(device);
    }
    run(device);
}
