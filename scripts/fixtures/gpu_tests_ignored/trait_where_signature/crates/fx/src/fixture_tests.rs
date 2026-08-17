// The boundary of the last-character rule. A signature-only fn is recognised
// by its trailing `;`, and a `where` clause pushes that `;` onto a later line
// — so this one DOES latch.
//
// This fixture pins only the SUB-CASE where nothing ever closes the latch, and
// it is loud only for that reason: the file below has no nested block, so no
// later line bares to the fn's indent. Add one and the latch closes silently —
// see `trait_where_signature_open_hole/`, which pins that fail-OPEN answer.
//
// So: not "where-split signatures fail closed". Only "a where-split signature
// with nothing after it to close the latch fails closed". The diagnostic also
// still talks about macro_rules! on a file that has none, which is a wording
// debt on top.

trait Loader {
    fn load<T>(&self, x: T) -> u32
    where
        T: Copy;
}

#[test]
fn after_where_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device);
}
