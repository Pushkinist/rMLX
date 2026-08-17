// The boundary of the last-character rule. A signature-only fn is recognised
// by its trailing `;`, and a `where` clause pushes that `;` onto a later line
// — so this one DOES latch, and the capture never terminates.
//
// That is the fail-closed side and it is the intended outcome: the run stops
// loudly instead of classifying the rest of the file as clean. What this
// fixture pins is that it stays loud. The diagnostic still talks about
// macro_rules! on a file that has none, which is a wording debt, not a
// classification one.

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
