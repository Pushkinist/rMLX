// A signature-only fn whose `where` clause puts the closing `;` on a later line. Until the
// body's `{` has been seen, a line that ends in `;` ends the declaration, so it neither
// latches nor swallows the test after it, which is classified and, being un-ignored and
// reaching Device::Gpu, named.

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
