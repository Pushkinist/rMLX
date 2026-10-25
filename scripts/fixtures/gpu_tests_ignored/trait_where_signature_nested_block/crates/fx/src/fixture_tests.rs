// The same declaration, and a test after it with a nested block. A declaration that latched
// would be closed by that block's `    }` and the test swallowed with it, passing as clean.
// Ending the declaration at its `;` classifies the test, which is un-ignored and reaches
// Device::Gpu, and so is named.

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
