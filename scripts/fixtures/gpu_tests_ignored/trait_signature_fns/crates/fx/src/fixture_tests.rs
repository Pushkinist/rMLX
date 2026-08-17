// A trait's method signatures have no brace at all on their `fn` line. A rule
// that latches whenever no brace is OPEN captures them forever, and the gate
// then hard-fails the whole run blaming a macro_rules! this file does not
// contain — while everything after the trait goes unclassified. The violation
// below is what proves classification survived the trait.

trait MockLoader {
    fn load(&self, path: &str) -> u32;
    fn unload(&self);
}

extern "C" {
    fn c_helper(x: u32) -> u32;
}

#[test]
fn after_trait_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device);
}
