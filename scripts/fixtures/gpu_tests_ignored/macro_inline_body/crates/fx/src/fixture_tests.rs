// The generated fn opens and closes on ONE line, so no later `}` matches its
// close marker. A classifier that latches on it swallows the rest of the file
// and stops classifying — the plain violation below then goes unreported,
// which is worse than not reading the macro at all.
//
// The `$(...)*` repetition is load-bearing: rustfmt refuses to reformat this
// body (verified byte-identical after `rustfmt --edition 2021`), so `make
// fmt-check` does not keep the shape out of the tree.

macro_rules! gpu_cell {
    ($name:ident, $($arg:expr),*) => {
        #[ignore]
        #[test]
        fn $name() { let device = Device::Gpu; run(device, $($arg),*); }
    };
}
gpu_cell!(cell_a, 1, 2);

#[test]
fn later_plain_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device);
}
