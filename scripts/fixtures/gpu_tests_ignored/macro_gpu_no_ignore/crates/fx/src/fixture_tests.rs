// A macro-generated test that binds Device::Gpu directly and carries no
// #[ignore]. The gate must fail on the macro BODY — the invocations below
// emit no `fn` line of their own.

macro_rules! gpu_cell {
    ($name:ident, $n:expr) => {
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run($n, device);
        }
    };
}

gpu_cell!(cell_a, 1);
gpu_cell!(cell_b, 2);
