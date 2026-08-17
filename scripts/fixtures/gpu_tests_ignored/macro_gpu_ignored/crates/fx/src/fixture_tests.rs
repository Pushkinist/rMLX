// Negative control: the compliant shape must NOT fire. The plain test below
// it pins the other half of the decision — a macro cell is enforced but stays
// out of `--list`, while a normal GPU test still goes in.

macro_rules! gpu_cell {
    ($name:ident, $n:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run($n, device);
        }
    };
}

gpu_cell!(cell_a, 1);

#[ignore = "GPU Metal context"]
#[test]
fn plain_gpu_test() {
    let device = Device::Gpu;
    run(0, device);
}
