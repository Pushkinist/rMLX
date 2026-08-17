// The shape that actually exists in the tree: the macro body names no device
// at all and reaches Metal one call deep. This is the mutation of a compliant
// cell — the same body with its #[ignore] deleted.

fn run_cell(n: usize) {
    let device = Device::Gpu;
    dispatch(n, device);
}

macro_rules! cell {
    ($name:ident, $n:expr) => {
        #[test]
        fn $name() {
            run_cell($n);
        }
    };
}

cell!(cell_a, 1);
cell!(cell_b, 2);
