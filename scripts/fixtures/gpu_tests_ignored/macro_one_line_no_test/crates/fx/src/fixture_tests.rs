// A one-line macro_rules! that declares NO test is ordinary code and must be
// stepped over, not latched. If the parser instead treats it as an open body,
// it stays "inside a macro" for the rest of the file: the next macro_rules!
// opener is skipped, and that macro's violation is reported under THIS
// macro's name. Recall survives, blame does not — so the label below is what
// this fixture pins.

macro_rules! noop_cell { ($n:ident) => { let _ = $n; }; }

macro_rules! gpu_cell {
    ($name:ident) => {
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run(device);
        }
    };
}

gpu_cell!(cell_a);
