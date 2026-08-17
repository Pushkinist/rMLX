// The first macro's closing brace carries a comment. An exact string compare
// never closes the body, so the second macro's opener is skipped and its
// violation gets reported under the FIRST macro's name — recall survives but
// the operator is sent to the wrong macro.

macro_rules! first_cell {
    ($name:ident) => {
        #[ignore]
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run(device);
        }
    };
} // end of first_cell

first_cell!(cell_a);

macro_rules! second_cell {
    ($name:ident) => {
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run(device);
        }
    };
}

second_cell!(cell_b);
