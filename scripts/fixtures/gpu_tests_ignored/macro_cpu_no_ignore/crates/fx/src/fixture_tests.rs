// Negative control: a macro-generated CPU test is legitimately un-ignored.
// Flagging it would push authors to #[ignore] a test that then never runs.

macro_rules! cpu_cell {
    ($name:ident) => {
        #[test]
        fn $name() {
            let device = Device::Cpu;
            assert!(policy(device));
        }
    };
}

cpu_cell!(cell_a);
cpu_cell!(cell_b);
