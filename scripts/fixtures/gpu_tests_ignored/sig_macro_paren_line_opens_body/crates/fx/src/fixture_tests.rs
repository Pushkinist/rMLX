macro_rules! gpu_cell {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name(
            x: $t,
        ) { let device = Device::Gpu;
            run(device, x);
        }
    };
}

gpu_cell!(cell_h, u32);
