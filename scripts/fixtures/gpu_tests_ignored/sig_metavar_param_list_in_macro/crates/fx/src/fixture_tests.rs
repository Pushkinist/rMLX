macro_rules! gpu_cell2 {
    ($name:ident, $($p:ident: $t:ty),*) => {
        #[test]
        fn $name(
            $($p: $t),*
        ) { let device = Device::Gpu; }
    };
}

gpu_cell2!(cell_m, a: u32);
