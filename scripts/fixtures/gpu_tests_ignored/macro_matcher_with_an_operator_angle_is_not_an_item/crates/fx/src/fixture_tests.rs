macro_rules! lt {
    (
        fn $name:ident where $a:literal < $b:literal;
    ) => {
        #[test]
        fn $name() {
            let d = Device::Gpu;
            assert!($a < $b);
        }
    };
}

lt!(fn gpu_lt where 1 < 2;);
