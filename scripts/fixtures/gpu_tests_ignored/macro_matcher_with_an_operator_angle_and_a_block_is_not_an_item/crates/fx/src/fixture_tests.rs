macro_rules! lt {
    (
        fn $name:ident() where $a:literal < $b:literal { $($body:tt)* }
    ) => {
        #[test]
        fn $name() {
            let d = Device::Gpu;
            assert!($a < $b);
            $($body)*
        }
    };
}

lt!(fn gpu_lt() where 1 < 2 {});
