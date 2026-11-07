#[macro_export] macro_rules! cell {
    ($name:ident) => {
        #[test]
        fn $name() {
            let d = Device::Gpu;
        }
    };
}
