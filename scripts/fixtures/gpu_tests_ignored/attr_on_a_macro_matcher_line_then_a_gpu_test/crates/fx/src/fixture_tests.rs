macro_rules! gpu_tests {
    (
        #[$m:meta] fn $name:ident
    ) => {
        #[test]
        fn $name() {
            let d = Device::Gpu;
        }
    };
}
