macro_rules! gpu_tests {
    (
        #[$m:meta]
        unsafe fn $name:ident
    ) => {
        #[test]
        fn $name() {
            let d = Device::Gpu;
        }
    };
}
