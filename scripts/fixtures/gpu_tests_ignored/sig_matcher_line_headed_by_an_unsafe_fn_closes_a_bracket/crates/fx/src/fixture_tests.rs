macro_rules! wrap {
    (
        unsafe fn foo($x:expr) $b:block
    ) => {
        fn generated() {}
    };
}

mod tests {
    #[test]
    fn gpu_hidden() {
        let d = Device::Gpu;
        if true {
            let _ = d;
        }
    }
}
