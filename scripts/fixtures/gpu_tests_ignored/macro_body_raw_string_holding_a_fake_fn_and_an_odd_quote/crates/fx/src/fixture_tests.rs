macro_rules! emit {
    () => {
        const SRC: &str = r#"
        fn fake() {
            let s = "abc
        }
        "#;
    };
}

mod m {
    #[test]
    fn gpu_t03() {
        let d = Device::Gpu;
    }

    struct S;
    impl S {
    }
}
