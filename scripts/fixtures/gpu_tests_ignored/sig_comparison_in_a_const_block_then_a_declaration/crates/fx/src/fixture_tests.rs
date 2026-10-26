trait T {
    fn f() -> Foo<{ N < 4 }
    >;
}

#[test]
fn gpu_lt() {
    if true {
        let d = Device::Gpu;
    }
}
