fn first() -> Foo<{ N < 4 }> { 0 }

#[test]
fn gpu_g2() {
    let d = Device::Gpu;
}
