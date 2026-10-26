#[test]
fn gpu_bs2() {
    const SRC: &str = "\
fn outer() {
    let a = MARK;
}
";
    let d = Device::Gpu;
    read(SRC);
}
