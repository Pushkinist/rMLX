#[test]
#[ignore = "slow"]
fn a() { let _s = r#"a"b"#; }

#[test]
fn gpu_rs1() {
    let d = Device::Gpu;
}
