decl!(
fn foo() -> u8);

#[test]
fn gpu_after_header_decl() {
    let _ = Device::Gpu;
}
