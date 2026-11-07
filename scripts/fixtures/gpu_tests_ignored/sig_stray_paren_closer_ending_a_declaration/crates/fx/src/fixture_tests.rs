decl!(
    fn foo() -> u8
);

#[test]
fn gpu_after_paren_macro() {
    let _ = Device::Gpu;
}
