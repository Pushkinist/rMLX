#[test]
#[cfg_attr(
    feature = "a", /* a comment that
    ends in a bracket ]
    */ feature = "b"
)]
fn gpu_bc() {
    let d = Device::Gpu;
}
