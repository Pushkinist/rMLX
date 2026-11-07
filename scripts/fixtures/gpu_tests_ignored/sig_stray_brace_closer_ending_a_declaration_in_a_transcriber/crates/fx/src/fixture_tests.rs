macro_rules! gen_test {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() $body
    };
}

gen_test!(gpu_generated, {
    let _ = Device::Gpu;
});

#[test]
fn gpu_after_transcriber() {
    let _ = Device::Gpu;
}
