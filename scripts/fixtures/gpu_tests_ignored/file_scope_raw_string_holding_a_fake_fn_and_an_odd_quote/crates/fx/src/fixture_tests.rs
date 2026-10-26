const SRC: &str = r#"
fn fake() {
    let s = "abc
}
"#;

#[test]
fn gpu_t01() {
    let d = Device::Gpu;
}

struct S;
impl S {
}
