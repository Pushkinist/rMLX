#[test]
fn gpu_ml<T>(x: T)
where
    T: Copy,
{
    let s = r#"
};
"#;
    let d = Device::Gpu;
    run(d, s, x);
}
