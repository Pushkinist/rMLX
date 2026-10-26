#[test]
fn gpu_all_brackets<T>(
    a: [u8; 4],
    b: (u8, [u8; { 2 }]),
) -> [u8; 2]
where
    T: Into<[u8; 2]>,
{
    let _ = Device::Gpu;
}
